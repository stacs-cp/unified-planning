# Copyright 2021-2023 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""This module defines the integers remover class."""
import operator
import time

import unified_planning as up
import unified_planning.engines as engines
from ortools.sat.python import cp_model
from bidict import bidict
from typing import Any
from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, Action, ProblemKind, Variable, Effect, EffectKind, Object, FNode, Fluent, \
    InstantaneousAction, Axiom, Parameter
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import get_fresh_name, replace_action, updated_minimize_action_costs
from typing import Optional, Iterator, OrderedDict, Union
from functools import partial
from unified_planning.shortcuts import And, Or, Equals, Not, FALSE, Iff, UserType, TRUE, ObjectExp, Plus, Int, Times, \
    Minus, GT, GE, Bool, BoolType
from typing import List, Dict, Tuple

class CPSolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collects all unique solutions from CP-SAT solver."""

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solutions = []
        self.__seen = set()  # To detect duplicates

    def on_solution_callback(self):
        solution = {str(v): self.value(v) for v in self.__variables}
        sol_tuple = tuple(sorted(solution.items()))

        if sol_tuple not in self.__seen:
            self.__seen.add(sol_tuple)
            self.__solutions.append(solution)

    @property
    def solutions(self) -> list[dict[str, int]]:
        return self.__solutions

class IntegersRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes bounded integers from a planning problem.

    Converts integer fluents to object-typed fluents where objects represent numeric values (n0, n1, n2, ...).
    Integer arithmetic and comparisons are handled by enumerating possible value combinations.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGERS_REMOVING)
        self._domains: Dict[str, Tuple[int, int]] = {}
        self._number_objects_cache: Dict[int, FNode] = {}
        self._static_fluents: Dict[int, FNode] = {}
        self._action_static_fluents: Dict[int, FNode] = {}

    @property
    def name(self):
        return "irm"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        #supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_parameters("REAL_ACTION_PARAMETERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_fluents_type("DERIVED_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_conditions_kind("COUNTING")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        supported_kind.set_time("CONTINUOUS_TIME")
        supported_kind.set_time("DISCRETE_TIME")
        supported_kind.set_time("INTERMEDIATE_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("EXTERNAL_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("TIMED_EFFECTS")
        supported_kind.set_time("TIMED_GOALS")
        supported_kind.set_time("DURATION_INEQUALITIES")
        supported_kind.set_time("SELF_OVERLAPPING")
        supported_kind.set_expression_duration("STATIC_FLUENTS_IN_DURATIONS")
        supported_kind.set_expression_duration("FLUENTS_IN_DURATIONS")
        supported_kind.set_expression_duration("INT_TYPE_DURATIONS")
        supported_kind.set_expression_duration("REAL_TYPE_DURATIONS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_constraints_kind("STATE_INVARIANTS")
        supported_kind.set_constraints_kind("TRAJECTORY_CONSTRAINTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_actions_cost_kind("STATIC_FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_quality_metrics("OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("TEMPORAL_OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_oversubscription_kind("INT_NUMBERS_IN_OVERSUBSCRIPTION")
        supported_kind.set_oversubscription_kind("REAL_NUMBERS_IN_OVERSUBSCRIPTION")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= IntegersRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INTEGERS_REMOVING

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("INT_FLUENTS")
        new_kind.unset_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        return new_kind

    # Operators that can appear inside arithmetic expressions
    ARITHMETIC_OPS = {
        OperatorKind.PLUS: 'plus',
        OperatorKind.MINUS: 'minus',
        OperatorKind.DIV: 'div',
        OperatorKind.TIMES: 'mult',
    }

    # ==================== METHODS ====================

    def _get_number_object(self, problem: Problem, value: int) -> FNode:
        """Get or create object representing numeric value (e.g., n5 for 5)."""
        if value in self._number_objects_cache:
            return self._number_objects_cache[value]

        new_object = Object(f'n{value}', UserType('Number'))
        problem.add_object(new_object)
        new_object_expression = ObjectExp(new_object)
        self._number_objects_cache[value] = new_object_expression
        return new_object_expression

    def _is_value_in_bounds(self, fluent_name: str, value: int) -> bool:
        """Check if a value is within the bounds of a fluent's domain."""
        if fluent_name not in self._domains:
            return True
        lb, ub = self._domains[fluent_name]
        return lb <= value <= ub

    def _has_arithmetic(self, node: FNode) -> bool:
        """Check if expression contains arithmetic operations."""
        if (node.node_type in self.ARITHMETIC_OPS or
                node.is_le() or node.is_lt()):
            return True
        return any(self._has_arithmetic(arg) for arg in node.args)

    def _find_integer_fluents(self, node: FNode) -> dict[str, list[int]]:
        """Extract all integer fluents and their domains from expression."""
        fluents = {}
        if node.is_fluent_exp():
            if not node.fluent().type.is_int_type():
                return fluents

            fluent_type = node.fluent().type
            fluents[node.fluent().name] = list(range(
                fluent_type.lower_bound,
                fluent_type.upper_bound + 1
            ))
            return fluents
        for arg in node.args:
            fluents.update(self._find_integer_fluents(arg))
        return fluents

    # ==================== CP-SAT Constraint Solving ====================

    def _register_usertype_mapping(self, user_type, objects):
        """Register object to type indexes mapping"""
        if not hasattr(self, '_object_to_index'):
            self._object_to_index = {}
        if not hasattr(self, '_index_to_object'):
            self._index_to_object = {}

        for idx, obj in enumerate(objects):
            self._object_to_index[(user_type, obj)] = idx
            self._index_to_object[(user_type, idx)] = obj

    def _add_cp_constraints(self, problem: Problem, node: FNode, variables: bidict, model: cp_model.CpModel):
        """Recursively build CP-SAT constraints from expression tree."""
        # Constants
        if node.is_constant():
            return model.new_constant(node.constant_value())

        # Fluents
        if node.is_fluent_exp():
            if node in variables:
                return variables[node]
            fluent = node.fluent()

            if fluent.type.is_int_type():
                var = model.new_int_var(
                    fluent.type.lower_bound,
                    fluent.type.upper_bound,
                    str(node)
                )
            elif fluent.type.is_user_type():
                # Obtenir tots els objectes d'aquest tipus
                objects = list(problem.objects(fluent.type))
                # Crear variable entera amb domini [0, len(objects)-1]
                var = model.new_int_var(0, len(objects) - 1, str(node))
                # Guardar mapping objecte -> índex
                if not hasattr(self, '_object_to_index'):
                    self._object_to_index = {}
                for idx, obj in enumerate(objects):
                    self._object_to_index[(fluent.type, obj)] = idx
            else:
                var = model.new_bool_var(str(node))

            variables[node] = var
            return var

        # Parameters
        if node.is_parameter_exp():
            if node in variables:
                return variables[node]

            param = node.parameter()
            assert param.type.is_user_type(), f"Parameter type {param.type} not supported"
            # Obtenir tots els objectes d'aquest tipus
            objects = list(problem.objects(param.type))
            if not objects:
                # Si no hi ha objectes, pot ser un tipus sense instàncies
                var = model.new_int_var(0, 0, str(node))
            else:
                var = model.new_int_var(0, len(objects) - 1, str(node))
                # Guardar mapping
                self._register_usertype_mapping(param.type, objects)

            variables[node] = var
            return var

        # Equality
        if node.is_equals():
            left = node.arg(0)
            right = node.arg(1)
            if left.type.is_user_type():
                left_var = self._add_cp_constraints(problem, left, variables, model)
                # Si right és un objecte/paràmetre constant
                if right.is_object_exp():
                    obj = right.object()
                    idx = self._object_to_index.get((left.type, obj))
                    if idx is not None:
                        eq_var = model.new_bool_var(f"eq_{id(node)}")
                        model.add(left_var == idx).only_enforce_if(eq_var)
                        model.add(left_var != idx).only_enforce_if(eq_var.negated())
                        return eq_var

                # Si right és un altre fluent/variable
                else:
                    right_var = self._add_cp_constraints(problem, right, variables, model)
                    eq_var = model.new_bool_var(f"eq_{id(node)}")
                    model.add(left_var == right_var).only_enforce_if(eq_var)
                    model.add(left_var != right_var).only_enforce_if(eq_var.negated())
                    return eq_var
            else:
                left = self._add_cp_constraints(problem, node.arg(0), variables, model)
                right = self._add_cp_constraints(problem, node.arg(1), variables, model)
                eq_var = model.new_bool_var(f"eq_{id(node)}")
                model.add(left == right).only_enforce_if(eq_var)
                model.add(left != right).only_enforce_if(eq_var.negated())
                return eq_var

        # AND
        if node.is_and():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            and_var = model.new_bool_var(f"and_{id(node)}")
            # Això crea la implicació: and_var == true <=> tots els child_vars == true
            model.add_bool_and(*child_vars).only_enforce_if(and_var)
            # També necessitem: si algun child és false, and_var és false
            for child in child_vars:
                model.add_implication(and_var, child)
            return and_var

        # OR
        if node.is_or():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            or_var = model.new_bool_var(f"or_{id(node)}")
            # or_var == true <=> almenys un child_var == true
            model.add_bool_or(*child_vars).only_enforce_if(or_var)
            # Si or_var és false, tots els children són false
            for child in child_vars:
                model.add_implication(child, or_var)
            return or_var

        # Implies: A -> B  equivalent to  (not A) or B
        if node.is_implies():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)

            impl_var = model.new_bool_var(f"impl_{id(node)}")

            # impl_var == true <=> (not left) or right
            # Equivalent: impl_var == true <=> left == false OR right == true

            # Si impl_var és true: not(left) or right ha de ser true
            model.add_bool_or(left.negated(), right).only_enforce_if(impl_var)

            # Si impl_var és false: left ha de ser true AND right ha de ser false
            model.add(left == 1).only_enforce_if(impl_var.negated())
            model.add(right == 0).only_enforce_if(impl_var.negated())
            return impl_var

        # Not
        if node.is_not():
            inner_var = self._add_cp_constraints(problem, node.arg(0), variables, model)
            not_var = model.new_bool_var(f"not_{id(node)}")
            # not_var és la negació d'inner_var
            model.add(not_var == (1 - inner_var))
            return not_var

        # Comparisons and arithmetic
        if node.is_lt():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)

            lt_var = model.new_bool_var(f"lt_{id(node)}")
            model.add(left < right).only_enforce_if(lt_var)
            model.add(left >= right).only_enforce_if(lt_var.negated())

            return lt_var

        if node.is_le():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)

            le_var = model.new_bool_var(f"le_{id(node)}")
            model.add(left <= right).only_enforce_if(le_var)
            model.add(left > right).only_enforce_if(le_var.negated())

            return le_var

        # Arithmetic - retorna expressions lineals
        if node.is_plus():
            args = [self._add_cp_constraints(problem, arg, variables, model)
                    for arg in node.args]
            return sum(args)

        if node.is_minus():
            args = [self._add_cp_constraints(problem, arg, variables, model)
                    for arg in node.args]
            if len(args) == 1:
                return -args[0]
            return args[0] - sum(args[1:])

        if node.is_times():
            args = [self._add_cp_constraints(problem, arg, variables, model)
                    for arg in node.args]
            result = args[0]
            for arg in args[1:]:
                # CP-SAT requereix multiplicació explícita
                temp = model.new_int_var(arg.type.lower_bound, arg.type.upper_bound, f"mult_{id(node)}")
                model.add_multiplication_equality(temp, result, arg)
                result = temp
            return result

        raise NotImplementedError(f"Node type {node.node_type} not implemented in CP-SAT")

    def _solutions_to_dnf(self, new_problem: Problem, solutions: List[dict], variables: bidict) -> Optional[FNode]:
        """Convert CP-SAT solutions to DNF formula, handling UserTypes."""
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        or_clauses = []
        for solution in solutions:
            and_clauses = []

            for var_str, value in solution.items():
                fnode = var_str_to_fnode.get(var_str)
                if not fnode:
                    continue
                if fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                    if fluent.type.is_int_type():
                        and_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                    elif fluent.type.is_user_type():
                        obj = self._get_object_from_index(fluent.type, value)
                        if obj:
                            and_clauses.append(Equals(new_fluent, ObjectExp(obj)))
                    elif fluent.type.is_bool_type():
                        if value == 1:
                            and_clauses.append(new_fluent)
                        else:
                            and_clauses.append(Not(new_fluent))
                elif fnode.is_parameter_exp():
                    param = fnode.parameter()
                    if param.type.is_user_type():
                        obj = self._get_object_from_index(param.type, value)
                        if obj:
                            and_clauses.append(Equals(fnode, ObjectExp(obj)))

            if and_clauses:
                or_clauses.append(And(and_clauses) if len(and_clauses) > 1 else and_clauses[0])
        if not or_clauses:
            return None
        return Or(or_clauses).simplify() if len(or_clauses) > 1 else or_clauses[0].simplify()

    def _add_effects_dnf_mode(
            self,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            variables: bidict,
            solutions: List[dict],
            normalized_effects: List[Effect]
    ) -> None:
        """Add effects for DNF mode, handling arithmetic with conditional effects if needed."""
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        for effect in normalized_effects:
            if effect.is_increase() or effect.is_decrease():
                for new_effect in self._transform_increase_decrease_effect(effect, new_problem):
                    new_action.add_effect(new_effect.fluent, new_effect.value,
                                          new_effect.condition, new_effect.forall)

            elif effect.value.node_type in self.ARITHMETIC_OPS:
                # Arithmetic effect - need conditional effects per solution
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                base_cond = self._transform_node(problem, new_problem, effect.condition)
                if base_cond is None:
                    base_cond = TRUE()

                # Group solutions by result value
                result_to_conditions = {}
                for solution in solutions:
                    evaluated = self._evaluate_with_solution(
                        new_problem, effect.value, solution
                    )
                    if evaluated:
                        result_key = str(evaluated)
                        if result_key not in result_to_conditions:
                            result_to_conditions[result_key] = {'value': evaluated, 'solutions': []}
                        result_to_conditions[result_key]['solutions'].append(solution)

                # Create one conditional effect per unique result
                for result_key, data in result_to_conditions.items():
                    # Build condition: base_cond AND (solution1 OR solution2 OR ...)
                    solution_clauses = []
                    for sol in data['solutions']:
                        sol_clause = []
                        for var_str, value in sol.items():
                            fnode = var_str_to_fnode.get(var_str)
                            if fnode and fnode.is_fluent_exp():
                                fluent = fnode.fluent()
                                new_fl = new_problem.fluent(fluent.name)(*fnode.args)
                                if fluent.type.is_int_type():
                                    sol_clause.append(Equals(new_fl, self._get_number_object(new_problem, value)))
                        if sol_clause:
                            solution_clauses.append(And(sol_clause) if len(sol_clause) > 1 else sol_clause[0])

                    if solution_clauses:
                        solutions_or = Or(solution_clauses) if len(solution_clauses) > 1 else solution_clauses[0]
                        full_cond = And(base_cond, solutions_or).simplify() if base_cond != TRUE() else solutions_or
                        new_action.add_effect(new_fluent, data['value'], full_cond)

            else:
                # Simple assignment
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_value = self._transform_node(problem, new_problem, effect.value)
                new_cond = self._transform_node(problem, new_problem, effect.condition)
                if new_cond is None:
                    new_cond = TRUE()
                if new_fluent and new_value:
                    new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)

    def _solve_with_cp_sat(self, variables, cp_model_obj):
        """Use CP-SAT solver to enumerate valid value assignments."""
        # Solve
        solver = cp_model.CpSolver()
        collector = CPSolutionCollector(list(variables.values()))
        solver.parameters.enumerate_all_solutions = True
        status = solver.solve(cp_model_obj, collector)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        solutions = collector.solutions
        return solutions

    # ==================== NODE TRANSFORMATION ====================

    def _transform_node(
            self, old_problem: Problem, new_problem: Problem, node: FNode
    ) -> Union[Union[None, str, FNode], Any]:
        """Transform expression node to use Number objects instead of integers."""
        em = new_problem.environment.expression_manager

        # Integer constants become Number objects
        if node.is_int_constant():
            return self._get_number_object(new_problem, node.constant_value())

        # Integer fluents
        if node.is_fluent_exp():
            if node.fluent().type.is_int_type():
                return new_problem.fluent(node.fluent().name)(*node.args)
            return node

        # Other terminals
        if node.is_object_exp() or node.is_constant() or node.is_parameter_exp():
            return node

        # Check for arithmetic operations
        if node.node_type in self.ARITHMETIC_OPS:
            raise UPProblemDefinitionError(
                f"Arithmetic operation {self.ARITHMETIC_OPS[node.node_type]} "
                f"not supported as external expression"
            )

        # Recursively transform children
        new_args = []
        for arg in node.args:
            transformed = self._transform_node(old_problem, new_problem, arg)
            if transformed is None:
                return None
            new_args.append(transformed)

        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _extract_constants_and_variables(self, node: FNode) -> tuple[int, list[tuple[int, FNode]]]:
        """
        Extreu la suma de constants i la llista de variables d'una expressió.
        Retorna (constant_sum, [variables])

        Exemples:
        - cols(i1) + 2 + 3 -> (5, [cols(i1)])
        - 10 -> (10, [])
        - cols(i1) + rows(i2) -> (0, [cols(i1), rows(i2)])
        """
        if node.is_constant():
            return node.constant_value(), []

        if node.is_plus():
            total_const = 0
            all_vars = []
            for arg in node.args:
                const_val, vars_list = self._extract_constants_and_variables(arg)
                total_const += const_val
                all_vars.extend(vars_list)
            return total_const, all_vars

        if node.is_minus() and len(node.args) == 2:
            left_const, left_vars = self._extract_constants_and_variables(node.arg(0))
            right_const, right_vars = self._extract_constants_and_variables(node.arg(1))

            # a - b: neguem els coeficients del costat dret
            negated_right_vars = [(-coef, var) for coef, var in right_vars]
            return left_const - right_const, left_vars + negated_right_vars

        # Si no és ni constant ni suma/resta, és una variable
        return 0, [(1, node)]

    def _build_expression(self, terms: list[tuple[int, FNode]]) -> FNode:
        """Construeix una expressió a partir d'una llista de termes"""
        if not terms:
            return Int(0)
        # Agrupa termes per variable per combinar coeficients
        var_to_coef = {}
        for coef, var in terms:
            var_key = id(var)  # Usa ID per comparar variables
            if var_key in var_to_coef:
                var_to_coef[var_key] = (var_to_coef[var_key][0] + coef, var)
            else:
                var_to_coef[var_key] = (coef, var)

        # Construeix l'expressió final
        result_terms = []
        for coef, var in var_to_coef.values():
            if coef == 0:
                continue  # Elimina termes amb coeficient 0
            elif coef == 1:
                result_terms.append(var)
            elif coef == -1:
                result_terms.append(Minus(Int(0), var))
            elif coef > 0:
                result_terms.append(Times(Int(coef), var))
            else:  # coef < 0
                result_terms.append(Minus(Int(0), Times(Int(-coef), var)))

        if not result_terms:
            return Int(0)
        elif len(result_terms) == 1:
            return result_terms[0].simplify()
        else:
            return Plus(result_terms).simplify()

    def _negate_expr(self, problem: Problem, expr: FNode) -> FNode:
        """Nega una expressió distribuint la negació recursivament"""
        # Si és (0 - x), retorna x
        if expr.node_type == OperatorKind.MINUS and expr.arg(0).is_constant():
            if expr.arg(0).constant_value() == 0:
                return expr.arg(1)
        if expr.node_type in self.constants:
            return Minus(Int(0), expr)
        #if expr.node_type == OperatorKind.PLUS and len(expr.args) == 2 and expr.arg(1).node_type == OperatorKind.MINUS and \
        #    expr.arg(1).arg(0).constant_value() == 0:
        #    print("AAAAA", expr, "to", Minus(expr.arg(1).arg(1), expr.arg(0)))
        #    return Minus(expr.arg(1).arg(1), expr.arg(0))
        #if expr.node_type == OperatorKind.PLUS and len(expr.args) == 2 and expr.arg(0).node_type == OperatorKind.MINUS and \
        #    expr.arg(0).arg(0).constant_value() == 0:
        #    print("AAAAA", expr, "to", Minus(expr.arg(0).arg(1), expr.arg(1)))
        #    return Minus(expr.arg(1), expr.arg(0).arg(1))
        # Recursivament negar els arguments
        if expr.args:
            em = problem.environment.expression_manager
            new_args = [self._negate_expr(problem, a) for a in expr.args]
            return em.create_node(expr.node_type, tuple(new_args)).simplify()
        else:
            raise UPProblemDefinitionError("WWWWWWW")

    def _normalise_comparison(self, problem: Problem, op_type, left: FNode, right: FNode) -> FNode:
        # Extreu constants i variables de cada costat
        left_const, left_vars = self._extract_constants_and_variables(left)
        right_const, right_vars = self._extract_constants_and_variables(right)

        # Mou totes les variables a l'esquerra: left_vars - right_vars
        all_vars = left_vars + [(-coef, var) for coef, var in right_vars]

        # Mou totes les constants a la dreta: right_const - left_const
        all_const = right_const - left_const

        # Construeix els costats normalitzats
        new_left = self._build_expression(all_vars)
        new_right = Int(all_const)

        # Constant folding: si tots dos són constants, avalua directament
        if new_left.is_constant():
            left_val = new_left.constant_value()
            right_val = new_right.constant_value()

            if op_type == OperatorKind.EQUALS:
                return TRUE() if left_val == right_val else FALSE()
            elif op_type == OperatorKind.LE:
                return TRUE() if left_val <= right_val else FALSE()
            elif op_type == OperatorKind.LT:
                return TRUE() if left_val < right_val else FALSE()

        # If the costant side is negative and more variables have negative coef then switch sides
        num_negative = sum(1 for coef, _ in all_vars if coef < 0)
        num_positive = sum(1 for coef, _ in all_vars if coef > 0)
        switch = num_negative >= num_positive

        em = problem.environment.expression_manager
        if switch:
            new_left_switched = self._negate_expr(problem, new_right).simplify()
            new_right_switched = self._negate_expr(problem, new_left).simplify()
            return em.create_node(op_type, (new_left_switched, new_right_switched)).simplify()

        return em.create_node(op_type, (new_left, new_right)).simplify()

    # ==================== EFFECT TRANSFORMATION ====================

    def _transform_increase_decrease_effect(
            self,
            effect,
            new_problem: Problem,
    ) -> Iterator[Effect]:
        """Convert increase/decrease effects to conditional assignments."""
        fluent = effect.fluent.fluent()
        lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
        new_fluent = new_problem.fluent(fluent.name)(*effect.fluent.args)

        # Calculate the valid bounds
        try:
            int_value = effect.value.constant_value()
        except:
            int_value = effect.value

        if effect.is_increase():
            # Per increase: valor final = i + delta, per tant i ha d'estar en [lb, ub-delta]
            valid_range = range(max(lb, lb), min(ub - int_value, ub) + 1) if isinstance(int_value, int) else range(lb, ub + 1)
        else:
            # Per decrease: valor final = i - delta, per tant i ha d'estar en [lb+delta, ub]
            valid_range = range(max(lb + int_value, lb), min(ub, ub) + 1) if isinstance(int_value, int) else range(lb, ub + 1)

        returned = set()

        for i in valid_range:
            next_val = (i + int_value) if effect.is_increase() else (i - int_value)
            try:
                next_val_int = next_val.simplify().constant_value() if hasattr(next_val, 'simplify') else next_val
            except:
                continue

            old_obj = self._get_number_object(new_problem, i)
            new_obj = self._get_number_object(new_problem, next_val_int)
            new_effect = Effect(
                new_fluent,
                new_obj,
                And(Equals(new_fluent, old_obj), effect.condition).simplify(),
                EffectKind.ASSIGN,
                effect.forall
            )
            if new_effect not in returned:
                yield new_effect
                returned.add(new_effect)

    # ==================== ACTION TRANSFORMATION ====================

    def _simplify_arithmetic_expression(self, problem: Problem, node: FNode) -> FNode:
        """
        Simplify arithmetic expressions by moving all constants to one side and variables to the other.
        """
        if node.is_constant() or node.is_variable_exp() or node.is_timing_exp() or node.is_parameter_exp() or node.is_fluent_exp():
            return node
        if (node.is_equals() and node.arg(0).type.is_int_type()) or node.is_le() or node.is_lt():
            left = self._simplify_arithmetic_expression(problem, node.arg(0))
            right = self._simplify_arithmetic_expression(problem, node.arg(1))

            result = self._normalise_comparison(problem, node.node_type, left, right)
            return result

        simplified_args = [self._simplify_arithmetic_expression(problem, arg) for arg in node.args]
        em = problem.environment.expression_manager
        return em.create_node(node.node_type, tuple(simplified_args))

    constants = {
        OperatorKind.BOOL_CONSTANT,
        OperatorKind.INT_CONSTANT,
        OperatorKind.ARRAY_CONSTANT,
        OperatorKind.REAL_CONSTANT,
        OperatorKind.FLUENT_EXP,
        OperatorKind.OBJECT_EXP,
        OperatorKind.PARAM_EXP,
        OperatorKind.VARIABLE_EXP
    }

    def _apply_negation(self, node):
        """
        Applies negation to expression node.
        """
        if node.node_type in self.constants:
            return Not(node)
        elif node.node_type == OperatorKind.LE:
            return GT(node.arg(0), node.arg(1))
        elif node.node_type == OperatorKind.LT:
            return GE(node.arg(0), node.arg(1))
        elif node.node_type == OperatorKind.AND:
            return Or(Not(a) for a in node.args)
        elif node.node_type == OperatorKind.OR:
            return And(Not(a) for a in node.args)
        elif node.node_type == OperatorKind.NOT:
            return node.arg(0)
        else:
            raise UPProblemDefinitionError(f"Negation not supported for {node.node_type}")

    def _to_nnf(self, new_problem, node):
        """
        Convert expression node to nnf form.
        """
        em = new_problem.environment.expression_manager
        if node.is_fluent_exp():
            if node in self._action_static_fluents:
                return self._action_static_fluents[node]
            return node
        elif node.node_type in self.constants:
            return node
        elif node.node_type == OperatorKind.NOT:
            if node.arg(0).node_type in self.constants:
                new_node = Not(self._to_nnf(new_problem, node.arg(0)))
            elif node.arg(0).is_equals(): # Not Equals() remaining
                new_node = em.create_node(node.node_type, tuple([self._to_nnf(new_problem, a) for a in node.args]))
            else:
                new_node = self._to_nnf(new_problem, self._apply_negation(node.arg(0)))
        elif node.node_type == OperatorKind.IMPLIES:
            new_node = Or(self._to_nnf(new_problem, Not(node.arg(0))), self._to_nnf(new_problem, node.arg(1)))
        elif node.node_type == OperatorKind.IFF:
            new_node = And(Or(self._to_nnf(new_problem, Not(node.arg(0))), self._to_nnf(new_problem, node.arg(1))),
                       Or(self._to_nnf(new_problem, node.arg(0)), self._to_nnf(new_problem, Not(node.arg(1)))))
        else:
            new_node = em.create_node(node.node_type, tuple([self._to_nnf(new_problem, a) for a in node.args]))

        # Simplify expressions
        return self._simplify_arithmetic_expression(new_problem, new_node).simplify()

    # ---------------------------------
    # Compression Method
    # ---------------------------------

    def _create_entries(
            self,
            solutions: list[dict[str, int]],
            selected_mfi: list[tuple[frozenset, set[int]]],
            remaining_indices: list[int]
    ) -> tuple[list[dict], list[dict]]:
        """
        Create compressed entries from selected MFI.
        Subtable is stored as a list of partial tuples (valid combinations only).
        """
        if not solutions:
            return [], []

        all_variables = set(solutions[0].keys())
        entries = []

        for itemset, covered_indices in selected_mfi:
            # Get the solutions covered by this itemset
            covered_solutions = [solutions[idx] for idx in covered_indices]

            #  Find ALL variables that are constant across covered solutions
            pattern = {}
            subtable_vars = []
            #pattern_vars = set(pattern.keys())

            for var in all_variables:
                values = {sol[var] for sol in covered_solutions}
                if len(values) == 1:
                    # Constant across all covered solutions -> add to pattern
                    pattern[var] = next(iter(values))
                else:
                    # Varies -> add to subtable
                    subtable_vars.append(var)

            # Subtable variables
            subtable_vars = sorted(subtable_vars)

            # Collect unique subtable tuples
            seen = set()
            subtable_tuples = []
            for sol in covered_solutions:
                subtable_tuple = tuple((var, sol[var]) for var in subtable_vars)
                if subtable_tuple not in seen:
                    seen.add(subtable_tuple)
                    subtable_tuples.append({var: sol[var] for var in subtable_vars})

            entries.append({
                'pattern': pattern,
                'subtable_vars': subtable_vars,
                'subtable_tuples': subtable_tuples,
                'num_tuples': len(covered_indices)
            })

        # Default tuples
        default_tuples = [solutions[idx] for idx in remaining_indices]
        return entries, default_tuples

    def _select_non_overlapping_mfi(
            self,
            solutions: list[dict[str, int]],
            sorted_mfi: list[tuple[frozenset, int, int]]
    ) -> tuple[list[tuple[frozenset, set[int]]], list[int]]:
        """
        Greedily select MFI with highest area, removing overlapping ones.
        Optimized with pre-indexed solutions.
        """
        if not solutions or not sorted_mfi:
            return [], list(range(len(solutions)))

        # Pre-indexar: per cada (var, val), quines solucions el contenen?
        item_to_solutions = {}
        for idx, solution in enumerate(solutions):
            for var, val in solution.items():
                item = (var, val)
                if item not in item_to_solutions:
                    item_to_solutions[item] = set()
                item_to_solutions[item].add(idx)

        def get_coverage_fast(itemset: frozenset) -> set[int]:
            """Get coverage using pre-built index."""
            if not itemset:
                return set(range(len(solutions)))

            items = list(itemset)
            result = item_to_solutions.get(items[0], set()).copy()
            for item in items[1:]:
                result &= item_to_solutions.get(item, set())
            return result

        uncovered = set(range(len(solutions)))
        selected = []

        for itemset, freq, area in sorted_mfi:
            if not uncovered:
                break
            coverage = get_coverage_fast(itemset) & uncovered
            if len(coverage) >= 2:
                selected.append((itemset, coverage))
                uncovered -= coverage
        return selected, list(uncovered)

    def _sort_by_area(self, maximal: dict[frozenset, int]) -> list[tuple[frozenset, int, int]]:
        """
        Sort maximal itemsets by area (descending).
        """
        with_area = []
        for itemset, freq in maximal.items():
            area = len(itemset) * freq
            with_area.append((itemset, freq, area))

        # Sort by area descending, then by frequency descending (tie-breaker)
        with_area.sort(key=lambda x: (-x[2], -x[1]))
        return with_area

    # -----------------------------

    def _create_subtable_precondition(
            self,
            new_problem: Problem,
            var_str_to_fnode: dict,
            subtable_vars: List[str],
            subtable_tuples: List[dict]
    ) -> Optional[FNode]:
        """
        Create precondition for subtable variables.
        """
        if not subtable_vars or not subtable_tuples:
            return None

        # Case 1: Single variable in subtable
        if len(subtable_vars) == 1:
            var_str = subtable_vars[0]
            values = {t[var_str] for t in subtable_tuples}
            fnode = var_str_to_fnode.get(var_str)

            if not fnode:
                return None

            if fnode.is_fluent_exp():
                fluent = fnode.fluent()
                new_fluent = new_problem.fluent(fluent.name)(*fnode.args)

                if fluent.type.is_int_type():
                    lb = fluent.type.lower_bound
                    ub = fluent.type.upper_bound
                    all_values = set(range(lb, ub + 1))

                    if values == all_values:
                        # Full domain - no constraint needed!
                        return None
                    elif len(values) == 1:
                        # Single value
                        val = next(iter(values))
                        return Equals(new_fluent, self._get_number_object(new_problem, val))
                    else:
                        # Multiple values - create Or
                        or_clauses = [
                            Equals(new_fluent, self._get_number_object(new_problem, v))
                            for v in sorted(values)
                        ]
                        return Or(or_clauses)

                elif fluent.type.is_user_type():
                    # Get all objects of this type
                    all_objects = list(new_problem.objects(fluent.type))
                    all_indices = set(range(len(all_objects)))

                    if values == all_indices:
                        return None  # Full domain - no constraint needed
                    elif len(values) == 1:
                        val = next(iter(values))
                        obj = self._get_object_from_index(fluent.type, val)
                        if obj:
                            return Equals(new_fluent, ObjectExp(obj))
                    else:
                        or_clauses = []
                        for v in sorted(values):
                            obj = self._get_object_from_index(fluent.type, v)
                            if obj:
                                or_clauses.append(Equals(new_fluent, ObjectExp(obj)))
                        if or_clauses:
                            return Or(or_clauses) if len(or_clauses) > 1 else or_clauses[0]

                elif fluent.type.is_bool_type():
                    if values == {0, 1}:
                        return None  # Both values - no constraint
                    elif len(values) == 1:
                        val = next(iter(values))
                        return new_fluent if val == 1 else Not(new_fluent)

            elif fnode.is_parameter_exp():
                param = fnode.parameter()
                if param.type.is_user_type():
                    all_objects = list(new_problem.objects(param.type))
                    all_indices = set(range(len(all_objects)))
                    if values == all_indices:
                        return None
                    elif len(values) == 1:
                        val = next(iter(values))
                        obj = self._get_object_from_index(param.type, val)
                        if obj:
                            return Equals(fnode, ObjectExp(obj))
                    else:
                        or_clauses = []
                        for v in sorted(values):
                            obj = self._get_object_from_index(param.type, v)
                            if obj:
                                or_clauses.append(Equals(fnode, ObjectExp(obj)))
                        if or_clauses:
                            return Or(or_clauses) if len(or_clauses) > 1 else or_clauses[0]
            return None

        # Case 2: Multiple variables - need to enumerate valid combinations
        or_clauses = []
        for subtable_tuple in subtable_tuples:
            and_clauses = []
            for var_str, value in subtable_tuple.items():
                fnode = var_str_to_fnode.get(var_str)
                if not fnode:
                    continue

                if fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)

                    if fluent.type.is_int_type():
                        and_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                    elif fluent.type.is_user_type():
                        obj = self._get_object_from_index(fluent.type, value)
                        if obj:
                            and_clauses.append(Equals(new_fluent, ObjectExp(obj)))
                    elif fluent.type.is_bool_type():
                        if value == 1:
                            and_clauses.append(new_fluent)
                        else:
                            and_clauses.append(Not(new_fluent))

                elif fnode.is_parameter_exp():
                    param = fnode.parameter()
                    if param.type.is_user_type():
                        obj = self._get_object_from_index(param.type, value)
                        if obj:
                            and_clauses.append(Equals(fnode, ObjectExp(obj)))

            if and_clauses:
                if len(and_clauses) == 1:
                    or_clauses.append(and_clauses[0])
                else:
                    or_clauses.append(And(and_clauses))
        if not or_clauses:
            return None
        return Or(or_clauses) if len(or_clauses) > 1 else or_clauses[0]

    def _get_value_from_solution(
            self,
            node: FNode,
            solution: dict,
            var_str_to_fnode: dict
    ) -> Optional[int]:
        """
        Get the value of a node from the solution.
        Returns None if not found in solution.
        """
        var_str = str(node)

        # Direct lookup
        if var_str in solution:
            return solution[var_str]

        # Check if it's a fluent/parameter in variables
        if node.is_fluent_exp() or node.is_parameter_exp():
            if var_str in solution:
                return solution[var_str]

        # Constant
        if node.is_constant():
            return node.constant_value()

        # Object - get index
        if node.is_object_exp():
            obj = node.object()
            if hasattr(self, '_object_to_index'):
                return self._object_to_index.get((obj.type, obj))

        return None

    def _build_tuple_condition(
            self,
            new_problem: Problem,
            var_str_to_fnode: dict,
            subtable_tuple: dict
    ) -> Optional[FNode]:
        """Build a condition for a subtable tuple."""
        conditions = []
        for var_str, value in subtable_tuple.items():
            fnode = var_str_to_fnode.get(var_str)
            if not fnode:
                continue
            if fnode.is_fluent_exp():
                fluent = fnode.fluent()
                new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                if fluent.type.is_int_type():
                    conditions.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                elif fluent.type.is_user_type():
                    obj = self._get_object_from_index(fluent.type, value)
                    if obj:
                        conditions.append(Equals(new_fluent, ObjectExp(obj)))
                elif fluent.type.is_bool_type():
                    if value == 1:
                        conditions.append(new_fluent)
                    else:
                        conditions.append(Not(new_fluent))
            elif fnode.is_parameter_exp():
                param = fnode.parameter()
                if param.type.is_user_type():
                    obj = self._get_object_from_index(param.type, value)
                    if obj:
                        conditions.append(Equals(fnode, ObjectExp(obj)))
        if not conditions:
            return None
        return And(conditions) if len(conditions) > 1 else conditions[0]

    def _add_effects_for_entry(
            self,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            variables: bidict,
            entry: dict,
            normalized_effects: List[Effect]
    ) -> None:
        """
        Add effects for a compressed entry (or full solution if subtable is empty).
        Propagates pattern values to simplify conditional effects.
        For effects depending on subtable variables, creates conditional effects.
        """
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}
        pattern = entry['pattern']
        subtable_vars = entry.get('subtable_vars', [])
        subtable_tuples = entry.get('subtable_tuples', [])

        for effect in normalized_effects:
            # Evaluate condition with pattern
            if effect.condition != TRUE():
                cond_result = self._evaluate_with_solution(new_problem, effect.condition, pattern)
                # Skip effect if condition is definitely False
                if cond_result == FALSE():
                    continue
            else:
                cond_result = TRUE()

            if effect.is_increase() or effect.is_decrease():
                for new_effect in self._transform_increase_decrease_effect(effect, new_problem):
                    final_cond = TRUE() if cond_result == TRUE() else self._transform_node(problem, new_problem,
                                                                                  new_effect.condition)
                    if final_cond is None:
                        final_cond = TRUE()
                    new_action.add_effect(new_effect.fluent, new_effect.value, final_cond, new_effect.forall)

            elif effect.value.node_type in self.ARITHMETIC_OPS:
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)

                # Check if effect depends on subtable variables
                effect_vars = self._get_variables_in_expression(effect.value)
                condition_vars = self._get_variables_in_expression(effect.condition)
                all_effect_vars = effect_vars | condition_vars
                depends_on_subtable = subtable_vars and any(v in subtable_vars for v in all_effect_vars)

                if not depends_on_subtable or len(subtable_tuples) <= 1:
                    # Single value case
                    full_solution = dict(pattern)
                    if subtable_tuples:
                        full_solution.update(subtable_tuples[0])

                    value_result = self._evaluate_with_solution(new_problem, effect.value, full_solution)
                    if value_result is not None:
                        final_cond = TRUE() if cond_result == TRUE() else self._transform_node(problem, new_problem,
                                                                                      effect.condition)
                        if final_cond is None:
                            final_cond = TRUE()
                        new_action.add_effect(new_fluent, value_result, final_cond, effect.forall)
                else:
                    # Multiple subtable tuples - create conditional effects per unique result
                    result_to_tuples = {}
                    for subtable_tuple in subtable_tuples:
                        full_solution = dict(pattern)
                        full_solution.update(subtable_tuple)

                        full_cond_result = self._evaluate_with_solution(new_problem, effect.condition, full_solution)
                        if full_cond_result is False or full_cond_result == FALSE():
                            continue

                        value_result = self._evaluate_with_solution(new_problem, effect.value, full_solution)
                        if value_result is not None:
                            result_key = str(value_result)
                            if result_key not in result_to_tuples:
                                result_to_tuples[result_key] = {'value': value_result, 'tuples': []}
                            result_to_tuples[result_key]['tuples'].append(subtable_tuple)

                    for data in result_to_tuples.values():
                        tuple_conditions = [
                            self._build_tuple_condition(new_problem, var_str_to_fnode, t)
                            for t in data['tuples']
                        ]
                        tuple_conditions = [c for c in tuple_conditions if c]

                        if tuple_conditions:
                            subtable_cond = Or(tuple_conditions) if len(tuple_conditions) > 1 else tuple_conditions[0]
                            new_action.add_effect(new_fluent, data['value'], subtable_cond, effect.forall)

            else:
                # Simple assignment
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_value = self._transform_node(problem, new_problem, effect.value)

                if new_fluent is None or new_value is None:
                    continue

                final_cond = TRUE() if cond_result == TRUE() else self._transform_node(problem, new_problem, effect.condition)
                if final_cond is None:
                    final_cond = TRUE()

                new_action.add_effect(new_fluent, new_value, final_cond, effect.forall)

    def _add_effects_for_solution(
            self,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            variables: bidict,
            solution: dict,
            normalized_effects: List[Effect]
    ) -> None:
        """
        Add effects for a fully instantiated solution.
        This is just a wrapper around _add_effects_for_entry with empty subtable.
        """
        entry = {
            'pattern': solution,
            'subtable_vars': [],
            'subtable_tuples': []
        }
        self._add_effects_for_entry(
            new_action, problem, new_problem, variables, entry, normalized_effects
        )

    def _compute_maximal_itemsets_apriori(
            self,
            solutions: list[dict[str, int]],
            min_support: int = 2,
            max_itemset_size: int = None
    ) -> dict[frozenset, int]:
        """
        Compute ONLY maximal frequent itemsets directly.
        """
        if not solutions:
            return {}

        variables = list(solutions[0].keys())
        num_vars = len(variables)

        if max_itemset_size is None:
            max_itemset_size = num_vars - 1

        # Guardem itemsets per nivell
        levels = {}

        # Level 1
        level_1 = {}
        for var in variables:
            value_counts = {}
            for sol in solutions:
                val = sol[var]
                item = frozenset([(var, val)])
                value_counts[item] = value_counts.get(item, 0) + 1
            for item, count in value_counts.items():
                if count >= min_support:
                    level_1[item] = count

        if not level_1:
            return {}

        levels[1] = level_1
        current_level = level_1
        k = 2

        while current_level and k <= max_itemset_size:
            candidates = self._apriori_generate_candidates(current_level, k)

            if not candidates:
                break

            next_level = {}
            for candidate in candidates:
                count = sum(1 for sol in solutions
                            if all(sol.get(var) == val for var, val in candidate))
                if count >= min_support:
                    next_level[candidate] = count

            if not next_level:
                break

            levels[k] = next_level
            current_level = next_level
            k += 1

        # Ara trobem maximals: un itemset és maximal si cap superset és freqüent
        max_level = max(levels.keys())
        maximal = dict(levels[max_level])  # El nivell més alt sempre és maximal

        # Per cada nivell inferior
        for level in range(max_level - 1, 0, -1):
            for itemset, freq in levels[level].items():
                # Comprovar si té algun superset freqüent al nivell superior
                has_frequent_superset = False

                if level + 1 in levels:
                    for larger in levels[level + 1]:
                        if itemset < larger:
                            has_frequent_superset = True
                            break

                if not has_frequent_superset:
                    maximal[itemset] = freq
        return maximal

    def _compute_min_support(self, num_solutions: int, num_variables: int) -> int:
        """
        Adaptive min_support based on problem size.

        Intuition: With more variables, we need higher support to find
        meaningful patterns (not just noise).
        """
        # Base: sqrt del nombre de solucions
        base = max(2, int(num_solutions ** 0.5))

        # Ajustar segons variables: més variables → més suport
        factor = 1 + (num_variables / 20)  # Factor creix amb variables

        min_support = max(2, int(base / factor))
        return min_support

    def _get_object_from_index(self, user_type, index):
        """Get object corresponding to an index for a UserType."""
        if hasattr(self, '_index_to_object'):
            return self._index_to_object.get((user_type, index))
        return None

    def _create_precondition_from_variable(
            self,
            fnode: FNode,
            value: int,
            problem: Problem,
            new_problem: Problem
    ) -> Optional[FNode]:
        """
        Create a precondition from a variable and its value.
        Handles int fluents, UserType fluents, bool fluents, and parameters.
        """
        if fnode.is_fluent_exp():
            fluent = fnode.fluent()
            new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
            if fluent.type.is_int_type():
                num_obj = self._get_number_object(new_problem, value)
                return Equals(new_fluent, num_obj)
            elif fluent.type.is_user_type():
                obj = self._get_object_from_index(fluent.type, value)
                if obj:
                    return Equals(new_fluent, ObjectExp(obj))
            elif fluent.type.is_bool_type():
                if value == 1:
                    return new_fluent
                else:
                    return Not(new_fluent)

        elif fnode.is_parameter_exp():
            param = fnode.parameter()
            assert param.type.is_user_type(), "param type is not UserType"
            obj = self._get_object_from_index(param.type, value)
            if obj:
                return Equals(fnode, ObjectExp(obj))

        return None

    def _create_instantiated_actions(
            self,
            old_action: Action,
            problem: Problem,
            new_problem: Problem,
            params: OrderedDict,
            solutions: List[dict],
            variables: bidict,
            normalized_effects: List[Effect],
            compression_mode: str
    ) -> List[Action]:
        """
        Create instantiated actions from solutions.
        compression_mode:
          - "naive": One action per solution
          - "mfi": Compress with maximal frequent itemsets (MFI)
        """
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}
        new_actions = []

        # ===== COMPRESSION =====
        if compression_mode == "mfi" and len(solutions) > 1:
            min_support = self._compute_min_support(len(solutions), len(variables))
            maximal = self._compute_maximal_itemsets_apriori(
                solutions,
                min_support=min_support,
                max_itemset_size=len(variables)-1
            )
            if maximal:
                sorted_mfi = self._sort_by_area(maximal)
                selected, remaining = self._select_non_overlapping_mfi(solutions, sorted_mfi)
                entries, default_tuples = self._create_entries(solutions, selected, remaining)
            else:
                entries = []
                default_tuples = solutions
        else:
            entries = []
            default_tuples = solutions

        # Crear accions per cada entry
        for idx, entry in enumerate(entries):
            action_name = f"{old_action.name}_e{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)

            for var_str, value in entry['pattern'].items():
                fnode = var_str_to_fnode.get(var_str)
                if not fnode:
                    continue
                precond = self._create_precondition_from_variable(
                    fnode, value, problem, new_problem
                )
                if precond:
                    new_action.add_precondition(precond)

            # Add subtable constraint
            subtable_constraint = self._create_subtable_precondition(
                new_problem, var_str_to_fnode, entry['subtable_vars'], entry['subtable_tuples']
            )
            if subtable_constraint:
                new_action.add_precondition(subtable_constraint)

            # Add effects
            self._add_effects_for_entry(
                new_action, problem, new_problem, variables, entry, normalized_effects
            )
            new_actions.append(new_action)

        # Crear accions per cada default tuple (no compressed)
        for idx, solution in enumerate(default_tuples):
            action_name = f"{old_action.name}_d{idx}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)
            # Add preconditions
            for var_str, value in solution.items():
                fnode = var_str_to_fnode.get(var_str)
                if not fnode:
                    continue
                precond = self._create_precondition_from_variable(fnode, value, problem, new_problem)
                if precond:
                    new_action.add_precondition(precond)

            # Add effects
            self._add_effects_for_solution(
                new_action, problem, new_problem, variables, solution, normalized_effects
            )
            new_actions.append(new_action)

        return new_actions

    def _add_effect_bounds_constraints(
            self,
            problem: Problem,
            effects: List[Effect],
            variables: bidict,
            model: cp_model.CpModel
    ) -> None:
        """
        Add constraints to ensure all arithmetic effects produce in-bounds values.

        For unconditional effects: lb <= expr <= ub
        For conditional effects: condition => (lb <= expr <= ub)
        """
        for effect in effects:
            # Only handle arithmetic effects
            if effect.value.node_type not in self.ARITHMETIC_OPS:
                continue

            # Get target fluent bounds
            target_fluent = effect.fluent.fluent()
            if not target_fluent.type.is_int_type():
                continue

            lb = target_fluent.type.lower_bound
            ub = target_fluent.type.upper_bound

            # Build CP expression for the effect value
            effect_expr = self._add_cp_constraints(problem, effect.value, variables, model)

            # Check if effect is conditional
            is_conditional = (effect.condition is not None and
                              not effect.condition.is_true() and
                              effect.condition != TRUE())

            if not is_conditional:
                # Unconditional effect: directly add bounds
                model.add(effect_expr >= lb)
                model.add(effect_expr <= ub)
            else:
                # Conditional effect: Implies(condition, bounds)
                condition_var = self._add_cp_constraints(problem, effect.condition, variables, model)

                # condition => (effect_expr >= lb)
                model.add(effect_expr >= lb).only_enforce_if(condition_var)
                # condition => (effect_expr <= ub)
                model.add(effect_expr <= ub).only_enforce_if(condition_var)

    def _replace_static(self, node: FNode) -> FNode:
        """Substitueix static fluents"""
        # Static fluent -> valor
        if node.is_fluent_exp() and node in self._static_fluents:
            return self._static_fluents[node]
        # Terminals
        if node.is_constant() or node.is_parameter_exp():
            return node
        if not node.args:
            return node
        # Recursiu només si cal
        new_args = [self._replace_static(arg) for arg in node.args]
        if all(n is o for n, o in zip(new_args, node.args)):
            return node  # Sense canvis
        em = node.environment.expression_manager
        return em.create_node(node.node_type, tuple(new_args))

    def _transform_action_integers(
            self, problem: Problem, new_problem: Problem, old_action: Action
    ) -> List[Action]:
        """
        Transform an action with integer arithmetic into multiple instantiated actions.

        ACTION_MODE:
          - "none": Single action, simply replaces integer values with objects (no CP-SAT)
          - "dnf": Single action, uses CP-SAT to specify the preconditions of the actions
          COMPRESSION_MODE:
            - "naive": Single dnf in the precondition
            - "mfi": Compressed dnf using mfi compression
          - "actions": Multiple actions, uses CP-SAT to specify the preconditions of the actions
          COMPRESSION_MODE:
            - "naive": One per solution
            - "mfi": One action per entry using mfi compression
        """
        ACTION_MODE = "none"  # "none" or "dnf" or "actions"
        COMPRESSION_MODE = "naive"  # "naive" or "mfi"

        params = OrderedDict(((p.name, p.type) for p in old_action.parameters))
        self._action_static_fluents = self._static_fluents

        # Replace static fluents in preconditions --
        unstatic_preconditions = []
        for precondition in old_action.preconditions:
            np = self._replace_static(precondition)
            if np.is_and():
                for new_precondition in np.args:
                    if new_precondition is not TRUE():
                        unstatic_preconditions.append(new_precondition)
            else:
                if np is not TRUE():
                    unstatic_preconditions.append(np)

        # Replace static fluents in effects
        unstatic_effects = []
        for effect in old_action.effects:
            # propagar precondicions a efectes aqui???? o mes tard - s'haura de fer mes tard amb les solucions segur
            new_value = self._replace_static(effect.value)
            new_condition = self._replace_static(effect.condition)
            unstatic_effect = Effect(effect.fluent, new_value, new_condition, effect.kind, effect.forall)
            unstatic_effects.append(unstatic_effect)

        # Check if we have arithmetic that requires CP-SAT
        has_arithmetic = any(self._has_arithmetic(p) for p in unstatic_preconditions)
        has_arithmetic_effects = any(
            effect.value.node_type in self.ARITHMETIC_OPS or effect.is_increase() or effect.is_decrease()
            for effect in unstatic_effects
        )

        # ===== NONE MODE: Direct transformation (no CP-SAT) =====
        if ACTION_MODE == "none":
            assert not has_arithmetic and not has_arithmetic_effects, \
                "The model contains arithmetic! Change the ACTION_MODE to 'dnf' or 'actions' to compile."
            action_name = f"{old_action.name}"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)

            # Transform preconditions directly
            for precondition in unstatic_preconditions:
                transformed = self._transform_node(problem, new_problem, precondition)
                if transformed and transformed != TRUE():
                    new_action.add_precondition(transformed)

            # Simplification - Get domains with preconditions
            #domains = self._extract_domains_from_preconditions(problem, unstatic_preconditions)
            # Replace effect conditions and values according to the domains
            #for effect in unstatic_effects:
            #    new_condition = self._evaluate_with_solution(new_problem, effect.condition, domains)
            #    new_value = self._evaluate_with_solution(new_problem, effect.value, domains)
            #    new_fluent = self._transform_node(problem, new_problem, effect.fluent)
            #    if new_condition is TRUE():
            #        new_action.add_effect(new_fluent, new_value, True, effect.forall)
            #        continue
            #    if new_condition is FALSE():
            #        continue
            #    new_action.add_effect(new_fluent, new_value, new_condition, effect.forall)

            # Transform effects directly
            for effect in unstatic_effects:
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_value = self._transform_node(problem, new_problem, effect.value)
                new_cond = self._transform_node(problem, new_problem, effect.condition)
                if new_cond is None:
                    new_cond = TRUE()
                if new_fluent and new_value:
                    new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)
            return [new_action]

        # ===== NEEDS CP-SAT =====

        # Clear mappings from previous action
        self._object_to_index = {}
        self._index_to_object = {}

        # Build a CP-SAT model
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        # Add normalized preconditions as constraints
        result_var = self._add_cp_constraints(problem, And(unstatic_preconditions), variables, cp_model_obj)
        cp_model_obj.add(result_var == 1)

        # Add effect bounds constraints
        self._add_effect_bounds_constraints(problem, unstatic_effects, variables, cp_model_obj)

        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        # Solve
        solutions = self._solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            return []

        # ===== DNF MODE: Direct transformation (CP-SAT - 1 action with dnf) =====
        if ACTION_MODE == "dnf":
            action_name = f"{old_action.name}_dnf"
            new_action = InstantaneousAction(action_name, _parameters=params, _env=problem.environment)

            # MFI compression
            if COMPRESSION_MODE == "mfi" and len(solutions) > 1:
                min_support = self._compute_min_support(len(solutions), len(variables))
                maximal = self._compute_maximal_itemsets_apriori(
                    solutions,
                    min_support=min_support,
                    max_itemset_size=len(variables) - 1
                )
                if maximal:
                    sorted_mfi = self._sort_by_area(maximal)
                    selected, remaining = self._select_non_overlapping_mfi(solutions, sorted_mfi)
                    entries, default_tuples = self._create_entries(solutions, selected, remaining)

                    # Convertir entries + default_tuples a fórmula DNF comprimida
                    dnf_formula = self._entries_to_formula(new_problem, variables, entries, default_tuples)
                    if dnf_formula:
                        new_action.add_precondition(dnf_formula)

                    # Per als efectes, necessitem totes les solucions per avaluar aritmètica
                    # Però podem utilitzar els dominis extrets per simplificar condicions
                    domains = self._extract_domains_from_solutions(solutions)

                    for effect in unstatic_effects:
                        if effect.is_increase() or effect.is_decrease():
                            for new_effect in self._transform_increase_decrease_effect(effect, new_problem):
                                new_action.add_effect(new_effect.fluent, new_effect.value,
                                                      new_effect.condition, new_effect.forall)

                        elif effect.value.node_type in self.ARITHMETIC_OPS:
                            # Arithmetic effect - need conditional effects per unique result
                            new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                            base_cond = self._transform_node(problem, new_problem, effect.condition)
                            if base_cond is None:
                                base_cond = TRUE()

                            # Group solutions by result value
                            result_to_solutions = {}
                            for solution in solutions:
                                evaluated = self._evaluate_with_solution(new_problem, effect.value, solution)
                                if evaluated and evaluated != effect.value:
                                    result_key = str(evaluated)
                                    if result_key not in result_to_solutions:
                                        result_to_solutions[result_key] = {'value': evaluated, 'solutions': []}
                                    result_to_solutions[result_key]['solutions'].append(solution)

                            # Create conditional effects per unique result
                            for data in result_to_solutions.values():
                                if len(data['solutions']) == len(solutions):
                                    # Totes les solucions donen el mateix resultat
                                    new_action.add_effect(new_fluent, data['value'], base_cond, effect.forall)
                                else:
                                    # Crear condició per aquest subconjunt de solucions
                                    sol_clauses = []
                                    for sol in data['solutions']:
                                        clause = []
                                        for var_str, value in sol.items():
                                            fnode = var_str_to_fnode.get(var_str)
                                            if fnode and fnode.is_fluent_exp():
                                                fluent = fnode.fluent()
                                                new_fl = new_problem.fluent(fluent.name)(*fnode.args)
                                                if fluent.type.is_int_type():
                                                    clause.append(
                                                        Equals(new_fl, self._get_number_object(new_problem, value)))
                                                elif fluent.type.is_user_type():
                                                    obj = self._get_object_from_index(fluent.type, value)
                                                    if obj:
                                                        clause.append(Equals(new_fl, ObjectExp(obj)))
                                        if clause:
                                            sol_clauses.append(And(clause) if len(clause) > 1 else clause[0])

                                    if sol_clauses:
                                        sol_cond = Or(sol_clauses) if len(sol_clauses) > 1 else sol_clauses[0]
                                        full_cond = And(base_cond,
                                                        sol_cond).simplify() if base_cond != TRUE() else sol_cond
                                        new_action.add_effect(new_fluent, data['value'], full_cond, effect.forall)

                        else:
                            # Simple assignment - simplificar amb dominis
                            new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                            new_value = self._transform_node(problem, new_problem, effect.value)
                            new_cond = self._evaluate_with_solution(new_problem, effect.condition, domains)
                            if new_cond is None or new_cond == effect.condition:
                                new_cond = self._transform_node(problem, new_problem, effect.condition)
                            if new_cond is None:
                                new_cond = TRUE()
                            if new_cond == FALSE():
                                continue
                            if new_fluent and new_value:
                                new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)

                else:
                    # No maximal itemsets found, fall back to naive DNF
                    dnf_formula = self._solutions_to_dnf(new_problem, solutions, variables)
                    if dnf_formula:
                        new_action.add_precondition(dnf_formula)
                    self._add_effects_dnf_mode(
                        new_action, problem, new_problem, variables, solutions, unstatic_effects
                    )

            # Naive
            else:
                # Convert solutions to DNF formula
                dnf_formula = self._solutions_to_dnf(new_problem, solutions, variables)
                if dnf_formula:
                    new_action.add_precondition(dnf_formula)
                # Add effects with conditional arithmetic if needed
                self._add_effects_dnf_mode(new_action, problem, new_problem, variables, solutions, unstatic_effects)

            return [new_action]

        # ===== ACTIONS MODE: Multiple actions =====
        return self._create_instantiated_actions(
            old_action, problem, new_problem, params, solutions, variables, unstatic_effects,
            compression_mode=COMPRESSION_MODE
        )

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions by grounding integer parameters."""
        new_to_old = {}
        total_original = len(problem.actions)
        total_new = 0
        for action in problem.actions:
            new_actions = self._transform_action_integers(problem, new_problem, action)
            for new_action in new_actions:
                new_problem.add_action(new_action)
                new_to_old[new_action] = action
            total_new += len(new_actions)
        print(f"\n=== ACTION TRANSFORMATION SUMMARY ===")
        print(f"Original actions: {total_original}")
        print(f"New actions: {total_new}")
        return new_to_old

    def _get_variables_in_expression(self, node: FNode) -> set:
        """
        Get all variable names (as strings) that appear in an expression.
        """
        variables = set()
        if node.is_fluent_exp():
            variables.add(str(node))
        for arg in node.args:
            variables.update(self._get_variables_in_expression(arg))
        return variables

    def _extract_domains_from_preconditions(
            self,
            problem: Problem,
            preconditions: List[FNode]
    ) -> dict:
        """
        Extreu dominis de les variables a partir de les precondicions.

        Returns: {var_name: value} o {var_name: {values}}
        """
        # Primer, trobar TOTES les variables a les precondicions
        all_vars = {}  # var_str -> var_node

        def collect_vars(node: FNode):
            """Recull totes les variables d'una expressió."""
            if node.is_fluent_exp() or node.is_parameter_exp():
                all_vars[str(node)] = node
            for arg in node.args:
                collect_vars(arg)

        for precond in preconditions:
            collect_vars(precond)

        # Per cada variable, guardem dominis
        domains = {}  # var_str -> set de valors
        excluded_values = {}  # var_str -> set de valors exclosos

        # Restriccions per processar amb propagació
        disjunction_constraints = []  # (var_str, constants, variables) de ORs
        equality_constraints = []  # (var1_str, var2_str) de var1 == var2

        def get_full_domain(node: FNode) -> set:
            """Retorna el domini complet d'una variable."""
            if node.is_fluent_exp():
                fluent = node.fluent()
                if fluent.type.is_int_type():
                    return set(range(fluent.type.lower_bound, fluent.type.upper_bound + 1))
                elif fluent.type.is_bool_type():
                    return {0, 1}
                elif fluent.type.is_user_type():
                    objects = list(problem.objects(fluent.type))
                    return set(range(len(objects)))
            elif node.is_parameter_exp():
                param = node.parameter()
                if param.type.is_user_type():
                    objects = list(problem.objects(param.type))
                    return set(range(len(objects)))
            return set()

        def get_constant_value(node: FNode) -> Optional[int]:
            """Extreu el valor constant d'un node."""
            if node.is_int_constant():
                return node.constant_value()
            if node.is_object_exp():
                obj = node.object()
                if hasattr(self, '_object_to_index'):
                    idx = self._object_to_index.get((obj.type, obj))
                    if idx is not None:
                        return idx
                objects = list(problem.objects(obj.type))
                for idx, o in enumerate(objects):
                    if o == obj:
                        return idx
            return None

        def extract_var_and_value(eq_node: FNode):
            """
            D'un Equals, extreu info:
            - Si és var == constant: (var_str, constant_value, False)
            - Si és var == var2: (var_str, var2_str, True)
            """
            if not eq_node.is_equals():
                return None, None, None

            left, right = eq_node.arg(0), eq_node.arg(1)

            left_is_var = left.is_fluent_exp() or left.is_parameter_exp()
            right_is_var = right.is_fluent_exp() or right.is_parameter_exp()

            # var == constant
            if left_is_var and not right_is_var:
                val = get_constant_value(right)
                if val is not None:
                    return str(left), val, False

            # constant == var
            if right_is_var and not left_is_var:
                val = get_constant_value(left)
                if val is not None:
                    return str(right), val, False

            # var == var2
            if left_is_var and right_is_var:
                return str(left), str(right), True

            return None, None, None

        def process_disjunction(or_node: FNode):
            """Processa una disjunció."""
            if not or_node.is_or():
                return

            var_options = {}  # var_str -> list of (value_or_var, is_var)

            for arg in or_node.args:
                if arg.is_equals():
                    var_str, val, is_var = extract_var_and_value(arg)
                    if var_str is not None:
                        if var_str not in var_options:
                            var_options[var_str] = []
                        var_options[var_str].append((val, is_var))

            for var_str, options in var_options.items():
                count = sum(1 for arg in or_node.args
                            if arg.is_equals() and extract_var_and_value(arg)[0] == var_str)

                if count == len(or_node.args):
                    constants = {val for val, is_var in options if not is_var}
                    variables = [val for val, is_var in options if is_var]

                    if not variables:
                        if var_str not in domains:
                            domains[var_str] = constants.copy()
                        else:
                            domains[var_str] &= constants
                    else:
                        disjunction_constraints.append((var_str, constants, variables))

        # Inicialitzar dominis amb domini complet
        for var_str, var_node in all_vars.items():
            domains[var_str] = get_full_domain(var_node)

        # Primera passada: processar restriccions simples
        for precond in preconditions:
            # Cas: Not(Equals(x, v)) -> excloure v de x
            if precond.is_not() and precond.arg(0).is_equals():
                var_str, val, is_var = extract_var_and_value(precond.arg(0))
                if var_str is not None and not is_var:
                    if var_str not in excluded_values:
                        excluded_values[var_str] = set()
                    excluded_values[var_str].add(val)

            # Cas: Equals(x, v) constant -> x només pot ser v
            elif precond.is_equals():
                var_str, val, is_var = extract_var_and_value(precond)
                if var_str is not None:
                    if not is_var:
                        # var == constant
                        domains[var_str] &= {val}
                    else:
                        # var == var2 -> guardar per propagació
                        equality_constraints.append((var_str, val))

            # Cas: Or(...)
            elif precond.is_or():
                process_disjunction(precond)

        # Aplicar exclusions
        for var_str, excluded in excluded_values.items():
            if var_str in domains:
                domains[var_str] -= excluded

        # Propagació fins convergència
        changed = True
        max_iterations = 10
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # Propagar equality constraints: var1 == var2 -> intersecció de dominis
            for var1, var2 in equality_constraints:
                if var1 in domains and var2 in domains:
                    intersection = domains[var1] & domains[var2]

                    if domains[var1] != intersection:
                        domains[var1] = intersection
                        changed = True

                    if domains[var2] != intersection:
                        domains[var2] = intersection
                        changed = True

            # Propagar disjunction constraints
            for var_str, constants, variables in disjunction_constraints:
                possible = constants.copy()
                for other_var in variables:
                    if other_var in domains:
                        possible |= domains[other_var]

                old_domain = domains[var_str].copy()
                domains[var_str] &= possible

                if domains[var_str] != old_domain:
                    changed = True

        # Construir resultat final
        result = {}
        for var_str, domain in domains.items():
            if len(domain) == 1:
                result[var_str] = next(iter(domain))
            elif len(domain) > 1:
                result[var_str] = domain

        return result

    def _extract_domains_from_solutions(self, solutions: List[dict]) -> dict:
        """
        Analitza les solucions i retorna per cada variable:
        - Si és constant: el valor directament (int)
        - Si varia: un set amb els valors possibles

        Returns: {var_name: value} o {var_name: {values}}
        """
        if not solutions:
            return {}

        result = {}

        for var in solutions[0].keys():
            values = {sol[var] for sol in solutions}
            if len(values) == 1:
                result[var] = next(iter(values))
            else:
                result[var] = values
        return result

    def _evaluate_with_solution(
            self,
            new_problem: Problem,
            expr: FNode,
            solution: dict,
    ) -> Optional[FNode]:
        """
        Evaluate an expression using values from solution.
        """
        def evaluate_recursive(node: FNode):
            # Constant
            if node.is_constant():
                return node.constant_value()

            # Object - get index
            if node.is_object_exp():
                obj = node.object()
                idx = self._object_to_index.get((obj.type, obj))
                if idx is not None:
                    return idx
                return None

            # Fluent/Parameter - look up in solution
            if node.is_fluent_exp() or node.is_parameter_exp():
                var_str = str(node)
                if var_str not in solution:
                    return None
                val = solution[var_str]
                if isinstance(val, set):
                    # mirar si esta en el set, si hi es
                    return val  # Retorna el set directament
                # Valor concret
                if node.type.is_user_type():
                    node_type = node.fluent().type if node.is_fluent_exp() else node.parameter().type
                    return self._get_object_from_index(node_type, val)
                return val

            # TRUE/FALSE
            if node.is_true():
                return True
            if node.is_false():
                return False

            # Equals
            if node.is_equals():
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))

                # Both concrete
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left == right
                    return None

                # Left is set, right is concrete
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if right not in left:
                        return False
                    if len(left) == 1:
                        return True
                    return None

                # Right is set, left is concrete
                if isinstance(right, set) and not isinstance(left, set) and left is not None:
                    if left not in right:
                        return False
                    if len(right) == 1:
                        return True
                    return None
                return None

            # Not
            if node.is_not():
                inner = evaluate_recursive(node.arg(0))
                if inner is None or isinstance(inner, set):
                    return None
                if isinstance(inner, bool):
                    return not inner
                return None

            # And
            if node.is_and():
                for arg in node.args:
                    result = evaluate_recursive(arg)
                    if result is False:
                        return False
                all_true = all(evaluate_recursive(arg) is True for arg in node.args)
                if all_true:
                    return True
                return None

            # Or
            if node.is_or():
                for arg in node.args:
                    result = evaluate_recursive(arg)
                    if result is True:
                        return True
                all_false = all(evaluate_recursive(arg) is False for arg in node.args)
                if all_false:
                    return False
                return None

            # Plus
            if node.is_plus():
                values = [evaluate_recursive(arg) for arg in node.args]
                if any(isinstance(v, set) or v is None for v in values):
                    return None
                return sum(values)

            # Minus
            if node.is_minus():
                values = [evaluate_recursive(arg) for arg in node.args]
                if any(isinstance(v, set) or v is None for v in values):
                    return None
                if len(values) == 1:
                    return -values[0]
                return values[0] - sum(values[1:])

            # Times
            if node.is_times():
                values = [evaluate_recursive(arg) for arg in node.args]
                if any(isinstance(v, set) or v is None for v in values):
                    return None
                result = 1
                for v in values:
                    result *= v
                return result

            # Lt
            if node.is_lt():
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left < right
                    return None
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if all(v < right for v in left):
                        return True
                    if all(v >= right for v in left):
                        return False
                    return None
                if isinstance(right, set) and not isinstance(left, set) and left is not None:
                    if all(left < v for v in right):
                        return True
                    if all(left >= v for v in right):
                        return False
                    return None
                return None

            # Le
            if node.is_le():
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left <= right
                    return None
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if all(v <= right for v in left):
                        return True
                    if all(v > right for v in left):
                        return False
                    return None
                if isinstance(right, set) and not isinstance(left, set) and left is not None:
                    if all(left <= v for v in right):
                        return True
                    if all(left > v for v in right):
                        return False
                    return None
                return None

            # Gt
            if node.node_type == OperatorKind.GT:
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left > right
                    return None
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if all(v > right for v in left):
                        return True
                    if all(v <= right for v in left):
                        return False
                    return None
                return None

            # Ge
            if node.node_type == OperatorKind.GE:
                left = evaluate_recursive(node.arg(0))
                right = evaluate_recursive(node.arg(1))
                if not isinstance(left, set) and not isinstance(right, set):
                    if left is not None and right is not None:
                        return left >= right
                    return None
                if isinstance(left, set) and not isinstance(right, set) and right is not None:
                    if all(v >= right for v in left):
                        return True
                    if all(v < right for v in left):
                        return False
                    return None
                return None
            return None

        result = evaluate_recursive(expr)
        if result is None or isinstance(result, set):
            return expr
        elif isinstance(result, bool):
            return TRUE() if result else FALSE()
        elif isinstance(result, int):
            if new_problem is not None:
                return self._get_number_object(new_problem, result)
            return Int(result)
        elif isinstance(result, Object):
            return ObjectExp(result)
        return expr


    def _transform_effects_for_solution(
            self,
            old_action: Action,
            new_action: InstantaneousAction,
            problem: Problem,
            new_problem: Problem,
            variables: bidict,
            solution: dict
    ) -> Union[Bool, None]:
        """
        Transform effects for a fully instantiated solution.
        All variables have concrete values.
        """
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        for effect in old_action.effects:
            #new_condition = self._to_nnf(new_problem, effect.condition)
            #new_condition = self._to_nnf(new_problem, effect.condition)
            #new_value = self._to_nnf(new_problem, effect.value)
            #temp_effect = Effect(
            #    effect.fluent,
            #    new_value,
            #    new_condition,
            #    effect.kind,
            #    effect.forall
            #)
            temp_effect = effect
            new_value = effect.value
            new_condition = effect.condition

            if temp_effect.is_increase() or temp_effect.is_decrease():
                # Increase/decrease effects
                for new_effect in self._transform_increase_decrease_effect(temp_effect, new_problem):
                    new_action.add_effect(
                        new_effect.fluent, new_effect.value,
                        new_effect.condition, new_effect.forall
                    )
            elif new_value.node_type in self.ARITHMETIC_OPS:
                # Arithmetic assignment - evaluate with concrete solution values
                evaluated_value = self._evaluate_with_solution(
                    new_problem, new_value, solution
                )
                if evaluated_value is None:
                    return None
                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_cond = self._transform_node(problem, new_problem, new_condition)
                if new_cond is None:
                    new_cond = TRUE()
                new_action.add_effect(new_fluent, evaluated_value, new_cond, effect.forall)

            else:
                # Simple assignment
                if new_value is not None and new_condition not in [None, FALSE()]:
                    new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                    new_value_transformed = self._transform_node(problem, new_problem, new_value)
                    new_cond = self._transform_node(problem, new_problem, new_condition)
                    if new_cond is None:
                        new_cond = TRUE()
                    if new_fluent and new_value_transformed:
                        new_action.add_effect(new_fluent, new_value_transformed, new_cond, effect.forall)
        return True

    # ==================== FLUENT TRANSFORMATION ====================

    def _transform_fluents(self, problem: Problem, new_problem: Problem):
        """Transform integer fluents -> boolean fluents with user-type parameter."""
        number_ut = UserType('Number')

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            if fluent.type.is_int_type():
                # Integer fluent -> Boolean fluent with user-type parameter
                new_fluent = Fluent(fluent.name, number_ut, fluent.signature, new_problem.environment)
                lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
                assert lb is not None and ub is not None
                self._domains[fluent.name] = (lb, ub)

                if default_value is not None:
                    default_obj = self._get_number_object(new_problem, default_value.constant_value())
                    new_problem.add_fluent(new_fluent, default_initial_value=default_obj)
                else:
                    new_problem.add_fluent(new_fluent)

                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        new_problem.set_initial_value(
                            new_problem.fluent(fluent.name)(*f.args),
                            self._get_number_object(new_problem, v.constant_value())
                        )
            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        new_problem.set_initial_value(f, v)

    def _find_static_fluents(self, problem: Problem, fluents: dict[FNode, FNode]) -> Dict[FNode, FNode]:
        """Find all static fluents throughout the problem."""
        modifiable_fluents = []
        for action in problem.actions:
            for effect in action.effects:
                modifiable_fluents.append(effect.fluent)
        return dict((f, v) for f, v in fluents.items() if f not in modifiable_fluents)

    def _apriori_generate_candidates(
            self,
            prev_level: dict[frozenset, int],
            k: int
    ) -> set[frozenset]:
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets.

        Uses the Apriori principle: only combine itemsets that share k-2 items,
        and prune candidates where any (k-1)-subset is not frequent.
        """
        candidates = set()
        prev_itemsets = list(prev_level.keys())

        # Sort items within each itemset for consistent comparison
        # Convert to sorted tuples for easier joining
        sorted_itemsets = [tuple(sorted(itemset)) for itemset in prev_itemsets]

        for i in range(len(sorted_itemsets)):
            for j in range(i + 1, len(sorted_itemsets)):
                itemset1 = sorted_itemsets[i]
                itemset2 = sorted_itemsets[j]

                # Join condition: first k-2 items must be the same
                if k > 2 and itemset1[:-1] != itemset2[:-1]:
                    continue

                # Create candidate by union
                candidate = frozenset(itemset1) | frozenset(itemset2)

                if len(candidate) != k:
                    continue

                # Pruning: check that all (k-1)-subsets are frequent
                is_valid = True
                for item in candidate:
                    subset = candidate - {item}
                    if subset not in prev_level:
                        is_valid = False
                        break

                if is_valid:
                    candidates.add(candidate)

        return candidates

    def _entries_to_formula(
            self,
            new_problem: Problem,
            variables: bidict,
            entries: list[dict],
            default_tuples: list[dict]
    ) -> Optional[FNode]:
        """
        Convert compressed entries to a DNF formula.
        """
        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}
        or_clauses = []

        for entry in entries:
            pattern = entry['pattern']
            subtable_vars = entry['subtable_vars']
            subtable_tuples = entry['subtable_tuples']

            # Build pattern constraints
            pattern_constraints = []
            for var_str, value in pattern.items():
                fnode = var_str_to_fnode.get(var_str)
                if fnode:
                    if fnode.is_fluent_exp():
                        fluent = fnode.fluent()
                        new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                        if fluent.type.is_int_type():
                            num_obj = self._get_number_object(new_problem, value)
                            pattern_constraints.append(Equals(new_fluent, num_obj))
                        elif fluent.type.is_user_type():
                            obj = self._get_object_from_index(fluent.type, value)
                            if obj:
                                pattern_constraints.append(Equals(new_fluent, ObjectExp(obj)))
                        elif fluent.type.is_bool_type():
                            if value == 1:
                                pattern_constraints.append(new_fluent)
                            else:
                                pattern_constraints.append(Not(new_fluent))
                    elif fnode.is_parameter_exp():
                        param = fnode.parameter()
                        if param.type.is_user_type():
                            obj = self._get_object_from_index(param.type, value)
                            if obj:
                                pattern_constraints.append(Equals(fnode, ObjectExp(obj)))

            if not subtable_vars or not subtable_tuples:
                # No subtable - just the pattern
                if pattern_constraints:
                    if len(pattern_constraints) == 1:
                        or_clauses.append(pattern_constraints[0])
                    else:
                        or_clauses.append(And(pattern_constraints))
                continue

            # Has subtable
            if len(subtable_vars) == 1:
                # Single variable in subtable - try to simplify
                var_str = subtable_vars[0]
                values = {t[var_str] for t in subtable_tuples}
                fnode = var_str_to_fnode.get(var_str)

                if fnode:
                    # Get full domain
                    full_domain = None
                    if fnode.is_fluent_exp():
                        fluent = fnode.fluent()
                        new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                        if fluent.type.is_int_type():
                            lb = fluent.type.lower_bound
                            ub = fluent.type.upper_bound
                            full_domain = set(range(lb, ub + 1))
                        elif fluent.type.is_user_type():
                            objects = list(new_problem.objects(fluent.type))
                            full_domain = set(range(len(objects)))
                        elif fluent.type.is_bool_type():
                            full_domain = {0, 1}
                    elif fnode.is_parameter_exp():
                        param = fnode.parameter()
                        if param.type.is_user_type():
                            objects = list(new_problem.objects(param.type))
                            full_domain = set(range(len(objects)))

                    if full_domain is not None and values == full_domain:
                        # Full domain - no constraint needed for this variable
                        if pattern_constraints:
                            if len(pattern_constraints) == 1:
                                or_clauses.append(pattern_constraints[0])
                            else:
                                or_clauses.append(And(pattern_constraints))
                        continue

                    # Not full domain - need to add constraint
                    if fnode.is_fluent_exp():
                        fluent = fnode.fluent()
                        new_fluent = new_problem.fluent(fluent.name)(*fnode.args)

                        if len(values) == 1:
                            val = next(iter(values))
                            if fluent.type.is_int_type():
                                pattern_constraints.append(
                                    Equals(new_fluent, self._get_number_object(new_problem, val)))
                            elif fluent.type.is_user_type():
                                obj = self._get_object_from_index(fluent.type, val)
                                if obj:
                                    pattern_constraints.append(Equals(new_fluent, ObjectExp(obj)))
                            elif fluent.type.is_bool_type():
                                pattern_constraints.append(new_fluent if val == 1 else Not(new_fluent))
                        else:
                            # Multiple values - create OR
                            value_clauses = []
                            for val in sorted(values):
                                if fluent.type.is_int_type():
                                    value_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, val)))
                                elif fluent.type.is_user_type():
                                    obj = self._get_object_from_index(fluent.type, val)
                                    if obj:
                                        value_clauses.append(Equals(new_fluent, ObjectExp(obj)))
                                elif fluent.type.is_bool_type():
                                    value_clauses.append(new_fluent if val == 1 else Not(new_fluent))

                            if value_clauses:
                                if len(value_clauses) == 1:
                                    pattern_constraints.append(value_clauses[0])
                                else:
                                    pattern_constraints.append(Or(value_clauses))

                    elif fnode.is_parameter_exp():
                        param = fnode.parameter()
                        if param.type.is_user_type():
                            if len(values) == 1:
                                val = next(iter(values))
                                obj = self._get_object_from_index(param.type, val)
                                if obj:
                                    pattern_constraints.append(Equals(fnode, ObjectExp(obj)))
                            else:
                                value_clauses = []
                                for val in sorted(values):
                                    obj = self._get_object_from_index(param.type, val)
                                    if obj:
                                        value_clauses.append(Equals(fnode, ObjectExp(obj)))
                                if value_clauses:
                                    if len(value_clauses) == 1:
                                        pattern_constraints.append(value_clauses[0])
                                    else:
                                        pattern_constraints.append(Or(value_clauses))

                    if pattern_constraints:
                        if len(pattern_constraints) == 1:
                            or_clauses.append(pattern_constraints[0])
                        else:
                            or_clauses.append(And(pattern_constraints))
                    continue

            # Multiple variables in subtable - enumerate all combinations
            for subtable_tuple in subtable_tuples:
                all_constraints = list(pattern_constraints)  # Copy pattern constraints

                for var_str, value in subtable_tuple.items():
                    fnode = var_str_to_fnode.get(var_str)
                    if not fnode:
                        continue

                    if fnode.is_fluent_exp():
                        fluent = fnode.fluent()
                        new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                        if fluent.type.is_int_type():
                            all_constraints.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                        elif fluent.type.is_user_type():
                            obj = self._get_object_from_index(fluent.type, value)
                            if obj:
                                all_constraints.append(Equals(new_fluent, ObjectExp(obj)))
                        elif fluent.type.is_bool_type():
                            if value == 1:
                                all_constraints.append(new_fluent)
                            else:
                                all_constraints.append(Not(new_fluent))
                    elif fnode.is_parameter_exp():
                        param = fnode.parameter()
                        if param.type.is_user_type():
                            obj = self._get_object_from_index(param.type, value)
                            if obj:
                                all_constraints.append(Equals(fnode, ObjectExp(obj)))

                if all_constraints:
                    if len(all_constraints) == 1:
                        or_clauses.append(all_constraints[0])
                    else:
                        or_clauses.append(And(all_constraints))

        # Process default tuples
        for solution in default_tuples:
            and_clauses = []
            for var_str, value in solution.items():
                fnode = var_str_to_fnode.get(var_str)
                if not fnode:
                    continue

                if fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                    if fluent.type.is_int_type():
                        and_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                    elif fluent.type.is_user_type():
                        obj = self._get_object_from_index(fluent.type, value)
                        if obj:
                            and_clauses.append(Equals(new_fluent, ObjectExp(obj)))
                    elif fluent.type.is_bool_type():
                        if value == 1:
                            and_clauses.append(new_fluent)
                        else:
                            and_clauses.append(Not(new_fluent))
                elif fnode.is_parameter_exp():
                    param = fnode.parameter()
                    if param.type.is_user_type():
                        obj = self._get_object_from_index(param.type, value)
                        if obj:
                            and_clauses.append(Equals(fnode, ObjectExp(obj)))

            if and_clauses:
                if len(and_clauses) == 1:
                    or_clauses.append(and_clauses[0])
                else:
                    or_clauses.append(And(and_clauses))

        if not or_clauses:
            return None

        if len(or_clauses) == 1:
            return or_clauses[0].simplify()
        return Or(or_clauses).simplify()

    def _verify_compression(self, original_solutions, entries, default_tuples):
        """Verify that compressed representation covers exactly the same tuples."""
        reconstructed = set()

        for entry in entries:
            pattern = entry['pattern']
            subtable_tuples = entry['subtable_tuples']

            if not subtable_tuples:
                # Just the pattern
                reconstructed.add(frozenset(pattern.items()))
            else:
                # Combine pattern with each subtable tuple
                for subtable_tuple in subtable_tuples:
                    full_tuple = dict(pattern)
                    full_tuple.update(subtable_tuple)
                    reconstructed.add(frozenset(full_tuple.items()))

        # Add default tuples
        for t in default_tuples:
            reconstructed.add(frozenset(t.items()))

        # Compare
        original_set = {frozenset(s.items()) for s in original_solutions}

        if reconstructed == original_set:
            print("Compression is CORRECT - same tuples covered")
            return True
        else:
            missing = original_set - reconstructed
            extra = reconstructed - original_set
            print(f"Compression ERROR!")
            print(f"  Missing tuples: {len(missing)}")
            print(f"  Extra tuples: {len(extra)}")
            return False

    # ==================== AXIOMS TRANSFORMATION ====================

    def _transform_axioms(self, problem: Problem, new_problem: Problem, new_to_old: Dict):
        """Transform axioms"""
        for axiom in problem.axioms:
            params = OrderedDict((p.name, p.type) for p in axiom.parameters)
            # Clone and transform
            new_axiom_name = get_fresh_name(new_problem, axiom.name)
            new_axiom = Axiom(new_axiom_name, params, axiom.environment)

            skip_axiom = False
            new_axiom.set_head(axiom.head.fluent)
            for body in axiom.body:
                new_body = self._transform_node(problem, new_problem, body)
                if new_body is None:
                    skip_axiom = True
                    break
                else:
                    new_axiom.add_body_condition(new_body)
            if skip_axiom:
                continue
            new_problem.add_axiom(new_axiom)
            new_to_old[new_axiom] = axiom

    # ==================== GOALS TRANSFORMATION ====================

    def _transform_goals(self, problem: Problem, new_problem: Problem) -> None:
        """
        Transform goals, using CP-SAT only if there's arithmetic.
        """
        # Replace static fluents
        non_static_goals = []
        for goal in problem.goals:
            ng = self._replace_static(goal)
            if ng.is_and():
                for g in ng.args:
                    if g is not TRUE():
                        non_static_goals.append(g)
            else:
                if ng is not TRUE():
                    non_static_goals.append(ng)

        if not non_static_goals:
            return

        # Check if goals have arithmetic
        has_arithmetic = any(self._has_arithmetic(g) for g in non_static_goals)

        # ===== NO ARITHMETIC: Direct transformation =====
        if not has_arithmetic:
            for goal in non_static_goals:
                transformed = self._transform_node(problem, new_problem, goal)
                if transformed and transformed != TRUE():
                    new_problem.add_goal(transformed)
            return

        # ===== HAS ARITHMETIC: Use CP-SAT =====
        # Clear mappings
        self._object_to_index = {}
        self._index_to_object = {}

        # Build CP-SAT model
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        result_var = self._add_cp_constraints(new_problem, And(non_static_goals), variables, cp_model_obj)
        cp_model_obj.add(result_var == 1)

        # Solve
        solutions = self._solve_with_cp_sat(variables, cp_model_obj)
        if not solutions:
            raise UPProblemDefinitionError("No possible goal!")

        # Convert to DNF
        dnf_formula = self._solutions_to_dnf(new_problem, solutions, variables)

        if dnf_formula:
            new_problem.add_goal(dnf_formula)
        else:
            raise UPProblemDefinitionError("No possible goal!")

    def _compile(
            self,
            problem: "up.model.AbstractProblem",
            compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)
        self._number_objects_cache.clear()
        self._domains.clear()

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_axioms()
        new_problem.initial_values.clear()
        new_problem.clear_quality_metrics()

        # Transform components
        self._static_fluents = self._find_static_fluents(problem, problem.initial_values)
        self._transform_fluents(problem, new_problem)

        # ========== Transform Actions ==========
        new_to_old = self._transform_actions(problem, new_problem)

        # ========== Transform Axioms ==========
        self._transform_axioms(problem, new_problem, new_to_old)

        # ========== Transform Goals ==========
        self._transform_goals(problem, new_problem)

        # ========== Transform Quality Metrics ==========
        for metric in problem.quality_metrics:
            if metric.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(
                        metric,
                        new_to_old,
                        new_problem.environment
                    )
                )
            else:
                new_problem.add_quality_metric(metric)

        return CompilerResult(
            new_problem,
            partial(replace_action, map=new_to_old),
            self.name
        )
