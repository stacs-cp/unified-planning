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
import unified_planning as up
import unified_planning.engines as engines
from ortools.sat.python import cp_model
from bidict import bidict
from typing import List, Dict, Any, Tuple, Union
from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, Action, ProblemKind, Variable, Effect, EffectKind, Object, FNode, Fluent, \
    InstantaneousAction, Axiom
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import get_fresh_name, replace_action, updated_minimize_action_costs
from typing import Optional, Iterator, OrderedDict, Union
from functools import partial
from unified_planning.shortcuts import And, Or, Equals, Not, FALSE, Iff, UserType, TRUE, ObjectExp, Plus, Int, Times, \
    Minus, GT, GE, Bool
from typing import List, Dict, Tuple, Set
from itertools import combinations

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
        if node.node_type in self.ARITHMETIC_OPS or node.is_le() or node.is_lt():
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

    def _add_cp_constraints(self, problem: Problem, node: FNode, variables: bidict, model: cp_model.CpModel):
        """Recursively build CP-SAT constraints from expression tree."""
        final_constraint = []

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
            else:
                var = model.new_bool_var(str(node))
            variables[node] = var
            return var

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

        # Not Equality
        if node.is_not():
            inner_var = self._add_cp_constraints(problem, node.arg(0), variables, model)
            not_var = model.new_bool_var(f"not_{id(node)}")
            # not_var és la negació d'inner_var
            model.add(not_var == (1 - inner_var))
            return not_var

        # Equality
        if node.is_equals():
            # User types treated as variables
            if node.arg(0).type.is_user_type():
                if node in variables:
                    return variables[node]
                var = model.new_bool_var(str(node))
                variables[node] = var
                return var

            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)
            eq_var = model.new_bool_var(f"eq_{id(node)}")

            model.add(left == right).only_enforce_if(eq_var)
            model.add(left != right).only_enforce_if(eq_var.negated())
            return eq_var

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

    def _simplify_solutions(self, variables: bidict, solutions: list[dict[str, int]]) -> list[dict[str, int]]:
        """
        Compact solutions by grouping those differing in few variables.
        Remove variables that take all possible domain values.
        """
        if not solutions:
            return []
        all_vars = list(solutions[0].keys())
        simplified = []
        used = set()

        # Try grouping by each variable
        for var_name in all_vars:
            groups = {}

            for i, sol in enumerate(solutions):
                if i in used:
                    continue

                # Key = all values except the varying variable
                key = tuple((k, v) for k, v in sorted(sol.items()) if k != var_name)
                groups.setdefault(key, []).append((i, sol[var_name]))

            # Compact groups with multiple values
            for key, indices_vals in groups.items():
                if len(indices_vals) <= 1:
                    continue

                # Mark as used
                for idx, _ in indices_vals:
                    used.add(idx)

                # Find the fluent node
                fnode = None
                for node, var in variables.items():
                    if str(node) == var_name:
                        fnode = node
                        break
                if not fnode:
                    continue

                compact = dict(key)
                values_set = {val for _, val in indices_vals}

                # Check if covers entire domain
                if fnode.fluent().type.is_int_type():
                    lb = fnode.fluent().type.lower_bound
                    ub = fnode.fluent().type.upper_bound
                    domain = set(range(lb, ub + 1))

                    # Only include if not entire domain
                    if values_set != domain:
                        compact[var_name] = values_set
                elif fnode.fluent().type.is_bool_type():
                    if values_set != {0, 1}:
                        compact[var_name] = values_set
                else:
                    compact[var_name] = values_set

                simplified.append(compact)

        # Add ungrouped solutions
        for i, sol in enumerate(solutions):
            if i not in used:
                simplified.append(sol)

        return simplified

    def _convert_to_dnf(self, new_problem, solutions, variables):
        #solutions = self._simplify_solutions(variables, solutions) # ?

        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        # Convert solutions to formula
        or_clauses = []
        for solution in solutions:
            and_clauses = []

            for var_str, value in solution.items():
                # Find corresponding FNode
                fnode = var_str_to_fnode.get(var_str)
                if not fnode:
                    continue
                if fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)

                    if fluent.type.is_int_type():
                        if isinstance(value, set):
                            # Multiple values: (f = v1) OR (f = v2) OR ...
                            lb, ub = self._domains[new_fluent.fluent().name]
                            or_eq = [Equals(new_fluent, self._get_number_object(new_problem, v))
                                     for v in value if lb <= v <= ub]
                            if or_eq:
                                and_clauses.append(Or(or_eq))
                        else:
                            if self._is_value_in_bounds(new_fluent.fluent().name, value):
                                and_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                    elif fluent.type.is_bool_type():
                        bool_val = (value == 1)
                        and_clauses.append(Iff(fnode, bool_val))
                else:
                    # Boolean variable
                    and_clauses.append(fnode if value == 1 else Not(fnode))
            if and_clauses:
                or_clauses.append(And(and_clauses))
        return Or(or_clauses).simplify() if or_clauses else None

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

        # Expressions with integers need CP-SAT solving
        if self._has_arithmetic(node):
            return "has_arithmetic"

        # Recursively transform children
        new_args = []
        for arg in node.args:
            transformed = self._transform_node(old_problem, new_problem, arg)
            if transformed is None:
                return None
            new_args.append(transformed)

        # Handle quantifiers
        # aixo passara mai?
        if node.is_exists() or node.is_forall():
            new_vars = [
                Variable(v.name, UserType('Number')) if v.type.is_int_type() else v
                for v in node.variables()
            ]
            return em.create_node(node.node_type, tuple(new_args), payload=tuple(new_vars)).simplify()

        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _chain_minus(self, base, subtractions):
        result = base
        for s in subtractions:
            result = Minus(result, s)
        return result

    def _simplify_plus_minus(self, node: FNode) -> FNode:
        # Primer, simplifica recursivament els fills
        if node.is_plus() or node.is_minus():
            simplified_args = [self._simplify_plus_minus(arg) for arg in node.args]
            # Reconstrueix el node amb els fills simplificats
            if node.is_plus():
                node = Plus(*simplified_args)
            else:
                if len(simplified_args) == 1:
                    node = simplified_args[0]
                else:
                    node = self._chain_minus(simplified_args[0], simplified_args[1:])
        # Ara processa les simplificacions
        if node.is_plus():
            # Busca tots els (0 - x) i converteix-los
            new_args = []
            subtractions = []
            for arg in node.args:
                if arg.is_minus() and len(arg.args) == 2:
                    left, right = arg.arg(0), arg.arg(1)
                    if left.is_constant() and left.constant_value() == 0:
                        subtractions.append(right)
                        continue
                new_args.append(arg)
            if subtractions:
                # Converteix: a + b + (0-c) + (0-d) → (a + b) - c - d
                if len(new_args) == 0:
                    # Tot eren subtraccions: (0-a) + (0-b) → 0 - a - b
                    return Minus(Int(0), *subtractions)
                elif len(new_args) == 1 and len(subtractions) == 1:
                    return Minus(new_args[0], subtractions[0])
                else:
                    # Cas general
                    base = Plus(*new_args) if len(new_args) > 1 else new_args[0]
                    return self._chain_minus(base, subtractions)
        if node.is_minus():
            # x - y - (0-z) → x - y + z
            if len(node.args) >= 2:
                left = node.arg(0)
                new_subtractions = []
                additions = []

                for i in range(1, len(node.args)):
                    arg = node.arg(i)
                    if arg.is_minus() and len(arg.args) == 2:
                        arg_left, arg_right = arg.arg(0), arg.arg(1)
                        if arg_left.is_constant() and arg_left.constant_value() == 0:
                            additions.append(arg_right)
                            continue
                    new_subtractions.append(arg)
                if additions:
                    if new_subtractions:
                        # x - y - (0-z) → (x + z) - y
                        base = Plus(left, *additions)
                        return self._chain_minus(base, new_subtractions)
                    else:
                        # x - (0-y) → x + y
                        return Plus(left, *additions)
        return node

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
            new_left_switched = self._simplify_plus_minus(self._negate_expr(problem, new_right)).simplify()
            new_right_switched = self._simplify_plus_minus(self._negate_expr(problem, new_left)).simplify()
            return em.create_node(op_type, (new_left_switched, new_right_switched)).simplify()

        new_left = self._simplify_plus_minus(new_left)
        new_right = self._simplify_plus_minus(new_right)
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

    def _transform_arithmetic_assignment(
            self,
            effect: Effect,
            old_problem: Problem,
            new_problem: Problem,
            variables,
            cp_model_obj
    ) -> Iterator[Effect]:
        """Handle assignments with arithmetic expressions by enumerating combinations."""
        fluent_domains = self._find_integer_fluents(effect.value)
        if not fluent_domains:
            return

        lb, ub = self._domains[effect.fluent.fluent().name] # rang del fluent al que assignem

        # Per cada valor possible del fluent d'assignació, trobar combinacions vàlides
        for v in range(lb, ub + 1):
            temp_model = cp_model.CpModel() #cp_model_obj.clone()
            # how to propagate the rest of constraints too?
            temp_variables = bidict({})
            equality = Equals(effect.value, v)
            result_var = self._add_cp_constraints(new_problem, equality, temp_variables, temp_model)
            temp_model.add(result_var == 1)
            result = self._solve_with_cp_sat(temp_variables, temp_model)
            # what to do here?
            if result is None:
                continue

            # Compression MFI
            compressed_formula = self._compress_solutions_mfi(new_problem, variables, result)

            # or DNF
            #formula = self._convert_to_dnf(new_problem, result, temp_variables)

            # Create effects
            new_fluent = new_problem.fluent(effect.fluent.fluent().name)
            new_base_condition = self._transform_node(old_problem, new_problem, effect.condition)
            full_condition = And(new_base_condition, compressed_formula).simplify()

            yield Effect(
                new_fluent(*effect.fluent.args),
                self._get_number_object(new_problem, v),
                full_condition,
                EffectKind.ASSIGN,
                effect.forall
            )

    # ==================== ACTION TRANSFORMATION ====================

    def _simplify_arithmetic_expression(self, problem: Problem, node: FNode) -> FNode:
        """
        Simplify arithmetic expressions by moving all constants to one side and variables to the other
        """
        em = problem.environment.expression_manager
        if node.is_constant() or node.is_variable_exp() or node.is_timing_exp() or node.is_parameter_exp() or node.is_fluent_exp():
            return node
        elif (node.is_equals() and node.arg(0).type.is_int_type()) or node.is_le() or node.is_lt():
            left = self._simplify_arithmetic_expression(problem, node.arg(0))
            right = self._simplify_arithmetic_expression(problem, node.arg(1))
            return self._normalise_comparison(problem, node.node_type, left, right)

        simplified_args = [self._simplify_arithmetic_expression(problem, arg) for arg in node.args]
        node = em.create_node(node.node_type, tuple(simplified_args))
        return node

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

    def _normalize_pair(self, first: FNode, second: FNode) -> Tuple[FNode, FNode]:
        """Normalitza parella per evitar duplicats: (a,b) i (b,a) són el mateix."""
        # Ordena per algún criteri consistent (per exemple, per string representation)
        if str(first) <= str(second):
            return (first, second)
        return (second, first)

    def _both_variables(self, first: FNode, second: FNode) -> bool:
        """Check if both nodes are fluents or parameters."""
        return ((first.is_fluent_exp() or first.is_parameter_exp()) and
                (second.is_fluent_exp() or second.is_parameter_exp()))

    def _get_equalities(self, problem: Problem, node: FNode) -> Tuple[List[Tuple[FNode, FNode]], List[Tuple[FNode, FNode]]]:
        """
        Finds pairs of equalities and not equalities that are always true in preconditions.
        Returns: (equalities, not_equalities) as lists of tuples (a, b)
        """
        # And: totes les equalities dels fills són certes
        if node.is_and():
            all_eq = []
            all_neq = []
            for arg in node.args:
                eq, neq = self._get_equalities(problem, arg)
                all_eq.extend(eq)
                all_neq.extend(neq)

            # Eliminar duplicats mantenint ordre
            all_eq = list(dict.fromkeys(all_eq))
            all_neq = list(dict.fromkeys(all_neq))

            return all_eq, all_neq

        # Or: no podem garantir res
        if node.is_or():
            return [], []

        # Not Equals o LT
        if (node.is_not() and node.arg(0).is_equals()) or node.is_lt():
            subnode = node if node.is_lt() else node.arg(0)
            first, second = subnode.arg(0), subnode.arg(1)
            if self._both_variables(first, second):
                pair = self._normalize_pair(first, second)
                return [], [pair]

        # Equals o LE
        if node.is_equals() or node.is_le():
            first, second = node.arg(0), node.arg(1)
            if self._both_variables(first, second):
                pair = self._normalize_pair(first, second)
                return [pair], []

        return [], []

    # ---------------------------------
    # Heuristic Approach Combining Maximal Itemsets and Area Measure for Compressing Voluminous Table Constraints
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
            # Pattern: the fixed variables from the itemset
            pattern = dict(itemset)
            pattern_vars = set(pattern.keys())

            # Subtable variables
            subtable_vars = sorted(all_variables - pattern_vars)

            # Collect the actual subtable tuples (only the varying parts)
            subtable_tuples = []
            for idx in covered_indices:
                solution = solutions[idx]
                subtable_tuple = {var: solution[var] for var in subtable_vars}
                if subtable_tuple not in subtable_tuples:
                    subtable_tuples.append(subtable_tuple)

            entries.append({
                'pattern': pattern,
                'subtable_vars': subtable_vars,
                'subtable_tuples': subtable_tuples,  # List of valid combinations
                'num_tuples': len(covered_indices)
            })

        # Default tuples
        default_tuples = [solutions[idx] for idx in remaining_indices]

        return entries, default_tuples

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
    import unified_planning as up
    import unified_planning.engines as engines
    from ortools.sat.python import cp_model
    from bidict import bidict
    from typing import List, Dict, Any, Tuple, Union
    from unified_planning.model.expression import ListExpression
    from unified_planning.model.operators import OperatorKind
    from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
    from unified_planning.engines.results import CompilerResult
    from unified_planning.exceptions import UPProblemDefinitionError
    from unified_planning.model import Problem, Action, ProblemKind, Variable, Effect, EffectKind, Object, FNode, \
        Fluent, \
        InstantaneousAction
    from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
    from unified_planning.engines.compilers.utils import get_fresh_name, replace_action, updated_minimize_action_costs
    from typing import Optional, Iterator, OrderedDict, Union
    from functools import partial
    from unified_planning.shortcuts import And, Or, Equals, Not, FALSE, Iff, UserType, TRUE, ObjectExp, Plus, Int, \
        Times, \
        Minus, GT, GE, Bool
    from typing import List, Dict, Tuple, Set
    from itertools import combinations

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
            # supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
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
            if node.node_type in self.ARITHMETIC_OPS or node.is_le() or node.is_lt():
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

        def _add_cp_constraints(self, problem: Problem, node: FNode, variables: bidict, model: cp_model.CpModel):
            """Recursively build CP-SAT constraints from expression tree."""
            final_constraint = []

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
                else:
                    var = model.new_bool_var(str(node))
                variables[node] = var
                return var

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

            # Not Equality
            if node.is_not():
                inner_var = self._add_cp_constraints(problem, node.arg(0), variables, model)
                not_var = model.new_bool_var(f"not_{id(node)}")
                # not_var és la negació d'inner_var
                model.add(not_var == (1 - inner_var))
                return not_var

            # Equality
            if node.is_equals():
                # User types treated as variables
                if node.arg(0).type.is_user_type():
                    if node in variables:
                        return variables[node]
                    var = model.new_bool_var(str(node))
                    variables[node] = var
                    return var

                left = self._add_cp_constraints(problem, node.arg(0), variables, model)
                right = self._add_cp_constraints(problem, node.arg(1), variables, model)
                eq_var = model.new_bool_var(f"eq_{id(node)}")

                model.add(left == right).only_enforce_if(eq_var)
                model.add(left != right).only_enforce_if(eq_var.negated())
                return eq_var

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

        def _simplify_solutions(self, variables: bidict, solutions: list[dict[str, int]]) -> list[dict[str, int]]:
            """
            Compact solutions by grouping those differing in few variables.
            Remove variables that take all possible domain values.
            """
            if not solutions:
                return []
            all_vars = list(solutions[0].keys())
            simplified = []
            used = set()

            # Try grouping by each variable
            for var_name in all_vars:
                groups = {}

                for i, sol in enumerate(solutions):
                    if i in used:
                        continue

                    # Key = all values except the varying variable
                    key = tuple((k, v) for k, v in sorted(sol.items()) if k != var_name)
                    groups.setdefault(key, []).append((i, sol[var_name]))

                # Compact groups with multiple values
                for key, indices_vals in groups.items():
                    if len(indices_vals) <= 1:
                        continue

                    # Mark as used
                    for idx, _ in indices_vals:
                        used.add(idx)

                    # Find the fluent node
                    fnode = None
                    for node, var in variables.items():
                        if str(node) == var_name:
                            fnode = node
                            break
                    if not fnode:
                        continue

                    compact = dict(key)
                    values_set = {val for _, val in indices_vals}

                    # Check if covers entire domain
                    if fnode.fluent().type.is_int_type():
                        lb = fnode.fluent().type.lower_bound
                        ub = fnode.fluent().type.upper_bound
                        domain = set(range(lb, ub + 1))

                        # Only include if not entire domain
                        if values_set != domain:
                            compact[var_name] = values_set
                    elif fnode.fluent().type.is_bool_type():
                        if values_set != {0, 1}:
                            compact[var_name] = values_set
                    else:
                        compact[var_name] = values_set

                    simplified.append(compact)

            # Add ungrouped solutions
            for i, sol in enumerate(solutions):
                if i not in used:
                    simplified.append(sol)

            return simplified

        def _convert_to_dnf(self, new_problem, solutions, variables):
            # solutions = self._simplify_solutions(variables, solutions) # ?

            var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

            # Convert solutions to formula
            or_clauses = []
            for solution in solutions:
                and_clauses = []

                for var_str, value in solution.items():
                    # Find corresponding FNode
                    fnode = var_str_to_fnode.get(var_str)
                    if not fnode:
                        continue
                    if fnode.is_fluent_exp():
                        fluent = fnode.fluent()
                        new_fluent = new_problem.fluent(fluent.name)(*fnode.args)

                        if fluent.type.is_int_type():
                            if isinstance(value, set):
                                # Multiple values: (f = v1) OR (f = v2) OR ...
                                lb, ub = self._domains[new_fluent.fluent().name]
                                or_eq = [Equals(new_fluent, self._get_number_object(new_problem, v))
                                         for v in value if lb <= v <= ub]
                                if or_eq:
                                    and_clauses.append(Or(or_eq))
                            else:
                                if self._is_value_in_bounds(new_fluent.fluent().name, value):
                                    and_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                        elif fluent.type.is_bool_type():
                            bool_val = (value == 1)
                            and_clauses.append(Iff(fnode, bool_val))
                    else:
                        # Boolean variable
                        and_clauses.append(fnode if value == 1 else Not(fnode))
                if and_clauses:
                    or_clauses.append(And(and_clauses))
            return Or(or_clauses).simplify() if or_clauses else None

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

            # Expressions with integers need CP-SAT solving
            if self._has_arithmetic(node):
                return "has_arithmetic"

            # Recursively transform children
            new_args = []
            for arg in node.args:
                transformed = self._transform_node(old_problem, new_problem, arg)
                if transformed is None:
                    return None
                new_args.append(transformed)

            # Handle quantifiers
            # aixo passara mai?
            if node.is_exists() or node.is_forall():
                new_vars = [
                    Variable(v.name, UserType('Number')) if v.type.is_int_type() else v
                    for v in node.variables()
                ]
                return em.create_node(node.node_type, tuple(new_args), payload=tuple(new_vars)).simplify()

            return em.create_node(node.node_type, tuple(new_args)).simplify()

        def _chain_minus(self, base, subtractions):
            result = base
            for s in subtractions:
                result = Minus(result, s)
            return result

        def _simplify_plus_minus(self, node: FNode) -> FNode:
            # Primer, simplifica recursivament els fills
            if node.is_plus() or node.is_minus():
                simplified_args = [self._simplify_plus_minus(arg) for arg in node.args]
                # Reconstrueix el node amb els fills simplificats
                if node.is_plus():
                    node = Plus(*simplified_args)
                else:
                    if len(simplified_args) == 1:
                        node = simplified_args[0]
                    else:
                        node = self._chain_minus(simplified_args[0], simplified_args[1:])
            # Ara processa les simplificacions
            if node.is_plus():
                # Busca tots els (0 - x) i converteix-los
                new_args = []
                subtractions = []
                for arg in node.args:
                    if arg.is_minus() and len(arg.args) == 2:
                        left, right = arg.arg(0), arg.arg(1)
                        if left.is_constant() and left.constant_value() == 0:
                            subtractions.append(right)
                            continue
                    new_args.append(arg)
                if subtractions:
                    # Converteix: a + b + (0-c) + (0-d) → (a + b) - c - d
                    if len(new_args) == 0:
                        # Tot eren subtraccions: (0-a) + (0-b) → 0 - a - b
                        return Minus(Int(0), *subtractions)
                    elif len(new_args) == 1 and len(subtractions) == 1:
                        return Minus(new_args[0], subtractions[0])
                    else:
                        # Cas general
                        base = Plus(*new_args) if len(new_args) > 1 else new_args[0]
                        return self._chain_minus(base, subtractions)
            if node.is_minus():
                # x - y - (0-z) → x - y + z
                if len(node.args) >= 2:
                    left = node.arg(0)
                    new_subtractions = []
                    additions = []

                    for i in range(1, len(node.args)):
                        arg = node.arg(i)
                        if arg.is_minus() and len(arg.args) == 2:
                            arg_left, arg_right = arg.arg(0), arg.arg(1)
                            if arg_left.is_constant() and arg_left.constant_value() == 0:
                                additions.append(arg_right)
                                continue
                        new_subtractions.append(arg)
                    if additions:
                        if new_subtractions:
                            # x - y - (0-z) → (x + z) - y
                            base = Plus(left, *additions)
                            return self._chain_minus(base, new_subtractions)
                        else:
                            # x - (0-y) → x + y
                            return Plus(left, *additions)
            return node

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
            # if expr.node_type == OperatorKind.PLUS and len(expr.args) == 2 and expr.arg(1).node_type == OperatorKind.MINUS and \
            #    expr.arg(1).arg(0).constant_value() == 0:
            #    print("AAAAA", expr, "to", Minus(expr.arg(1).arg(1), expr.arg(0)))
            #    return Minus(expr.arg(1).arg(1), expr.arg(0))
            # if expr.node_type == OperatorKind.PLUS and len(expr.args) == 2 and expr.arg(0).node_type == OperatorKind.MINUS and \
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
                new_left_switched = self._simplify_plus_minus(self._negate_expr(problem, new_right)).simplify()
                new_right_switched = self._simplify_plus_minus(self._negate_expr(problem, new_left)).simplify()
                return em.create_node(op_type, (new_left_switched, new_right_switched)).simplify()

            new_left = self._simplify_plus_minus(new_left)
            new_right = self._simplify_plus_minus(new_right)
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
                valid_range = range(max(lb, lb), min(ub - int_value, ub) + 1) if isinstance(int_value, int) else range(
                    lb, ub + 1)
            else:
                # Per decrease: valor final = i - delta, per tant i ha d'estar en [lb+delta, ub]
                valid_range = range(max(lb + int_value, lb), min(ub, ub) + 1) if isinstance(int_value, int) else range(
                    lb, ub + 1)

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

        def _transform_arithmetic_assignment(
                self,
                effect: Effect,
                old_problem: Problem,
                new_problem: Problem,
                variables,
                cp_model_obj
        ) -> Iterator[Effect]:
            """Handle assignments with arithmetic expressions by enumerating combinations."""
            fluent_domains = self._find_integer_fluents(effect.value)
            if not fluent_domains:
                return

            lb, ub = self._domains[effect.fluent.fluent().name]  # rang del fluent al que assignem

            # Per cada valor possible del fluent d'assignació, trobar combinacions vàlides
            for v in range(lb, ub + 1):
                temp_model = cp_model.CpModel()  # cp_model_obj.clone()
                # how to propagate the rest of constraints too?
                temp_variables = bidict({})
                equality = Equals(effect.value, v)
                result_var = self._add_cp_constraints(new_problem, equality, temp_variables, temp_model)
                temp_model.add(result_var == 1)
                result = self._solve_with_cp_sat(temp_variables, temp_model)
                # what to do here?
                if result is None:
                    continue

                # Compression MFI
                compressed_formula = self._compress_solutions_mfi(new_problem, variables, result)

                # or DNF
                #compressed_formula = self._convert_to_dnf(new_problem, result, temp_variables)

                # Create effects
                new_fluent = new_problem.fluent(effect.fluent.fluent().name)
                new_base_condition = self._transform_node(old_problem, new_problem, effect.condition)
                full_condition = And(new_base_condition, compressed_formula).simplify()

                yield Effect(
                    new_fluent(*effect.fluent.args),
                    self._get_number_object(new_problem, v),
                    full_condition,
                    EffectKind.ASSIGN,
                    effect.forall
                )

        # ==================== ACTION TRANSFORMATION ====================

        def _simplify_arithmetic_expression(self, problem: Problem, node: FNode) -> FNode:
            """
            Simplify arithmetic expressions by moving all constants to one side and variables to the other
            """
            em = problem.environment.expression_manager
            if node.is_constant() or node.is_variable_exp() or node.is_timing_exp() or node.is_parameter_exp() or node.is_fluent_exp():
                return node
            elif (node.is_equals() and node.arg(0).type.is_int_type()) or node.is_le() or node.is_lt():
                left = self._simplify_arithmetic_expression(problem, node.arg(0))
                right = self._simplify_arithmetic_expression(problem, node.arg(1))
                return self._normalise_comparison(problem, node.node_type, left, right)

            simplified_args = [self._simplify_arithmetic_expression(problem, arg) for arg in node.args]
            node = em.create_node(node.node_type, tuple(simplified_args))
            return node

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
                elif node.arg(0).is_equals():  # Not Equals() remaining
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

        def _normalize_pair(self, first: FNode, second: FNode) -> Tuple[FNode, FNode]:
            """Normalitza parella per evitar duplicats: (a,b) i (b,a) són el mateix."""
            # Ordena per algún criteri consistent (per exemple, per string representation)
            if str(first) <= str(second):
                return (first, second)
            return (second, first)

        def _both_variables(self, first: FNode, second: FNode) -> bool:
            """Check if both nodes are fluents or parameters."""
            return ((first.is_fluent_exp() or first.is_parameter_exp()) and
                    (second.is_fluent_exp() or second.is_parameter_exp()))

        def _get_equalities(self, problem: Problem, node: FNode) -> Tuple[
            List[Tuple[FNode, FNode]], List[Tuple[FNode, FNode]]]:
            """
            Finds pairs of equalities and not equalities that are always true in preconditions.
            Returns: (equalities, not_equalities) as lists of tuples (a, b)
            """
            # And: totes les equalities dels fills són certes
            if node.is_and():
                all_eq = []
                all_neq = []
                for arg in node.args:
                    eq, neq = self._get_equalities(problem, arg)
                    all_eq.extend(eq)
                    all_neq.extend(neq)

                # Eliminar duplicats mantenint ordre
                all_eq = list(dict.fromkeys(all_eq))
                all_neq = list(dict.fromkeys(all_neq))

                return all_eq, all_neq

            # Or: no podem garantir res
            if node.is_or():
                return [], []

            # Not Equals o LT
            if (node.is_not() and node.arg(0).is_equals()) or node.is_lt():
                subnode = node if node.is_lt() else node.arg(0)
                first, second = subnode.arg(0), subnode.arg(1)
                if self._both_variables(first, second):
                    pair = self._normalize_pair(first, second)
                    return [], [pair]

            # Equals o LE
            if node.is_equals() or node.is_le():
                first, second = node.arg(0), node.arg(1)
                if self._both_variables(first, second):
                    pair = self._normalize_pair(first, second)
                    return [pair], []

            return [], []

        def _get_equalities_from_preconditions(
                self, problem: Problem, preconditions: List[FNode]
        ) -> Tuple[List[Tuple[FNode, FNode]], List[Tuple[FNode, FNode]]]:
            """Wrapper to process the preconditions list (AND implicit between them)."""
            if not preconditions:
                return [], []

            all_eq = []
            all_neq = []

            # Com les precondicions tenen un AND implícit entre elles, acumulem totes
            for precondition in preconditions:
                eq, neq = self._get_equalities(problem, precondition)
                all_eq.extend(eq)
                all_neq.extend(neq)

            # Eliminar duplicats
            all_eq = list(dict.fromkeys(all_eq))
            all_neq = list(dict.fromkeys(all_neq))

            return all_eq, all_neq

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
            new_to_old = self._transform_actions(problem, new_problem)

            # ========== Transform Axioms ==========
            for axiom in problem.axioms:
                new_axiom = axiom.clone()
                new_axiom.name = get_fresh_name(new_problem, axiom.name)
                new_axiom.clear_preconditions()
                new_axiom.clear_effects()
                # Transform preconditions
                skip_axiom = False
                for precondition in axiom.preconditions:
                    new_precondition = self._transform_node(problem, new_problem, precondition)
                    if new_precondition is None or new_precondition == FALSE():
                        skip_axiom = True
                        break
                    new_axiom.add_precondition(new_precondition)
                if skip_axiom:
                    continue

                # Transform effects
                for effect in axiom.effects:
                    new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                    new_condition = self._transform_node(problem, new_problem, effect.condition)
                    new_value = self._transform_node(problem, new_problem, effect.value)
                    if new_fluent is None or new_condition is None or new_value is None:
                        skip_axiom = True
                        break
                    new_axiom.add_effect(new_fluent, new_value, new_condition, effect.forall)
                if not skip_axiom:
                    new_to_old[new_axiom] = axiom
                    new_problem.add_axiom(new_axiom)

            # ========== Transform Goals ==========
            for goal in problem.goals:
                new_goal = self._transform_node(problem, new_problem, goal)
                if new_goal is None:
                    raise UPProblemDefinitionError(
                        f"Goal cannot be translated after integer removal: {goal}"
                    )
                new_problem.add_goal(new_goal)

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

    def _select_non_overlapping_mfi(
            self,
            solutions: list[dict[str, int]],
            sorted_mfi: list[tuple[frozenset, int, int]]
    ) -> tuple[list[tuple[frozenset, set[int]]], list[int]]:
        """
        Greedily select MFI with highest area, removing overlapping ones.

        Args:
            solutions: Original list of solution dicts
            sorted_mfi: List of (itemset, freq, area) sorted by area descending

        Returns:
            (selected_mfi, remaining_indices) where:
            - selected_mfi: List of (itemset, covered_tuple_indices)
            - remaining_indices: Indices of tuples not covered by any MFI
        """

        # Precompute which tuples each itemset covers
        def get_coverage(itemset: frozenset) -> set[int]:
            """Get indices of tuples that contain this itemset"""
            covered = set()
            for idx, solution in enumerate(solutions):
                # Check if all (var, val) pairs in itemset match the solution
                if all(solution.get(var) == val for var, val in itemset):
                    covered.add(idx)
            return covered

        # Track which tuples are still uncovered
        uncovered = set(range(len(solutions)))
        selected = []

        for itemset, freq, area in sorted_mfi:
            if not uncovered:
                break

            # Get coverage of this itemset (only among uncovered tuples)
            coverage = get_coverage(itemset) & uncovered

            if len(coverage) >= 2:  # min_support = 2
                selected.append((itemset, coverage))
                uncovered -= coverage

        return selected, list(uncovered)

    def _compute_area(self, itemset: frozenset, freq: int) -> int:
        """Compute area measure: size × frequency"""
        return len(itemset) * freq

    def _sort_by_area(self, maximal: dict[frozenset, int]) -> list[tuple[frozenset, int, int]]:
        """
        Sort maximal itemsets by area (descending).

        Returns:
            List of (itemset, frequency, area) tuples, sorted by area descending
        """
        with_area = []
        for itemset, freq in maximal.items():
            area = self._compute_area(itemset, freq)
            with_area.append((itemset, freq, area))

        # Sort by area descending, then by frequency descending (tie-breaker)
        with_area.sort(key=lambda x: (-x[2], -x[1]))
        return with_area

    def _filter_maximal_itemsets(self, frequencies: dict[frozenset, int]) -> dict[frozenset, int]:
        """
        Filter to keep only maximal frequent itemsets.
        An itemset is maximal if no superset is also frequent.

        Args:
            frequencies: Dict mapping itemset to frequency

        Returns:
            Dict with only maximal itemsets
        """
        all_itemsets = set(frequencies.keys())
        non_maximal = set()

        for itemset in all_itemsets:
            # Check if any superset exists in frequencies
            for other in all_itemsets:
                if itemset < other:  # itemset is strict subset of other
                    non_maximal.add(itemset)
                    break

        # Return only maximal itemsets
        return {itemset: freq for itemset, freq in frequencies.items()
                if itemset not in non_maximal}

    def _compute_itemset_frequencies(self, solutions: list[dict[str, int]], min_support: int = 2) -> dict[
        frozenset, int]:
        """
        Compute frequency of all itemsets that appear in at least min_support solutions.

        Args:
            solutions: List of solution dicts
            min_support: Minimum frequency threshold

        Returns:
            Dict mapping itemset (as frozenset of (var, val) tuples) to frequency
        """
        if not solutions:
            return {}

        variables = list(solutions[0].keys())
        num_vars = len(variables)

        # Count frequencies for all possible itemsets
        frequencies = {}

        # For each itemset size (1, 2, 3, ..., num_vars)
        for size in range(1, num_vars + 1):
            # For each combination of variables
            for var_combo in combinations(variables, size):
                # For each solution, extract the itemset for these variables
                itemset_counts = {}

                for solution in solutions:
                    # Create itemset: frozenset of (variable, value) pairs
                    itemset = frozenset((var, solution[var]) for var in var_combo)
                    itemset_counts[itemset] = itemset_counts.get(itemset, 0) + 1

                # Keep only frequent itemsets
                for itemset, count in itemset_counts.items():
                    if count >= min_support:
                        frequencies[itemset] = count

        return frequencies

    def _entries_to_formula(
            self,
            new_problem: Problem,
            variables: bidict,
            entries: list[dict],
            default_tuples: list[dict]
    ) -> FNode:
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
                if fnode and fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                    if fluent.type.is_int_type():
                        num_obj = self._get_number_object(new_problem, value)
                        pattern_constraints.append(Equals(new_fluent, num_obj))

            if not subtable_vars:
                # No subtable - just the pattern
                if pattern_constraints:
                    or_clauses.append(And(pattern_constraints))
            elif len(subtable_tuples) == 1:
                # Single subtable tuple - add to pattern
                subtable_constraints = []
                for var_str, value in subtable_tuples[0].items():
                    fnode = var_str_to_fnode.get(var_str)
                    if fnode and fnode.is_fluent_exp():
                        fluent = fnode.fluent()
                        new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                        if fluent.type.is_int_type():
                            num_obj = self._get_number_object(new_problem, value)
                            subtable_constraints.append(Equals(new_fluent, num_obj))
                or_clauses.append(And(pattern_constraints + subtable_constraints))
            else:
                # Multiple subtable tuples
                # Check if we can simplify (single variable covering full domain)
                if len(subtable_vars) == 1:
                    var_str = subtable_vars[0]
                    values = {t[var_str] for t in subtable_tuples}
                    fnode = var_str_to_fnode.get(var_str)

                    if fnode and fnode.is_fluent_exp():
                        fluent = fnode.fluent()
                        new_fluent = new_problem.fluent(fluent.name)(*fnode.args)

                        if fluent.type.is_int_type():
                            lb = fluent.type.lower_bound
                            ub = fluent.type.upper_bound
                            all_values = set(range(lb, ub + 1))

                            if values == all_values:
                                # Full domain - no constraint needed for this variable!
                                if pattern_constraints:
                                    or_clauses.append(And(pattern_constraints))
                                continue
                            elif len(values) == 1:
                                val = next(iter(values))
                                num_obj = self._get_number_object(new_problem, val)
                                pattern_constraints.append(Equals(new_fluent, num_obj))
                                or_clauses.append(And(pattern_constraints))
                                continue
                            else:
                                # Multiple values but not full domain
                                or_values = [
                                    Equals(new_fluent, self._get_number_object(new_problem, v))
                                    for v in sorted(values)
                                ]
                                pattern_constraints.append(Or(or_values))
                                or_clauses.append(And(pattern_constraints))
                                continue

                # General case: multiple variables or couldn't simplify
                # Create OR of all subtable combinations
                subtable_or_clauses = []
                for subtable_tuple in subtable_tuples:
                    subtable_constraints = []
                    for var_str, value in subtable_tuple.items():
                        fnode = var_str_to_fnode.get(var_str)
                        if fnode and fnode.is_fluent_exp():
                            fluent = fnode.fluent()
                            new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                            if fluent.type.is_int_type():
                                num_obj = self._get_number_object(new_problem, value)
                                subtable_constraints.append(Equals(new_fluent, num_obj))
                    if subtable_constraints:
                        subtable_or_clauses.append(And(subtable_constraints))

                if subtable_or_clauses:
                    if len(subtable_or_clauses) == 1:
                        full_clause = And(pattern_constraints + [subtable_or_clauses[0]])
                    else:
                        full_clause = And(pattern_constraints + [Or(subtable_or_clauses)])
                    or_clauses.append(full_clause)

        # Process default tuples
        for solution in default_tuples:
            and_clauses = []
            for var_str, value in solution.items():
                fnode = var_str_to_fnode.get(var_str)
                if fnode and fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
                    if fluent.type.is_int_type():
                        num_obj = self._get_number_object(new_problem, value)
                        and_clauses.append(Equals(new_fluent, num_obj))
            if and_clauses:
                or_clauses.append(And(and_clauses))

        if not or_clauses:
            return None

        return Or(or_clauses).simplify() if len(or_clauses) > 1 else or_clauses[0].simplify()

    def _compress_solutions_mfi(
            self,
            new_problem: Problem,
            variables: bidict,
            solutions: list[dict[str, int]],
            min_support: int = 2
    ) -> FNode:
        """
        Compress solutions using MFI + Area measure and convert to formula.

        Args:
            new_problem: The new problem being built
            variables: Mapping from FNode to CP variable
            solutions: List of solution dicts from CP-SAT
            min_support: Minimum frequency threshold

        Returns:
            Compressed DNF formula
        """
        if not solutions:
            return None

        # If too few solutions, don't bother compressing
        if len(solutions) <= min_support:
            return self._convert_to_dnf(new_problem, solutions, variables)

        # Step 1: Find frequent itemsets
        frequencies = self._compute_itemset_frequencies(solutions, min_support)

        if not frequencies:
            return self._convert_to_dnf(new_problem, solutions, variables)

        # Step 2: Filter to maximal
        maximal = self._filter_maximal_itemsets(frequencies)

        if not maximal:
            return self._convert_to_dnf(new_problem, solutions, variables)

        # Step 3: Sort by area
        sorted_mfi = self._sort_by_area(maximal)

        # Step 4: Greedy selection
        selected, remaining = self._select_non_overlapping_mfi(solutions, sorted_mfi)

        # Step 5: Create entries
        entries, default_tuples = self._create_entries(solutions, selected, remaining)

        ###
        #self._verify_compression(solutions, entries, default_tuples)
        ###

        # Step 6: Convert to formula
        formula = self._entries_to_formula(new_problem, variables, entries, default_tuples)
        return formula

    def _transform_action_integers(
            self, problem: Problem, new_problem: Problem, old_action: Action
    ) -> Union[Action, None]:
        """
        Change all integers in the action for their new user-type fluent.
        Returns new_actions
        """
        params = OrderedDict(((p.name, p.type) for p in old_action.parameters))
        new_action = InstantaneousAction(old_action.name, _parameters=params, _env=problem.environment)
        self._action_static_fluents = self._static_fluents

        # Create the CP model
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        # 0) Normalize and replace static fluents in preconditions
        normalized_preconditions = []
        for precondition in old_action.preconditions:
            np = self._to_nnf(new_problem, precondition)
            if np.is_and():
                for new_precondition in np.args:
                    if new_precondition is not TRUE():
                        normalized_preconditions.append(new_precondition)
            else:
                if np is not TRUE():
                    normalized_preconditions.append(np)

        # I si en comptes de guardar els bounds el que fem es passar les precondicions al cp-solver?
        # Ja se n'encarregara el cp-solver de fer tot aixo
        # el que faig es afegir les normalized

        # 1) Get the bounds of the fluents and parameters for this action
        #eq, neq = self._get_equalities_from_preconditions(new_problem, normalized_preconditions)
        # Transform preconditions
        #for precondition in old_action.preconditions:
        #    print("precondition", precondition)
        #    new_precondition = self._transform_node(problem, new_problem, precondition, variables, cp_model_obj)
        #    print("new_precondition", new_precondition)
        #    if new_precondition is None or new_precondition == FALSE():
        #        # Impossible action
        #        return None
        #    new_action.add_precondition(new_precondition)

        arithmetic_preconditions = []
        result_var = self._add_cp_constraints(problem, And(normalized_preconditions), variables, cp_model_obj)
        cp_model_obj.add(result_var == 1)
        for precondition in normalized_preconditions:
            new_precondition = self._transform_node(problem, new_problem, precondition)
            if new_precondition == "has_arithmetic":
                arithmetic_preconditions.append(precondition)
                continue
            if new_precondition is None or new_precondition == FALSE():
                # Impossible action
                return None
            new_action.add_precondition(new_precondition)

        if arithmetic_preconditions:
            result = self._solve_with_cp_sat(variables, cp_model_obj)
            # Compression MFI
            compressed_formula = self._compress_solutions_mfi(new_problem, variables, result)

            # or DNF
            #compressed_formula = self._convert_to_dnf(new_problem, result, variables)

            if compressed_formula:
                new_action.add_precondition(compressed_formula)

            #dnf_result = self._convert_to_dnf(new_problem, result, variables)
            #new_action.add_precondition(dnf_result)

        # Transform effects
        for effect in old_action.effects:
            new_condition = self._to_nnf(new_problem, effect.condition)
            new_value = self._to_nnf(new_problem, effect.value)

            temp_effect = Effect(
                effect.fluent,
                new_value,
                new_condition,
                effect.kind,
                effect.forall
            )

            if temp_effect.is_increase() or temp_effect.is_decrease():
                # Increase/decrease effects
                for new_effect in self._transform_increase_decrease_effect(
                        temp_effect, new_problem
                ):
                    new_action.add_effect(
                        new_effect.fluent, new_effect.value, new_effect.condition, new_effect.forall
                    )

            elif new_value.node_type in self.ARITHMETIC_OPS: # arreglar
                # Assignment with arithmetic
                effects_generated = False
                for new_effect in self._transform_arithmetic_assignment(temp_effect, problem, new_problem, variables, cp_model_obj):
                    new_action.add_effect(
                        new_effect.fluent,
                        new_effect.value,
                        new_effect.condition,
                        new_effect.forall
                    )
                    effects_generated = True

                if not effects_generated:
                    # No valid assignments -> skip action
                    return None

            else:
                # Check bounds for assignments ????
                if new_value is None or new_value is None:
                    return None
                if new_condition not in [None, FALSE()]:
                    new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                    new_value = self._transform_node(problem, new_problem, new_value)
                    new_condition = self._transform_node(problem, new_problem, new_condition)
                    new_action.add_effect(new_fluent, new_value, new_condition, effect.forall)
        return new_action

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions by grounding integer parameters."""
        new_to_old = {}
        for action in problem.actions:
            new_action = self._transform_action_integers(problem, new_problem, action)
            if new_action is not None:
                new_problem.add_action(new_action)
                new_to_old[new_action] = action
        return new_to_old


    # ==================== FLUENT TRANSFORMATION ====================

    def _transform_fluents(self, problem: Problem, new_problem: Problem):
        """Transform integer fluents -> user-type fluents."""
        number_ut = UserType('Number')

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            if fluent.type.is_int_type():
                # Integer fluent -> Number-typed fluent
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
        return dict((f,v) for f,v in fluents.items() if f not in modifiable_fluents)

    def _transform_axioms(self, problem: Problem, new_problem: Problem, new_to_old: Dict):
        """Transform axioms"""
        for axiom in problem.axioms:
            # Check for integer parameters
            for param in axiom.parameters:
                if param.type.is_int_type():
                    raise NotImplementedError(
                        "Integer parameters in axioms are not supported!"
                    )
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

        # ========== Transform Action ==========
        new_to_old = self._transform_actions(problem, new_problem)

        # ========== Transform Axioms ==========
        self._transform_axioms(problem, new_problem, new_to_old)

        # ========== Transform Goals ==========
        for goal in problem.goals:
            new_goal = self._transform_node(problem, new_problem, goal)
            if new_goal is None:
                raise UPProblemDefinitionError(
                    f"Goal cannot be translated after integer removal: {goal}"
                )
            if new_goal == "has_arithmetic":
                variables_goal = bidict({})
                cp_model_goal = cp_model.CpModel()
                result_var = self._add_cp_constraints(new_problem, goal, variables_goal, cp_model_goal)
                cp_model_goal.add(result_var == 1)
                result = self._solve_with_cp_sat(variables_goal, cp_model_goal)
                # Compression MFI
                compressed_formula = self._compress_solutions_mfi(new_problem, variables_goal, result)

                # or DNF
                #compressed_formula = self._convert_to_dnf(new_problem, result, variables_goal)

                if compressed_formula:
                    new_problem.add_goal(compressed_formula)
            else:
                new_problem.add_goal(new_goal)

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
