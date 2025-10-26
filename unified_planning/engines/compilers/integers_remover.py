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
from itertools import product
from bidict import bidict
from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError, UPValueError
from unified_planning.model import Problem, Action, ProblemKind, Variable, Effect, EffectKind, Object, FNode, Fluent
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import get_fresh_name, replace_action, updated_minimize_action_costs
from typing import Dict, Optional, Iterator
from functools import partial
from unified_planning.shortcuts import And, Or, Equals, Int, Not, FALSE, GT, GE, Iff

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
        self.lb = None
        self.ub = None

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
        em = problem.environment.expression_manager
        tm = problem.environment.type_manager
        number_type = tm.UserType('Number')

        obj_name = f'n{value}'
        if not problem.has_object(obj_name):
            problem.add_object(Object(obj_name, number_type))

        return em.ObjectExp(problem.object(obj_name))

    def _has_integers(self, node: FNode) -> bool:
        """Check if expression contains integer fluents or operations."""
        if node.is_fluent_exp() and node.fluent().type.is_int_type():
            return True
        if node.is_int_constant() or node.is_le() or node.is_lt():
            return True
        return any(self._has_integers(arg) for arg in node.args)

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

    # ==================== EXPRESSION TRANSFORMATION ====================

    def _evaluate_with_assignment(self, problem: Problem, node: FNode, assignment: dict[str, int]) -> FNode:
        """Evaluate expression by substituting fluent values from assignment."""
        em = problem.environment.expression_manager
        if node.is_fluent_exp():
            return Int(assignment[node.fluent().name])
        if not node.args:
            return node
        new_args = [
            self._evaluate_with_assignment(problem, arg, assignment)
            for arg in node.args
        ]
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    constants = {
        OperatorKind.BOOL_CONSTANT, OperatorKind.INT_CONSTANT, OperatorKind.LIST_CONSTANT, OperatorKind.REAL_CONSTANT,
        OperatorKind.FLUENT_EXP, OperatorKind.OBJECT_EXP, OperatorKind.PARAM_EXP, OperatorKind.VARIABLE_EXP
    }

    def _apply_negation(self, node: FNode) -> FNode:
        """Apply De Morgan's laws to push negation inward."""

        if node.node_type in self.constants:
            return Not(node)
        elif node.node_type == OperatorKind.LE:
            return GT(node.arg(0), node.arg(1)).simplify()
        elif node.node_type == OperatorKind.LT:
            return GE(node.arg(0), node.arg(1)).simplify()
        elif node.node_type == OperatorKind.AND:
            return Or(Not(arg) for arg in node.args).simplify()
        elif node.node_type == OperatorKind.OR:
            return And(Not(arg) for arg in node.args).simplify()
        elif node.node_type == OperatorKind.NOT:
            return node.arg(0)
        else:
            raise UPProblemDefinitionError(f"Cannot negate {node.node_type}")

    def _to_nnf(self, problem: Problem, node: FNode) -> FNode:
        """Convert expression to Negation Normal Form."""
        em = problem.environment.expression_manager
        if node.node_type in self.constants:
            return node
        elif node.is_not():
            if node.arg(0).node_type in self.constants or node.arg(0).is_equals():
                return Not(self._to_nnf(problem, node.arg(0)))
            else:
                return self._to_nnf(problem, self._apply_negation(node.arg(0)))
        elif node.is_implies():
            return Or(
                self._to_nnf(problem, Not(node.arg(0))),
                self._to_nnf(problem, node.arg(1))
            )
        elif node.is_iff():
            return And(
                Or(self._to_nnf(problem, Not(node.arg(0))), self._to_nnf(problem, node.arg(1))),
                Or(self._to_nnf(problem, node.arg(0)), self._to_nnf(problem, Not(node.arg(1))))
            )
        else:
            new_args = tuple(self._to_nnf(problem, arg) for arg in node.args)
            return em.create_node(node.node_type, new_args)

    # ==================== CP-SAT Constraint Solving ====================

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
                    fluent.name
                )
            else:
                var = model.new_bool_var(str(node))
            variables[node] = var
            return var

        # AND
        if node.is_and():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            and_var = model.new_bool_var(f"and_{id(node)}")
            model.add(sum(child_vars) == len(child_vars)).only_enforce_if(and_var)
            model.add(sum(child_vars) < len(child_vars)).only_enforce_if(~and_var)
            return and_var

        # OR
        if node.is_or():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            or_var = model.new_bool_var(f"or_{id(node)}")
            model.add(sum(child_vars) >= 1).only_enforce_if(or_var)
            model.add(sum(child_vars) == 0).only_enforce_if(~or_var)
            return or_var

        # NOT
        if node.is_not():
            inner = self._add_cp_constraints(problem, node.arg(0), variables, model)
            not_var = model.new_bool_var(f"not_{id(node)}")
            model.add(inner == 0).only_enforce_if(not_var)
            model.add(inner == 1).only_enforce_if(~not_var)
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
            model.add(left != right).only_enforce_if(~eq_var)
            return eq_var

        # Comparisons and arithmetic
        if node.is_lt() or node.is_le():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)
            var = model.new_bool_var(f"{'lt' if node.is_lt() else 'le'}_{id(node)}")
            if node.is_lt():
                model.add(left < right).only_enforce_if(var)
                model.add(left >= right).only_enforce_if(~var)
            else:
                model.add(left <= right).only_enforce_if(var)
                model.add(left > right).only_enforce_if(~var)
            return var
        if node.is_plus() or node.is_minus() or node.is_times():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)
            if node.is_plus():
                return left + right
            elif node.is_minus():
                return left - right
            else:
                return left * right

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

    def _solve_with_cp_sat(self, old_problem: Problem, new_problem: Problem, node: FNode) -> Optional[FNode]:
        """Use CP-SAT solver to enumerate valid value assignments."""
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        # Build constraints
        result = self._add_cp_constraints(old_problem, node, variables, cp_model_obj)

        # Ensure constraint is satisfied
        if isinstance(result, cp_model.IntVar):
            cp_model_obj.add(result == 1)
        else:
            cp_model_obj.add(result)

        # Solve
        solver = cp_model.CpSolver()
        collector = CPSolutionCollector(list(variables.values()))
        solver.parameters.enumerate_all_solutions = True
        status = solver.solve(cp_model_obj, collector)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        solutions = collector.solutions
        if len(solutions) > 1:
            solutions = self._simplify_solutions(variables, solutions)

        # Convert solutions to formula
        or_clauses = []
        for solution in solutions:
            and_clauses = []

            for var_str, value in solution.items():
                # Find corresponding FNode
                fnode = None
                for node_key, var in variables.items():
                    if str(node_key) == var_str:
                        fnode = node_key
                        break
                if not fnode:
                    continue

                if fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)

                    if fluent.type.is_int_type():
                        if isinstance(value, set):
                            # Multiple values: (f = v1) OR (f = v2) OR ...
                            or_eq = [Equals(new_fluent, self._get_number_object(new_problem, v)) for v in value]
                            and_clauses.append(Or(or_eq))
                        else:
                            # Single value
                            and_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                    elif fluent.type.is_bool_type():
                        bool_val = (value == 1)
                        and_clauses.append(Iff(fnode, bool_val))
                    else:
                        raise UPProblemDefinitionError(f"Unexpected fluent type in CP-SAT solution")
                else:
                    # Boolean variable
                    and_clauses.append(fnode if value == 1 else Not(fnode))

            or_clauses.append(And(and_clauses).simplify())
        return Or(or_clauses).simplify()

    # ==================== NODE TRANSFORMATION ====================

    def _transform_node(self, old_problem: Problem, new_problem: Problem, node: FNode) -> Optional[FNode]:
        """Transform expression node to use Number objects instead of integers."""
        em = new_problem.environment.expression_manager
        tm = new_problem.environment.type_manager

        # Integer constants become Number objects
        if node.is_int_constant():
            return self._get_number_object(new_problem, node.constant_value())

        # Integer fluents
        if node.is_fluent_exp() and node.fluent().type.is_int_type():
            return new_problem.fluent(node.fluent().name)(*node.args)

        # Other terminals
        if node.is_object_exp() or node.is_fluent_exp() or node.is_constant() or node.is_parameter_exp():
            return node

        # Check for arithmetic operations
        if node.node_type in self.ARITHMETIC_OPS:
            raise UPProblemDefinitionError(
                f"Arithmetic operation {self.ARITHMETIC_OPS[node.node_type]} "
                f"not supported as external expression"
            )

        # Expressions with integers need CP-SAT solving
        if self._has_integers(node):
            return self._solve_with_cp_sat(old_problem, new_problem, node)

        # Recursively transform children
        new_args = []
        for arg in node.args:
            transformed = self._transform_node(old_problem, new_problem, arg)
            if transformed is None:
                return None
            new_args.append(transformed)

        # Handle quantifiers
        if node.is_exists() or node.is_forall():
            new_vars = [
                Variable(v.name, tm.UserType('Number')) if v.type.is_int_type() else v
                for v in node.variables()
            ]
            return em.create_node(node.node_type, tuple(new_args), payload=tuple(new_vars)).simplify()

        return em.create_node(node.node_type, tuple(new_args)).simplify()

    # ==================== EFFECT TRANSFORMATION ====================

    def _transform_increase_decrease_effect(
            self,
            effect: Effect,
            old_problem: Problem,
            new_problem: Problem
    ) -> Iterator[Effect]:
        """Convert increase/decrease effects to conditional assignments."""
        fluent = effect.fluent.fluent()
        lb = fluent.type.lower_bound
        ub = fluent.type.upper_bound
        new_condition = self._transform_node(old_problem, new_problem, effect.condition)
        new_fluent = new_problem.fluent(fluent.name)(*effect.fluent.args)
        returned = set()

        for i in range(lb, ub + 1):
            if effect.is_increase():
                next_val = i + effect.value
            else:
                next_val = i - effect.value
            try:
                next_val_int = next_val.simplify().constant_value()
            except:
                continue

            if not (lb <= next_val_int <= ub):
                continue
            try:
                old_obj = self._get_number_object(new_problem, i)
                new_obj = self._get_number_object(new_problem, next_val_int)
                new_effect = Effect(
                    new_fluent,
                    new_obj,
                    And(Equals(new_fluent, old_obj), new_condition).simplify(),
                    EffectKind.ASSIGN,
                    effect.forall
                )
                if new_effect not in returned:
                    yield new_effect
                    returned.add(new_effect)
            except UPValueError:
                continue

    def _transform_arithmetic_assignment(
            self,
            effect: Effect,
            old_problem: Problem,
            new_problem: Problem
    ) -> Iterator[Effect]:
        """Handle assignments with arithmetic expressions by enumerating combinations."""
        fluent_domains = self._find_integer_fluents(effect.value)

        if not fluent_domains:
            return

        lb = effect.fluent.fluent().type.lower_bound
        ub = effect.fluent.fluent().type.upper_bound

        # Group assignments by result value
        value_to_conditions = {}

        for combination in product(*fluent_domains.values()):
            assignment = dict(zip(fluent_domains.keys(), combination))

            try:
                result_val = self._evaluate_with_assignment(new_problem, effect.value, assignment).constant_value()
            except:
                continue

            if not (lb <= result_val <= ub):
                continue

            # Build condition for this assignment
            conditions = []
            for fluent_name, value in assignment.items():
                fluent = new_problem.fluent(fluent_name)
                conditions.append(Equals(fluent, self._get_number_object(new_problem, value)))

            value_to_conditions.setdefault(result_val, []).append(And(conditions))

        # Create effects
        new_fluent = new_problem.fluent(effect.fluent.fluent().name)
        new_base_condition = self._transform_node(old_problem, new_problem, effect.condition)

        for value, condition_list in value_to_conditions.items():
            full_condition = And(new_base_condition, Or(condition_list)).simplify()
            yield Effect(
                new_fluent,
                self._get_number_object(new_problem, value),
                full_condition,
                EffectKind.ASSIGN,
                effect.forall
            )

    def _compile(
            self,
            problem: "up.model.AbstractProblem",
            compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_axioms()
        new_problem.initial_values.clear()
        new_problem.clear_quality_metrics()

        tm = new_problem.environment.type_manager
        number_type = tm.UserType('Number')
        new_to_old: Dict[Action, Action] = {}

        # ========== Transform Fluents ==========
        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            # Check signature
            new_signature = []
            for param in fluent.signature:
                if param.type.is_int_type():
                    raise NotImplementedError(
                        f"Fluent '{fluent.name}' has integer parameter '{param.name}', "
                        f"which is not supported"
                    )
                new_signature.append(param)

            # Transform fluent type
            if fluent.type.is_int_type():
                # Integer fluent -> Number-typed fluent
                new_fluent = Fluent(fluent.name, number_type, new_signature, new_problem.environment)

                if default_value is not None:
                    default_obj = self._get_number_object(new_problem, default_value)
                    new_problem.add_fluent(new_fluent, default_initial_value=default_obj)
                else:
                    new_problem.add_fluent(new_fluent)

                # Set initial values
                for fluent_exp, value in problem.initial_values.items():
                    if fluent_exp.fluent().name == fluent.name and value != default_value:
                        new_problem.set_initial_value(
                            new_problem.fluent(fluent.name)(*fluent_exp.args),
                            self._get_number_object(new_problem, value)
                        )
            else:
                # Non-integer fluent
                new_fluent = Fluent(
                    fluent.name,
                    fluent.type,
                    new_signature,
                    new_problem.environment
                )
                new_problem.add_fluent(new_fluent, default_initial_value=default_value)

                # Set initial values
                for fluent_exp, value in problem.initial_values.items():
                    if fluent_exp.fluent().name == fluent.name and value != default_value:
                        new_problem.set_initial_value(fluent_exp, value)

        # ========== Transform Actions ==========
        for action in problem.actions:
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()
            new_action.clear_effects()

            # Transform preconditions
            new_preconditions = self._transform_node(problem, new_problem, And(action.preconditions))
            if new_preconditions is None or new_preconditions == FALSE():
                # Unsatisfiable precondition -> skip action
                continue

            new_action.add_precondition(new_preconditions)

            # Transform effects
            skip_action = False
            for effect in action.effects:
                if effect.is_increase() or effect.is_decrease():
                    # Increase/decrease effects
                    for new_effect in self._transform_increase_decrease_effect(effect, problem, new_problem):
                        new_action.add_effect(
                            new_effect.fluent, new_effect.value, new_effect.condition, new_effect.forall
                        )

                elif effect.value.node_type in self.ARITHMETIC_OPS:
                    # Assignment with arithmetic
                    effects_generated = False
                    for new_effect in self._transform_arithmetic_assignment(
                            effect, problem, new_problem
                    ):
                        new_action.add_effect(
                            new_effect.fluent,
                            new_effect.value,
                            new_effect.condition,
                            new_effect.forall
                        )
                        effects_generated = True

                    if not effects_generated:
                        # No valid assignments -> skip action
                        skip_action = True
                        break

                else:
                    new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                    new_condition = self._transform_node(problem, new_problem, effect.condition)
                    new_value = self._transform_node(problem, new_problem, effect.value)
                    if new_fluent is None or new_condition is None or new_value is None:
                        skip_action = True
                        break
                    new_action.add_effect(new_fluent, new_value, new_condition, effect.forall)
            if not skip_action:
                new_to_old[new_action] = action
                new_problem.add_action(new_action)

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
