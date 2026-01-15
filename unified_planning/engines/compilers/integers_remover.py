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
from unified_planning.model import Problem, Action, ProblemKind, Variable, Effect, EffectKind, Object, FNode, Fluent, \
    InstantaneousAction
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import get_fresh_name, replace_action, updated_minimize_action_costs
from typing import Dict, Optional, Iterator, OrderedDict, List, Tuple, Union
from functools import partial
from unified_planning.shortcuts import And, Or, Equals, Int, Not, FALSE, GT, GE, Iff, UserType, TRUE, ObjectExp

class IntegersRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes bounded integers from a planning problem.

    Converts integer fluents to object-typed fluents where objects represent numeric values (n0, n1, n2, ...).
    Integer arithmetic and comparisons are handled by enumerating possible value combinations.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGERS_REMOVING)
        self.domains: Dict[str, Tuple[int, int]] = {}
        self._number_objects_cache: Dict[int, FNode] = {}

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
        self._number_objects_cache[value] = ObjectExp(new_object)
        return ObjectExp(new_object)

    def _is_value_in_bounds(self, fluent_name: str, value: int) -> bool:
        """Check if a value is within the bounds of a fluent's domain."""
        if fluent_name not in self.domains:
            return True
        lb, ub = self.domains[fluent_name]
        return lb <= value <= ub

    def _has_arithmetic(self, node: FNode) -> bool:
        """Check if expression contains arithmetic operations."""
        if node.node_type in self.ARITHMETIC_OPS or node.is_le() or node.is_lt():
            return True
        return any(self._has_arithmetic(arg) for arg in node.args)

    def _find_integer_fluents(self, node: FNode) -> dict[FNode, list[int]]:
        """Extract all integer fluents and their domains from expression."""
        fluents = {}
        if node.is_fluent_exp() and node.fluent().type.is_int_type():
            fluent_type = node.fluent().type
            fluents[node] = list(range(
                fluent_type.lower_bound,
                fluent_type.upper_bound + 1
            ))
        for arg in node.args:
            fluents.update(self._find_integer_fluents(arg))
        return fluents

    def _evaluate_with_assignment(self, problem: Problem, node: FNode, assignment: dict[FNode, int]) -> FNode:
        """Evaluate expression by substituting fluent values from assignment."""
        em = problem.environment.expression_manager
        if node.is_fluent_exp():
            return Int(assignment[node])
        if not node.args:
            return node
        new_args = [
            self._evaluate_with_assignment(problem, arg, assignment)
            for arg in node.args
        ]
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _enumerate_arithmetic_combinations(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode
    ) -> Optional[FNode]:
        """
        Enumerate all possible combinations of integer fluent values and
        return a formula representing when the expression is true.
        """
        fluent_domains = self._find_integer_fluents(node)
        if not fluent_domains:
            return None

        assignments = []
        for combination in product(*fluent_domains.values()):
            assignment = dict(zip(fluent_domains.keys(), combination))

            try:
                result_val = self._evaluate_with_assignment(old_problem, node, assignment).constant_value()
                if result_val is True:
                    assignments.append(assignment)
            except:
                continue

        if assignments:
            # primer simplificar!!
            print("old --------------", len(assignments))
            simplified_assignments = self._simplify_solutions(assignments)
            print("new --------------", len(simplified_assignments))
            return self._generate_node_with_assignments(old_problem, new_problem, simplified_assignments)
        return FALSE()

    def _generate_node_with_assignments(self, old_problem, new_problem, assignments: List[dict[FNode, int]]) -> FNode:
        all_conditions = []
        for assignment in assignments:
            print("assignment", assignment)
            conditions = []
            for fluent_node, value in assignment.items():
                # Transformar el FNode de old_problem a new_problem
                new_fluent_fnode = self._transform_node(old_problem, new_problem, fluent_node)
                if isinstance(value, set):
                    # Multiple values: (f = v1) OR (f = v2) OR ...
                    conditions.append(Or(
                        [Equals(new_fluent_fnode, self._get_number_object(new_problem, v)) for v in value]
                    ))
                else:
                    # Single value
                    conditions.append(Equals(new_fluent_fnode, self._get_number_object(new_problem, value)))

            all_conditions.append(And(conditions))
        print("all conditions")
        return Or(all_conditions).simplify()

    def _simplify_solutions(self, solutions: list[dict[FNode, int]]) -> list[dict[FNode, int]]:
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
        for var_node in all_vars:
            groups = {}

            for i, sol in enumerate(solutions):
                if i in used:
                    continue

                # Key = all values except the varying variable
                key = tuple((k, v) for k, v in sorted(sol.items()) if k != var_node)
                groups.setdefault(key, []).append((i, sol[var_node]))

            # Compact groups with multiple values
            for key, indices_vals in groups.items():
                if len(indices_vals) <= 1:
                    continue

                # Mark as used
                for idx, _ in indices_vals:
                    used.add(idx)

                compact = dict(key)
                values_set = {val for _, val in indices_vals}

                # Check if covers entire domain
                if var_node.fluent().type.is_int_type():
                    lb = var_node.fluent().type.lower_bound
                    ub = var_node.fluent().type.upper_bound
                    domain = set(range(lb, ub + 1))

                    # Only include if not entire domain
                    if values_set != domain:
                        compact[var_node] = values_set
                elif var_node.fluent().type.is_bool_type():
                    if values_set != {0, 1}:
                        compact[var_node] = values_set
                else:
                    compact[var_node] = values_set

                simplified.append(compact)

        # Add ungrouped solutions
        for i, sol in enumerate(solutions):
            if i not in used:
                simplified.append(sol)

        return simplified


    # ==================== NODE TRANSFORMATION ====================

    def _transform_node(self, old_problem: Problem, new_problem: Problem, node: FNode) -> Optional[FNode]:
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

        # Expressions with comparisons involving arithmetic - enumerate combinations
        if self._has_arithmetic(node):
            return self._enumerate_arithmetic_combinations(old_problem, new_problem, node)

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
                Variable(v.name, UserType('Number')) if v.type.is_int_type() else v
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
        lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
        new_condition = self._transform_node(old_problem, new_problem, effect.condition)
        new_fluent = new_problem.fluent(fluent.name)(*effect.fluent.args)

        # Calculate the valid bounds
        try:
            int_value = effect.value.simplify().constant_value()
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
                And(Equals(new_fluent, old_obj), new_condition).simplify(),
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
            new_problem: Problem
    ) -> Iterator[Effect]:
        """Handle assignments with arithmetic expressions by enumerating combinations."""
        fluent_domains = self._find_integer_fluents(effect.value)
        if not fluent_domains:
            return

        lb, ub = effect.fluent.fluent().type.lower_bound, effect.fluent.fluent().type.upper_bound

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
            for fluent_node, value in assignment.items():
                new_fluent_node = self._transform_node(old_problem, new_problem, fluent_node)
                conditions.append(Equals(new_fluent_node, self._get_number_object(new_problem, value)))

            value_to_conditions.setdefault(result_val, []).append(And(conditions))

        # Create effects
        new_fluent = self._transform_node(old_problem, new_problem, effect.fluent)
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

    # ==================== ACTION TRANSFORMATION ====================

    def _transform_action_integers(
            self, problem: Problem, new_problem: Problem, old_action: Action
    ) -> Union[Action, None]:
        """
        Change all integers in the action for their new user-type fluent.
        Returns new_actions
        """
        params = OrderedDict(((p.name, p.type) for p in old_action.parameters))
        new_action = InstantaneousAction(old_action.name, _parameters=params, _env=problem.environment)
        # Transform preconditions
        for precondition in old_action.preconditions:
            new_precondition = self._transform_node(problem, new_problem, precondition)
            if new_precondition is None or new_precondition == FALSE():
                # Impossible action
                return None
            new_action.add_precondition(new_precondition)

        # Transform effects
        for effect in old_action.effects:
            if effect.is_increase() or effect.is_decrease():
                # Increase/decrease effects
                for new_effect in self._transform_increase_decrease_effect(effect, problem, new_problem):
                    new_action.add_effect(
                        new_effect.fluent, new_effect.value, new_effect.condition, new_effect.forall
                    )

            elif effect.value.node_type in self.ARITHMETIC_OPS:
                # Assignment with arithmetic
                effects_generated = False
                for new_effect in self._transform_arithmetic_assignment(effect, problem, new_problem):
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
                # Check bounds for assignments
                if effect.fluent.fluent().type.is_int_type() and effect.value.is_int_constant():
                    value = effect.value.constant_value()
                    if not self._is_value_in_bounds(effect.fluent.fluent().name, value):
                        if effect.condition == TRUE():
                            return None
                        continue

                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_condition = self._transform_node(problem, new_problem, effect.condition)
                new_value = self._transform_node(problem, new_problem, effect.value)

                if new_value is None or new_value is None:
                    return None
                if new_condition not in [None, FALSE()]:
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
                self.domains[fluent.name] = (lb, ub)

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

        # Transform components
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
                print("---------------------- precondition:", precondition)
                new_precondition = self._transform_node(problem, new_problem, precondition)
                print("---------------------- new_precondition:", new_precondition)
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
