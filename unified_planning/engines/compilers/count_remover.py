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
"""This module defines the count remover class."""
import itertools
import unified_planning as up
import unified_planning.engines as engines
from unified_planning.exceptions import UPValueError
from unified_planning.model.walkers import simplifier
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import Problem, Action, ProblemKind, OperatorKind, FNode
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action, get_fresh_name, updated_minimize_action_costs
from typing import Dict, Optional, Tuple, List
from functools import partial
from unified_planning.shortcuts import Not, And, Or, FALSE, TRUE

class CountRemover(engines.engine.Engine, CompilerMixin):
    """
    Removes Count expressions that appear inside comparisons by expanding them into boolean formulas.

    Supports:
    - Count vs constant
    - Count vs Count
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.COUNT_REMOVING)

    @property
    def name(self):
        return "crm"

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
        supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_parameters("REAL_ACTION_PARAMETERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_fluents_type("ARRAY_FLUENTS")
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
        return problem_kind <= CountRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.COUNT_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("COUNTING")
        return new_kind

    # ==================== EXPRESSION TRANSFORMATION ====================

    def _transform_count_comparison(self, new_problem: Problem, node: FNode) -> FNode:
        """
        Transform comparisons involving Count expressions.
        Handles:
        - Count(args) op constant
        - constant op Count(args)
        - Count(args1) op Count(args2)
        """
        left = node.arg(0)
        right = node.arg(1)
        op = node.node_type
        left_is_count = left.is_count()
        right_is_count = right.is_count()

        if left_is_count and right.is_int_constant():
            return self._expand_count_vs_constant(left, right.constant_value(), op)
        elif right_is_count and left.is_int_constant():
            return self._expand_constant_vs_count(left.constant_value(), right, op)
        elif left_is_count and right_is_count:
            return self._expand_count_vs_count(left, right, op)
        else:
            raise UPValueError(f"Unexpected Count comparison structure: {node}")

    def _transform_expression(self, new_problem: Problem, node: FNode) -> FNode:
        """Transform expressions recursively."""
        # Base cases: no transformation needed
        if node.is_fluent_exp() or node.is_parameter_exp() or node.is_constant():
            return node

        # Check if this node is a comparison involving Count
        comparison_ops = {OperatorKind.LT, OperatorKind.LE, OperatorKind.EQUALS}
        if node.node_type in comparison_ops and any(arg.is_count() for arg in node.args):
            return self._transform_count_comparison(new_problem, node)

        # Transform all arguments
        em = new_problem.environment.expression_manager
        new_args = [
            self._transform_expression(new_problem, arg)
            for arg in node.args
        ]
        return em.create_node(node.node_type, tuple(new_args))

    # ==================== COUNT VS CONSTANT ====================

    def _exactly_k_combinations(self, arguments: List[FNode], k: int) -> List[FNode]:
        """Generate all formulas where exactly k arguments are true."""
        n = len(arguments)
        if k > n or k < 0:
            return []
        combinations = []
        for true_indices in itertools.combinations(range(n), k):
            true_set = set(true_indices)
            literals = [
                arguments[i] if i in true_set else Not(arguments[i])
                for i in range(n)
            ]
            combinations.append(And(*literals))
        return combinations

    def _exactly_k_true_formula(self, arguments: List[FNode], min_true: int, max_true: int) -> FNode:
        """
        Generate formula: "between min_true and max_true arguments are true".
        Returns: Or of all combinations where exactly k args are true,
                 for k in [min_true, max_true].
        """
        n = len(arguments)

        # Edge cases
        if min_true > n or max_true < 0:
            return FALSE()
        if min_true < 0:
            min_true = 0
        if max_true > n:
            max_true = n
        if min_true == 0 and max_true >= n:
            return TRUE()

        # OPTIMIZED: Special cases
        if min_true == max_true:
            # Exact count - enumerate
            clauses = self._exactly_k_combinations(arguments, min_true)
        elif min_true == 0 and max_true == n - 1:
            # At most n-1 = not all true
            return Not(And(*arguments))
        elif min_true == 1 and max_true == n:
            # At least 1
            return Or(*arguments)
        elif min_true == 0 and max_true == 1:
            # At most 1
            clauses = []
            clauses.append(And(*[Not(a) for a in arguments]))  # all false
            clauses.extend(self._exactly_k_combinations(arguments, 1))  # exactly 1
        else:
            # General range - enumerate all k values
            clauses = []
            for k in range(min_true, max_true + 1):
                clauses.extend(self._exactly_k_combinations(arguments, k))

        if len(clauses) == 0:
            return FALSE()
        elif len(clauses) == 1:
            return clauses[0]
        else:
            return Or(*clauses)

    def _expand_count_vs_constant(
            self, count_node: FNode, value: int, op: OperatorKind
    ) -> FNode:
        """
        Expand Count(args) op value into boolean formula.

        Examples:
        - Count(a, b, c) <= 2  →  "at most 2 of {a,b,c} are true"
        - Count(a, b, c) == 2   →  "exactly 2 of {a,b,c} is true"
        - Count(a, b, c) < 2   →  "at most 1 of {a,b,c} is true"
        """
        arguments = list(count_node.args)
        n = len(arguments)

        # Determine min and max number of true arguments
        if op == OperatorKind.LE:
            min_true, max_true = 0, min(value, n)
        elif op == OperatorKind.EQUALS:
            min_true, max_true = value, value
        elif op == OperatorKind.LT:
            min_true, max_true = 0, min(value - 1, n)
        else:
            raise UPValueError(f"Operator {op} not supported (should be LT/LE/EQUALS)")

        # Generate boolean formula
        return self._exactly_k_true_formula(arguments, min_true, max_true)

    def _expand_constant_vs_count(
            self, value: int, count_node: FNode, op: OperatorKind
    ) -> FNode:
        """
        Expand constant op Count(args) into boolean formula.
        """
        arguments = list(count_node.args)
        n = len(arguments)

        # Flip the comparison semanticsx
        if op == OperatorKind.LT:
            # value < Count means Count > value means Count >= value+1
            min_true, max_true = value + 1, n
        elif op == OperatorKind.LE:
            # value <= Count means Count >= value
            min_true, max_true = value, n
        elif op == OperatorKind.EQUALS:
            # value = Count means Count = value
            min_true, max_true = value, value
        else:
            raise UPValueError(f"Operator {op} not supported (should be LT/LE/EQUALS)")

        # Generate boolean formula
        return self._exactly_k_true_formula(arguments, min_true, max_true)

    def _satisfies_operator(self, k1: int, k2: int, op: OperatorKind) -> bool:
        """Check if (k1, k2) satisfies the operator."""
        if op == OperatorKind.LT:
            return k1 < k2
        elif op == OperatorKind.LE:
            return k1 <= k2
        elif op == OperatorKind.EQUALS:
            return k1 == k2
        else:
            raise UPValueError(f"Operator {op} should be normalized by UP")

    # ==================== COUNT VS COUNT ====================

    def _expand_count_vs_count(
            self, count1: FNode, count2: FNode, op: OperatorKind
    ) -> FNode:
        """
        Expand Count(args1) op Count(args2) into boolean formula.

        Strategy: Enumerate all possible (k1, k2) pairs that satisfy the comparison,
        then generate: Or over all valid pairs of (exactly k1 true in args1) ∧ (exactly k2 true in args2)
        """
        args1 = list(count1.args)
        args2 = list(count2.args)
        n1 = len(args1)
        n2 = len(args2)

        # Generate all valid (k1, k2) pairs
        valid_pairs = self._get_valid_count_pairs(op, n1, n2)
        if len(valid_pairs) == 0:
            return FALSE()

        # Generate formula for each pair
        k1_to_k2s = {}
        for k1, k2 in valid_pairs:
            k1_to_k2s.setdefault(k1, []).append(k2)

        # Generate formula for each k1 group
        clauses = []
        for k1 in sorted(k1_to_k2s.keys()):
            k2_list = sorted(k1_to_k2s[k1])
            k2_min = min(k2_list)
            k2_max = max(k2_list)

            formula1 = self._exactly_k_true_formula(args1, k1, k1)  # exactly k1

            # Check if k2_list is continuous range
            if k2_list == list(range(k2_min, k2_max + 1)):
                # Continuous range - use optimized formula
                formula2 = self._exactly_k_true_formula(args2, k2_min, k2_max)
            else:
                # Non-continuous - enumerate each k2
                k2_clauses = [self._exactly_k_true_formula(args2, k2, k2) for k2 in k2_list]
                formula2 = Or(*k2_clauses) if len(k2_clauses) > 1 else k2_clauses[0]

            clauses.append(And(formula1, formula2))
        if len(clauses) == 1:
            return clauses[0]
        else:
            return Or(*clauses)

    def _get_valid_count_pairs(self, op: OperatorKind, n1: int, n2: int) -> List[Tuple[int, int]]:
        """Get all valid (k1, k2) pairs for Count(args1) op Count(args2)."""
        pairs = []
        for k1 in range(n1 + 1):
            for k2 in range(n2 + 1):
                # Check if (k1, k2) satisfies the operator
                if (
                        (op == OperatorKind.LT and k1 < k2) or
                        (op == OperatorKind.LE and k1 <= k2) or
                        (op == OperatorKind.EQUALS and k1 == k2)
                ):
                    pairs.append((k1, k2))
        return pairs

    # ==================== ACTION TRANSFORMATION ====================

    def _transform_action(
            self, problem: Problem, new_problem: Problem, action: Action
    ) -> Action:
        """Transform an action by transforming all its expressions."""
        new_action = action.clone()
        new_action.name = get_fresh_name(new_problem, action.name)
        new_action.clear_preconditions()
        new_action.clear_effects()

        # Transform preconditions
        for precondition in action.preconditions:
            new_precondition = self._transform_expression(new_problem, precondition)
            new_action.add_precondition(new_precondition)

        # Transform effects
        for effect in action.effects:
            self._transform_effect(new_problem, new_action, effect)

        return new_action

    def _transform_effect(
            self,
            new_problem: Problem,
            new_action: Action,
            effect: "up.model.effect.Effect"
    ):
        """Transform a single effect."""
        new_fluent = self._transform_expression(new_problem, effect.fluent)
        new_value = self._transform_expression(new_problem, effect.value)
        new_condition = self._transform_expression(new_problem, effect.condition)

        # Add the appropriate type of effect
        if effect.is_increase():
            new_action.add_increase_effect(
                new_fluent, new_value, new_condition, effect.forall
            )
        elif effect.is_decrease():
            new_action.add_decrease_effect(
                new_fluent, new_value, new_condition, effect.forall
            )
        else:
            new_action.add_effect(
                new_fluent, new_value, new_condition, effect.forall
            )

    # ==================== MAIN COMPILATION ====================

    def _compile(
            self,
            problem: Problem,
            compilation_kind: CompilationKind,
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)

        # Clone problem
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_quality_metrics()

        # Transform actions
        new_to_old: Dict[Action, Action] = {}
        for action in problem.actions:
            new_action = self._transform_action(problem, new_problem, action)
            new_problem.add_action(new_action)
            new_to_old[new_action] = action

        # Transform goals
        for goal in problem.goals:
            new_goal = self._transform_expression(new_problem, goal)
            new_problem.add_goal(new_goal)

        # Transform quality metrics
        for qm in problem.quality_metrics:
            if qm.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(qm, new_to_old, new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(qm)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
