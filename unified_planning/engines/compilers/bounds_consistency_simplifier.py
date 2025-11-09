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
"""This module defines the bounds consistency simplifier class."""
import bisect
import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, Action, ProblemKind, InstantaneousAction, FNode, AbstractProblem, \
    OperatorKind, Object
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action, updated_minimize_action_costs
from typing import Dict, List, Optional, Tuple, OrderedDict, Any, Set, Union
from functools import partial
from unified_planning.shortcuts import FALSE, TRUE, Int, Equals, Not, And, Or, GT, GE, ObjectExp


class UnionFind:
    """Union-Find data structure for equivalence classes."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: FNode) -> FNode:
        """Find representative of x's equivalence class."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: FNode, y: FNode):
        """Merge equivalence classes of x and y."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_class(self, x: FNode) -> Set[FNode]:
        """Get all elements in x's equivalence class."""
        root = self.find(x)
        return {k for k, v in self.parent.items() if self.find(k) == root}

class BoundsConsistencySimplifier(engines.engine.Engine, CompilerMixin):
    """
    Compiler that simplifies actions by detecting fixed or bounded fluent values in preconditions
    and substituting them throughout the action (bounds consistency).

    This reduces the complexity for subsequent compilers by:
    - Substituting known values into arithmetic expressions
    - Simplifying arithmetic (e.g., 0 + x → x, x - x → 0)
    - Eliminating impossible preconditions/effects (out of bounds)
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.BOUNDS_CONSISTENCY_SIMPLIFIER)

    @property
    def name(self):
        return "bcs"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
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
        return problem_kind <= BoundsConsistencySimplifier.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.BOUNDS_CONSISTENCY_SIMPLIFIER

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        return new_kind

    def _is_trackable(self, node: FNode) -> bool:
        """Check if a node can be tracked (fluent or parameter)."""
        return node.is_fluent_exp() or node.is_parameter_exp()

    def _get_initial_bounds(self, problem: Problem, node: FNode) -> Set[Union[int, Object]]:
        """Get the initial domain of a trackable expression."""
        if node.type.is_int_type():
            return set(range(node.type.lower_bound, node.type.upper_bound + 1))
        elif node.type.is_user_type():
            return set(problem.objects(node.type))
        else:
            return set()

    # ==================== EQUIVALENCE & BOUNDS EXTRACTION ====================

    def _analyze_preconditions(
            self,
            problem: Problem,
            preconditions: List[FNode]
    ) -> Tuple[Optional[UnionFind], Optional[Dict[FNode, Set[Union[int, Object]]]], List[FNode]]:
        """
        Analyze preconditions to extract:
        1. Equivalence classes (UnionFind)
        2. Bounds for each equivalence class representative
        3. Core preconditions (equivalences and direct bounds constraints)

        Returns (uf, bounds, core_preconditions) or (None, None, []) if impossible
        """
        uf = UnionFind()
        bounds = {}
        core_preconditions = []  # Preconditions that define bounds/equivalences

        # First pass: extract equivalences and core constraints
        for precond in preconditions:
            if self._is_core_constraint(precond):
                core_preconditions.append(precond)
            self._extract_equivalences_from_node(precond, uf)

        # Initialize bounds for all trackable expressions
        for precond in preconditions:
            self._collect_trackable_nodes(precond, problem, uf, bounds)

        # Second pass: extract bounds from constraints
        for precond in preconditions:
            if not self._update_bounds_from_constraint(problem, precond, bounds, uf):
                return None, None, []  # Impossible

        return uf, bounds, core_preconditions

    def _is_core_constraint(self, node: FNode) -> bool:
        """
        Check if a precondition is a core constraint that defines bounds/equivalences.
        Core constraints should be kept in the simplified action.
        """
        # Direct equality between trackables
        if node.is_equals():
            left, right = node.arg(0), node.arg(1)
            if self._is_trackable(left) and self._is_trackable(right):
                return True
            if self._is_trackable(left) and right.is_constant():
                return True
            if self._is_trackable(right) and left.is_constant():
                return True

        # NOT(x == constant) - bounds constraint
        if node.is_not() and node.arg(0).is_equals():
            left, right = node.arg(0).arg(0), node.arg(0).arg(1)
            if (self._is_trackable(left) and right.is_constant()) or \
                    (self._is_trackable(right) and left.is_constant()):
                return True

        # Comparisons with trackables
        if node.is_lt() or node.is_le():
            left, right = node.arg(0), node.arg(1)
            if self._is_trackable(left) or self._is_trackable(right):
                return True

        return False

    def _collect_trackable_nodes(self, node: FNode, problem: Problem, uf: UnionFind, bounds: Dict):
        """Recursively collect all trackable nodes and initialize their bounds."""
        if self._is_trackable(node):
            rep = uf.find(node)
            if rep not in bounds:
                bounds[rep] = self._get_initial_bounds(problem, node)

        for arg in node.args:
            self._collect_trackable_nodes(arg, problem, uf, bounds)

    def _extract_equivalences_from_node(self, node: FNode, uf: UnionFind):
        """Recursively extract equivalences from a node."""
        if node.is_and():
            for arg in node.args:
                self._extract_equivalences_from_node(arg, uf)

        elif node.is_equals():
            # x == y → x and y are equivalent
            left, right = node.arg(0), node.arg(1)
            if self._is_trackable(left) and self._is_trackable(right):
                uf.union(left, right)

    def _update_bounds_from_constraint(
            self,
            problem: Problem,
            constraint: FNode,
            bounds: Dict[FNode, Set[Union[int, Object]]],
            uf: UnionFind
    ) -> bool:
        """Update bounds based on a constraint. Returns False if impossible."""
        if constraint.is_and():
            for arg in constraint.args:
                if not self._update_bounds_from_constraint(problem, arg, bounds, uf):
                    return False
            return True

        # x == constant: bounds[x] = {constant}
        if constraint.is_equals():
            left, right = constraint.arg(0), constraint.arg(1)

            if self._is_trackable(left) and right.is_constant():
                return self._constrain_to_value(uf.find(left), right, bounds)
            elif self._is_trackable(right) and left.is_constant():
                return self._constrain_to_value(uf.find(right), left, bounds)

        # NOT(x == constant): remove constant from bounds
        if constraint.is_not() and constraint.arg(0).is_equals():
            left, right = constraint.arg(0).arg(0), constraint.arg(0).arg(1)

            if self._is_trackable(left) and right.is_constant():
                return self._remove_value_from_bounds(uf.find(left), right, bounds)
            elif self._is_trackable(right) and left.is_constant():
                return self._remove_value_from_bounds(uf.find(right), left, bounds)

        # x < constant or x <= constant
        if constraint.is_lt() or constraint.is_le():
            left, right = constraint.arg(0), constraint.arg(1)

            if self._is_trackable(left) and right.is_int_constant():
                return self._apply_upper_bound(uf.find(left), right.constant_value(),
                                               constraint.is_le(), bounds)

        return True

    def _constrain_to_value(
            self,
            rep: FNode,
            const: FNode,
            bounds: Dict[FNode, Set[Union[int, Object]]]
    ) -> bool:
        """Constrain representative to a single value."""
        if rep.type.is_int_type():
            value = const.constant_value()
        elif rep.type.is_user_type():
            value = const.object()
        else:
            return True

        if value not in bounds[rep]:
            return False  # Impossible

        bounds[rep] = {value}
        return True

    def _remove_value_from_bounds(
            self,
            rep: FNode,
            const: FNode,
            bounds: Dict[FNode, Set[Union[int, Object]]]
    ) -> bool:
        """Remove a value from bounds."""
        if rep.type.is_int_type():
            value = const.constant_value()
        elif rep.type.is_user_type():
            value = const.object()
        else:
            return True

        bounds[rep].discard(value)

        if not bounds[rep]:
            return False  # No possible values left

        return True

    def _apply_upper_bound(
            self,
            rep: FNode,
            value: int,
            inclusive: bool,
            bounds: Dict[FNode, Set[Union[int, Object]]]
    ) -> bool:
        """Apply upper bound constraint."""
        if not rep.type.is_int_type():
            return True

        current_list = sorted(bounds[rep])

        if inclusive:
            idx = bisect.bisect_right(current_list, value)
        else:
            idx = bisect.bisect_left(current_list, value)

        new_bounds = set(current_list[:idx])

        if not new_bounds:
            return False

        bounds[rep] = new_bounds
        return True

    # ==================== SIMPLIFICATION ====================

    def _simplify_expression(
            self,
            problem: Problem,
            node: FNode,
            bounds: Dict[FNode, Set[Union[int, Object]]],
            uf: UnionFind
    ) -> FNode:
        """
        Simplify an expression using bounds information.
        Only substitute when we know a concrete single value.
        Use equivalences to detect tautologies/contradictions.
        """
        em = problem.environment.expression_manager

        # Check if this trackable has a single concrete value
        if self._is_trackable(node):
            rep = uf.find(node)
            if rep in bounds and len(bounds[rep]) == 1:
                value = next(iter(bounds[rep]))
                if node.type.is_int_type():
                    return Int(value)
                elif node.type.is_user_type():
                    return ObjectExp(value)

        # Base cases
        if node.is_constant() or node.is_object_exp() or node.is_parameter_exp():
            return node

        if node.is_fluent_exp():
            return node

        # Handle EQUALS
        if node.is_equals():
            left, right = node.arg(0), node.arg(1)

            # Check equivalence
            if self._is_trackable(left) and self._is_trackable(right):
                if uf.find(left) == uf.find(right):
                    return TRUE()

            # Check against bounds
            if self._is_trackable(left) and right.is_constant():
                rep = uf.find(left)
                if rep in bounds:
                    value = right.constant_value() if left.type.is_int_type() else right.object()
                    if value not in bounds[rep]:
                        return FALSE()
                    if len(bounds[rep]) == 1:
                        return TRUE()

            if self._is_trackable(right) and left.is_constant():
                rep = uf.find(right)
                if rep in bounds:
                    value = left.constant_value() if right.type.is_int_type() else left.object()
                    if value not in bounds[rep]:
                        return FALSE()
                    if len(bounds[rep]) == 1:
                        return TRUE()

        # Handle OR
        if node.is_or():
            new_args = []
            for arg in node.args:
                simplified_arg = self._simplify_expression(problem, arg, bounds, uf)
                if simplified_arg == TRUE():
                    return TRUE()
                if simplified_arg != FALSE():
                    new_args.append(simplified_arg)

            if not new_args:
                return FALSE()
            if len(new_args) == 1:
                return new_args[0]
            return em.create_node(OperatorKind.OR, tuple(new_args))

        # Handle AND
        if node.is_and():
            new_args = []
            for arg in node.args:
                simplified_arg = self._simplify_expression(problem, arg, bounds, uf)
                if simplified_arg == FALSE():
                    return FALSE()
                if simplified_arg != TRUE():
                    new_args.append(simplified_arg)

            if not new_args:
                return TRUE()
            if len(new_args) == 1:
                return new_args[0]
            return em.create_node(OperatorKind.AND, tuple(new_args))

        # Handle NOT
        if node.is_not():
            arg = self._simplify_expression(problem, node.arg(0), bounds, uf)
            if arg == TRUE():
                return FALSE()
            if arg == FALSE():
                return TRUE()
            return em.create_node(OperatorKind.NOT, (arg,))

        # Recursively simplify children
        new_args = [self._simplify_expression(problem, arg, bounds, uf) for arg in node.args]

        # Simplify arithmetic
        simplified = self._simplify_arithmetic(em, node.node_type, new_args)
        if simplified is not None:
            return simplified

        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _simplify_arithmetic(self, em, op: OperatorKind, args: List[FNode]) -> Optional[FNode]:
        """Simplify arithmetic expressions with constants."""
        if len(args) != 2:
            return None

        left, right = args

        # Evaluate constant expressions
        if left.is_int_constant() and right.is_int_constant():
            l_val, r_val = left.constant_value(), right.constant_value()

            if op == OperatorKind.PLUS:
                return Int(l_val + r_val)
            elif op == OperatorKind.MINUS:
                return Int(l_val - r_val)
            elif op == OperatorKind.TIMES:
                return Int(l_val * r_val)
            elif op == OperatorKind.DIV and r_val != 0:
                return Int(l_val // r_val)

        # Algebraic simplifications
        if op == OperatorKind.PLUS:
            if left.is_int_constant() and left.constant_value() == 0:
                return right
            if right.is_int_constant() and right.constant_value() == 0:
                return left

        elif op == OperatorKind.MINUS:
            if right.is_int_constant() and right.constant_value() == 0:
                return left
            if left == right:
                return Int(0)

        elif op == OperatorKind.TIMES:
            if left.is_int_constant():
                if left.constant_value() == 0:
                    return Int(0)
                if left.constant_value() == 1:
                    return right
            if right.is_int_constant():
                if right.constant_value() == 0:
                    return Int(0)
                if right.constant_value() == 1:
                    return left

        elif op == OperatorKind.DIV:
            if right.is_int_constant() and right.constant_value() == 1:
                return left

        return None

    # ==================== ACTION TRANSFORMATION ====================

    def _simplify_action(self, problem: Problem, action: Action) -> Optional[Action]:
        """Simplify an action using bounds consistency."""
        preconditions_list = list(action.preconditions)

        # Analyze preconditions
        uf, bounds, core_preconditions = self._analyze_preconditions(problem, preconditions_list)

        if uf is None or bounds is None:
            return None  # Impossible action

        # Create new action
        params = OrderedDict((p.name, p.type) for p in action.parameters)
        new_action = InstantaneousAction(action.name, _parameters=params, _env=problem.environment)

        # Add core preconditions (always kept)
        for precond in core_preconditions:
            new_action.add_precondition(precond)

        # Add other preconditions (simplified)
        for precond in preconditions_list:
            if precond in core_preconditions:
                continue  # Already added

            simplified = self._simplify_expression(problem, precond, bounds, uf)

            if simplified == FALSE():
                return None  # Impossible action
            if simplified != TRUE():
                new_action.add_precondition(simplified)

        # Add effects (simplified)
        for effect in action.effects:
            new_condition = self._simplify_expression(problem, effect.condition, bounds, uf)

            if new_condition == FALSE():
                continue  # Skip impossible effect

            new_value = self._simplify_expression(problem, effect.value, bounds, uf)

            # Check out of bounds
            if effect.fluent.fluent().type.is_int_type() and new_value.is_int_constant():
                lb = effect.fluent.fluent().type.lower_bound
                ub = effect.fluent.fluent().type.upper_bound
                val = new_value.constant_value()

                if val < lb or val > ub:
                    if new_condition == TRUE():
                        return None  # Impossible action
                    continue

            # Add effect
            if effect.is_increase():
                new_action.add_increase_effect(effect.fluent, new_value, new_condition, effect.forall)
            elif effect.is_decrease():
                new_action.add_decrease_effect(effect.fluent, new_value, new_condition, effect.forall)
            else:
                new_action.add_effect(effect.fluent, new_value, new_condition, effect.forall)

        return new_action

    def _simplify_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions."""
        new_to_old = {}

        for action in problem.actions:
            new_action = self._simplify_action(new_problem, action)

            if new_action is not None:
                new_problem.add_action(new_action)
                new_to_old[new_action] = action

        return new_to_old

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation method."""
        assert isinstance(problem, Problem)

        # Create new problem
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_actions()
        new_problem.clear_quality_metrics()

        # Transform actions
        new_to_old = self._simplify_actions(problem, new_problem)

        # Transform quality metrics
        for metric in problem.quality_metrics:
            if metric.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(metric, new_to_old, new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(metric)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
