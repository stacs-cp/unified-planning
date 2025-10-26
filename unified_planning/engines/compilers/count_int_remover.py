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
"""This module defines the count int remover class."""

import unified_planning.engines as engines
import re

from itertools import product
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import Problem, Action, ProblemKind, OperatorKind, FNode, AbstractProblem
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action, updated_minimize_action_costs
from typing import Dict, List, Optional, Union, Set
from functools import partial
from unified_planning.shortcuts import Int, Not, And, Equals

class CountIntRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes Count expressions by converting them to integer fluents.

    Each boolean expression inside a Count becomes a 0/1 fluent, and the Count expression is replaced by the sum of these fluents.

    Example:
        Count((a > b), my_bool) >= 1
    Becomes:
        count_0 + count_1 >= 1
    Where:
        count_0 represents (a > b) (0 or 1)
        count_1 represents my_bool (0 or 1)
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.COUNT_INT_REMOVING)

    @property
    def name(self):
        return "crim"

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
        return problem_kind <= CountIntRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.COUNT_INT_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("COUNTING")
        new_kind.set_fluents_type("INT_FLUENTS")
        return new_kind

    # ==================== EXPRESSION EVALUATION ====================

    def _evaluate_expression(
            self,
            problem: Problem,
            expression: FNode,
            fluent_to_update: Optional[FNode] = None,
            new_value: Optional[FNode] = None,
            effect_type: Optional[str] = None
    ) -> FNode:
        """
        Evaluate expression, used to compute initial values and to determine how effects change count fluents.
        """
        em = problem.environment.expression_manager
        # Base cases
        if expression.is_constant() or expression.is_parameter_exp() or expression.is_object_exp():
            return expression
        # Fluent expression
        if expression.is_fluent_exp():
            # Handle array fluents with indices in name (e.g., "fluent[0][1]")

            # Maybe I should change this?
            if '[' in expression.fluent().name:
                base_name = expression.fluent().name.split('[')[0]
                indices = [int(i) for i in re.findall(r'\[(.*?)\]', expression.fluent().name)]
                fluent_exp = problem.fluent(base_name)(*expression.args)
                value = problem.initial_value(fluent_exp)
                # Navigate through array indices
                for idx in indices:
                    value = value.constant_value()[idx]
                return value

            # Check if this is the fluent being updated
            if fluent_to_update is not None and fluent_to_update == expression:
                if effect_type == 'increase':
                    return em.Plus(expression, new_value).simplify()
                elif effect_type == 'decrease':
                    return em.Minus(expression, new_value).simplify()
                else:
                    return new_value

            # Regular fluent evaluation
            return problem.initial_value(expression)

        # Recursive case
        new_args = [
            self._evaluate_expression(problem, arg, fluent_to_update, new_value, effect_type)
            for arg in expression.args
        ]
        return em.create_node(expression.node_type, tuple(new_args)).simplify()

    def _find_affected_fluents(self, expression: FNode) -> List[FNode]:
        """Extract all fluent expressions from an expression tree."""
        if expression.is_fluent_exp():
            return [expression]
        fluents = []
        for arg in expression.args:
            fluents.extend(self._find_affected_fluents(arg))
        return fluents

    # ==================== COUNT EXPRESSION REPLACEMENT ====================
    def _depends_on_params_action(self, expression: FNode) -> bool:
        if expression.is_parameter_exp():
            return True
        for a in expression.args:
            if self._depends_on_params_action(a):
                return True
        return False

    def _replace_count_with_fluents(
            self, problem: Problem, expression: FNode, count_registry: Dict[str, FNode]
    ) -> FNode:
        """Replace Count expressions with sums of fluents."""
        em = problem.environment.expression_manager
        tm = problem.environment.type_manager

        # Base cases
        if expression.is_fluent_exp() or expression.is_parameter_exp() or expression.is_constant():
            return expression

        # Count expression
        if expression.is_count():
            sum_args = []
            for arg in expression.args:
                # If the argument contains a parameter - not fully instantiated (depends on each specific action)
                if self._depends_on_params_action(arg):
                    # Return the same Count, the translation is not possible in this case
                    return expression
                # Skip trivial cases
                if arg.is_false():
                    continue
                elif arg.is_true():
                    sum_args.append(Int(1))
                    continue
                # Check if we already have a fluent for this expression
                existing_name = next(
                    (name for name, expr in count_registry.items() if expr == arg),
                    None
                )
                if existing_name:
                    sum_args.append(problem.fluent(existing_name)())
                else:
                    # Create new count fluent
                    fluent_name = f'count_{len(count_registry)}'
                    count_registry[fluent_name] = arg
                    # Evaluate initial value
                    # pero que passa quan no esta totalment grounded?
                    initial_eval = self._evaluate_expression(problem, arg)
                    assert initial_eval.is_bool_constant(), \
                        f"Count argument initial value must be boolean constant, got: {initial_eval}"
                    initial_value = Int(1) if initial_eval.is_true() else Int(0)
                    # Add fluent to problem
                    problem.add_fluent(fluent_name, tm.IntType(0, 1))
                    problem.set_initial_value(problem.fluent(fluent_name), initial_value)
                    sum_args.append(problem.fluent(fluent_name)())

            # Return sum of all count fluents
            if not sum_args:
                return Int(0)
            elif len(sum_args) == 1:
                return sum_args[0]
            else:
                return em.Plus(*sum_args)

        # Recursive case
        new_args = [self._replace_count_with_fluents(problem, arg, count_registry) for arg in expression.args]
        return em.create_node(expression.node_type, tuple(new_args))

    # ==================== EFFECT GENERATION ====================

    def _add_count_effect_to_action(
            self, action: Action, problem: Problem, count_name: str, new_expr: FNode, condition: Optional[FNode]
    ):
        """Add conditional effect to update count fluent."""
        count_fluent = problem.fluent(count_name)
        if new_expr.is_bool_constant():
            # Expression evaluates to constant
            value = 1 if new_expr.is_true() else 0
            if condition:
                action.add_effect(count_fluent, value, condition)
            else:
                action.add_effect(count_fluent, value)
        else:
            # Expression is conditional
            if condition:
                # True case
                action.add_effect(
                    count_fluent, 1,
                    And(new_expr, condition).simplify()
                )
                # False case
                action.add_effect(
                    count_fluent, 0,
                    And(Not(new_expr), condition).simplify()
                )
            else:
                action.add_effect(count_fluent, 1, new_expr)
                action.add_effect(count_fluent, 0, Not(new_expr).simplify())

    def _generate_count_effects(self, problem: Problem, action: Action, count_registry: Dict[str, FNode]) -> Action:
        """
        Generate effects for count fluents based on action effects.
        For each count fluent tracking expression E:
        - If action effects change fluents in E, add conditional effects to update the count fluent accordingly
        """
        for count_name, count_expr in count_registry.items():
            # Find which fluents in count_expr are affected by action
            affected_fluents = self._find_affected_fluents(count_expr)

            # Separate direct and indirect effects
            direct_effects = []
            indirect_effects = []
            param_to_objects: Dict[FNode, Set[FNode]] = {}

            for effect in action.effects:
                for tracked_fluent in affected_fluents:
                    if effect.fluent == tracked_fluent:
                        # Direct effect on tracked fluent
                        direct_effects.append(effect)
                    elif effect.fluent.fluent().name == tracked_fluent.fluent().name:
                        # Same fluent but different parameters
                        indirect_effects.append(effect)

                        # Track parameter mappings
                        for i, param in enumerate(effect.fluent.args):
                            obj = tracked_fluent.arg(i)
                            param_to_objects.setdefault(param, set()).add(obj)
            if not (direct_effects or indirect_effects):
                continue

            # Process direct effects
            combined_condition = None
            new_expr = count_expr
            for effect in direct_effects:
                if effect.is_conditional():
                    combined_condition = And(combined_condition, effect.condition).simplify() \
                        if combined_condition else effect.condition
                effect_type = 'increase' if effect.is_increase() else 'decrease' if effect.is_decrease() else None
                new_expr = self._evaluate_expression(problem, new_expr, effect.fluent, effect.value, effect_type)

            # Process indirect effects (with parameter instantiation)
            if indirect_effects:
                # Generate all parameter combinations
                param_list = list(param_to_objects.keys())
                object_lists = [list(param_to_objects[p]) for p in param_list]

                for obj_combination in product(*object_lists):
                    inst_expr = new_expr
                    inst_condition = combined_condition
                    param_map = dict(zip(param_list, obj_combination))
                    for effect in indirect_effects:
                        # Build instantiated fluent
                        inst_fluent_args = [param_map[arg] for arg in effect.fluent.args]
                        inst_fluent = effect.fluent.fluent()(*inst_fluent_args)

                        # Add parameter equality conditions
                        for param, obj in param_map.items():
                            eq_cond = Equals(obj, param)
                            inst_condition = And(inst_condition, eq_cond).simplify() if inst_condition else eq_cond
                        # Add effect condition
                        if effect.is_conditional():
                            inst_condition = And(inst_condition, effect.condition).simplify()

                        # Update expression
                        effect_type = 'increase' if effect.is_increase() else 'decrease' if effect.is_decrease() else None
                        inst_expr = self._evaluate_expression(problem, inst_expr, inst_fluent, effect.value, effect_type)
                    # Add effect to action
                    self._add_count_effect_to_action(
                        action, problem, count_name,
                        inst_expr, inst_condition
                    )
            else:
                # No indirect effects, just add the direct effect
                self._add_count_effect_to_action(
                    action, problem, count_name,
                    new_expr, combined_condition
                )
        return action

    def _compile(
        self,
        problem: AbstractProblem,
        compilation_kind: CompilationKind,
    ) -> CompilerResult:
        """Main compilation method."""
        assert isinstance(problem, Problem)

        # Setup new problem
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_quality_metrics()

        count_registry: Dict[str, FNode] = {}
        new_to_old: Dict[Action, Action] = {}

        # Transform actions (preconditions only, effects later)
        temp_actions = []
        for action in problem.actions:
            new_action = action.clone()
            new_action.clear_preconditions()
            for precondition in action.preconditions:
                new_precondition = self._replace_count_with_fluents(new_problem, precondition, count_registry)
                new_action.add_precondition(new_precondition)
            temp_actions.append(new_action)

        # Transform goals
        for goal in problem.goals:
            new_goal = self._replace_count_with_fluents(new_problem, goal, count_registry)
            new_problem.add_goal(new_goal)

        # Add effects for count fluents
        final_actions = []
        for temp_action, old_action in zip(temp_actions, problem.actions):
            final_action = self._generate_count_effects(
                new_problem, temp_action, count_registry
            )
            final_actions.append(final_action)
            new_problem.add_action(final_action)
            new_to_old[final_action] = old_action

        # Transform quality metrics
        for metric in problem.quality_metrics:
            if metric.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(
                        metric, new_to_old, new_problem.environment
                    )
                )
            else:
                new_problem.add_quality_metric(metric)

        return CompilerResult(new_problem, partial(replace_action, map=new_to_old), self.name)