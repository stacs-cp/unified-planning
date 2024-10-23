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
"""This module defines the quantifiers remover class."""
import itertools
from itertools import product

from unified_planning.exceptions import UPValueError
from unified_planning.model.walkers import simplifier

import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import (
    Problem,
    InstantaneousAction,
    DurativeAction,
    Action,
    ProblemKind,
    Oversubscription,
    TemporalOversubscription,
    Object,
    Variable,
    Expression,
    Effect, OperatorKind,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    replace_action,
)
from typing import Dict, List, Optional, Tuple, OrderedDict, Any, Union
from functools import partial
from unified_planning.shortcuts import Int, Plus, Not, Minus, And, Equals, TRUE, FALSE, Or
import re

class CountRemover(engines.engine.Engine, CompilerMixin):
    """
    Count expression remover class: Only appears in the goal
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

    def _generate_combinations(self, arguments, min_true, max_true) -> "up.model.fnode.FNode":
        """Generates combinations of expressions with a fixed amount of True arguments."""
        return Or(
            And(*(arguments[idx] if idx in true_indices else Not(arguments[idx])
                 for idx in range(len(arguments))))
            for i in range(min_true, max_true + 1)
            for true_indices in itertools.combinations(range(len(arguments)), i)
        )

    def _get_expression(self, node_type: str, arguments: list["up.model.fnode.FNode"], value: int):
        """Returns the expression according to the node type and value."""
        operation_map = {
            'le': (0, value),
            'eq': (value, value),
            'lt': (0, value - 1),
        }
        if node_type in operation_map:
            return self._generate_combinations(arguments, *operation_map[node_type])
        else:
            raise UPValueError(f"Count in an expression type {node_type} not supported!")

    def _manage_counts(
            self,
            new_problem: "up.model.Problem",
            expression: "up.model.fnode.FNode",
            count_expressions: Dict[str, "up.model.fnode.FNode"]
    ) -> Union["up.model.fnode.FNode", "up.model.fluent.Fluent", bool]:
        """Manage expressions type Count and generates new expressions."""
        if expression.is_fluent_exp() or expression.is_parameter_exp() or expression.is_constant():
            return expression

        if expression.arg(0).is_count() or expression.arg(1).is_count():
            count_expression, value = (
                (expression.arg(0), expression.arg(1).constant_value())
                if expression.arg(0).is_count() else (expression.arg(1), expression.arg(0).constant_value())
            )
            operation_map = {
                OperatorKind.EQUALS: 'eq',
                OperatorKind.LT: 'lt',
                OperatorKind.LE: 'le'
            }
            operation = operation_map.get(expression.node_type)
            return self._get_expression(operation, count_expression.args, value)

        new_args = [self._manage_counts(new_problem, arg, count_expressions) for arg in expression.args]
        return new_problem.environment.expression_manager.create_node(expression.node_type, tuple(new_args))

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """
        """
        assert isinstance(problem, Problem)
        new_to_old: Dict[Action, Action] = {}
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        count_expressions: Dict[str, "up.model.fnode.FNode"] = {}
        for action in problem.actions:
            new_to_old[action] = action

        new_problem.clear_goals()
        for goal in problem.goals:
            new_goal = self._manage_counts(new_problem, goal, count_expressions)
            new_problem.add_goal(new_goal)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
