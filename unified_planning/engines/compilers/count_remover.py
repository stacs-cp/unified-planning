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


from itertools import product
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
from unified_planning.model.walkers import ExpressionQuantifiersRemover
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action,
    updated_minimize_action_costs,
)
from typing import Dict, List, Optional, Tuple, OrderedDict, Any, Union
from functools import partial
from unified_planning.shortcuts import Int, Plus, Not
import re

class CountRemover(engines.engine.Engine, CompilerMixin):
    """
    Count expression remover class: ...
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
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_conditions_kind("COUNTINGS")
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
        return problem_kind.clone()

    '''
    def find_value_effect(
            self,
            expression: "up.model.fnode.FNode",
            value: "up.model.fnode.FNode",
            fluent_name: str,
            fluents_affected: Dict[str, List[str]]
    ) -> "up.model.fnode.FNode":
        assert expression.type.is_bool_type()
        if expression.is_constant():
            return expression
        elif expression.is_fluent_exp():
            fluent = expression.fluent()
            # sustituir el valor
            return Int(1) if new_problem.initial_value(arg).is_true() else Int(0)
        else:
            new_args = []
            for a in arg.args:
                new_args.append(self.decompose_expression(a, new_problem, count_arg_name, fluents_affected))
            return Int(1) if em.create_node(arg.node_type, tuple(new_args)).simplify().is_true() else Int(0)
    '''

    def decompose_expression(
            self,
            new_problem: "up.model.Problem",
            expression: "up.model.fnode.FNode",
            fluent: "up.model.fnode.FNode" = None,
            value: "up.model.fnode.FNode" = None,
    ) -> "up.model.fnode.FNode":
        env = new_problem.environment
        em = env.expression_manager
        if expression.is_constant():
            return expression
        elif expression.is_fluent_exp():
            if fluent is None:
                return new_problem.initial_value(expression.fluent())
            else:
                if fluent == expression.fluent():
                    return value
                else:
                    return expression
        else:
            new_args = []
            for arg in expression.args:
                new_args.append(self.decompose_expression(new_problem, arg, fluent, value))
            return em.create_node(expression.node_type, tuple(new_args))

    def expression_value(
            self,
            new_problem: "up.model.Problem",
            expression: "up.model.fnode.FNode",
            fluent: Optional["up.model.fnode.FNode"] = None,
            value: Optional["up.model.fnode.FNode"] = None,
    ) -> "up.model.fnode.FNode":
        assert expression.type.is_bool_type()
        env = new_problem.environment
        em = env.expression_manager
        if expression.is_constant():
            return expression
        elif expression.is_fluent_exp():
            assert expression.fluent().type.is_bool_type()
            if value is None:
                return new_problem.initial_value(expression.fluent())
            else:
                assert value.is_bool_constant()
                if fluent == expression.fluent():
                    return value
                else:
                    return expression
        else:
            new_args = []
            for arg in expression.args:
                new_args.append(self.decompose_expression(new_problem, arg, fluent, value))
            return em.create_node(expression.node_type, tuple(new_args)).simplify()

    def find_fluents_affected(
            self,
            expression: "up.model.fnode.FNode",
    ) -> List[str]:
        fluents = []
        if expression.is_constant():
            return fluents
        elif expression.is_fluent_exp():
            fluents.append(expression.fluent().name)
        else:
            for arg in expression.args:
                arg_fluents = self.find_fluents_affected(arg)
                if arg_fluents:
                    fluents.append(*arg_fluents)
        return fluents

    def manage_node(
            self,
            new_problem: "up.model.Problem",
            new_to_old: Dict[Action, Action],
            goal: "up.model.fnode.FNode",
            n_count: int,
    ) -> Union["up.model.fnode.FNode", "up.model.fluent.Fluent", bool]:
        env = new_problem.environment
        em = env.expression_manager
        tm = env.type_manager

        # 0 = arg del count, 1 = fluents affected
        fluents_affected: Dict[str, List[str]] = {}

        new_args = []
        for arg in goal.args:
            if arg.is_fluent_exp() or arg.is_parameter_exp() or arg.is_constant():
                new_args.append(arg)
            elif arg.is_count():
                new_ca_args = []
                for ca in arg.args:
                    fluent_name = 'count_' + str(n_count)
                    fluents_affected[fluent_name] = self.find_fluents_affected(ca)

                    # controlar valor (en aquest cas inicial de l'expressio) per tant value=None
                    # retorna un boolea
                    initial_value = self.expression_value(new_problem, ca)
                    assert initial_value.is_bool_constant()
                    if initial_value.is_true():
                        fluent_value = Int(1)
                    else:
                        fluent_value = Int(0)
                    new_problem.add_fluent(fluent_name, tm.IntType(), default_initial_value=fluent_value)
                    new_fluent = new_problem.fluent(fluent_name)
                    new_ca_args.append(new_fluent())

                    actions = new_problem.actions
                    new_problem.clear_actions()
                    # new conditional effects to the actions

                    for action in actions:
                        new_action = action.clone()
                        print(action)
                        print("ca: ", ca)
                        new_expression = ca
                        fluent_in_action = False
                        for effect in action.effects:
                            if effect.fluent.fluent().name in fluents_affected[fluent_name]:
                                fluent_in_action = True
                                new_expression = self.expression_value(new_problem, new_expression, effect.fluent.fluent(), effect.value)
                        print("new_expression: ", new_expression)
                        print(new_expression.is_bool_constant())
                        if fluent_in_action:
                            if new_expression.is_bool_constant():
                                if new_expression.is_true():
                                    new_action.add_effect(new_fluent, 1)
                                else:
                                    new_action.add_effect(new_fluent, 0)
                            else:
                                new_action.add_effect(new_fluent, 1, new_expression)
                                new_action.add_effect(new_fluent, 0, Not(new_expression))

                        # afegir la nova condicio amb en nou valor (effect.value) del fluent
                        new_problem.add_action(new_action)
                        new_to_old[new_action] = action

                    n_count += 1
                new_args.append(em.create_node(OperatorKind.PLUS, tuple(new_ca_args)))
            else:
                new_args.append(self.manage_node(new_problem, new_to_old, arg, n_count))
        return em.create_node(goal.node_type, tuple(new_args))

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
        new_problem.clear_goals()
        n_count = 0
        for goal in problem.goals:
            new_goal = self.manage_node(new_problem, new_to_old, goal, n_count)
            new_problem.add_goal(new_goal)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
