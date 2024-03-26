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
    Effect,
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
        supported_kind.set_fluents_type("ARRAY_FLUENTS")
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

    def check_initial_value(
            self, arg: "up.model.fnode.FNode", new_problem: "up.model.Problem") -> Int:
        assert arg.type.is_bool_type()
        if arg.is_true():
            return Int(1)
        elif arg.is_false():
            return Int(0)
        elif arg.is_fluent_exp():
            fluent = arg.fluent()
            assert fluent.type.is_bool_type()
            return Int(1) if new_problem.initial_value(arg).is_true() else Int(0)
        else:
            # search initial value?
            return Int(0)

    def manage_node(
            self,
            new_problem: "up.model.Problem",
            goal: "up.model.fnode.FNode",
            n_count: int,
    ) -> Union["up.model.fnode.FNode", "up.model.fluent.Fluent", bool]:
        env = new_problem.environment
        em = env.expression_manager
        tm = env.type_manager

        new_args = []
        for arg in goal.args:
            if arg.is_fluent_exp() or arg.is_parameter_exp() or arg.is_constant():
                new_args.append(arg)
            elif arg.is_count():
                new_ca_args = []
                for ca in arg.args:
                    fluent_name = 'count_' + str(n_count)
                    new_problem.add_fluent(fluent_name, tm.IntType(),
                                           default_initial_value=self.check_initial_value(ca, new_problem))
                    new_fluent = new_problem.fluent(fluent_name)
                    new_ca_args.append(new_fluent)

                    actions = new_problem.actions
                    new_problem.clear_actions()
                    # new conditional effects to the actions
                    for action in actions:
                        new_action = action.clone()
                        print(new_action)

                        #new_action.clear_effects()

                        #for effect in action.effects:

                            #if effect.is_increase():
                            #    new_action.add_increase_effect(new_fnode, new_value, new_condition, effect.forall)
                            #elif effect.is_decrease():
                            #    new_action.add_decrease_effect(new_fnode, new_value, new_condition, effect.forall)
                            #else:
                            #    new_action.add_effect(new_fnode, new_value, new_condition, effect.forall)
                        #new_problem.add_action(new_action)
                        #new_to_old[new_action] = action

                    new_action_true = InstantaneousAction("set_true_"+fluent_name)
                    new_action_true.add_precondition(ca)
                    new_action_true.add_effect(new_fluent, Int(1))

                    new_action_false = InstantaneousAction("set_false_"+fluent_name)
                    new_action_false.add_precondition(Not(ca))
                    new_action_false.add_effect(new_fluent, Int(0))

                    new_problem.add_action(new_action_true)
                    new_problem.add_action(new_action_false)
                    n_count += 1

                new_args.append(Plus(new_ca_args))
            else:
                new_args.append(self.manage_node(new_problem, arg, n_count))
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
        # new_problem.clear_timed_goals()
        # new_problem.clear_quality_metrics()
        new_problem.clear_goals()
        n_count = 0
        for goal in problem.goals:
            new_goal = self.manage_node(new_problem, goal, n_count)
            new_problem.add_goal(new_goal)

        print(problem.goals)
        print(new_problem.goals)
        print("-------")
        print(new_problem.actions)


        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
