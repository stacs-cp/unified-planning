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
from typing import Dict, List, Optional, Tuple, OrderedDict, Any
from functools import partial
from unified_planning.shortcuts import Int
import re

class ArraysRemover(engines.engine.Engine, CompilerMixin):
    """
    Arrays remover class: ...
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.ARRAYS_REMOVING)

    @property
    def name(self):
        return "arm"

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
        supported_kind.set_fluents_type("ARRAY_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
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
        return problem_kind <= ArraysRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.ARRAYS_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        return problem_kind.clone()

    def _manage_node(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
            position: Optional[int] = None
    ) -> "up.model.fnode.FNode":
        env = new_problem.environment
        em = env.expression_manager
        print("node:", node.type)
        print(node.is_fluent_exp())
        print(node.type.is_array_type())
        if node.is_fluent_exp():
            fluent = node.fluent()
            new_name = fluent.name
            while '[' in str(new_name):
                new_name = re.sub(r'\[(\d+)\]', r'_\1', new_name)
            if position is not None:
                new_name = new_name + '_' + str(position)
            return up.model.Fluent(new_name, fluent.type, fluent.signature, fluent.environment)(*fluent.signature)
        elif node.is_parameter_exp():
            print("param: ", node.parameter())
            return node
        elif node.is_constant():
            if position is not None:
                return node.constant_value()[position]
            else:
                return node
        else:
            new_args = []
            for arg in node.args:
                new_args.append(self._manage_node(new_problem, arg))
            return em.create_node(node.node_type, tuple(new_args))

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """
        """
        assert isinstance(problem, Problem)

        new_to_old: Dict[Action, Action] = {}

        env = problem.environment
        em = env.expression_manager
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        # new_problem.clear_timed_goals()
        # new_problem.clear_quality_metrics()
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()

        # FLUENTS AND DEFAULT_VALUES
        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent).constant_value()
            this_fluent = fluent.type
            if this_fluent.is_array_type():
                new_type = this_fluent.elements_type
                domain = []
                while this_fluent.is_array_type():
                    domain_in = []
                    for i in range(0, this_fluent.size):
                        domain_in.append(i)
                    domain.append(domain_in)
                    new_type = this_fluent.elements_type
                    this_fluent = this_fluent.elements_type

                combinations = list(product(*domain))
                for combination in combinations:
                    new_name = fluent.name + ''.join(f'_{str(c)}' for c in combination)
                    new_default_value = default_value
                    for i in combination:
                        new_default_value = new_default_value[i].constant_value()
                    new_problem.add_fluent(model.Fluent(new_name, new_type, fluent.signature, fluent.environment),
                                           default_initial_value=new_default_value)
            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)

        # ACTIONS
        for action in problem.actions:
            new_parameters = OrderedDict()
            if isinstance(action, InstantaneousAction):
                new_action = action.clone()
                new_action.clear_preconditions()
                new_action.clear_effects()

                for precondition in action.preconditions:
                    new_precondition = self._manage_node(new_problem, precondition)
                    new_action.add_precondition(new_precondition)
                for effect in action.effects:
                    new_fnode = self._manage_node(new_problem, effect.fluent)
                    new_value = self._manage_node(new_problem, effect.value)
                    if effect.is_increase():
                        new_action.add_increase_effect(new_fnode, new_value, effect.condition, effect.forall)
                    elif effect.is_decrease():
                        new_action.add_decrease_effect(new_fnode, new_value, effect.condition, effect.forall)
                    else:
                        new_action.add_effect(new_fnode,new_value, effect.condition, effect.forall)
                new_problem.add_action(new_action)

        # GOALS
        for g in problem.goals:
            left = g.arg(0)
            right = g.arg(1)
            if left.type.is_array_type() and right.type.is_array_type():
                #for i in range(left.type.size):
                print("args:", tuple([self._manage_node(new_problem, left, i), self._manage_node(new_problem, right, i)]))
                new_problem.add_goal(em.create_node(g.node_type, tuple([self._manage_node(new_problem, left, i), self._manage_node(new_problem, right, i)])))
            else:
                new_problem.add_goal(em.create_node(g.node_type, tuple(
                    [self._manage_node(new_problem, left), self._manage_node(new_problem, right)])))

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
