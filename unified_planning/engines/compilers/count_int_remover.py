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

from unified_planning.model.walkers import simplifier

import unified_planning as up
import unified_planning.engines as engines
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import (
    Problem,
    Action,
    ProblemKind,
    OperatorKind,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    replace_action,
)
from typing import Dict, List, Optional, Union
from functools import partial
from unified_planning.shortcuts import Int, Not, And, Equals

class CountIntRemover(engines.engine.Engine, CompilerMixin):
    """
    Count expression remover class: ...
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

    def expression_value(
            self,
            new_problem: "up.model.Problem",
            expression: "up.model.fnode.FNode",
            fluent: Optional["up.model.fnode.FNode"] = None,
            value: Optional["up.model.fnode.FNode"] = None,
            type_effect: Optional[str] = None
    ) -> "up.model.fnode.FNode":
        env = new_problem.environment
        em = env.expression_manager
        if expression.is_constant() or expression.is_parameter_exp():
            return expression
        elif expression.is_fluent_exp():
            if fluent is None:
                return new_problem.initial_value(expression)
            else:
                if fluent == expression:
                    if type_effect == 'increase':
                        new_expression = em.create_node(OperatorKind.PLUS, tuple([expression, value])).simplify()
                    elif type_effect == 'decrease':
                        new_expression = em.create_node(OperatorKind.MINUS, tuple([expression, value])).simplify()
                    else:
                        new_expression = value
                else:
                    new_expression = expression
                return new_expression
        else:
            new_args = []
            for arg in expression.args:
                new_args.append(self.expression_value(new_problem, arg, fluent, value, type_effect))
            return em.create_node(expression.node_type, tuple(new_args)).simplify()

    def find_fluents_affected(
            self,
            expression: "up.model.fnode.FNode",
    ) -> List["up.model.fnode.FNode"]:
        fluents = []
        if expression.is_fluent_exp():
            fluents.append(expression)
        else:
            for arg in expression.args:
                fluents += self.find_fluents_affected(arg)
        return fluents

    def add_count_effects(
            self,
            new_problem: "up.model.Problem",
            action: "up.model.action.Action",
            count_expressions: Dict[str, "up.model.fnode.FNode"]
    ) -> "up.model.action.Action":
        for count, expression in count_expressions.items():
            new_expression = expression
            effects_conditions = True
            direct_effect_fluents = []
            indirect_effect_fluents = []
            possible_parameters: Dict["up.model.fnode.FNode", List["up.model.fnode.FNode"]] = {}

            for effect in action.effects:
                for fe in self.find_fluents_affected(expression):
                    if effect.fluent == fe:
                        direct_effect_fluents.append(effect)
                    elif effect.fluent.fluent().name == fe.fluent().name:
                        if indirect_effect_fluents.count(effect) == 0:
                            indirect_effect_fluents.append(effect)
                        for i in range(len(effect.fluent.args)):
                            this_parameter = effect.fluent.arg(i)
                            this_object = fe.arg(i)
                            if this_parameter in possible_parameters.keys():
                                possible_parameters[this_parameter].append(this_object)
                            else:
                                possible_parameters[this_parameter] = [this_object]

            if direct_effect_fluents or indirect_effect_fluents:
                for effect in direct_effect_fluents:
                    if effect.is_conditional():
                        effects_conditions = And(effects_conditions, effect.condition).simplify()
                    if effect.is_increase():
                        type_effect = 'increase'
                    elif effect.is_decrease():
                        type_effect = 'decrease'
                    else:
                        type_effect = None
                    new_expression = self.expression_value(new_problem, new_expression, effect.fluent, effect.value,
                                                               type_effect)

                if indirect_effect_fluents:
                    combinations = list(product(*possible_parameters.values()))
                    keys = list(possible_parameters.keys())
                    for c in combinations:
                        comb_new_expression = new_expression
                        comb_effects_conditions = effects_conditions
                        for effect in indirect_effect_fluents:
                            new_args_fluent = []
                            for arg in effect.fluent.args:
                                i = keys.index(arg)
                                new_args_fluent.append(c[i])
                                comb_effects_conditions = And(comb_effects_conditions, Equals(c[i], arg)).simplify()

                            if effect.is_conditional():
                                comb_effects_conditions = And(comb_effects_conditions, effect.condition).simplify()
                            if effect.is_increase():
                                type_effect = 'increase'
                            elif effect.is_decrease():
                                type_effect = 'decrease'
                            else:
                                type_effect = None
                            comb_new_expression = self.expression_value(new_problem, comb_new_expression, effect.fluent.fluent()(*new_args_fluent),
                                                                   effect.value, type_effect)
                        if comb_new_expression.is_bool_constant():
                            if comb_new_expression.is_true():
                                new_value = 1
                            else:
                                new_value = 0
                            if comb_effects_conditions is None:
                                action.add_effect(new_problem.fluent(count), new_value)
                            else:
                                action.add_effect(new_problem.fluent(count), new_value, comb_effects_conditions)
                        else:
                            if comb_effects_conditions is None:
                                action.add_effect(new_problem.fluent(count), 1, comb_new_expression)
                                action.add_effect(new_problem.fluent(count), 0, Not(comb_new_expression).simplify())
                            else:
                                action.add_effect(new_problem.fluent(count), 1, And(comb_new_expression, comb_effects_conditions).simplify())
                                action.add_effect(new_problem.fluent(count), 0,
                                                  And(Not(comb_new_expression), comb_effects_conditions).simplify())
                else:
                    if new_expression.is_bool_constant():
                        if new_expression.is_true():
                            new_value = 1
                        else:
                            new_value = 0
                        if effects_conditions is None:
                            action.add_effect(new_problem.fluent(count), new_value)
                        else:
                            action.add_effect(new_problem.fluent(count), new_value, effects_conditions)
                    else:
                        if effects_conditions is None:
                            action.add_effect(new_problem.fluent(count), 1, new_expression)
                            action.add_effect(new_problem.fluent(count), 0, Not(new_expression).simplify())
                        else:
                            action.add_effect(new_problem.fluent(count), 1, And(new_expression, effects_conditions).simplify())
                            action.add_effect(new_problem.fluent(count), 0, And(Not(new_expression), effects_conditions).simplify())
        return action

    def add_counts(
            self,
            new_problem: "up.model.Problem",
            expression: "up.model.fnode.FNode",
            count_expressions: Dict[str, "up.model.fnode.FNode"]
    ) -> Union["up.model.fnode.FNode", "up.model.fluent.Fluent", bool]:
        env = new_problem.environment
        em = env.expression_manager
        tm = env.type_manager
        if expression.is_fluent_exp() or expression.is_parameter_exp() or expression.is_constant():
            return expression
        elif expression.is_count():
            new_ca_args = []
            for ca in expression.args:
                if ca.is_false():
                    pass
                elif ca.is_true():
                    new_ca_args.append(Int(1))
                else:
                    existing_key = next((key for key, value in count_expressions.items() if value == ca), None)
                    if existing_key is not None:
                        existing_fluent = new_problem.fluent(existing_key)
                        new_ca_args.append(existing_fluent())
                    else:
                        n_count = len(count_expressions)
                        fluent_name = 'count_' + str(n_count)
                        count_expressions[fluent_name] = ca
                        initial_value = self.expression_value(new_problem, ca)
                        assert initial_value.is_bool_constant()
                        if initial_value.is_true():
                            fluent_value = Int(1)
                        else:
                            fluent_value = Int(0)
                        new_problem.add_fluent(fluent_name, tm.IntType(0, 1))
                        new_problem.set_initial_value(new_problem.fluent(fluent_name), fluent_value)
                        new_fluent = new_problem.fluent(fluent_name)
                        new_ca_args.append(new_fluent())
                        n_count += 1
            return em.create_node(OperatorKind.PLUS, tuple(new_ca_args))
        else:
            new_args = []
            for arg in expression.args:
                new_args.append(self.add_counts(new_problem, arg, count_expressions))
        return em.create_node(expression.node_type, tuple(new_args))

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
        new_problem.clear_actions()
        for action in problem.actions:
            new_action = action.clone()
            new_action.clear_preconditions()
            for pre in action.preconditions:
                new_precondition = self.add_counts(new_problem, pre, count_expressions)
                new_action.add_precondition(new_precondition)
            new_problem.add_action(new_action)

        new_problem.clear_goals()
        for goal in problem.goals:
            new_goal = self.add_counts(new_problem, goal, count_expressions)
            new_problem.add_goal(new_goal)

        new_actions = []
        changed_actions = new_problem.actions
        new_problem.clear_actions()
        for action in changed_actions:
            new_action = self.add_count_effects(new_problem, action, count_expressions)
            new_actions.append(new_action)
            new_problem.add_action(new_action)

        for i in range(0, len(problem.actions)):
            new_to_old[new_actions[i]] = problem.actions[i]

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )