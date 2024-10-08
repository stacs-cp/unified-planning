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
from unified_planning.exceptions import UPProblemDefinitionError, UPValueError
from unified_planning.model import (
    Problem,
    Action,
    ProblemKind,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action,
)
from typing import Dict, List, Optional
from functools import partial
from unified_planning.shortcuts import Int, FALSE
import re

class ArraysRemover(engines.engine.Engine, CompilerMixin):
    """
    Arrays remover class: ...
    """

    def __init__(self, mode: str = 'strict'):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.ARRAYS_REMOVING)
        self.mode = mode

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
        return problem_kind <= ArraysRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.ARRAYS_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_fluents_type("ARRAY_FLUENTS")
        return new_kind

    def _get_new_fluent(
        self,
        fluent: "up.model.fluent.Fluent"
    ) -> "up.model.fluent.Fluent":
        new_name = fluent.name
        pattern = r'\[(.*?)\]'
        this_ints = re.findall(pattern, new_name)
        if this_ints:
            new_name = new_name.split('[')[0] + '_' + '_'.join(map(str, this_ints))
        new_fluent = up.model.fluent.Fluent(new_name, fluent.type, fluent.signature, fluent.environment)
        return new_fluent

    def _get_new_fnodes(
        self,
        new_problem: "up.model.AbstractProblem",
        node: "up.model.fnode.FNode",
    ) -> List["up.model.fnode.FNode"]:
        env = new_problem.environment
        em = env.expression_manager

        if node.is_fluent_exp():
            new_fluent = self._get_new_fluent(node.fluent())
            try:
                new_problem.fluent(new_fluent.name)(*node.fluent().signature)
            except (KeyError, UPValueError):
                if self.mode == 'strict':
                    print(f"Fluent {new_fluent.name} out of range!")
                    exit(1)
                else:
                    if new_fluent.type.is_bool_type():
                        return [FALSE()]
                    return [None]
            return [new_fluent(*node.args)]
        elif node.is_parameter_exp() or node.is_constant():
            return [node]
        else:
            if node.arg(0).type.is_array_type():
                assert all(arg.type.is_array_type() for arg in node.args), "Argument is not an array type"

                this_type = node.arg(0).type
                domain = []
                while this_type.is_array_type():
                    domain.append(range(this_type.size))
                    this_type = this_type.elements_type

                new_fnodes = []
                for combination in list(product(*domain)):
                    new_args = []
                    for arg in node.args:
                        if arg.is_fluent_exp():
                            new_fluent = self._get_new_fluent(arg.fluent())
                            new_name = new_fluent.name + ''.join(f'_{str(i)}' for i in combination)
                            try:
                                new_arg = new_problem.fluent(new_name)(*arg.args)
                            except (KeyError, UPValueError):
                                if self.mode == 'strict':
                                    print(f"Fluent {new_fluent.name} out of range!")
                                    exit(1)
                                else:
                                    new_arg = FALSE() if new_fluent.type.is_bool_type() else None
                        elif arg.constant_value():
                            new_arg = arg
                            for i in combination:
                                new_arg = new_arg.constant_value()[i]
                        else:
                            new_arg = arg
                        new_args.append(new_arg)
                    if None in new_args:
                        new_fnodes.append(FALSE() if node.type.is_bool_type() else None)
                    else:
                        new_fnodes.append(em.create_node(node.node_type, tuple(new_args)))
                return new_fnodes
            else:
                new_args = [nla for arg in node.args for nla in self._get_new_fnodes(new_problem, arg)]
                if None in new_args:
                    return [FALSE() if node.type.is_bool_type() else None]
                return [em.create_node(node.node_type, tuple(new_args))]

    def get_element_value(self, v, combination):
        """Obtain the value of the element for a given combination of access."""
        element_value = v
        for c in combination:
            element_value = element_value.constant_value()[c]
        return element_value

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
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.initial_values.clear()
        assert self.mode == 'strict' or self.mode == 'permissive'
        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent, None)

            if fluent.type.is_array_type():
                this_type = fluent.type
                domain = []
                while this_type.is_array_type():
                    domain.append(range(this_type.size))
                    this_type = this_type.elements_type
                for combination in list(product(*domain)):
                    fluent_name = get_fresh_name(new_problem, fluent.name, list(map(str, combination)))
                    new_fluent = model.Fluent(fluent_name, this_type, fluent.signature, fluent.environment)
                    new_problem.add_fluent(new_fluent, default_initial_value=default_value)

                    for f, v in problem.explicit_initial_values.items():
                        if f.fluent() == fluent:
                            element_value = self.get_element_value(v, combination)
                            if element_value != default_value:
                                new_problem.set_initial_value(new_fluent(*f.args), element_value)
                        elif f.fluent().name.split('[')[0] == fluent.name and not f.fluent().type.is_array_type():
                            indices = [int(i) for i in re.findall(r'\[([0-9]+)\]', f.fluent().name)]
                            if tuple(indices) == combination and v != default_value:
                                new_problem.set_initial_value(new_fluent(*f.args), v)
                        # falta el cas que es indexat pero segueix sent array
            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent and v != default_value:
                        new_problem.set_initial_value(fluent(*f.args), v)

        for action in problem.actions:
            for p in action.parameters:
                assert not p.type.is_int_type(), \
                    f"Integer parameter '{p.name}' in action '{action.name}' must be removed before processing."
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()
            new_action.clear_effects()

            for precondition in action.preconditions:
                new_preconditions = self._get_new_fnodes(new_problem, precondition)
                for np in new_preconditions:
                    # si una precondicio es falsa -> accio mai passara -> no afegir accio
                    new_action.add_precondition(np)
            try:
                for effect in action.effects:
                    new_fnode = self._get_new_fnodes(new_problem, effect.fluent)
                    new_value = self._get_new_fnodes(new_problem, effect.value)
                    new_condition = self._get_new_fnodes(new_problem, effect.condition)
                    if effect.is_increase():
                        new_action.add_increase_effect(new_fnode, new_value, new_condition, effect.forall)
                    elif effect.is_decrease():
                        new_action.add_decrease_effect(new_fnode, new_value, new_condition, effect.forall)
                    else:
                        new_action.add_effect(new_fnode, new_value, new_condition, effect.forall)
            except Exception:
                # print(f"Action {action.name} eliminated due to an access to a fluent out of range in the effects.")
                continue
            else:
                new_problem.add_action(new_action)
                new_to_old[new_action] = action

        for g in problem.goals:
            new_goals = self._get_new_fnodes(new_problem, g)
            for ng in new_goals:
                new_problem.add_goal(ng)
        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
