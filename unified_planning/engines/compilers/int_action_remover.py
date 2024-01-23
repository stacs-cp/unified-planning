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
"""This module defines the int action remover class."""


import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import Problem, ProblemKind, Fluent, FNode, Action, InstantaneousAction
from unified_planning.model.fluent import get_all_fluent_exp
from unified_planning.model.types import _RealType, _IntType
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.model.walkers import FluentsSubstituter
from unified_planning.engines.compilers.utils import (
    add_invariant_condition_apply_function_to_problem_expressions,
    replace_action,
)
from typing import List, Dict, OrderedDict, Optional, Union, cast, Any
from functools import partial
from itertools import product
from unified_planning.shortcuts import Equals, Int


class IntActionRemover(engines.engine.Engine, CompilerMixin):
    """
    Bounded types remover class: this class offers the capability
    to transform a :class:`~unified_planning.model.Problem` with Bounded :class:`Types <unified_planning.model.Type>`
    into a `Problem` without bounded `Types` (only IntType and RealType can be bounded).
    This capability is offered by the :meth:`~unified_planning.engines.compilers.BoundedTypesRemover.compile`
    method, that returns a :class:`~unified_planning.engines.CompilerResult` in which the :meth:`problem <unified_planning.engines.CompilerResult.problem>` field
    is the compiled Problem.

    This is done by changing the type of the fluents to unbounded types, and adding to every action's condition and
    every goal of the problem the artificial condition that emulates the typing bound.

    For example, if we have a fluent `F` of type `int[0, 5]`, the added condition would be `0 <= F <= 5`.

    This `Compiler` supports only the the `BOUNDED_TYPES_REMOVING` :class:`~unified_planning.engines.CompilationKind`.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INT_ACTION_REMOVING)

    @property
    def name(self):
        return "iarac"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        # canviar!!!!!!!
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
        return problem_kind <= IntActionRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INT_ACTION_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        return problem_kind.clone()

    def _get_new_value(
            self,
            value: "up.model.fnode.FNode",
            int_parameters: dict[str, int],
            c: Any
    ) -> "up.model.fnode.FNode":
        new_value = value
        print(value.node_type)
        print(value.args)
        print(value.environment)
        for key in int_parameters.keys():
            if key.split('] ')[1] in str(new_value):
                new_value = c[int_parameters.get(key)]
        return Int(new_value)

    def _get_new_fnode(
            self,
            problem: "up.model.AbstractProblem",
            fluent: Fluent,
            int_parameters: dict[str, int],
            c: Any
    ) -> "up.model.fnode.FNode":
        new_name = fluent.name
        for key in int_parameters.keys():
            if key in str(new_name):
                fluent_0 = new_name.split(key)[0]
                fluent_1 = new_name.split(key)[1]
                new_name = fluent_0 + str(c[int_parameters.get(key)]) + fluent_1
        fluent = problem.fluent(new_name)
        print("new_fluent: ", fluent)
        # arreglar (+1 parametre)
        if fluent.signature:
            fluent_parameter = fluent.signature[0]
            return fluent(fluent_parameter)
        else:
            return fluent()

    def _manage_node(
            self,
            problem: "up.model.AbstractProblem",
            int_parameters: dict[str, int],
            c: Any,
            node_type: "up.model.OperatorKind",
            args: List["up.model.fnode.FNode"],
    ) -> "up.model.fnode.FNode":

        env = problem.environment
        em = env.expression_manager
        new_arguments = []

        for arg in args:
            print(arg)
            print(arg.type)
            print(arg.node_type)
            print(arg.args)
            print("isfluent: ", arg.is_fluent_exp())
            print("isconst: ", arg.is_constant())
            print("isint: ", arg.is_int_constant())
            print("isparam: ", arg.is_parameter_exp())
            if arg.is_fluent_exp():
                print("fluent: ", arg.fluent())
                new_arguments.append(self._get_new_fnode(problem, arg.fluent(), int_parameters, c))
            elif arg.is_parameter_exp():
                print("arg: ", arg, type(arg))
                new = self._get_new_value(arg, int_parameters, c)
                print("new: ", new, type(new))
                new_arguments.append(new)
            else:
                new_arguments.append(self._manage_node(problem, int_parameters, c, arg.node_type, arg.args))
            print("new arguments: ", new_arguments)
        print(new_arguments)
        if not new_arguments:
            return em.create_node(node_type, ())
        else:
            return em.create_node(node_type, tuple(new_arguments))

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
        new_problem.clear_actions()

        conditions: List[FNode] = []

        parameters = {}
        int_parameters = {}
        int_domains = []

        # per cada accio mirar els parametres i treure el que es enter
        for action in problem.actions:
            if isinstance(action, InstantaneousAction):
                n_i = 0
                # separar els parametres UserType i IntType
                for old_parameter in action.parameters:
                    if old_parameter.type.is_user_type():
                        parameters[old_parameter.name] = old_parameter.type
                    else:
                        int_parameters[str(old_parameter.type) + ' ' + old_parameter.name] = n_i
                        n_i = n_i+1
                        domain = []
                        for i in range(old_parameter.type.lower_bound, old_parameter.type.upper_bound+1):
                            domain.append(i)
                        int_domains.append(domain)

                combinations = list(product(*int_domains))
                # per cada combinacio possible dels enters -> creem una accio
                for c in combinations:
                    new_action = InstantaneousAction(action.name+str(c), parameters, action.environment)

                    # mirem les precondicions
                    for precondition in action.preconditions:
                        new_precondition = self._manage_node(new_problem, int_parameters, c, precondition.node_type, precondition.args)
                        new_action.add_precondition(new_precondition)

                    for effect in action.effects:
                        new_fnode = self._get_new_fnode(problem, effect.fluent.fluent(), int_parameters, c)
                        new_value = self._get_new_value(effect.value, int_parameters, c)
                        if effect.is_increase():
                            new_action.add_increase_effect(new_fnode, new_value, effect.condition, effect.forall)
                        elif effect.is_decrease():
                            new_action.add_decrease_effect(new_fnode, new_value, effect.condition, effect.forall)
                        else:
                            new_action.add_effect(new_fnode, new_value, effect.condition, effect.forall)

                    new_problem.add_action(new_action)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
