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

from unified_planning.model.operators import OperatorKind

import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
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
from typing import Dict, List, Optional, OrderedDict, Union
from functools import partial
from unified_planning.model.types import _UserType
from unified_planning.shortcuts import Int, FALSE
import re


class IntegersRemover(engines.engine.Engine, CompilerMixin):
    """
    Integers remover class: ...
    """

    def __init__(self, mode: str = 'strict'):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGERS_REMOVING)
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
        return problem_kind <= IntegersRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INTEGERS_REMOVING

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        return problem_kind.clone()

    def _get_new_fnode(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
    ) -> up.model.fnode.FNode:
        env = new_problem.environment
        em = env.expression_manager
        tm = env.type_manager
        print("args of node: ", node.args)
        if node.is_int_constant():
            print("int: ", node.int_constant_value())
            number_user_type = tm.UserType('Number')
            new_number = model.Object('n' + str(node.int_constant_value()), number_user_type)
            print("new_number: ", new_number, new_number.type)
            return em.ObjectExp(new_number)
        elif node.is_fluent_exp() and node.fluent().type.is_int_type():
            print("accedint a new fluent.. ", node.fluent().name, *node.fluent().signature)
            return new_problem.fluent(node.fluent().name)(*node.fluent().signature)
        elif node.args == ():
            # if node.node_type == OperatorKind.PLUS:
            # elif node.node_type == OperatorKind.MINUS:
            # elif node.node_type == OperatorKind.DIV:
            # elif node.node_type == OperatorKind.LE:
            # elif node.node_type == OperatorKind.LT:
            # elif node.node_type == OperatorKind.:
            return node
        else:
            new_args = []
            for arg in node.args:
                new = self._get_new_fnode(new_problem, arg)
                print(new.type)
                new_args.append(new)
                print("new_args: ", new, new.type)
            return em.create_node(node.node_type, tuple(new_args))

    def _add_object_numbers(
            self,
            new_problem: "up.model.AbstractProblem",
            lower_bound: int,
            upper_bound: int,
    ):
        for i in range(lower_bound, upper_bound):
            new_number = model.Object('n' + str(i), new_problem.user_type('Number'))
            new_problem.add_object(new_number)

    def _add_relationships(
            self,
            new_problem: "up.model.AbstractProblem",
            lower_bound: int,
            mid_low_bound: Union[int, None],
            mid_up_bound: Union[int, None],
            upper_bound: int,
    ):
        eq = new_problem.fluent("eq")
        lt = new_problem.fluent("lt")
        for i in range(lower_bound, upper_bound + 1):
            if mid_low_bound is None or i < mid_low_bound:
                for j in range(i, upper_bound + 1):
                    if mid_up_bound is None or j >= mid_up_bound:
                        if i == j:
                            new_problem.set_initial_value(
                                eq(new_problem.object('n' + str(i)), new_problem.object('n' + str(j))), True
                            )
                        if i < j:
                            new_problem.set_initial_value(
                                lt(new_problem.object('n' + str(i)), new_problem.object('n' + str(j))), True
                            )

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
        env = new_problem.environment
        tm = env.type_manager
        assert self.mode == 'strict' or self.mode == 'permissive'
        lb = None
        ub = None
        ut_number = tm.UserType('Number')
        # Relationships between objects
        params = OrderedDict()
        params['n1'] = ut_number
        params['n2'] = ut_number
        lt = model.Fluent('lt', _signature=params, environment=env)
        eq = model.Fluent('eq', _signature=params, environment=env)
        new_problem.add_fluent(lt, default_initial_value=False)
        new_problem.add_fluent(eq, default_initial_value=False)

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)
            # si el fluent es integer
            if fluent.type.is_int_type():
                # crear nou fluent objecte
                tlb = fluent.type.lower_bound
                tub = fluent.type.upper_bound
                print("this int: ", tlb, tub)
                new_fluent = model.Fluent(fluent.name, ut_number, fluent.signature, env)
                print("new fluent: ", new_fluent)

                # First integer fluent! - control of ranges
                if lb is None and ub is None:
                    self._add_object_numbers(new_problem, tlb, tub + 1)
                    self._add_relationships(new_problem, tlb, None, None, tub)
                    ub = tub
                    lb = tlb
                # si aquest fluent te rang amb numeros superiors a l'anterior, afegir-los
                elif tub > ub or tlb < lb:
                    if tub > ub:
                        self._add_object_numbers(new_problem, ub + 1, tub + 1)
                        self._add_relationships(new_problem, lb, None, ub+1, tub)
                        ub = tub
                    if tlb < lb:
                        self._add_object_numbers(new_problem, tlb, lb)
                        self._add_relationships(new_problem, tlb, lb, None, tub)
                        lb = tlb

                # default value
                if default_value is not None:
                    new_problem.add_fluent(new_fluent,
                                           default_initial_value=new_problem.object('n' + str(default_value)))
                else:
                    new_problem.add_fluent(new_fluent)

                # initial value
                objects = []
                for s in fluent.signature:
                    objects.append(problem.objects(s.type))
                fluent_parameters = list(product(*objects))
                if fluent_parameters == [()]:
                    fluent_parameters = []

                if fluent_parameters:
                    for fp in fluent_parameters:
                        iv = problem.initial_value(fluent(*fp))
                        if iv is None:
                            raise UPProblemDefinitionError(
                                f"Initial value not set for fluent: {fluent(*fp)}"
                            )
                        elif iv != default_value:
                            new_initial_value = model.Object('n' + str(iv), ut_number)
                            # afegir objecte si no hi es
                            new_problem.set_initial_value(new_fluent(*fp), new_initial_value)
                else:
                    iv = problem.initial_value(fluent())
                    if iv is None:
                        raise UPProblemDefinitionError(
                            f"Initial value not set for fluent: {fluent()}"
                        )
                    elif iv != default_value:
                        new_initial_value = model.Object('n' + str(iv), ut_number)
                        new_problem.set_initial_value(new_fluent(), new_initial_value)

            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                iv = problem.initial_value(fluent(fluent.signature))
                if iv is None:
                    raise UPProblemDefinitionError(
                        f"Initial value not set for fluent: {fluent(fluent.signature)}"
                    )
                elif iv != default_value:
                    new_problem.set_initial_value(fluent(fluent.signature), iv)

        print("INITIAL VALUES: ", new_problem.initial_values)

        # canviar numeros en precondicions i efectes d'accions
        for action in problem.actions:
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()
            new_action.clear_effects()

            for precondition in action.preconditions:
                new_preconditions = self._get_new_fnode(new_problem, precondition)
                print("new preconditions: ", new_preconditions)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
