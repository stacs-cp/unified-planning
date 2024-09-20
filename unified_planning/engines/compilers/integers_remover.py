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
        return "irm"

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
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("INT_FLUENTS")
        return new_kind

    def _get_new_fnode(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
    ) -> up.model.fnode.FNode:
        env = new_problem.environment
        em = env.expression_manager
        tm = env.type_manager
        if node.is_int_constant():
            number_user_type = tm.UserType('Number')
            new_number = model.Object('n' + str(node.int_constant_value()), number_user_type)
            return em.ObjectExp(new_number)
        elif node.is_fluent_exp() and node.fluent().type.is_int_type():
            return new_problem.fluent(node.fluent().name)(*node.fluent().signature)
        elif node.is_parameter_exp() or node.is_object_exp() or node.is_fluent_exp() or node.is_constant() or node.is_variable_exp():
            return node
        else:
            new_args = []
            for arg in node.args:
                new = self._get_new_fnode(new_problem, arg)
                new_args.append(new)
            if node.node_type == OperatorKind.PLUS:
                operation = 'plus'
            elif node.node_type == OperatorKind.MINUS:
                operation = 'minus'
            elif node.node_type == OperatorKind.DIV:
                operation = 'div'
            elif node.node_type == OperatorKind.TIMES:
                operation = 'mult'
            elif node.node_type == OperatorKind.LT:
                operation = 'lt'
            elif node.node_type == OperatorKind.LE:
                # trobar el rang d'enters
                lb = None
                ub = None
                print(node, new_args)
                for arg in new_args:
                    print(arg, arg.type)
                    if arg.is_fluent_exp() and arg.type.is_int_type():
                        if lb is None or arg.type.lower_bound < lb:
                            lb = arg.type.lower_bound
                        if ub is None or arg.type.upper_bound > ub:
                            ub = arg.type.upper_bound
                print(lb,ub)
                self._add_relationships(new_problem, 'lt', lb, ub)
                if len(new_args) > 2:
                    result = em.Or(new_problem.fluent('lt')(new_args[0], new_args[1]), em.Equals(new_args[0], new_args[1]))
                    for arg in new_args[2:]:
                        em.Or(new_problem.fluent('lt')(result, arg), em.Equals(result, arg))
                    return result
                else:
                    return em.Or(new_problem.fluent('lt')(*new_args), em.Equals(*new_args))
            else:
                return em.create_node(node.node_type, tuple(new_args))
            # trobar el rang d'enters
            lb = None
            ub = None
            print(node, new_args)
            for arg in new_args:
                print(arg, arg.is_fluent_exp(), arg.type.is_int_type())
                if arg.is_fluent_exp() and arg.type.is_int_type():
                    if lb is None or arg.type.lower_bound < lb:
                        lb = arg.type.lower_bound
                    if ub is None or arg.type.upper_bound > ub:
                        ub = arg.type.upper_bound
            print(lb, ub)
            self._add_relationships(new_problem, operation, lb, ub)
            if len(new_args) > 2:
                result = new_problem.fluent(operation)(new_args[0], new_args[1])
                for arg in new_args[2:]:
                    result = new_problem.fluent(operation)(result, arg)
                return result
            else:
                return new_problem.fluent(operation)(*new_args)

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
            relationship: str,
            lower_bound: int,
            upper_bound: int,
    ):
        # lt, plus, minus, div, mult
        # crear fluent de relacio si no hi es
        params = OrderedDict()
        ut_number = new_problem.user_type('Number')
        params['n1'] = ut_number
        params['n2'] = ut_number
        if new_problem.fluent(relationship):
            relationship_fluent = new_problem.fluent(relationship)
        else:
            relationship_fluent = model.Fluent(relationship, _signature=params, environment=new_problem.environment)
            if relationship == 'lt':
                def_value = False
            else:
                def_value = new_problem.object('null')
            new_problem.add_fluent(relationship_fluent, default_initial_value = def_value)

        # mirar si les relacions del rang d'aquests numeros estan inicialitzats
        for i in range(lower_bound, upper_bound + 1):
            for j in range(i, upper_bound + 1):
                ni = new_problem.object('n' + str(i))
                nj = new_problem.object('n' + str(j))
                if new_problem.initial_value(relationship_fluent(ni, nj)) is None:
                    if relationship == 'lt':
                        new_problem.set_initial_value(relationship_fluent(ni, nj), True)
                    elif relationship == 'plus':
                        try:
                            plus_i_j = new_problem.object('n' + str(i+j))
                            if plus_i_j:
                                new_problem.set_initial_value(relationship_fluent(ni, nj), plus_i_j)
                                new_problem.set_initial_value(relationship_fluent(nj, ni), plus_i_j)
                        except UPValueError:
                            pass
                    elif relationship == 'minus':
                        try:
                            minus_i_j = new_problem.object('n' + str(i-j))
                            if minus_i_j:
                                new_problem.set_initial_value(relationship_fluent(ni, nj), minus_i_j)
                        except UPValueError:
                            pass
                        try:
                            minus_j_i = new_problem.object('n' + str(j-i))
                            if minus_j_i:
                                new_problem.set_initial_value(relationship_fluent(nj, ni), minus_j_i)
                        except UPValueError:
                            pass
                    # Div
                    elif relationship == 'div':
                        try:
                            if j > 0:
                                div_i_j = new_problem.object('n' + str(i/j))
                                if div_i_j:
                                    new_problem.set_initial_value(relationship_fluent(ni, nj), div_i_j)
                        except UPValueError:
                            pass
                        try:
                            if i > 0:
                                div_j_i = new_problem.object('n' + str(j/i))
                                if div_j_i:
                                    new_problem.set_initial_value(relationship_fluent(nj, ni), div_j_i)
                        except UPValueError:
                            pass
                    # Mult
                    elif relationship == 'mult':
                        try:
                            mult_i_j = new_problem.object('n' + str(i*j))
                            if mult_i_j:
                                new_problem.set_initial_value(relationship_fluent(ni, nj), mult_i_j)
                        except UPValueError:
                            pass
                        try:
                            mult_j_i = new_problem.object('n' + str(j*i))
                            if mult_j_i:
                                new_problem.set_initial_value(relationship_fluent(nj, ni), mult_j_i)
                        except UPValueError:
                            pass

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
        null = model.Object('null', ut_number)
        new_problem.add_object(null)
        # Relationships between objects
        #params = OrderedDict()
        #params['n1'] = ut_number
        #params['n2'] = ut_number
        #lt = model.Fluent('lt', _signature=params, environment=env)
        #plus = model.Fluent('plus', ut_number, _signature=params, environment=env)
        #minus = model.Fluent('minus', ut_number, _signature=params, environment=env)
        #div = model.Fluent('div', ut_number, _signature=params, environment=env)
        #mult = model.Fluent('mult', ut_number, _signature=params, environment=env)
        #new_problem.add_fluent(lt, default_initial_value=False)
        #new_problem.add_fluent(plus, default_initial_value=null)
        #new_problem.add_fluent(minus, default_initial_value=null)
        #new_problem.add_fluent(div, default_initial_value=null)
        #new_problem.add_fluent(mult, default_initial_value=null)

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)
            if fluent.type.is_int_type():
                tlb = fluent.type.lower_bound
                tub = fluent.type.upper_bound
                new_fluent = model.Fluent(fluent.name, ut_number, fluent.signature, env)
                # First integer fluent! - control of ranges
                if lb is None and ub is None:
                    self._add_object_numbers(new_problem, tlb, tub + 1)
                    #self._add_relationships(new_problem, tlb, None, None, tub)
                    ub = tub
                    lb = tlb
                # if another fluent has lower or upper range add them
                elif tub > ub or tlb < lb:
                    if tub > ub:
                        self._add_object_numbers(new_problem, ub + 1, tub + 1)
                        #self._add_relationships(new_problem, lb, None, ub+1, tub)
                        ub = tub
                    if tlb < lb:
                        self._add_object_numbers(new_problem, tlb, lb)
                        #self._add_relationships(new_problem, tlb, lb, None, tub)
                        lb = tlb
                # Default initial values
                if default_value is not None:
                    new_problem.add_fluent(new_fluent,
                                           default_initial_value=new_problem.object('n' + str(default_value)))
                else:
                    new_problem.add_fluent(new_fluent)
                # Initial values
                if fluent.signature:
                    objects = []
                    for s in fluent.signature:
                        objects.append(problem.objects(s.type))
                    fluent_parameters = list(product(*objects))
                    for fp in fluent_parameters:
                        iv = problem.initial_value(fluent(*fp))
                        if iv is None:
                            raise UPProblemDefinitionError(
                                f"Initial value not set for fluent: {fluent(*fp)}"
                            )
                        elif iv != default_value:
                            new_initial_value = model.Object('n' + str(iv), ut_number)
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
                # Default initial values
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                # Initial values
                if fluent.signature:
                    objects = []
                    for s in fluent.signature:
                        objects.append(problem.objects(s.type))
                    fluent_parameters = list(product(*objects))
                    for fp in fluent_parameters:
                        iv = problem.initial_value(fluent(*fp))
                        if iv is None:
                            raise UPProblemDefinitionError(
                                f"Initial value not set for fluent: {fluent(*fp)}"
                            )
                        elif iv != default_value:
                            new_problem.set_initial_value(fluent(*fp), iv)
                else:
                    iv = problem.initial_value(fluent())
                    if iv is None:
                        raise UPProblemDefinitionError(
                            f"Initial value not set for fluent: {fluent()}"
                        )
                    elif iv != default_value:
                        new_problem.set_initial_value(fluent(), iv)

        # Actions
        for action in problem.actions:
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()
            new_action.clear_effects()
            for precondition in action.preconditions:
                new_precondition = self._get_new_fnode(new_problem, precondition)
                new_action.add_precondition(new_precondition)
            for effect in action.effects:
                new_fnode = self._get_new_fnode(new_problem, effect.fluent)
                new_value = self._get_new_fnode(new_problem, effect.value)
                new_condition = self._get_new_fnode(new_problem, effect.condition)
                if effect.is_increase():
                    new_action.add_increase_effect(new_fnode, new_value, new_condition, effect.forall)
                elif effect.is_decrease():
                    new_action.add_decrease_effect(new_fnode, new_value, new_condition, effect.forall)
                else:
                    new_action.add_effect(new_fnode, new_value, new_condition, effect.forall)
            new_problem.add_action(new_action)
            new_to_old[new_action] = action

        for goal in problem.goals:
            new_problem.add_goal(self._get_new_fnode(new_problem, goal))

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )

