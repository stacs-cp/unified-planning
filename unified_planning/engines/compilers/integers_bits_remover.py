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
import math
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


class IntegersBitsRemover(engines.engine.Engine, CompilerMixin):
    """
    Integers remover class: ...
    """

    def __init__(self, mode: str = 'strict'):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGERS_BITS_REMOVING)
        self.lb = None
        self.ub = None

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
        return problem_kind <= IntegersBitsRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INTEGERS_BITS_REMOVING

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("INT_FLUENTS")
        return new_kind

    def _get_new_fnode(
            self,
            old_problem: "up.model.AbstractProblem",
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode"
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
                new = self._get_new_fnode(old_problem, new_problem, arg)
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
                operation = 'le'
            else:
                return em.create_node(node.node_type, tuple(new_args))
            if operation == 'le':
                self._add_relationships(new_problem, 'lt')
                if len(new_args) > 2:
                    result = em.Or(new_problem.fluent('lt')(new_args[0], new_args[1]),
                                   em.Equals(new_args[0], new_args[1]))
                    for arg in new_args[2:]:
                        em.Or(new_problem.fluent('lt')(result, arg), em.Equals(result, arg))
                    return result
                else:
                    return em.Or(new_problem.fluent('lt')(*new_args), em.Equals(*new_args))
            else:
                self._add_relationships(new_problem, operation)
                if len(new_args) > 2:
                    result = new_problem.fluent(operation)(new_args[0], new_args[1])
                    for arg in new_args[2:]:
                        result = new_problem.fluent(operation)(result, arg)
                    return result
                else:
                    return new_problem.fluent(operation)(*new_args)

    def _convert_int_fluent(
            self,
            old_problem: "up.model.AbstractProblem",
            new_problem: "up.model.AbstractProblem",
            fluent: "up.model.Fluent"
    ):
        env = new_problem.environment
        em = env.expression_manager
        tm = env.type_manager
        params = []
        n_bits = math.ceil(math.log2(fluent.type.upper_bound - fluent.type.lower_bound + 1))
        for i in range(0, n_bits):
            params.append(up.model.Parameter('b' + str(i), tm.BoolType()))
        new_fluent = model.Fluent(fluent.name, _signature=fluent.signature + params, environment=new_problem.environment)
        default_value = old_problem.fluents_defaults.get(fluent)
        # Default initial values
        if default_value is not None:
            new_problem.add_fluent(new_fluent, default_initial_value=False)
        else:
            new_problem.add_fluent(new_fluent)
        # Initial values
        for k, v in old_problem.initial_values.items():
            if k.type.is_int_type() and k.fluent().name == fluent.name:
                n_binari = bin(v.constant_value())[2:]
                number_with_bits = n_binari.zfill(n_bits)
                bits_param = [b == '1' for b in number_with_bits]
                new_problem.set_initial_value(new_problem.fluent(k.fluent().name)(*k.args + tuple(bits_param)),
                                              True)

    def _add_relationships(
            self,
            new_problem: "up.model.AbstractProblem",
            relationship: str,
    ):
        # lt, plus, minus, div, mult
        # crear fluent de relacio si no hi es
        params = OrderedDict()
        ut_number = new_problem.user_type('Number')
        params['n1'] = ut_number
        params['n2'] = ut_number
        try:
            relationship_fluent = new_problem.fluent(relationship)
        except UPValueError:
            relationship_fluent = model.Fluent(relationship, _signature=params, environment=new_problem.environment)
            new_problem.add_fluent(relationship_fluent)

        # mirar si les relacions del rang d'aquests numeros estan inicialitzats
        for i in range(self.lb, self.ub + 1):
            for j in range(i, self.ub + 1):
                ni = new_problem.object('n' + str(i))
                nj = new_problem.object('n' + str(j))
                if new_problem.initial_values.get(relationship_fluent(ni, nj)) is None:
                    if relationship == 'lt':
                        if i < j:
                            new_problem.set_initial_value(relationship_fluent(ni, nj), True)
                        else:
                            new_problem.set_initial_value(relationship_fluent(ni, nj), False)
                        if j < i:
                            new_problem.set_initial_value(relationship_fluent(nj, ni), True)
                        else:
                            new_problem.set_initial_value(relationship_fluent(nj, ni), False)
                    elif relationship == 'plus':
                        try:
                            plus_i_j = new_problem.object('n' + str(i+j))
                            new_problem.set_initial_value(relationship_fluent(ni, nj), plus_i_j)
                            new_problem.set_initial_value(relationship_fluent(nj, ni), plus_i_j)
                        except UPValueError:
                            try:
                                null = new_problem.object('null')
                            except UPValueError:
                                new_problem.add_object(model.Object('null', ut_number))
                                null = new_problem.object('null')
                            new_problem.set_initial_value(relationship_fluent(ni, nj), null)
                            new_problem.set_initial_value(relationship_fluent(nj, ni), null)
                    elif relationship == 'minus':
                        try:
                            minus_i_j = new_problem.object('n' + str(i-j))
                            new_problem.set_initial_value(relationship_fluent(ni, nj), minus_i_j)
                        except UPValueError:
                            try:
                                null = new_problem.object('null')
                            except UPValueError:
                                new_problem.add_object(model.Object('null', ut_number))
                                null = new_problem.object('null')
                            new_problem.set_initial_value(relationship_fluent(ni, nj), null)
                        try:
                            minus_j_i = new_problem.object('n' + str(j-i))
                            new_problem.set_initial_value(relationship_fluent(nj, ni), minus_j_i)
                        except UPValueError:
                            try:
                                null = new_problem.object('null')
                            except UPValueError:
                                new_problem.add_object(model.Object('null', ut_number))
                                null = new_problem.object('null')
                            new_problem.set_initial_value(relationship_fluent(nj, ni), null)

                    # Div
                    elif relationship == 'div':
                        try:
                            if j > 0:
                                div_i_j = new_problem.object('n' + str(i/j))
                                new_problem.set_initial_value(relationship_fluent(ni, nj), div_i_j)
                        except UPValueError:
                            try:
                                null = new_problem.object('null')
                            except UPValueError:
                                new_problem.add_object(model.Object('null', ut_number))
                                null = new_problem.object('null')
                            new_problem.set_initial_value(relationship_fluent(ni, nj), null)
                        try:
                            if i > 0:
                                div_j_i = new_problem.object('n' + str(j/i))
                                new_problem.set_initial_value(relationship_fluent(nj, ni), div_j_i)
                        except UPValueError:
                            try:
                                null = new_problem.object('null')
                            except UPValueError:
                                new_problem.add_object(model.Object('null', ut_number))
                                null = new_problem.object('null')
                            new_problem.set_initial_value(relationship_fluent(nj, ni), null)
                    # Mult
                    elif relationship == 'mult':
                        try:
                            mult_i_j = new_problem.object('n' + str(i*j))
                            new_problem.set_initial_value(relationship_fluent(ni, nj), mult_i_j)
                        except UPValueError:
                            try:
                                null = new_problem.object('null')
                            except UPValueError:
                                new_problem.add_object(model.Object('null', ut_number))
                                null = new_problem.object('null')
                            new_problem.set_initial_value(relationship_fluent(ni, nj), null)
                        try:
                            mult_j_i = new_problem.object('n' + str(j*i))
                            new_problem.set_initial_value(relationship_fluent(nj, ni), mult_j_i)
                        except UPValueError:
                            try:
                                null = new_problem.object('null')
                            except UPValueError:
                                new_problem.add_object(model.Object('null', ut_number))
                                null = new_problem.object('null')
                            new_problem.set_initial_value(relationship_fluent(nj, ni), null)

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

        # Search range of integers in this problem
        for fluent in problem.fluents:
            if fluent.type.is_int_type():
                # First integer fluent! - control of ranges
                if self.lb is None and self.ub is None:
                    self.ub = fluent.type.upper_bound
                    self.lb = fluent.type.lower_bound
                # if another fluent has lower or upper range add them
                elif fluent.type.upper_bound > self.ub or fluent.type.lower_bound < self.lb:
                    if fluent.type.upper_bound > self.ub:
                        self.ub = fluent.type.upper_bound
                    if fluent.type.lower_bound < self.lb:
                        self.lb = fluent.type.lower_bound
        # Change the integers fluents
        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)
            if fluent.type.is_int_type():
                self._convert_int_fluent(problem, new_problem, fluent)
            else:
                # Default initial values
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                # Initial values
                for k, v in problem.initial_values.items():
                    if k.fluent().name == fluent.name and v != default_value:
                        new_problem.set_initial_value(k, v)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )

