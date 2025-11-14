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
from unified_planning.model import (
    Problem,
    Action,
    ProblemKind, Object, MinimizeActionCosts,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action,
)
from typing import Dict, List, Optional, OrderedDict, Iterable, Tuple, Iterator
from functools import partial
from unified_planning.shortcuts import And, Not, Iff, Equals, Or
import re


class ArraysLogarithmicRemover(engines.engine.Engine, CompilerMixin):
    """
    Arrays Logarithmic Removerr class:
    The problem has to contain arrays/multiarrays of integers. They will be transformed into a bit boolean format to
    represent the numbers.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.ARRAYS_LOGARITHMIC_REMOVING)
        self.n_bits = OrderedDict()

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
        # supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_parameters("REAL_ACTION_PARAMETERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("ARRAY_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_fluents_type("DERIVED_FLUENTS")
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
        return problem_kind <= ArraysLogarithmicRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.ARRAYS_LOGARITHMIC_REMOVING

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("INT_FLUENTS")
        new_kind.unset_conditions_kind("ARRAY_FLUENTS")
        return new_kind

    def _get_new_fluent(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
            indexes: Optional["up.model.fnode.FNode"] = None,
    ) -> List["up.model.fnode.FNode"]:
        assert node.is_fluent_exp()

        name_fluent = node.fluent().name.split('[')[0]
        n_bits = self.n_bits[name_fluent]

        if node.fluent().type.is_int_type():
            indexes = [int(i) for i in re.findall(r'\[([0-9]+)]', node.fluent().name)]
        if not indexes:
            return [
                new_problem.fluent(f"{name_fluent}_{i}")(*node.args)
                for i in range(n_bits)
            ]
        else:
            position_object_fluent = new_problem.object(f'p_{"_".join(map(str, indexes))}')
            return [
                new_problem.fluent(f"{name_fluent}_{i}")(*node.args, position_object_fluent)
                for i in range(n_bits)
            ]

    def _get_fluent_domain(
            self,
            fluent: "up.model.Fluent",
            save: bool = False
    ) -> Iterable[int]:
        domain = []
        inner_fluent = fluent.type

        while inner_fluent.is_array_type():
            domain.append(range(inner_fluent.size))
            inner_fluent = inner_fluent.elements_type

        assert inner_fluent.is_int_type(), f"Fluent {fluent.name} has not type int. Only integer arrays supported."

        #  save number of bits of the fluent
        if save:
            self.n_bits[fluent.name] = math.ceil(math.log2(inner_fluent.upper_bound + 1))

        return tuple(product(*domain))

    def _get_element_value(self, v, combination) -> "up.model.fnode.FNode":
        """Obtain the value of the element for a given combination of access."""
        element_value = v
        for c in combination:
            element_value = element_value.constant_value()[c]
        return element_value

    def _get_new_expression(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode"
    ) -> up.model.fnode.FNode:
        operation_map = {
            OperatorKind.EQUALS: 'equals',  # The only operator that works within arrays
            OperatorKind.LT: 'lt',
            OperatorKind.LE: 'le',
            #OperatorKind.PLUS: 'plus',
            #OperatorKind.MINUS: 'minus',
            #OperatorKind.DIV: 'div',
            #OperatorKind.TIMES: 'mult',
        }
        operation = operation_map.get(node.node_type)
        # de mom suporta equals i lt
        if operation is not None:
            fluent = node.arg(0)
            value = node.arg(1)

            if fluent.type.is_int_type():
                new_fluents, new_values = self._convert_fluent_and_value(new_problem, fluent, value)
                if operation == 'equals':
                    and_node = []
                    for f, v in zip(new_fluents, new_values):
                        if value.is_fluent_exp():
                            and_node.append(Iff(f, v))
                        else:
                            and_node.append(f if v else Not(f))
                    return And(and_node)

                elif operation == 'lt':
                    if value.is_fluent_exp():
                        or_node = []
                        iff_node = []
                        for f, v in zip(new_fluents, new_values):
                            new_and_node = And(Not(f), v)
                            or_node.append(And(iff_node, new_and_node))
                            iff_node.append(Iff(f, v))
                        return Or(or_node)
                    else:
                        # es podria millorar ?
                        or_node = []
                        for i in range(value.constant_value()-1):
                            lower_value_bits = self._convert_value(i, len(new_values))
                            and_node = []
                            for f, v in zip(new_fluents, lower_value_bits):
                                and_node.append(f if v else Not(f))
                            or_node.append(And(and_node))
                        return Or(or_node)

                elif operation == 'le':
                    and_node = []
                    for f, v in zip(new_fluents, new_values):
                        if value.is_fluent_exp():
                            and_node.append(Iff(f, v))
                        else:
                            and_node.append(f if v else Not(f))
                    equals_node = And(and_node)

                    if value.is_fluent_exp():
                        or_node = []
                        iff_node = []
                        for f, v in zip(new_fluents, new_values):
                            new_and_node = And(Not(f), v)
                            or_node.append(And(iff_node, new_and_node))
                            iff_node.append(Iff(f, v))
                        lt_node = Or(or_node)

                    else:
                        or_node = []
                        for i in range(value.constant_value()-1):
                            lower_value_bits = self._convert_value(i, len(new_values))
                            and_node = []
                            for f, v in zip(new_fluents, lower_value_bits):
                                and_node.append(f if v else Not(f))
                            or_node.append(And(and_node))
                        lt_node = Or(or_node)

                    return Or(equals_node, lt_node)
                elif operation == 'plus' or operation == 'minus' or operation == 'div' or operation == 'mult':
                    raise NotImplementedError(f"Operation {operation} not supported")
                #elif operation == 'minus':
                #elif operation == 'div':
                #elif operation == 'mult':

            elif fluent.type.is_array_type() and operation == 'equals':
                new_fluents, new_values = self._convert_fluent_and_value(new_problem, fluent, value)
                and_node = []
                for i in range(len(new_fluents)):
                    for f, v in zip(new_fluents[i], new_values[i]):
                        if value.is_fluent_exp():
                            and_node.append(Iff(f, v))
                        else:
                            and_node.append(f if v else Not(f))
                return And(and_node)
        return node

    def _convert_value(self, value, n_bits):
        """Convert integer value to binary list of n_bits."""
        return [b == '1' for b in bin(value)[2:].zfill(n_bits)]

    def _set_fluent_bits(self, problem, fluent, k_args, new_value, n_bits, object_ref: Optional[Object] = None):
        for bit_index in range(n_bits):
            this_fluent = problem.fluent(f"{fluent.name}_{bit_index}")(*k_args, *(object_ref,) if object_ref is not None else ())
            problem.set_initial_value(this_fluent, new_value[bit_index])

    def _convert_fluent_and_value(
            self,
            new_problem: "up.model.AbstractProblem",
            fluent: "up.model.fnode.FNode",
            value: "up.model.fnode.FNode",
    ) -> Tuple[List["up.model.fnode.FNode"], List["up.model.fnode.FNode"]]:
        """Convert fluent and value to bits."""
        name_fluent = fluent.fluent().name.split('[')[0]
        n_bits = self.n_bits[name_fluent]
        if fluent.type.is_int_type():
            new_fluents = self._get_new_fluent(new_problem, fluent)
            if value.is_fluent_exp():
                new_values = self._get_new_fluent(new_problem, value)
            else:
                assert value.is_constant(), "Value must be a constant!"
                new_values = self._convert_value(value.constant_value(), n_bits)
        else:
            assert fluent.type.is_array_type()
            indices = tuple(int(i) for i in re.findall(r'\[([0-9]+)]', fluent.fluent().name))
            new_fluents = []
            new_values = []
            if value.is_fluent_exp():
                value_fluent_domain = tuple(int(i) for i in re.findall(r'\[([0-9]+)]', value.fluent().name))
                for combination in self._get_fluent_domain(fluent.fluent()):
                    new_fluents.append(self._get_new_fluent(new_problem, fluent, indices + combination))
                    new_values.append(self._get_new_fluent(new_problem, value, value_fluent_domain + combination))
            else:
                for combination in self._get_fluent_domain(fluent.fluent()):
                    new_fluents.append(self._get_new_fluent(new_problem, fluent, indices + combination))
                    element_value = self._get_element_value(value, combination)
                    new_values.append(self._convert_value(element_value.constant_value(), n_bits))
        return new_fluents, new_values

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
        new_problem.clear_quality_metrics()
        new_problem.initial_values.clear()

        for fluent in problem.fluents:
            # Change the integer array fluents
            if fluent.type.is_array_type():
                Position = new_problem.environment.type_manager.UserType('Position')
                #new_user_type = new_problem.environment.type_manager.UserType(fluent.name.capitalize(), Position)
                new_parameter = up.model.Parameter('p', Position)

                combination = self._get_fluent_domain(fluent, True)
                print(combination)
                for c in combination:
                    new_object = model.Object(f'p_{"_".join(map(str, c))}', Position)
                    if not new_problem.has_object(new_object.name):
                        new_problem.add_object(new_object)
                n_bits = self.n_bits[fluent.name]
                # Default initial values
                default_value = problem.fluents_defaults.get(fluent)
                default_bits = self._convert_value(default_value.constant_value(), n_bits) if default_value else [None] * n_bits
                for i in range(n_bits):
                    new_fluent = model.Fluent(f"{fluent.name}_{i}", _signature=fluent.signature + [new_parameter],
                                              environment=new_problem.environment)
                    new_problem.add_fluent(new_fluent, default_initial_value=default_bits[i])

                # Initial values
                for k, v in problem.explicit_initial_values.items():
                    fluent_name = k.fluent().name.split('[')[0]
                    # For entire arrays (e.g., puzzle = [[8,7,6], [0,4,1], [2,5,3]])
                    if k.fluent() == fluent:
                        for positions in combination:
                            element_value = self._get_element_value(v, positions)
                            new_value = self._convert_value(element_value.constant_value(), n_bits)
                            object_ref = new_problem.object(f'p_{"_".join(map(str, positions))}')
                            self._set_fluent_bits(new_problem, fluent, k.args, new_value, n_bits, object_ref)

                    # For sub-arrays (e.g., puzzle[0] = [8,7,6]) or specific elements (e.g., puzzle[0][0] = 8)
                    elif fluent_name == fluent.name:
                        explicit_domain = tuple(int(i) for i in re.findall(r'\[([0-9]+)]', k.fluent().name))
                        domain = self._get_fluent_domain(k.fluent()) if k.fluent().type.is_array_type() else [()]

                        for c in domain:
                            element_value = self._get_element_value(v, c)
                            new_value = self._convert_value(element_value.constant_value(), n_bits)

                            if new_value != default_bits:
                                combined_domain = explicit_domain + c if c else explicit_domain
                                object_ref = new_problem.object(f'p_{"_".join(map(str, combined_domain))}')
                                self._set_fluent_bits(new_problem, fluent, k.args, new_value, n_bits, object_ref)
            elif fluent.type.is_int_type():
                self._get_fluent_domain(fluent, True)
                n_bits = self.n_bits[fluent.name]
                # Default initial values
                default_value = problem.fluents_defaults.get(fluent)
                default_bits = self._convert_value(default_value.constant_value(), n_bits) if default_value \
                    else [None] * n_bits
                for i in range(n_bits):
                    new_fluent = model.Fluent(f"{fluent.name}_{i}", _signature=fluent.signature,
                                              environment=new_problem.environment)
                    new_problem.add_fluent(new_fluent, default_initial_value=default_bits[i])
                # Initial values
                for k, v in problem.explicit_initial_values.items():
                    if k.fluent() == fluent:
                        new_value = self._convert_value(v.constant_value(), n_bits)
                        self._set_fluent_bits(new_problem, fluent, k.args, new_value, n_bits)
            else:
                default_value = problem.fluents_defaults.get(fluent)
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for k, v in problem.explicit_initial_values.items():
                    if k.fluent().name == fluent.name and v != default_value:
                        new_problem.set_initial_value(k, v)

        for action in problem.actions:
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()
            new_action.clear_effects()
            for precondition in action.preconditions:
                new_precondition = self._get_new_expression(new_problem, precondition)
                new_action.add_precondition(new_precondition)
            for effect in action.effects:
                fluent = effect.fluent
                value = effect.value

                new_condition = self._get_new_expression(new_problem, effect.condition)

                fluent_name = fluent.fluent().name.split('[')[0]
                if fluent_name in self.n_bits:
                    new_fluents, new_values = self._convert_fluent_and_value(new_problem, fluent, value)
                    # For specific elements (e.g., puzzle[0][0] := 8)
                    if fluent.fluent().type.is_int_type():
                        for f, v in zip(new_fluents, new_values):
                            new_action.add_effect(f, v, new_condition, effect.forall)
                    # For arrays (e.g., puzzle := [[8,7,6],[0,4,1],[2,5,3]]) or sub-arrays (e.g., puzzle[0] := [8,7,6])
                    else:
                        for i in range(len(new_fluents)):
                            for f, v in zip(new_fluents[i], new_values[i]):
                                new_action.add_effect(f, v, new_condition, effect.forall)
                else:
                    new_action.add_effect(fluent, value, new_condition, effect.forall)
            new_problem.add_action(new_action)
            new_to_old[new_action] = action

        for goal in problem.goals:
            new_problem.add_goal(self._get_new_expression(new_problem, goal))

        for qm in problem.quality_metrics:
            if qm.is_minimize_sequential_plan_length() or qm.is_minimize_makespan():
                new_problem.add_quality_metric(qm)
            elif qm.is_minimize_action_costs():
                assert isinstance(qm, MinimizeActionCosts)
                new_costs: Dict["up.model.Action", "up.model.Expression"] = {}
                for new_act, old_act in new_to_old.items():
                    if old_act is None:
                        continue
                    new_costs[new_act] = qm.get_action_cost(old_act)
                new_problem.add_quality_metric(
                    MinimizeActionCosts(new_costs, environment=new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(qm)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
