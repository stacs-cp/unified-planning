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
import re
from itertools import product
from unified_planning.model.fnode import FNode
import unified_planning.engines as engines
from collections import OrderedDict
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import (
    Problem,
    InstantaneousAction,
    DurativeAction,
    Action,
    ProblemKind,
    Fluent,
    MinimizeActionCosts,
    RangeVariable,
    OperatorKind,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    lift_action_instance,
)
from typing import Dict, List, Optional, Tuple, OrderedDict, Union
from functools import partial

from unified_planning.shortcuts import Int, FALSE, TRUE


class IntParameterActionsRemover(engines.engine.Engine, CompilerMixin):
    """
    Arrays remover class: ...
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INT_PARAMETER_ACTIONS_REMOVING)

    @property
    def name(self):
        return "ipar"

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
        supported_kind.set_conditions_kind("COUNTING")
        supported_kind.set_conditions_kind("RANGE_VARIABLES")
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
        return problem_kind <= IntParameterActionsRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INT_PARAMETER_ACTIONS_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        new_kind.unset_conditions_kind("RANGE_VARIABLES")
        return new_kind

    def _get_fluent_name(self, problem, fluent_name, integer_parameters: dict[str, int], instantiations) -> Union[str, None]:
        pattern = r'\[(.*?)\]'
        new_name = fluent_name
        for access in re.findall(pattern, fluent_name):
            new_access = access
            for key, index in integer_parameters.items():
                if key == access or any(part.strip() == key for part in re.split(r'[\s+\-*/()]', new_access)):
                    new_access = new_access.replace(key, str(instantiations[index]))
            new_name = new_name.replace('[' + access + ']', '[' + str(eval(new_access)) + ']')
        original_fluent = problem.fluent(fluent_name.split('[')[0])
        undefined_positions = original_fluent.undefined_positions
        sizes = original_fluent.sizes
        try:
            indexes = re.findall(pattern, new_name)
            index_0 = eval(indexes[0])
            if type(sizes) == tuple:
                index_1 = eval(indexes[1])
                assert len(indexes) == 2, "Expected 2 indexes for double array."
                assert 0 <= index_0 < sizes[0], "Index 0 out of bounds."
                assert 0 <= index_1 < sizes[1], "Index 1 out of bounds."
                if undefined_positions is not None:
                    assert (index_0, index_1) not in undefined_positions, "This position is undefined."
            else:
                assert len(indexes) == 1
                assert 0 <= index_0 < sizes, "Index out of bounds."
                if undefined_positions is not None:
                    assert index_0 not in undefined_positions, "This position is undefined."
            return new_name
        except AssertionError as e:
            return None

    def _manage_node(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
            integer_parameters: Optional[dict[str, int]] = {},
            instantiations: Optional[tuple] = ()
    ) -> Union["up.model.fnode.FNode", None]:
        em = new_problem.environment.expression_manager
        if node.is_fluent_exp():
            original_fluent = new_problem.fluent(node.fluent().name.split('[')[0])
            if original_fluent.type.is_array_type():
                fluent = node.fluent()
                new_name = self._get_fluent_name(new_problem, fluent.name, integer_parameters, instantiations)
                if new_name is None:
                    return None
                return Fluent(new_name, fluent.type, fluent.signature, fluent.environment)(*node.args)
            return node
        elif node.is_parameter_exp():
            if integer_parameters.get(node.parameter().name) is not None:
                return Int(instantiations[integer_parameters.get(node.parameter().name)])
            return node
        elif node.is_constant() or node.is_variable_exp() or node.is_timing_exp():
            return node
        else:
            if node.is_forall() or node.is_exists():
                new_forall, range_vars = self._process_forall(node.variables())
                if not new_forall:
                    this_integer_parameters = integer_parameters.copy()
                    for key in range_vars.keys():
                        this_integer_parameters[key] = len(this_integer_parameters)

                    updated_range_vars = self._update_range_vars(range_vars, this_integer_parameters,
                                                                 instantiations) if range_vars else {}
                    ranges = [updated_range_vars[n] for n in updated_range_vars] if updated_range_vars else []
                    instantiation_combinations = self._get_instantiations(ranges) if ranges else [()]
                    new_args = []
                    for ti in instantiation_combinations:
                        this_instantiations = instantiations + ti
                        for arg in node.args:
                            new_args.append(self._manage_node(new_problem, arg, this_integer_parameters, this_instantiations))
                    new_node_type = OperatorKind.AND if node.is_forall() else OperatorKind.OR
                    if None in new_args:
                        new_args = self._is_problematic(new_node_type, new_args)
                    if new_args is None:
                        return None
                    return em.create_node(new_node_type, tuple(new_args)).simplify()
                else:
                    new_args = [self._manage_node(new_problem, arg, integer_parameters, instantiations) for arg in
                                node.args]
                    if None in new_args:
                        new_args = self._is_problematic(node.node_type, new_args)
                    if new_args is None or new_args is []:
                        return None
                    return em.create_node(node.node_type, tuple(new_args), new_forall).simplify()
            new_args = [self._manage_node(new_problem, arg, integer_parameters, instantiations) for arg in node.args]
            if None in new_args:
                new_args = self._is_problematic(node.node_type, new_args)
            if new_args is None or new_args is []:
                return None
            return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _is_problematic(self, node_type, args) -> Union[list[FNode], None]:
        if node_type in {OperatorKind.OR, OperatorKind.COUNT, OperatorKind.EXISTS}:
            return [arg for arg in args if arg is not None]
        elif node_type in {OperatorKind.IMPLIES}:
            return [args[0], FALSE()] if (args[1] is None and args[0] is not None) else args[1]
        # LE, LT, NOT, EQUALS, FORALL, IFF, AND, ARITHMETIC OPERATIONS ...
        else:
            return None if None in args else args

    def _process_forall(self, forall):
        new_forall = []
        range_info = {}
        for f in forall:
            if isinstance(f, RangeVariable):
                range_info[f.name] = (f.initial, f.last)
            else:
                new_forall.append(f)
        return tuple(new_forall), range_info

    def _update_range_vars(self, range_vars, int_parameters: dict[str, int], instantiations) -> dict[str, Tuple[int, int]]:
        for n, r in range_vars.items():
            new_r_0 = str(r[0])
            new_r_1 = str(r[1])
            for key, index in int_parameters.items():
                if key in new_r_0:
                    new_r_0 = str(new_r_0).replace(key, str(instantiations[index]))
                if key in new_r_1:
                    new_r_1 = str(new_r_1).replace(key, str(instantiations[index]))
            range_vars[n] = (eval(new_r_0), eval(new_r_1))
        return range_vars

    def _get_instantiations(self, ranges: list[(int, int)]):
        """
        Generates all possible combinations given a list of ranges.

        :param ranges: A list of tuples (start, end), where each tuple defines an inclusive range.
        :return: A list of tuples with all the possible combinations.
        """
        ranges_as_iterables = [range(start, end + 1) for start, end in ranges]
        return list(product(*ranges_as_iterables))

    def _add_effect(self, action, effect_type: str, fluent, value, condition, forall):
        """
        Adds an effect to the specified action.

        :param action: The action to which the effect will be added.
        :param effect_type: A string indicating the type of the effect (e.g., "add", "delete", or "modify").
        :param fluent: The fluent being affected by the effect.
        :param value: The value to be assigned to the fluent as part of the effect.
        :param condition: The condition under which the effect is applied. This can be a logical condition or a fluent expression.
        :param forall: A list or iterable representing the variables that must satisfy the effect's condition (used for universally quantified effects).
        :return: None. The effect is directly added to the action.
        """
        if effect_type == 'increase':
            action.add_increase_effect(fluent, value, condition, forall)
        elif effect_type == 'decrease':
            action.add_decrease_effect(fluent, value, condition, forall)
        else:
            action.add_effect(fluent, value, condition, forall)

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """
        Takes an instance of a :class:`~unified_planning.model.Problem` and the `INT_PARAMETER_ACTIONS_REMOVING` `~unified_planning.engines.CompilationKind`
        and returns a `CompilerResult` where the `Problem` does not have `Actions` with integer parameters.

        :param problem: The instance of the `Problem` that must be returned without integer parameters.
        :param compilation_kind: The `CompilationKind` that must be applied on the given problem;
            only `INT_PARAMETER_ACTIONS_REMOVING` is supported by this compiler
        :return: The resulting `CompilerResult` data structure.
        """
        assert isinstance(problem, Problem)

        trace_back_map: Dict[Action, Tuple[Action, List["up.model.fnode.FNode"]]] = {}

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_actions()
        new_problem.clear_axioms()
        new_problem.clear_quality_metrics()

        for action in problem.actions:
            new_parameters = OrderedDict()
            integer_parameters = {}
            integers_domains = []
            for old_parameter in action.parameters:
                if old_parameter.type.is_user_type():
                    new_parameters.update({old_parameter.name: old_parameter.type})
                else:
                    assert old_parameter.type.is_int_type(), "Type of parameter not supported"
                    integers_domains.append((old_parameter.type.lower_bound, old_parameter.type.upper_bound))
                    integer_parameters[old_parameter.name] = len(integer_parameters)

            for instantiations in self._get_instantiations(integers_domains):
                new_action_name = get_fresh_name(new_problem, action.name, list(map(str, instantiations)))
                if isinstance(action, InstantaneousAction):
                    new_action = InstantaneousAction(new_action_name, new_parameters, action.environment)
                elif isinstance(action, DurativeAction):
                    new_action = DurativeAction(new_action_name, new_parameters, action.environment)
                else:
                    new_action = Action(new_action_name, new_parameters, action.environment)
                remove_action = False
                for precondition in action.preconditions:
                    new_precondition = self._manage_node(new_problem, precondition, integer_parameters, instantiations)
                    if new_precondition is None or new_precondition == FALSE():
                        remove_action = True
                        break
                    new_action.add_precondition(new_precondition)
                if not remove_action:
                    for effect in action.effects:
                        effect_type = 'increase' if effect.is_increase() else 'decrease' if effect.is_decrease() else 'none'
                        new_forall, range_vars = self._process_forall(effect.forall)
                        this_integer_parameters = integer_parameters.copy()
                        for key in range_vars.keys():
                            this_integer_parameters[key] = len(this_integer_parameters)

                        updated_range_vars = self._update_range_vars(range_vars, this_integer_parameters,
                                                                     instantiations) if range_vars else {}
                        ranges = [updated_range_vars[n] for n in updated_range_vars] if updated_range_vars else []
                        instantiation_combinations = self._get_instantiations(ranges) if ranges else [()]

                        for ti in instantiation_combinations:
                            this_instantiations = instantiations + ti
                            new_fluent = self._manage_node(new_problem, effect.fluent, this_integer_parameters, this_instantiations)
                            new_value = self._manage_node(new_problem, effect.value, this_integer_parameters, this_instantiations)
                            new_condition = self._manage_node(new_problem, effect.condition, this_integer_parameters, this_instantiations)

                            # Invalid value if it is out of range
                            if new_fluent.type.is_int_type():
                                t = new_fluent.type
                                if (new_value < t.lower_bound).simplify() == TRUE() or (new_value > t.upper_bound).simplify() == TRUE():
                                    new_value = None
                            # Unconditional effect
                            if effect.condition == TRUE():
                                if None in (new_fluent, new_value):
                                    remove_action = True
                                    break
                                else:
                                    self._add_effect(new_action, effect_type, new_fluent, new_value, new_condition, new_forall)
                            # Conditional effect
                            else:
                                if new_condition not in (None, FALSE()) and None not in (new_fluent, new_value):
                                    self._add_effect(new_action, effect_type, new_fluent, new_value, new_condition, new_forall)
                        if remove_action:
                            break
                    if not remove_action and new_action.effects != []:
                        new_problem.add_action(new_action)
                        trace_back_map[new_action] = (action, instantiations)

        for axiom in problem.axioms:
            new_axiom = axiom.clone()
            new_axiom.name = get_fresh_name(new_problem, new_axiom.name)
            new_axiom.clear_preconditions()
            new_axiom.clear_effects()
            for parameter in axiom.parameters:
                if parameter.type.is_int_type():
                    raise NotImplementedError(
                        "Integer parameters in axioms are not supported!"
                    )
            for precondition in axiom.preconditions:
                new_precondition = self._manage_node(new_problem, precondition)
                new_axiom.add_precondition(new_precondition)
            for effect in axiom.effects:
                new_fluent = self._manage_node(new_problem, effect.fluent)
                new_value = self._manage_node(new_problem, effect.value)
                new_condition = self._manage_node(new_problem, effect.condition)
                if not new_condition.is_false() and new_fluent is not None:
                    if effect.is_increase():
                        new_axiom.add_increase_effect(new_fluent, new_value, new_condition, effect.forall)
                    elif effect.is_decrease():
                        new_axiom.add_decrease_effect(new_fluent, new_value, new_condition, effect.forall)
                    else:
                        new_axiom.add_effect(new_fluent, new_value, new_condition, effect.forall)

            new_problem.add_axiom(new_axiom)
            trace_back_map[new_axiom] = axiom

        for qm in problem.quality_metrics:
            if qm.is_minimize_sequential_plan_length() or qm.is_minimize_makespan():
                new_problem.add_quality_metric(qm)
            elif qm.is_minimize_action_costs():
                assert isinstance(qm, MinimizeActionCosts)
                new_costs: Dict["up.model.Action", "up.model.Expression"] = {}
                for new_act, old_act in trace_back_map.items():
                    if old_act is None:
                        continue
                    new_cost = qm.get_action_cost(old_act[0])
                    if new_cost.is_parameter_exp():
                        i = 0
                        for p in old_act[0].parameters:
                            if p.name == str(new_cost):
                                break
                            if p.type.is_int_type():
                                i += 1
                        new_costs[new_act] = old_act[1][i]
                    else:
                        new_costs[new_act] = new_cost
                new_problem.add_quality_metric(
                    MinimizeActionCosts(new_costs, environment=new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(qm)

        return CompilerResult(
            new_problem,
            partial(lift_action_instance, map=trace_back_map),
            self.name,
        )
