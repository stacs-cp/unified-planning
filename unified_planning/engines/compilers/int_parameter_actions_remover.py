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
import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType, Union, Iterable
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
    Effect, Fluent,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.model.walkers import ExpressionQuantifiersRemover
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action,
    updated_minimize_action_costs,
)
from typing import Dict, List, Optional, Tuple, Any, OrderedDict
from functools import partial

from unified_planning.shortcuts import Int


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
        return problem_kind <= IntParameterActionsRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INT_PARAMETER_ACTIONS_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        new_kind.unset_conditions_kind("EXISTENTIAL_CONDITIONS")
        new_kind.unset_conditions_kind("UNIVERSAL_CONDITIONS")
        new_kind.unset_effects_kind("FORALL_EFFECTS")
        if problem_kind.has_existential_conditions():
            new_kind.set_conditions("DISJUNCTIVE_CONDITIONS")
        return new_kind

    def _manage_node(
            self,
            em: "up.model.expression.ExpressionManager",
            node: "up.model.fnode.FNode",
            int_parameters: dict[str, int],
            c: Any,
            n_i: int
    ) -> Union[List["up.model.fnode.FNode"], "up.model.fnode.FNode"]:
        # mirar si es forall
        if node.is_exists():
            print(node)
            print(node.variables())

            print(int_parameters)
            print(c)
            vars_domains = []
            new_n_i = n_i
            new_variables = []
            for v in node.variables():
                # si el tipus es enter!!!!!!
                if v.type.is_int_type():
                    print(v)
                    int_parameters[v.name] = new_n_i
                    new_n_i = new_n_i + 1
                    domain = []
                    for i in range(v.type.lower_bound, v.type.upper_bound + 1):
                        domain.append(i)
                    vars_domains.append(domain)
                else:
                    new_variables.append(v)
            new_domains = list(product(*vars_domains))
            new_fnodes = []
            for i in new_domains:
                new_c = c + i
                new_args = []
                for arg in node.args:
                    # cridar manage node amb el nou c i int_domains
                    new_args.append(self._manage_node(em, arg, int_parameters, new_c, new_n_i))
                # o no crear l'exists un altre cop si es que no hi ha variables normals?
                print("new_variables", new_variables)
                if new_variables:
                    print("ex")
                    new_node = em.create_node(node.node_type, tuple(new_args), tuple(new_variables))
                else:
                    print("or")
                    new_node = em.create_node(up.model.operators.OperatorKind.OR, tuple(new_args))
                new_fnodes.append(new_node)
            print("new_fnodes: ", new_fnodes)
            return new_fnodes
        elif node.is_fluent_exp():
            print("es fluent!!!", node)
            fluent = node.fluent()
            new_name = fluent.name
            pattern = r'\[(.*?)\]'
            for ti in re.findall(pattern, new_name):
                for key in int_parameters.keys():
                    if key in ti:
                        new_ti = '[' + ti.replace(key, str(c[int_parameters.get(key)])) + ']'
                        new_name = new_name.replace('[' + ti + ']', str(eval(new_ti)))
            print("new_name fluent: ", new_name)
            return Fluent(new_name, fluent.type, fluent.signature, fluent.environment)(*fluent.signature)
        elif node.is_variable_exp():
            # si es int substituir?
            print(node.variable())
            if node.variable().type.is_int_type():
                new_int = c[int_parameters.get(node.variable().name)]
                return Int(new_int)
            else:
                return node
        elif node.is_parameter_exp():
            if int_parameters.get(node.parameter().name):
                new_int = c[int_parameters.get(node.parameter().name)]
                return Int(new_int)
            else:
                return node
        elif node.is_constant():
            return node
        else:
            new_args = []
            for arg in node.args:
                new_args.append(self._manage_node(em, arg, int_parameters, c, n_i))
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
        new_problem.clear_actions()

        for action in problem.actions:
            new_parameters = OrderedDict()
            int_parameters = {}
            int_domains = []
            n_i = 0
            for old_parameter in action.parameters:
                if old_parameter.type.is_user_type():
                    new_parameters.update({old_parameter.name: old_parameter.type})
                else:
                    assert old_parameter.type.is_int_type()
                    int_parameters[old_parameter.name] = n_i
                    n_i = n_i + 1
                    domain = []
                    for i in range(old_parameter.type.lower_bound, old_parameter.type.upper_bound + 1):
                        domain.append(i)
                    int_domains.append(domain)
            combinations = list(product(*int_domains))
            for c in combinations:
                if isinstance(action, InstantaneousAction):
                    new_action = InstantaneousAction(action.name + '_' + '_'.join(map(str, c)), new_parameters,
                                                     action.environment)
                elif isinstance(action, DurativeAction):
                    new_action = DurativeAction(action.name + '_' + '_'.join(map(str, c)), new_parameters,
                                                action.environment)
                else:
                    new_action = Action(action.name + '_' + '_'.join(map(str, c)), new_parameters,
                                        action.environment)
                for precondition in action.preconditions:
                    new_precondition = self._manage_node(em, precondition, int_parameters, c, n_i)
                    print(new_precondition)
                    print(isinstance(new_precondition, List))
                    if isinstance(new_precondition, List):
                        for p in new_precondition:
                            new_action.add_precondition(p)
                    else:
                        new_action.add_precondition(new_precondition)
                for effect in action.effects:
                    new_fnode = self._manage_node(em, effect.fluent, int_parameters, c, n_i)
                    new_value = self._manage_node(em, effect.value, int_parameters, c, n_i)
                    new_condition = self._manage_node(em, effect.condition, int_parameters, c, n_i)
                    if effect.is_increase():
                        new_action.add_increase_effect(new_fnode, new_value, new_condition, effect.forall)
                    elif effect.is_decrease():
                        new_action.add_decrease_effect(new_fnode, new_value, new_condition, effect.forall)
                    else:
                        new_action.add_effect(new_fnode, new_value, new_condition, effect.forall)
                new_problem.add_action(new_action)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
