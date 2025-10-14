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
    ProblemKind, MinimizeActionCosts, Oversubscription, BoolExpression, NumericConstant,
    MinimizeExpressionOnFinalState, MaximizeExpressionOnFinalState,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action, updated_minimize_action_costs,
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

    def _get_domain_and_type(self, fluent: "up.model.fluent.Fluent"):
        this_type = fluent.type
        domain = []
        while this_type.is_array_type():
            domain.append(range(this_type.size))
            this_type = this_type.elements_type
        positions = list(product(*domain))
        if fluent.undefined_positions is not None:
            return [p for p in positions if p not in fluent.undefined_positions], this_type
        else:
            return positions, this_type

    def _process_arg(self, new_problem, arg, combination):
        """Process an argument depending on the type."""
        if arg.is_fluent_exp():
            new_fluent = self._get_new_fluent(arg.fluent())
            new_name = new_fluent.name + ''.join(f'_{str(i)}' for i in combination)
            try:
                return new_problem.fluent(new_name)(*arg.args)
            except (KeyError, UPValueError):
                if self.mode == 'strict':
                    print(f"Fluent {new_fluent.name} out of range!")
                    exit(1)
                return FALSE() if new_fluent.type.is_bool_type() else None
        elif arg.constant_value():
            new_arg = arg
            for i in combination:
                new_arg = new_arg.constant_value()[i]
            return new_arg
        return arg

    def _get_new_nodes(
        self,
        new_problem: "up.model.AbstractProblem",
        node: "up.model.fnode.FNode",
    ) -> List["up.model.fnode.FNode"]:
        env = new_problem.environment
        em = env.expression_manager
        if node.is_fluent_exp():
            print("args fluent :", node.args)
            new_args = []
            for a in node.args:
                new_args.extend(self._get_new_nodes(new_problem, a))
            new_fluent = self._get_new_fluent(node.fluent())
            print(new_args)
            if new_problem.has_fluent(new_fluent.name):
                return [new_fluent(*new_args)]
            else:
                if self.mode == 'strict':
                    print(f"Fluent {new_fluent.name} out of range!")
                    exit(1)
                else:
                    print(f"Unexpected error - {new_fluent.name} out of range!")
                    exit(1)
        elif node.is_parameter_exp() or node.is_constant():
            return [node]
        else:
            if node.arg(0).type.is_array_type():
                assert all(arg.type.is_array_type() for arg in node.args), "Argument is not an array type"
                domain, this_type = self._get_domain_and_type(node.arg(0).fluent())
                new_nodes = []
                for combination in domain:
                    new_args = [
                        self._process_arg(new_problem, arg, combination)
                        for arg in node.args
                    ]
                    new_nodes.append(em.create_node(node.node_type, tuple(new_args)).simplify())
                return new_nodes
            else:
                new_args = [
                    nla for arg in node.args for nla in self._get_new_nodes(new_problem, arg)
                ]
                if node.is_exists() or node.is_forall():
                    return [em.create_node(node.node_type, tuple(new_args), tuple(node.variables())).simplify()]
                else:
                    return [em.create_node(node.node_type, tuple(new_args)).simplify()]


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
        new_problem.clear_axioms()
        new_problem.initial_values.clear()
        new_problem.clear_quality_metrics()
        assert self.mode == 'strict' or self.mode == 'permissive'

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent, None)

            if fluent.type.is_array_type():
                domain, this_type = self._get_domain_and_type(fluent)

                for combination in domain:
                    fluent_name = get_fresh_name(new_problem, fluent.name, list(map(str, combination)))
                    new_fluent = model.Fluent(fluent_name, this_type, fluent.signature, fluent.environment)
                    new_problem.add_fluent(new_fluent, default_initial_value=default_value)

                for f, v in problem.explicit_initial_values.items():
                    fluent_name = f.fluent().name.split('[')[0]
                    if f.fluent() == fluent:
                        domain, _ = self._get_domain_and_type(f.fluent())
                        for d in domain:
                            this_fluent = new_problem.fluent(f'{fluent_name}_{"_".join(map(str, d))}')
                            this_value = self.get_element_value(v, d)
                            new_problem.set_initial_value(this_fluent(*f.args), this_value)

                    elif fluent_name == fluent.name:
                        indices = tuple(int(i) for i in re.findall(r'\[([0-9]+)\]', f.fluent().name))
                        if not f.fluent().type.is_array_type():
                            this_fluent = new_problem.fluent(f'{fluent_name}_{"_".join(map(str, indices))}')
                            new_problem.set_initial_value(this_fluent(*f.args), v)
                        else:
                            post_domain, _ = self._get_domain_and_type(f.fluent())
                            for i in post_domain:
                                combined_domain = indices + i
                                this_fluent = new_problem.fluent(f'{fluent_name}_{"_".join(map(str, combined_domain))}')
                                new_problem.set_initial_value(this_fluent(*f.args), self.get_element_value(v, i))

            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent and v != default_value:
                        new_problem.set_initial_value(fluent(*f.args), v)

        for axiom in problem.axioms:
            new_axiom = axiom.clone()
            new_axiom.name = get_fresh_name(new_problem, new_axiom.name)
            new_axiom.clear_preconditions()
            new_axiom.clear_effects()
            for precondition in axiom.preconditions:
                new_preconditions = self._get_new_nodes(new_problem, precondition)
                for np in new_preconditions:
                    new_axiom.add_precondition(np)
            for effect in axiom.effects:
                new_fnode = self._get_new_nodes(new_problem, effect.fluent)[0]
                new_value = self._get_new_nodes(new_problem, effect.value)[0]
                new_condition = self._get_new_nodes(new_problem, effect.condition)[0]
                if not new_condition.is_false() and new_fnode is not None:
                    if effect.is_increase():
                        new_axiom.add_increase_effect(new_fnode, new_value, new_condition, effect.forall)
                    elif effect.is_decrease():
                        new_axiom.add_decrease_effect(new_fnode, new_value, new_condition, effect.forall)
                    else:
                        new_axiom.add_effect(new_fnode, new_value, new_condition, effect.forall)

            new_problem.add_axiom(new_axiom)
            new_to_old[new_axiom] = axiom

        for action in problem.actions:
            for p in action.parameters:
                assert not p.type.is_int_type(), \
                    f"Integer parameter '{p.name}' in action '{action.name}' must be removed before processing."
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()
            new_action.clear_effects()

            for precondition in action.preconditions:
                new_preconditions = self._get_new_nodes(new_problem, precondition)
                if FALSE() in new_preconditions:
                    print(f"Action {action.name} removed as it will never be applied.")
                    break
                for np in new_preconditions:
                    new_action.add_precondition(np)
            else:
                for effect in action.effects:
                    new_fnode = self._get_new_nodes(new_problem, effect.fluent)[0]
                    new_value = self._get_new_nodes(new_problem, effect.value)[0]
                    new_condition = self._get_new_nodes(new_problem, effect.condition)[0]
                    if not new_condition.is_false() and new_fnode is not None:
                        if effect.is_increase():
                            new_action.add_increase_effect(new_fnode, new_value, new_condition, effect.forall)
                        elif effect.is_decrease():
                            new_action.add_decrease_effect(new_fnode, new_value, new_condition, effect.forall)
                        else:
                            new_action.add_effect(new_fnode, new_value, new_condition, effect.forall)
                    else:
                        print(f"Effect {effect} not added.")
                        continue
                else:
                    new_problem.add_action(new_action)
                    new_to_old[new_action] = action

        for g in problem.goals:
            new_goals = self._get_new_nodes(new_problem, g)
            for ng in new_goals:
                new_problem.add_goal(ng)

        for qm in problem.quality_metrics:
            if qm.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(
                        qm, new_to_old, new_problem.environment
                    )
                )
            # ...
            else:
                new_problem.add_quality_metric(qm)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
