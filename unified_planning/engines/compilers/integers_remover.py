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
from unified_planning.exceptions import UPProblemDefinitionError, UPValueError, UPConflictingEffectsException
from unified_planning.model import (
    Problem,
    Action,
    ProblemKind, Variable, MinimizeActionCosts, Effect, EffectKind,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action, updated_minimize_action_costs,
)
from typing import Dict, Optional, OrderedDict, Set, Iterator
from functools import partial
from unified_planning.shortcuts import Exists, And, Or, Equals

class IntegersRemover(engines.engine.Engine, CompilerMixin):
    """
    Integers remover class: ...
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGERS_REMOVING)
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
        #supported_kind.set_parameters("UNBOUNDED_INT_ACTION_PARAMETERS")
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
        new_kind.unset_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        return new_kind

    def _process_two_args_operation(self, first_arg, second_arg, target_value, operation, sub_operation, old_problem, new_problem):
        if operation == 'equals':
            return new_problem.fluent(sub_operation)(first_arg, second_arg, target_value)
        else:
            self._add_relationships(new_problem, 'lt')
            intermediate_var = Variable('value', new_problem.environment.type_manager.UserType('Number'))
            new_inner_expr = new_problem.fluent(sub_operation)(first_arg, second_arg, intermediate_var)
            new_lt_expr = new_problem.fluent('lt')(intermediate_var, target_value)

            if operation == 'lt':
                return Exists(And(new_inner_expr, new_lt_expr), intermediate_var)
            else:
                new_eq_expr = Equals(intermediate_var, target_value)
                return Exists(And(new_inner_expr, Or(new_lt_expr, new_eq_expr)), intermediate_var)

    def _process_multi_args_operation(self, inner_expr, first_arg, target_value, operation, sub_operation, old_problem, new_problem):
        and_node = []
        intermediate_vars = []
        n = 1

        for arg in inner_expr.args[1:-1]:
            next_arg = self._get_new_node(old_problem, new_problem, arg)
            intermediate_var = Variable(f'{sub_operation}_{n}', new_problem.environment.type_manager.UserType('Number'))
            intermediate_vars.append(intermediate_var)
            and_node.append(new_problem.fluent(sub_operation)(first_arg, next_arg, intermediate_var))
            first_arg = intermediate_var
            n += 1

        last_arg = self._get_new_node(old_problem, new_problem, inner_expr.arg(-1))

        if operation == 'equals':
            and_node.append(new_problem.fluent(sub_operation)(first_arg, last_arg, target_value))
            return Exists(And(and_node), *intermediate_vars)
        else:
            self._add_relationships(new_problem, 'lt')
            value_var = Variable('value', new_problem.environment.type_manager.UserType('Number'))
            and_node.append(new_problem.fluent(sub_operation)(first_arg, last_arg, value_var))
            new_lt_expr = new_problem.fluent('lt')(value_var, target_value)
            if operation == 'lt':
                return Exists(And(and_node, new_lt_expr), *(intermediate_vars + [value_var]))
            else:
                new_eq_expr = Equals(value_var, target_value)
                return Exists(And(and_node, Or(new_lt_expr, new_eq_expr)), *(intermediate_vars + [value_var]))

    def _handle_equals_lt_le(self, node, operation, operation_inner_map, old_problem, new_problem):
        inner_expr, value, sub_operation = None, None, None
        for i in range(2):
            sub_operation = operation_inner_map.get(node.arg(i).node_type)
            if sub_operation is not None:
                inner_expr, value = (node.arg(i), node.arg(1 - i))
                break

        if sub_operation is not None:
            target_value = self._get_new_node(old_problem, new_problem, value)
            first_arg = self._get_new_node(old_problem, new_problem, inner_expr.arg(0))
            if sub_operation == 'plus':
                max_value = 0
                for a in inner_expr.args:
                    max_value += a.type.upper_bound
                self._add_object_numbers(new_problem, 0, max_value)
            else:
                print(f"Operation {sub_operation} not supported yet!")
            self._add_relationships(new_problem, sub_operation)

            if len(inner_expr.args) == 2:
                second_arg = self._get_new_node(old_problem, new_problem, inner_expr.arg(1))
                return self._process_two_args_operation(first_arg, second_arg, target_value, operation, sub_operation,
                                                        old_problem, new_problem)
            else:
                return self._process_multi_args_operation(inner_expr, first_arg, target_value, operation, sub_operation,
                                                          old_problem, new_problem)

        # Operacions sense sub-operacions internes
        new_args = [self._get_new_node(old_problem, new_problem, arg) for arg in node.args]
        if operation == 'equals':
            return new_problem.environment.expression_manager.create_node(node.node_type, tuple(new_args))
        else:
            self._add_relationships(new_problem, 'lt')
            lt_expr = new_problem.fluent('lt')(*new_args)
            return Or(Equals(*new_args), lt_expr) if operation == 'le' else lt_expr

    def _get_new_node(
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
            object_name = f'n{node.int_constant_value()}'
            new_number = model.Object(object_name, number_user_type)
            if not new_problem.has_object(object_name):
                self._add_object_numbers(new_problem, node.int_constant_value(), node.int_constant_value())
            return em.ObjectExp(new_number)
        elif node.is_fluent_exp() and node.fluent().type.is_int_type():
            return new_problem.fluent(node.fluent().name)(*node.args)
        elif node.is_object_exp() or node.is_fluent_exp() or node.is_constant() or node.is_parameter_exp():
            return node
        else:
            operation_map = {
                OperatorKind.EQUALS: 'equals',
                OperatorKind.LT: 'lt',
                OperatorKind.LE: 'le',
            }
            operation_inner_map = {
                OperatorKind.PLUS: 'plus',
                OperatorKind.MINUS: 'minus',
                OperatorKind.DIV: 'div',
                OperatorKind.TIMES: 'mult',
            }
            operation = operation_map.get(node.node_type)

            if operation is not None:
                return self._handle_equals_lt_le(node, operation, operation_inner_map, old_problem, new_problem)

            # Plus/Minus/Div/Mult in an outer expression
            operation_inner = operation_inner_map.get(node.node_type)
            if operation_inner is not None:
                raise UPProblemDefinitionError(
                    f"Operation {operation_inner} not supported as an external expression!")

            # Other nodes
            new_args = [self._get_new_node(old_problem, new_problem, arg) for arg in node.args]
            if node.is_exists() or node.is_forall():
                new_variables = [
                    model.Variable(v.name, tm.UserType('Number')) if v.type.is_int_type() else v
                    for v in node.variables()
                ]
                return em.create_node(node.node_type, tuple(new_args), payload=tuple(new_variables))
            return em.create_node(node.node_type, tuple(new_args))

    def _convert_effect(
            self,
            effect: "up.model.Effect",
            old_problem: "up.model.AbstractProblem",
            new_problem: "up.model.AbstractProblem",
    ) -> Iterator[Effect]:
        em = new_problem.environment.expression_manager
        returned_effects: Set[Effect] = set()

        lower = effect.fluent.fluent().type.lower_bound
        upper = effect.fluent.fluent().type.upper_bound

        new_condition = self._get_new_node(old_problem, new_problem, effect.condition)
        for i in range(lower, upper + 1):
            next_value = (i + effect.value if effect.is_increase() else i - effect.value).simplify()
            try:
                old_obj_num = em.ObjectExp(new_problem.object(f'n{i}'))
                new_obj_num = em.ObjectExp(new_problem.object(f'n{next_value}'))
                new_fluent = new_problem.fluent(effect.fluent.fluent().name)(*effect.fluent.args)

                new_effect = Effect(
                    new_fluent,
                    new_obj_num,
                    And(Equals(new_fluent, old_obj_num), new_condition).simplify(),
                    EffectKind.ASSIGN,
                    effect.forall,
                )
                if new_effect not in returned_effects:
                    yield new_effect
                    returned_effects.add(new_effect)
            except UPValueError:
                continue


    def _add_object_numbers(
            self,
            new_problem: "up.model.AbstractProblem",
            lower_bound: Optional[int] = None,
            upper_bound: Optional[int] = None,
    ):
        ut_number = new_problem.environment.type_manager.UserType('Number')
        if self.lb is None and self.ub is None:
            for i in range(lower_bound, upper_bound + 1):
                new_problem.add_object(model.Object(f'n{i}', ut_number))
            self.lb = lower_bound
            self.ub = upper_bound
        else:
            if upper_bound > self.ub:
                for i in range(self.ub + 1, upper_bound + 1):
                    new_problem.add_object(model.Object(f'n{i}', ut_number))
                self.ub = upper_bound
            if lower_bound < self.lb:
                for i in range(lower_bound, self.lb):
                    new_problem.add_object(model.Object(f'n{i}', ut_number))
                self.lb = lower_bound

    def _add_relationships(
            self,
            new_problem: "up.model.AbstractProblem",
            relationship: str,
    ):
        ut_number = new_problem.user_type('Number')
        params = OrderedDict({'n1': ut_number, 'n2': ut_number})
        if not new_problem.has_fluent(relationship):
            if relationship == 'lt':
                relationship_fluent = model.Fluent(relationship, _signature=params, environment=new_problem.environment)
                new_problem.add_fluent(relationship_fluent, default_initial_value=False)
            elif relationship == 'plus' or relationship == 'minus' or relationship == 'div' or relationship == 'mult':
                params['n3'] = ut_number
                relationship_fluent = model.Fluent(relationship, _signature=params, environment=new_problem.environment)
                new_problem.add_fluent(relationship_fluent, default_initial_value=False)
            else:
                raise NotImplementedError

            for i in range(self.lb, self.ub + 1):
                ni = new_problem.object(f'n{i}')
                for j in range(i, self.ub + 1):
                    nj = new_problem.object(f'n{j}')
                    if relationship == 'lt':
                        if i < j:
                            new_problem.set_initial_value(relationship_fluent(ni, nj), True)
                        if j < i:
                            new_problem.set_initial_value(relationship_fluent(nj, ni), True)
                    else:
                        operation_map = {
                            'plus': lambda a, b: a + b,
                            'minus': lambda a, b: a - b,
                            'div': lambda a, b: a / b if b > 0 else None,
                            'mult': lambda a, b: a * b
                        }
                        if relationship in operation_map:
                            for a, b in [(i, j), (j, i)]:
                                result = operation_map[relationship](a, b)
                                if result is not None:
                                    try:
                                        result_obj = new_problem.object(f'n{result}')
                                        new_problem.set_initial_value(
                                            relationship_fluent(new_problem.object(f'n{a}'), new_problem.object(f'n{b}'),
                                                                result_obj), True)
                                    except UPValueError:
                                        continue

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
        new_problem.clear_quality_metrics()
        env = new_problem.environment
        tm = env.type_manager
        ut_number = tm.UserType('Number')
        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)
            new_signature = []
            for s in fluent.signature:
                if s.type.is_int_type():
                    assert s.type.lower_bound is not None and s.type.upper_bound, f"Integer {s} not bounded"
                    self._add_object_numbers(new_problem, s.type.lower_bound, s.type.upper_bound)
                    new_signature.append(up.model.Parameter(s.name, ut_number))
                else:
                    new_signature.append(s)
            if fluent.type.is_int_type():
                assert fluent.type.lower_bound is not None and fluent.type.upper_bound, f"Integer {fluent} not bounded"
                new_fluent = model.Fluent(fluent.name, ut_number, new_signature, env)
                self._add_object_numbers(new_problem, fluent.type.lower_bound, fluent.type.upper_bound)
                if default_value is not None:
                    new_problem.add_fluent(new_fluent,
                                           default_initial_value=new_problem.object(f'n{default_value}'))
                else:
                    new_problem.add_fluent(new_fluent)
                for k, v in problem.initial_values.items():
                    if k.type.is_int_type() and k.fluent().name == fluent.name and v != default_value:
                        new_problem.set_initial_value(new_problem.fluent(k.fluent().name)(*k.args),
                                                      new_problem.object(f'n{v}'))
            else:
                new_fluent = model.Fluent(fluent.name, fluent.type, new_signature, env)
                new_problem.add_fluent(new_fluent, default_initial_value=default_value)
                for k, v in problem.initial_values.items():
                    if k.fluent().name == fluent.name and v != default_value:
                        new_problem.set_initial_value(k, v)

        for action in problem.actions:
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()
            new_action.clear_effects()
            for precondition in action.preconditions:
                new_precondition = self._get_new_node(problem, new_problem, precondition)
                new_action.add_precondition(new_precondition)
            for effect in action.effects:
                if effect.is_increase() or effect.is_decrease():
                    for ne in self._convert_effect(effect, problem, new_problem):
                        new_action.add_effect(ne.fluent, ne.value, ne.condition, ne.forall)
                        #new_action._add_effect_instance(ne)
                        # new_action.add_effect(ne.fluent, ne.value, ne.condition)
                else:
                    new_node = self._get_new_node(problem, new_problem, effect.fluent)
                    new_value = self._get_new_node(problem, new_problem, effect.value)
                    new_condition = self._get_new_node(problem, new_problem, effect.condition)
                    new_action.add_effect(new_node, new_value, new_condition, effect.forall)

            new_to_old[new_action] = action
            new_problem.add_action(new_action)

        for goal in problem.goals:
            new_problem.add_goal(self._get_new_node(problem, new_problem, goal))

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