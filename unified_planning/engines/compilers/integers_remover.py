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
import operator
from itertools import product

from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind

import unified_planning as up
import unified_planning.engines as engines
import bisect
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
from unified_planning.shortcuts import Exists, And, Or, Equals, Int, Plus, Not, LT, FALSE, LE, GT, GE


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

    operation_inner_map = {
        OperatorKind.PLUS: 'plus',
        OperatorKind.MINUS: 'minus',
        OperatorKind.DIV: 'div',
        OperatorKind.TIMES: 'mult',
    }

    operation_outer_map = {
        OperatorKind.OR: 'or',
        OperatorKind.AND: 'and',
    }

    def _get_bounds(self, expression):
        if expression.is_fluent_exp() and expression.fluent().type.is_int_type():
            fluent = expression.fluent().type
            return list(range(fluent.lower_bound, fluent.upper_bound + 1))
        elif expression.is_constant():
            return expression.constant_value()
        return []

    def _find_fluents(self, node):
        info = {}
        if node.is_fluent_exp():
            assert node.fluent().type.is_int_type(), "Fluents inside arithmetic expressions must have integer type!"
            info[node.fluent().name] = [i for i in
                                       range(node.fluent().type.lower_bound, node.fluent().type.upper_bound + 1)]
            return info
        for arg in node.args:
            child_info = self._find_fluents(arg)
            for k, v in child_info.items():
                info[k] = v
        return info

    def _check_expression(self, new_problem, node, assignation):
        new_args = []
        for arg in node.args:
            if arg.is_fluent_exp():
                new_args.append(Int(assignation[arg.fluent().name]))
            elif arg.args == ():
                new_args.append(arg)
            else:
                new_args.append(self._check_expression(new_problem, arg, assignation))
        return new_problem.environment.expression_manager.create_node(node.node_type, tuple(new_args)).simplify()

    def _negate(self, v):
        return {var: -coef for var, coef in v.items()}

    def _merge(self, v1, v2):
        result = dict(v1)  # copia
        for var, coef in v2.items():
            result[var] = result.get(var, 0) + coef
            if result[var] == 0:
                del result[var]  # per netejar zeros
        return result

    def _decompose(self, expr):
        if expr.is_fluent_exp():
            return {expr: 1}, 0
        elif expr.is_constant():
            return {}, expr.constant_value()
        elif expr.is_plus():
            v1, c1 = self._decompose(expr.arg(0))
            v2, c2 = self._decompose(expr.arg(1))
            return self._merge(v1, v2), c1 + c2
        elif expr.is_minus():
            v1, c1 = self._decompose(expr.arg(0))
            v2, c2 = self._decompose(expr.arg(1))
            return self._merge(v1, self._negate(v2)), c1 - c2
        elif expr.is_times():
            v1, c1 = self._decompose(expr.arg(0))
            v2, c2 = self._decompose(expr.arg(1))
            if len(v1) == 0:  # constant is the first argument
                return {var: coef * c1 for var, coef in v2.items()}, c1 * c2
            elif len(v2) == 0:  # constant is the second argument
                return {var: coef * c2 for var, coef in v1.items()}, c1 * c2
            else:
                raise UPProblemDefinitionError("Multiplication of 2 fluents not supported!")
        elif expr.is_div():
            v1, c1 = self._decompose(expr.arg(0))
            v2, c2 = self._decompose(expr.arg(1))
            # division per constant
            if len(v2) == 0:
                return {var: coef / c2 for var, coef in v1.items()}, c1 / c2
            else:
                raise UPProblemDefinitionError("Division of 2 fluents not supported!")
        #elif expr.is_implies():
        #    return self._decompose((~expr.arg(0)) | expr.arg(1))
        else:
            raise UPProblemDefinitionError(
                f"Operation {expr.node_type} not supported!")

    def _normalize(self, node):
        v1, c1 = self._decompose(node.arg(0))
        v2, c2 = self._decompose(node.arg(1))
        vars_norm = self._merge(v1, self._negate(v2))
        const_norm = c1 - c2

        # If there's only a variable
        if len(vars_norm) == 1:
            (var, coef), = vars_norm.items()
            if coef != 1:
                vars_norm = {var: 1}
                const_norm = const_norm / coef
        return vars_norm, const_norm

    def _handle_comparison(self, new_problem, node):
        # Normalitzar
        normalized_vars, normalized_const = self._normalize(node)
        target = -normalized_const
        var_items = list(normalized_vars.items())

        if normalized_vars == {}:
            return []
        # Cas 1 variable
        if len(var_items) == 1:
            (var, coef) = var_items[0]
            assert coef == 1, "Coefficient of not 1 not supported!"
            bounds = self._get_bounds(var)

            if node.node_type == OperatorKind.EQUALS:
                solutions = [{var: target}] if target in bounds else []
            elif node.node_type in (OperatorKind.LE, OperatorKind.LT):
                bisect_fn = bisect.bisect_right if node.node_type == OperatorKind.LE else bisect.bisect_left
                idx = bisect_fn(bounds, target)
                solutions = [{var: v} for v in bounds[:idx]]
            else:
                raise UPProblemDefinitionError(
                    f"Node type {node.node_type} not supported here!"
                )

        # Cas 2 variables
        elif len(var_items) == 2:
            (var1, coef1), (var2, coef2) = var_items
            bounds_1 = self._get_bounds(var1)
            bounds_2 = self._get_bounds(var2)
            solutions = []
            # Iterem per la variable amb menys valors, però simplifiquem en els dos sentits
            if node.node_type == OperatorKind.EQUALS:
                op = operator.eq
            elif node.node_type == OperatorKind.LE:
                op = operator.le
            elif node.node_type == OperatorKind.LT:
                op = operator.lt
            else:
                raise UPProblemDefinitionError("Node type not supported here!")

            if len(bounds_1) <= len(bounds_2):
                for v1 in bounds_1:
                    valid_v2 = [v2 for v2 in bounds_2 if op(coef1 * v1 + coef2 * v2 + normalized_const, 0)]

                    if set(valid_v2) == set(bounds_2):
                        # Fixant var1 = v1, TOTS els valors de var2 funcionen
                        solutions.append({var1: v1})
                    else:
                        # Normal: només afegim els parells vàlids
                        solutions.extend({var1: v1, var2: v2} for v2 in valid_v2)
            else:
                for v2 in bounds_2:
                    valid_v1 = [v1 for v1 in bounds_1 if op(coef1 * v1 + coef2 * v2 + normalized_const, 0)]

                    if set(valid_v1) == set(bounds_1):
                        # Fixant var2 = v2, TOTS els valors de var1 funcionen
                        solutions.append({var2: v2})
                    else:
                        solutions.extend({var1: v1, var2: v2} for v1 in valid_v1)

        # Cas més de 2 variables
        else:
            def backtrack(idx, current_assign, current_sum):
                if idx == len(var_items) - 1:
                    var, coef = var_items[idx]
                    rest = target - current_sum
                    if rest % coef == 0:
                        v = rest // coef
                        if v in self._get_bounds(var):
                            yield {**current_assign, var: v}
                else:
                    var, coef = var_items[idx]
                    bounds = self._get_bounds(var)
                    valid_values = []

                    for v in bounds:
                        new_sum = current_sum + coef * v
                        for _ in backtrack(idx + 1, {**current_assign, var: v}, new_sum):
                            valid_values.append(v)
                            break

                    if set(valid_values) == set(bounds):
                        # tots els valors del domini funcionen - nomes guardem la variable
                        yield current_assign
                    else:
                        for v in valid_values:
                            yield from backtrack(idx + 1, {**current_assign, var: v}, current_sum + coef * v)

            solutions = list(backtrack(0, {}, 0))
        return solutions

    def _has_arithmetics(self, node: "up.model.fnode.FNode"):
        # mirar si algun argument es enter
        for a in node.args:
            operation_inner = self.operation_inner_map.get(a.node_type)
            if operation_inner is not None:
                return True
            if self._has_arithmetics(a):
                return True
        return False

    def _has_integers(self, node: "up.model.fnode.FNode"):
        # Check if any child of node has integers in it
        for a in node.args:
            if a.is_fluent_exp():
                if a.fluent().type.is_int_type():
                    return True
            if a.is_int_constant() or a.is_le() or a.is_lt():
                return True
        return False

    def _consistent(self, assign1, assign2):
        for k, v in assign1.items():
            if k in assign2 and assign2[k] != v:
                return False
        return True

    def _combine_constraints(self, node, old_problem, new_problem):
        all_solutions = []
        for child in node.args:
            if self.operation_outer_map.get(child.node_type) is not None:
                sols = self._combine_constraints(child, old_problem, new_problem)
            elif self._has_integers(child):
                sols = self._handle_comparison(new_problem, child)
            else:
                sols = self._get_new_node(old_problem, new_problem, child)

            if not sols:  # cap solució -> AND impossible
                return None
            all_solutions.append(sols)

        results = []

        # aixo ns
        def backtrack(i, current):
            if i == len(all_solutions):
                results.append(current.copy())
                return
            for sol in all_solutions[i]:
                if self._consistent(current, sol):
                    merged = {**current, **sol}
                    backtrack(i + 1, merged)

        backtrack(0, {})
        return results

    def _treat_integer_expression(self, node, old_problem, new_problem):
        if node.node_type == OperatorKind.AND: # controlar que hi ha enters un altre cop..
            return self._combine_constraints(node, old_problem, new_problem)
        elif node.node_type == OperatorKind.NOT:
            return self._treat_integer_expression(node, old_problem, new_problem)
        elif node.node_type == OperatorKind.OR:
            possible_cases = []
            for arg in node.args:
                possible_cases.extend(self._treat_integer_expression(arg, old_problem, new_problem))
            return possible_cases
        else:  # comparison operators (equals, lt, le)
            possible_cases = self._handle_comparison(new_problem, node)
            return possible_cases

    def _apply_negation(self, node):
        constants = {
            OperatorKind.BOOL_CONSTANT,
            OperatorKind.INT_CONSTANT,
            OperatorKind.LIST_CONSTANT,
            OperatorKind.REAL_CONSTANT,
            OperatorKind.FLUENT_EXP,
            OperatorKind.OBJECT_EXP,
            OperatorKind.PARAM_EXP,
            OperatorKind.VARIABLE_EXP
        }
        if node.node_type in constants:
            return Not(node)
        if node.node_type == OperatorKind.EQUALS:
            return Or(LT(node.arg(0), node.arg(1)),
                      GT(node.arg(0), node.arg(1))).simplify()
        elif node.node_type == OperatorKind.LE:
            return GT(node.arg(0), node.arg(1)).simplify()
        elif node.node_type == OperatorKind.LT:
            return GE(node.arg(0), node.arg(1)).simplify()
        elif node.node_type == OperatorKind.AND:
            return Or(node.args).simplify()
        elif node.node_type == OperatorKind.OR:
            return And(node.args).simplify()
        elif node.node_type == OperatorKind.NOT:
            return node.arg(0)
        else:
            raise UPProblemDefinitionError(f"Negation not supported for {node.node_type}")

    def _to_nnf(self, new_problem, node):
        constants = {
            OperatorKind.BOOL_CONSTANT,
            OperatorKind.INT_CONSTANT,
            OperatorKind.LIST_CONSTANT,
            OperatorKind.REAL_CONSTANT,
            OperatorKind.FLUENT_EXP,
            OperatorKind.OBJECT_EXP,
            OperatorKind.PARAM_EXP,
            OperatorKind.VARIABLE_EXP
        }
        if node.node_type in constants:
            return node
        elif node.node_type == OperatorKind.NOT:
            return self._apply_negation(self._to_nnf(new_problem, node.arg(0)))
        elif node.node_type == OperatorKind.IMPLIES:
            return Or(self._to_nnf(new_problem, Not(node.arg(0))), self._to_nnf(new_problem, node.arg(1)))
        elif node.node_type == OperatorKind.IFF:
            return And(Or(self._to_nnf(new_problem, Not(node.arg(0))), self._to_nnf(new_problem, node.arg(1))),
                       Or(self._to_nnf(new_problem, node.arg(0)), self._to_nnf(new_problem, Not(node.arg(1)))))
        else:
            em = new_problem.environment.expression_manager
            return em.create_node(node.node_type, tuple([self._to_nnf(new_problem, a) for a in node.args]))

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
            return self._get_object_n(new_problem, node.constant_value())
        elif node.is_fluent_exp() and node.fluent().type.is_int_type():
            return new_problem.fluent(node.fluent().name)(*node.args)
        elif node.is_object_exp() or node.is_fluent_exp() or node.is_constant() or node.is_parameter_exp():
            return node
        else:
            # Plus/Minus/Div/Mult in an outer expression
            operation_inner = self.operation_inner_map.get(node.node_type)
            if operation_inner is not None:
                raise UPProblemDefinitionError(
                    f"Operation {operation_inner} not supported as an external expression!")

            # Expression that contains integers !!!

            # negation normal form!!!
            new_node = self._to_nnf(new_problem, node)
            if self._has_integers(new_node):
                cases = self._treat_integer_expression(new_node, old_problem, new_problem)
                if not cases:
                    return None
                or_cases = []
                for c in cases:
                    and_cases = []
                    for f,v in c.items():
                        and_cases.append(Equals(self._get_new_node(old_problem, new_problem, f),
                                                self._get_new_node(old_problem, new_problem, Int(v))).simplify())
                    or_cases.append(And(*and_cases).simplify())
                return Or(*or_cases).simplify()
            else:
                # Other nodes
                new_args = [self._get_new_node(old_problem, new_problem, arg) for arg in new_node.args]
                if new_node.is_exists() or new_node.is_forall():
                    new_variables = [
                        model.Variable(v.name, tm.UserType('Number')) if v.type.is_int_type() else v
                        for v in new_node.variables()
                    ]
                    return em.create_node(new_node.node_type, tuple(new_args), payload=tuple(new_variables))
                if None in new_args:
                    return None
                return em.create_node(new_node.node_type, tuple(new_args)).simplify()

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

    def _get_object_n(
            self,
            problem: "up.model.AbstractProblem",
            n: Optional[int] = None,
    ):
        number_type = problem.environment.type_manager.UserType('Number')
        if not problem.has_object(f'n{n}'):
            ut_number = model.Object(f'n{n}', number_type)
            problem.add_object(ut_number)
        else:
            ut_number = problem.object(f'n{n}')
        return problem.environment.expression_manager.ObjectExp(ut_number)

    def _transform_fluent(self, fluent, default_value, new_signature, new_problem, ut_number):
        if fluent.type.is_int_type():
            lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
            # cal afegir-los?
            new_fluent = model.Fluent(fluent.name, ut_number, new_signature, new_problem.environment)
            if default_value is not None:
                default_obj = self._get_object_n(new_problem, default_value)
                new_problem.add_fluent(new_fluent, default_initial_value=default_obj)
            else:
                new_problem.add_fluent(new_fluent)
            return new_fluent, True
        else:
            new_fluent = model.Fluent(fluent.name, fluent.type, new_signature, new_problem.environment)
            new_problem.add_fluent(new_fluent, default_initial_value=default_value)
            return new_fluent, False

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

        # Fluents
        ut_number = tm.UserType('Number')
        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)
            new_signature = []
            for s in fluent.signature:
                if s.type.is_int_type():
                    raise NotImplementedError(
                        f"Fluent '{fluent.name}' has a parameter '{s.name}' of integer type, which is not supported."
                    )
                else:
                    new_signature.append(s)

            new_fluent, is_transformed_int = self._transform_fluent(
                fluent, default_value, new_signature, new_problem, ut_number)

            for k, v in problem.initial_values.items():
                if k.fluent().name == fluent.name and v != default_value:
                    if is_transformed_int and k.type.is_int_type():
                        new_problem.set_initial_value(new_problem.fluent(fluent.name)(*k.args),
                                                      self._get_object_n(new_problem, v))
                    else:
                        new_problem.set_initial_value(k, v)

        # Actions
        for action in problem.actions:
            remove_action = False
            new_action = action.clone()
            new_action.name = get_fresh_name(new_problem, action.name)
            new_action.clear_preconditions()
            new_action.clear_effects()
            for precondition in action.preconditions:
                new_precondition = self._get_new_node(problem, new_problem, precondition)
                if new_precondition is None or new_precondition == FALSE():
                    remove_action = True
                    break
                new_action.add_precondition(new_precondition)
            if not remove_action:
                for effect in action.effects:
                    if effect.is_increase() or effect.is_decrease():
                        for ne in self._convert_effect(effect, problem, new_problem):
                            new_action.add_effect(ne.fluent, ne.value, ne.condition, ne.forall)
                    else:
                        operation_inner = self.operation_inner_map.get(effect.value.node_type)
                        # Plus/Minus/Div/Mult in an outer expression
                        if operation_inner is not None:
                            fluent_values = self._find_fluents(effect.value)
                            grouped_assignments = {}
                            for combination in product(*fluent_values.values()):
                                assignation = dict(zip(fluent_values.keys(), combination))
                                # now assign all values to the fluents and check if the expression is satisfied
                                lb = effect.fluent.fluent().type.lower_bound
                                ub = effect.fluent.fluent().type.upper_bound
                                possible_value = self._check_expression(
                                        new_problem, effect.value, assignation).constant_value()
                                if lb <= possible_value <= ub:
                                    grouped_assignments.setdefault(possible_value, []).append(assignation.copy())

                            if not grouped_assignments:
                                remove_action = True

                            for v, assignments in grouped_assignments.items():
                                possible_cases = []
                                for a in assignments:
                                    inner_assignments = []
                                    fluent_cache = {f: new_problem.fluent(f) for f in fluent_values.keys()}
                                    for f, fv in a.items():
                                        inner_assignments.append(
                                            Equals(fluent_cache[f], self._get_object_n(new_problem, fv))
                                        )
                                    possible_cases.append(And(inner_assignments))

                                new_action.add_effect(
                                    new_problem.fluent(effect.fluent.fluent().name),
                                    self._get_object_n(new_problem, v),
                                    Or(possible_cases), effect.forall)
                        else:
                            new_node = self._get_new_node(problem, new_problem, effect.fluent)
                            new_condition = self._get_new_node(problem, new_problem, effect.condition)
                            new_value = self._get_new_node(problem, new_problem, effect.value)
                            new_action.add_effect(new_node, new_value, new_condition, effect.forall)
                if not remove_action:
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