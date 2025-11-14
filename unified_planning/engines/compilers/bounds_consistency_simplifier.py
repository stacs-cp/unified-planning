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
"""This module defines the bounds consistency simplifier class."""
import bisect
import unified_planning as up
import unified_planning.engines as engines
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, Action, ProblemKind, InstantaneousAction, FNode, OperatorKind
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action, updated_minimize_action_costs
from typing import Dict, List, Optional, Tuple, OrderedDict, Union
from functools import partial
from unified_planning.shortcuts import FALSE, TRUE, Int, Equals, Not, Plus, Minus, Times, ObjectExp, Iff, Or, And, GT, \
    GE


class BoundsConsistencySimplifier(engines.engine.Engine, CompilerMixin):
    """
    Compiler that simplifies actions by detecting fixed (static) or bounded (bounds consistency) fluent values in
    preconditions and substituting them throughout, eliminating impossible preconditions/effects (out of bounds).
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.BOUNDS_CONSISTENCY_SIMPLIFIER)
        self._static_fluents: Dict[FNode, FNode] = {}

    @property
    def name(self):
        return "bcs"

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
        return problem_kind <= BoundsConsistencySimplifier.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.BOUNDS_CONSISTENCY_SIMPLIFIER

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = problem_kind.clone()
        return new_kind

    # ==================== METHODS ====================

    def _handle_none_args(self, node_type: OperatorKind, args: List) -> Union[List[FNode], None]:
        """Handle undefined (None) values in arguments based on operator semantics."""
        if None not in args:
            return args

        if node_type in {OperatorKind.OR, OperatorKind.COUNT, OperatorKind.EXISTS}:
            filtered = [arg for arg in args if arg is not None]
            return filtered if filtered else None
        elif node_type == OperatorKind.IMPLIES:
            if args[1] is None and args[0] is not None:
                return [args[0], FALSE()]
            return [TRUE(), args[1]] if args[1] is not None else None
        else:
            return None

    # ==================== BOUNDS EXTRACTION ====================

    def _get_initial_bounds(self, problem: Problem, fluent: FNode) -> List:
        """Get initial domain of a FLUENT."""
        if fluent.type.is_int_type():
            return list(range(fluent.type.lower_bound, fluent.type.upper_bound + 1))
        elif fluent.type.is_bool_type():
            return [True, False]
        elif fluent.type.is_user_type():
            return list(problem.objects(fluent.type))
        else:
            return []

    #def _get_bounds(self, problem: Problem, node: FNode, bounds: Optional[Dict[FNode, List[FNode]]] = None):
    #    """Get the possible bounds of an expression. Only implemented with constants for now."""
    #    # pot ser que es doni pot ser que no?
    #    # and fara interseccio
#
    #    # fem normalitzacio abans??? aixi implies es treuria.. i tindriem conjuncions de coses
    #    # any way, he d'implementar per trobar els bounds de qualsevol expressio...
#
#
    #    print("get bounds", node)
    #    # And
    #    if node.is_and():
    #        for arg in node.args:
    #            result = self._get_bounds(problem, arg, bounds)
    #            if result is None:
    #                return None
    #        return True
#
    #    elif node.is_or():
    #        print("or")
    #        new_bounds = dict()
    #        union_bounds = dict()
    #        for arg in node.args:
    #            result = self._get_bounds(problem, arg, new_bounds)
    #            if result is None:
    #                return None
    #        return True
    #    # Not Equals()
    #    elif node.is_not() and node.arg(0).is_equals():
    #        print("not equals", node)
    #        subnode = node.arg(0)
    #        first = subnode.arg(0)
    #        second = subnode.arg(1)
    #        # Integers
    #        if first.type.is_int_type() and (first.is_constant() or second.is_constant()):
    #            switch = first.is_constant()
    #            fluent, value = (
    #                (first, second.constant_value())  # CONSTANT O NO?
    #                if not switch else
    #                (second, first.constant_value())
    #            )
    #            current_bounds = bounds.get(
    #                fluent,
    #                self._get_initial_bounds(problem, fluent)
    #            )
    #            idx = bisect.bisect_left(current_bounds, value)
#
    #            if idx < len(current_bounds) and current_bounds[idx] == value:
    #                new_bounds = current_bounds[:idx] + current_bounds[idx + 1:]
    #                if not new_bounds:
    #                    return None
    #                bounds[fluent] = new_bounds
    #            return True
#
    #        # UserType
    #        elif first.type.is_user_type() and (first.is_constant or second.is_constant):
    #            switch = first.is_object_exp()
    #            fluent, value = (
    #                (first, second)  # CONSTANT O NO?
    #                if not switch else
    #                (second, first)
    #            )
    #            current_bounds = bounds.get(
    #                fluent,
    #                self._get_initial_bounds(problem, fluent)
    #            )
    #            new_bounds = [b for b in current_bounds if b != value]
    #            if any(b not in current_bounds for b in new_bounds):
    #                return None
    #            bounds[fluent] = new_bounds
    #            return True
#
    #    # Equals(), LT(), LE() - Integers
    #    elif node.is_equals() or node.is_lt() or node.is_le():
    #        print("equals, lt, le")
    #        first, second = node.arg(0), node.arg(1)
    #        if first.type.is_int_type() and (first.is_constant or second.is_constant):
    #            switch = first.is_constant()
    #            fluent, value = (
    #                (first, second.constant_value())
    #                if not switch else
    #                (second, first.constant_value())
    #            )
    #            current_bounds = bounds.get(
    #                fluent,
    #                self._get_initial_bounds(problem, fluent)
    #            )
#
    #            print("current bounds", current_bounds)
#
    #            # Decide bisect mode based on operator and side
    #            if node.is_lt():
    #                left_fn, right_fn = bisect.bisect_left, bisect.bisect_right
    #                take_lower = not switch
    #            elif node.is_le():
    #                print("le")
    #                left_fn, right_fn = bisect.bisect_right, bisect.bisect_left
    #                take_lower = not switch
    #            else:  # equals
    #                if value not in current_bounds:
    #                    return None
    #                bounds[fluent] = [Int(value)]
    #                return True
#
    #            # Apply slicing
    #            if take_lower:
    #                idx = left_fn(current_bounds, value)
    #                print(current_bounds[:idx])
    #                new_bounds = current_bounds[:idx]
    #            else:
    #                print("AQUI")
    #                idx = right_fn(current_bounds, value)
    #                print(idx)
    #                print(current_bounds[idx:])
    #                new_bounds = current_bounds[idx:]
#
    #            if any(n not in current_bounds for n in new_bounds):
    #                return None
    #            bounds[fluent] = new_bounds
#
    #        # UserType
    #        elif first.type.is_user_type() and (first.is_constant or second.is_constant):
    #            switch = first.is_object_exp()
    #            fluent, value = (
    #                (first, second)
    #                if not switch else
    #                (second, first)
    #            )
    #            current_bounds = bounds.get(
    #                fluent,
    #                self._get_initial_bounds(problem, fluent)
    #            )
    #            if value not in current_bounds:
    #                return None
    #            bounds[fluent] = [value]
    #            return True
#
    #    # Not Fluent - Boolean
    #    elif node.is_not() and node.arg(0).is_fluent_exp():
    #        fluent = node.arg(0)
    #        current_bounds = bounds.get(
    #            fluent,
    #            [TRUE(), FALSE()]
    #        )
    #        if FALSE() not in current_bounds:
    #            return None
    #        bounds[fluent] = [FALSE()]
#
    #    # Fluent - Boolean
    #    elif node.is_fluent_exp():
    #        assert node.type.is_bool_type(), "Error!!!"
    #        fluent = node
    #        current_bounds = bounds.get(
    #            fluent,
    #            [TRUE(), FALSE()]
    #        )
    #        if TRUE() not in current_bounds:
    #            return None
    #        bounds[fluent] = [TRUE()]
    #    return True

    def _get_bounds(
            self, problem: Problem, node: FNode
    ) -> Union[Dict[FNode, List[FNode]], None]:
        """
        Get the possible bounds of an expression. Only implemented with constants. Parameter-fluents in progress
        Supports:
            fluent - constant
            parameter - constant
        """
        bounds = {}
        # And
        all_new_bounds = dict()
        if node.is_and():
            for i, arg in enumerate(node.args):
                new_bounds = self._get_bounds(problem, arg)
                if i == 0:
                    # Primer argument: inicialitzar amb còpia
                    all_new_bounds = {k: v.copy() for k, v in new_bounds.items()}
                else:
                    # Arguments següents: fusionar amb intersecció
                    for key, values in new_bounds.items():
                        if key in all_new_bounds:
                            # Si la clau existeix, fer intersecció
                            all_new_bounds[key] = [v for v in all_new_bounds[key] if v in values]
                        else:
                            # Si la clau no existeix, afegir-la directament
                            all_new_bounds[key] = values.copy()
            return all_new_bounds
        # Or
        if node.is_or():
            all_new_bounds = dict()
            for arg in node.args:
                new_bounds = self._get_bounds(problem, arg)
                # Union new_bounds amb all_new_bounds
                for key, values in new_bounds.items():
                    all_new_bounds.setdefault(key, []).extend(
                        [v for v in values if v not in all_new_bounds.get(key, [])])
            return all_new_bounds

        # Not Equals()
        if node.is_not() and node.arg(0).is_equals():
            subnode = node.arg(0)
            first = subnode.arg(0)
            second = subnode.arg(1)
            if ((first.is_constant() and (second.is_parameter_exp() or second.is_fluent_exp())) or
                (second.is_constant()) and (first.is_parameter_exp() or first.is_fluent_exp())):
                # Integers
                if first.type.is_int_type():
                    switch = first.is_constant()
                    fluent, value = (
                        (first, second.constant_value())
                        if not switch else
                        (second, first.constant_value())
                    )
                    all_bounds = self._get_initial_bounds(problem, fluent)

                    idx = bisect.bisect_left(all_bounds, value)

                    if idx < len(all_bounds) and all_bounds[idx] == value:
                        new_bounds = all_bounds[:idx] + all_bounds[idx + 1:]
                        if not new_bounds:
                            return None
                        bounds[fluent] = new_bounds
                    return bounds

                # UserType
                elif first.type.is_user_type():
                    switch = first.is_object_exp()
                    fluent, value = (
                        (first, second.object())  # CONSTANT O NO?
                        if not switch else
                        (second, first.object())
                    )
                    all_bounds = self._get_initial_bounds(problem, fluent)
                    new_bounds = [b for b in all_bounds if b != value]
                    if any(b not in all_bounds for b in new_bounds):
                        return None
                    bounds[fluent] = new_bounds
                    return bounds

            elif (first.is_fluent_exp() or first.is_parameter_exp()) and (second.is_fluent_exp() or second.is_parameter_exp()):
                first_all_bounds = self._get_initial_bounds(problem, first)
                second_all_bounds = self._get_initial_bounds(problem, second)
                bounds[first] = first_all_bounds
                bounds[second] = second_all_bounds
                return bounds

        # Equals(), LT(), LE()
        if node.is_equals() or node.is_lt() or node.is_le():

            first, second = node.arg(0), node.arg(1)
            if ((first.is_constant() and (second.is_parameter_exp() or second.is_fluent_exp())) or
                    (second.is_constant()) and (first.is_parameter_exp() or first.is_fluent_exp())):
                if first.type.is_int_type():
                    switch = first.is_constant()
                    fluent, value = (
                        (first, second.constant_value())
                        if not switch else
                        (second, first.constant_value())
                    )
                    all_bounds = self._get_initial_bounds(problem, fluent)
                    # Decide bisect mode based on operator and side
                    if node.is_lt():
                        left_fn, right_fn = bisect.bisect_left, bisect.bisect_right
                        take_lower = not switch
                    elif node.is_le():
                        left_fn, right_fn = bisect.bisect_right, bisect.bisect_left
                        take_lower = not switch
                    else:  # equals
                        if value not in all_bounds:
                            return None
                        bounds[fluent] = [value]
                        return bounds

                    # Apply slicing
                    if take_lower:
                        idx = left_fn(all_bounds, value)
                        new_bounds = all_bounds[:idx]
                    else:
                        idx = right_fn(all_bounds, value)
                        new_bounds = all_bounds[idx:]

                    if any(n not in all_bounds for n in new_bounds):
                        return None
                    bounds[fluent] = new_bounds
                    return bounds

                # UserType
                if first.type.is_user_type():
                    switch = first.is_object_exp()
                    fluent, value = (
                        (first, second.object())
                        if not switch else
                        (second, first.object())
                    )
                    all_bounds = self._get_initial_bounds(problem, fluent) # aixo no caldria
                    if value not in all_bounds:
                        return None
                    bounds[fluent] = [value]
                    return bounds

            elif first.is_fluent_exp() or first.is_parameter_exp() and second.is_fluent_exp() or second.is_parameter_exp():
                first_all_bounds = self._get_initial_bounds(problem, first)
                second_all_bounds = self._get_initial_bounds(problem, second)
                bounds[first] = first_all_bounds
                bounds[second] = second_all_bounds
                return bounds

        # Not Fluent - Boolean
        if node.is_not() and node.arg(0).is_fluent_exp():
            fluent = node.arg(0)
            bounds[fluent] = [False]
            return bounds

        # Fluent - Boolean
        if node.is_fluent_exp():
            assert node.type.is_bool_type(), "Error!!!"
            fluent = node
            bounds[fluent] = [True]
            return bounds

        return {}

    # ==================== SIMPLIFICATION ====================

    def _negate_expr(self, problem: Problem, expr: FNode) -> FNode:
        """Nega una expressió distribuint la negació recursivament"""
        # Si és (0 - x), retorna x
        if expr.is_constant():
            return Int(-expr.constant_value())
        if expr.node_type == OperatorKind.MINUS and expr.arg(0).constant_value() == 0:
            return expr.arg(1)
        if expr.node_type in self.constants:
            return Minus(Int(0), expr)
        #if expr.node_type == OperatorKind.PLUS and len(expr.args) == 2 and expr.arg(1).node_type == OperatorKind.MINUS and \
        #    expr.arg(1).arg(0).constant_value() == 0:
        #    print("AAAAA", expr, "to", Minus(expr.arg(1).arg(1), expr.arg(0)))
        #    return Minus(expr.arg(1).arg(1), expr.arg(0))
        #if expr.node_type == OperatorKind.PLUS and len(expr.args) == 2 and expr.arg(0).node_type == OperatorKind.MINUS and \
        #    expr.arg(0).arg(0).constant_value() == 0:
        #    print("AAAAA", expr, "to", Minus(expr.arg(0).arg(1), expr.arg(1)))
        #    return Minus(expr.arg(1), expr.arg(0).arg(1))
        # Recursivament negar els arguments
        if expr.args:
            em = problem.environment.expression_manager
            new_args = [self._negate_expr(problem, a) for a in expr.args]
            return em.create_node(expr.node_type, tuple(new_args)).simplify()
        else:
            raise UPProblemDefinitionError("WWWWWWW")

    def _build_expression(self, terms: list[tuple[int, FNode]]) -> FNode:
        """Construeix una expressió a partir d'una llista de termes"""
        if not terms:
            return Int(0)
        # Agrupa termes per variable per combinar coeficients
        var_to_coef = {}
        for coef, var in terms:
            var_key = id(var)  # Usa ID per comparar variables
            if var_key in var_to_coef:
                var_to_coef[var_key] = (var_to_coef[var_key][0] + coef, var)
            else:
                var_to_coef[var_key] = (coef, var)

        # Construeix l'expressió final
        result_terms = []
        for coef, var in var_to_coef.values():
            if coef == 0:
                continue  # Elimina termes amb coeficient 0
            elif coef == 1:
                result_terms.append(var)
            elif coef == -1:
                result_terms.append(Minus(Int(0), var))
            elif coef > 0:
                result_terms.append(Times(Int(coef), var))
            else:  # coef < 0
                result_terms.append(Minus(Int(0), Times(Int(-coef), var)))

        if not result_terms:
            return Int(0)
        elif len(result_terms) == 1:
            return result_terms[0].simplify()
        else:
            return Plus(result_terms).simplify()

    def _extract_constants_and_variables(self, node: FNode) -> tuple[int, list[tuple[int, FNode]]]:
        """
        Extreu la suma de constants i la llista de variables d'una expressió.
        Retorna (constant_sum, [variables])

        Exemples:
        - cols(i1) + 2 + 3 -> (5, [cols(i1)])
        - 10 -> (10, [])
        - cols(i1) + rows(i2) -> (0, [cols(i1), rows(i2)])
        """
        if node.is_constant():
            return node.constant_value(), []

        if node.is_plus():
            total_const = 0
            all_vars = []
            for arg in node.args:
                const_val, vars_list = self._extract_constants_and_variables(arg)
                total_const += const_val
                all_vars.extend(vars_list)
            return total_const, all_vars

        if node.is_minus() and len(node.args) == 2:
            left_const, left_vars = self._extract_constants_and_variables(node.arg(0))
            right_const, right_vars = self._extract_constants_and_variables(node.arg(1))

            # a - b: neguem els coeficients del costat dret
            negated_right_vars = [(-coef, var) for coef, var in right_vars]
            return left_const - right_const, left_vars + negated_right_vars

        # Si no és ni constant ni suma/resta, és una variable
        return 0, [(1, node)]

    def _normalise_comparison(self, problem: Problem, op_type, left: FNode, right: FNode) -> FNode:
        # Extreu constants i variables de cada costat
        left_const, left_vars = self._extract_constants_and_variables(left)
        right_const, right_vars = self._extract_constants_and_variables(right)

        # Mou totes les variables a l'esquerra: left_vars - right_vars
        all_vars = left_vars + [(-coef, var) for coef, var in right_vars]

        # Mou totes les constants a la dreta: right_const - left_const
        all_const = right_const - left_const

        # Construeix els costats normalitzats
        new_left = self._build_expression(all_vars)
        new_right = Int(all_const)

        # Constant folding: si tots dos són constants, avalua directament
        if new_left.is_constant():
            left_val = new_left.constant_value()
            right_val = new_right.constant_value()

            if op_type == OperatorKind.EQUALS:
                return TRUE() if left_val == right_val else FALSE()
            elif op_type == OperatorKind.LE:
                return TRUE() if left_val <= right_val else FALSE()
            elif op_type == OperatorKind.LT:
                return TRUE() if left_val < right_val else FALSE()

        # If the costant side is negative and more variables have negative coef then switch sides
        num_negative = sum(1 for coef, _ in all_vars if coef < 0)
        num_positive = sum(1 for coef, _ in all_vars if coef > 0)
        switch = new_right.constant_value() <= 0 and num_negative >= num_positive

        em = problem.environment.expression_manager
        if switch:
            new_left_switched = self._negate_expr(problem, new_left).simplify()
            new_right_switched = self._negate_expr(problem, new_right).simplify()
            new_node = em.create_node(op_type, (new_right_switched, new_left_switched)).simplify()
            return new_node
        return em.create_node(op_type, (new_left, new_right)).simplify()

    def _simplify_arithmetic_expression(self, problem: Problem, node: FNode) -> FNode:
        """
        Simplify arithmetic expressions by moving all constants to one side and variables to the other
        """
        em = problem.environment.expression_manager
        if node.is_constant() or node.is_variable_exp() or node.is_timing_exp() or node.is_parameter_exp() or node.is_fluent_exp():
            return node
        elif (node.is_equals() and node.arg(0).type.is_int_type()) or node.is_le() or node.is_lt():
            left = self._simplify_arithmetic_expression(problem, node.arg(0))
            right = self._simplify_arithmetic_expression(problem, node.arg(1))
            return self._normalise_comparison(problem, node.node_type, left, right)

        simplified_args = [self._simplify_arithmetic_expression(problem, arg) for arg in node.args]
        node = em.create_node(node.node_type, tuple(simplified_args)).simplify()
        return node

    #def _replace_static(self, new_problem: Problem, node: FNode) -> FNode:
    #    """Transform expression by substituting static fluents."""
    #    if node.is_constant() or node.is_variable_exp() or node.is_timing_exp() or node.is_parameter_exp():
    #        return node
    #    if node.is_fluent_exp():
    #        if node in self._static_fluents:
    #            return self._static_fluents[node]
    #        return node
    #    # Generic recursive transformation
    #    em = new_problem.environment.expression_manager
    #    new_args = [self._replace_static(new_problem, arg) for arg in node.args]
    #    new_node = em.create_node(node.node_type, tuple(new_args)).simplify()
    #    # Simplify expressions
    #    new_node_simplified = self._simplify_arithmetic_expression(new_problem, new_node)
    #    return new_node_simplified

    def _replace_values(
            self, new_problem: Problem, node: FNode,
            constants: dict[FNode, FNode], bounded_fluents: dict[FNode, list[FNode]]
    ) -> Union[FNode, None]:
        """
        Transform expression by substituting fluent constant values.
        It handles the implies operator
        """
        em = new_problem.environment.expression_manager
        if node.is_constant() or node.is_variable_exp() or node.is_timing_exp():
            return node
        if node.is_fluent_exp() or node.is_parameter_exp():
            if node in constants.keys():
                if node.type.is_int_type():
                    return Int(constants[node])
                elif node.type.is_user_type():
                    return ObjectExp(constants[node])
                else:
                    return TRUE() if constants[node] else FALSE()
            if node in bounded_fluents:
                return node
            return node
        if node.is_equals():
            fluent, value = (node.arg(0), node.arg(1)) if node.arg(1).is_constant() else (node.arg(1), node.arg(0))
            if fluent in constants.keys():
                value_constant = value.constant_value() if value.is_constant() else value.object()
                if value_constant == constants[fluent]:
                    return TRUE()
                else:
                    return None
            if fluent in bounded_fluents:
                value_constant = value.constant_value() if value.is_constant() else value.object()
                if value_constant in bounded_fluents[fluent]:
                    return node
                else:
                    return None
        if node.is_le():
            switch = node.arg(0).is_constant()
            fluent, value = (node.arg(1), node.arg(0).constant_value()) if switch else (node.arg(0), node.arg(1).constant_value())
            if fluent in constants:
                if value <= constants[fluent] if switch else constants[fluent] <= value:
                    return TRUE()
                else:
                    return None
            if fluent in bounded_fluents:
                is_le = all(b <= value for b in bounded_fluents[fluent]) if not switch else all(value <= b for b in bounded_fluents[fluent])
                if is_le:
                    return TRUE()
                else:
                    return node

        if node.is_lt():
            switch = node.arg(0).is_constant()
            fluent, value = (node.arg(1), node.arg(0).constant_value()) if switch else (node.arg(0), node.arg(1).constant_value())
            if fluent in constants:
                if value < constants[fluent] if switch else constants[fluent] < value:
                    return TRUE()
                else:
                    return None
            if fluent in bounded_fluents:
                is_lt = all(b < value for b in bounded_fluents[fluent]) if not switch else all(value < b for b in bounded_fluents[fluent])
                if is_lt:
                    return TRUE()
                else:
                    return node
        if node.is_not() and node.arg(0).is_equals():
            subnode = node.arg(0)
            fluent, value = (subnode.arg(0), subnode.arg(1)) if subnode.arg(1).is_constant() else (subnode.arg(1), subnode.arg(0))
            if fluent in constants:
                value_constant = value.constant_value() if value.is_constant() else value.object()
                if value_constant != constants[fluent]:
                    return TRUE()
                else:
                    return None
            if fluent in bounded_fluents:
                value_constant = value.constant_value() if value.is_constant() else value.object()
                if value_constant in bounded_fluents[fluent]:
                    return node
                else:
                    return TRUE()

        new_args = [self._replace_values(new_problem, arg, constants, bounded_fluents) for arg in node.args]
        new_args = self._handle_none_args(node.node_type, new_args)
        if new_args is None:
            return None
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    # ==================== ACTION TRANSFORMATION ====================

    def _add_effect_to_action(
            self, action: Action, effect_type: str, fluent: FNode, value: FNode, condition: FNode, forall: Tuple
    ):
        """Add effect to action based on type."""
        if effect_type == 'increase':
            action.add_increase_effect(fluent, value, condition, forall)
        elif effect_type == 'decrease':
            action.add_decrease_effect(fluent, value, condition, forall)
        else:
            action.add_effect(fluent, value, condition, forall)

    constants = {
        OperatorKind.BOOL_CONSTANT,
        OperatorKind.INT_CONSTANT,
        OperatorKind.ARRAY_CONSTANT,
        OperatorKind.REAL_CONSTANT,
        OperatorKind.FLUENT_EXP,
        OperatorKind.OBJECT_EXP,
        OperatorKind.PARAM_EXP,
        OperatorKind.VARIABLE_EXP
    }

    def _apply_negation(self, node):
        if node.node_type in self.constants:
            return Not(node).simplify()
        elif node.node_type == OperatorKind.LE:
            return GT(node.arg(0), node.arg(1)).simplify()
        elif node.node_type == OperatorKind.LT:
            return GE(node.arg(0), node.arg(1)).simplify()
        elif node.node_type == OperatorKind.AND:
            return Or(Not(a) for a in node.args).simplify()
        elif node.node_type == OperatorKind.OR:
            return And(Not(a) for a in node.args).simplify()
        elif node.node_type == OperatorKind.NOT:
            return node.arg(0).simplify()
        else:
            raise UPProblemDefinitionError(f"Negation not supported for {node.node_type}")

    def _to_nnf(self, new_problem, node):
        em = new_problem.environment.expression_manager
        if node.is_fluent_exp():
            if node in self._static_fluents:
                return self._static_fluents[node]
            return node
        elif node.node_type in self.constants:
            return node
        elif node.node_type == OperatorKind.NOT:
            if node.arg(0).node_type in self.constants:
                new_node = Not(self._to_nnf(new_problem, node.arg(0)))
            elif node.arg(0).is_equals(): # Not Equals() remaining
                new_node = em.create_node(node.node_type, tuple([self._to_nnf(new_problem, a) for a in node.args])).simplify()
            else:
                new_node = self._to_nnf(new_problem, self._apply_negation(node.arg(0)))
        elif node.node_type == OperatorKind.IMPLIES:
            new_node = Or(self._to_nnf(new_problem, Not(node.arg(0))), self._to_nnf(new_problem, node.arg(1)))
        elif node.node_type == OperatorKind.IFF:
            new_node = And(Or(self._to_nnf(new_problem, Not(node.arg(0))), self._to_nnf(new_problem, node.arg(1))),
                       Or(self._to_nnf(new_problem, node.arg(0)), self._to_nnf(new_problem, Not(node.arg(1)))))
        else:
            new_node = em.create_node(node.node_type, tuple([self._to_nnf(new_problem, a) for a in node.args])).simplify()

        # Simplify expressions
        new_node_simplified = self._simplify_arithmetic_expression(new_problem, new_node)
        return new_node_simplified

    def _simplify_action(self, new_problem: Problem, old_action: Action) -> Union[Action, None]:
        """Simplify an action using bounds consistency."""
        # primer pas es analitzar les precondicions per veure els bounds dels fluents/parametres en aquella accio concreta!
        # llavors podrem substituir aquells valors en les precondicions i en els efectes!

        # 0) Normalize and replace static fluents in preconditions
        normalized_preconditions = []
        for precondition in old_action.preconditions:
            np = self._to_nnf(new_problem, precondition)
            if np is not TRUE():
                if np.is_and():
                    for new_precondition in np.args:
                        normalized_preconditions.append(new_precondition)
                else:
                    normalized_preconditions.append(np)

        # 1) Replace static fluents in preconditions
        #if self._static_fluents:
        #    temporal_preconditions = []
        #    for precondition in normalized_preconditions:
        #        tp = self._replace_static(new_problem, precondition)
        #        if tp is not TRUE():
        #            temporal_preconditions.append(tp)
        #else:
        #    temporal_preconditions = normalized_preconditions

        # 2) Get the possible bounds or constant value for the fluents used in this action
        bounds = self._get_bounds(new_problem, And(*normalized_preconditions))
        if bounds is None:
            return None

        # bounds pot tenir algun

        constants = dict((f,v[0]) for f, v in bounds.items() if len(v) == 1)
        bounded_fluents = dict((f,v) for f,v in bounds.items() if len(v) > 1)

        # Create the new action
        params = OrderedDict((p.name, p.type) for p in old_action.parameters)
        new_action = InstantaneousAction(old_action.name, _parameters=params, _env=new_problem.environment)

        # 3) Add the fluent values that are constant within the action -- no
        #for f,v in constants.items():
        #    if f.type.is_bool_type():
        #        new_action.add_precondition(f) if v else new_action.add_precondition(Not(f))
        #    else:
        #        new_action.add_precondition(Equals(f, v))

        # 4) Replace fluent preconditions with constant values!!! yes
        #if constants or bounded_fluents:
        if constants:
            for precondition in normalized_preconditions:
                new_precondition = self._replace_values(new_problem, precondition, constants, dict())
                if new_precondition is None:
                    return None
                new_action.add_precondition(new_precondition)
        else:
            for precondition in normalized_preconditions:
                new_action.add_precondition(precondition)

        # 5) Afegir precondicions amb els bounds - ajudaran al cp tools -- no!!!!
        #for f, bounds in bounded_fluents.items():
        #    if f.type.is_bool_type():
        #        or_expr = [Iff(f, b) for b in bounds]
        #    else:
        #        or_expr = [Equals(f, b) for b in bounds]
        #    print("afegint: ", Or(*or_expr))
        #    new_action.add_precondition(Or(*or_expr))

        # 6) Replace fluents with constant values and control the bounds in the effects conditions and values
        for effect in old_action.effects:
            if effect.is_increase():
                effect_type = 'increase'
            elif effect.is_decrease():
                effect_type = 'decrease'
            else:
                effect_type = 'none'

            new_value = self._to_nnf(new_problem, effect.value)
            new_condition = self._to_nnf(new_problem, effect.condition)
            if constants or bounded_fluents:
                new_value = self._replace_values(new_problem, new_value, constants, bounded_fluents)
                new_condition = self._replace_values(new_problem, new_condition, constants, bounded_fluents)
            self._add_single_effect(new_action, effect_type, effect.fluent, new_value, new_condition, effect.forall)

        return new_action

    def _add_single_effect(
            self,
            action: Action,
            effect_type: str,
            fluent: FNode,
            value: FNode,
            condition: FNode,
            forall: Tuple
    ) -> bool:
        """Add single effect to action. Returns False if action should be pruned."""
        # Handle unconditional effects
        if condition == TRUE():
            if fluent is None or value is None:
                # Invalid unconditional effect -> prune action
                return False
            if fluent.type.is_int_type() and value.is_constant():
                if not fluent.type.lower_bound <= value.constant_value() <= fluent.type.upper_bound:
                    return False
            self._add_effect_to_action( action, effect_type, fluent, value, condition, forall)
        # Handle conditional effects
        else:
            if condition not in [None, FALSE()] and fluent is not None and value is not None:
                if (fluent.type.is_int_type() and
                        not fluent.type.lower_bound <= value.constant_value() <= fluent.type.upper_bound):
                    return True
                self._add_effect_to_action(action, effect_type, fluent, value, condition, forall)
        return True

    def _simplify_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions."""
        new_to_old = {}
        for action in problem.actions:
            new_action = self._simplify_action(new_problem, action)
            if new_action is not None:
                new_problem.add_action(new_action)
                new_to_old[new_action] = action
        return new_to_old

    def _find_static_fluents(self, problem: Problem, fluents: dict[FNode, FNode]) -> Dict[FNode, FNode]:
        """Find all static fluents throughout the problem."""
        modifiable_fluents = []
        for action in problem.actions:
            for effect in action.effects:
                modifiable_fluents.append(effect.fluent)
        return dict((f,v) for f,v in fluents.items() if f not in modifiable_fluents)

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation method."""
        assert isinstance(problem, Problem)

        # Create new problem
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_actions()
        new_problem.clear_quality_metrics()

        # Find static fluents throughout the problem
        all_fluents = problem.initial_values
        self._static_fluents = self._find_static_fluents(problem, all_fluents)

        # Transform actions
        new_to_old = self._simplify_actions(problem, new_problem)

        # Transform quality metrics
        for metric in problem.quality_metrics:
            if metric.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(metric, new_to_old, new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(metric)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
