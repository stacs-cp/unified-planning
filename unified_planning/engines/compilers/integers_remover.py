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
from operator import and_, or_

from ortools.sat.python import cp_model
from itertools import product

from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind

import unified_planning as up
import unified_planning.engines as engines
from collections import defaultdict
import bisect
from unified_planning import model
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError, UPValueError, UPConflictingEffectsException
from unified_planning.model import (
    Problem,
    Action,
    ProblemKind, Variable, MinimizeActionCosts, Effect, EffectKind, AbstractProblem,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action, updated_minimize_action_costs,
)
from typing import Dict, Optional, OrderedDict, Set, Iterator, List, Tuple, Union
from functools import partial, reduce

from unified_planning.model.types import _IntType
from unified_planning.shortcuts import Exists, And, Or, Equals, Int, Plus, Not, LT, FALSE, LE, GT, GE, IntType, Iff


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solutions = []
        self.__seen = set()  # Per detectar duplicats

    def on_solution_callback(self):
        solution = {str(v): self.value(v) for v in self.__variables}

        # Convertir a tupla per fer hash
        sol_tuple = tuple(sorted(solution.items()))

        if sol_tuple not in self.__seen:
            self.__seen.add(sol_tuple)
            self.__solutions.append(solution)
            self.__solution_count += 1

    @property
    def solution_count(self) -> int:
        return self.__solution_count

    @property
    def solutions(self) -> list[dict[str, int]]:
        return self.__solutions

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

    def _has_integers(self, node: "up.model.fnode.FNode"):
        # Check if any child of node has integers in it
        for a in node.args:
            if a.is_fluent_exp():
                if a.fluent().type.is_int_type():
                    return True
            if a.is_int_constant() or a.is_le() or a.is_lt():
                return True
            if self._has_integers(a):
                return True
        return False

    def _add_constraints(self, problem, node, variables, expressions, cpmodel):
        if node.is_constant():
            return cpmodel.new_constant(node.constant_value())
        elif node.is_fluent_exp():
            fluent = node.fluent()
            if node in variables:
                return variables[node]
            else:
                if fluent.type.is_int_type():
                    f = cpmodel.new_int_var(fluent.type.lower_bound, fluent.type.upper_bound, fluent.name)
                    variables[node] = f
                    return f
                else:
                    f = cpmodel.new_bool_var(fluent.name)
                    variables[node] = f
                    return f

        elif node.is_and():
            child_vars = []
            for arg in node.args:
                child_result = self._add_constraints(problem, arg, variables, expressions, cpmodel)
                child_vars.append(child_result)

            and_var = cpmodel.new_bool_var(f"and_{id(node)}")
            cpmodel.add(sum(child_vars) == len(child_vars)).only_enforce_if(and_var)
            cpmodel.add(sum(child_vars) < len(child_vars)).only_enforce_if(~and_var)
            return and_var

        elif node.is_or():
            child_vars = []
            for arg in node.args:
                child_result = self._add_constraints(problem, arg, variables, expressions, cpmodel)
                child_vars.append(child_result)

            or_var = cpmodel.new_bool_var(f"or_{id(node)}")
            cpmodel.add(sum(child_vars) >= 1).only_enforce_if(or_var)
            cpmodel.add(sum(child_vars) == 0).only_enforce_if(~or_var)
            return or_var

        elif node.is_not():
            inner_var = self._add_constraints(problem, node.arg(0), variables, expressions, cpmodel)
            # Crear una nova variable booleana per la negació
            not_var = cpmodel.new_bool_var(f"not_{id(node)}")
            cpmodel.add(inner_var == 0).only_enforce_if(not_var)
            cpmodel.add(inner_var == 1).only_enforce_if(~not_var)
            return not_var

        elif node.is_equals():
            if node.arg(0).type.is_user_type():
                if node in variables:
                    return variables[node]
                var = cpmodel.new_bool_var(str(node))
                variables[node] = var
                expressions[str(var)] = node
                return var
            else:
                left = self._add_constraints(problem, node.arg(0), variables, expressions, cpmodel)
                right = self._add_constraints(problem, node.arg(1), variables, expressions, cpmodel)
                eq_var = cpmodel.new_bool_var(f"eq_{node}")
                cpmodel.add(left == right).only_enforce_if(eq_var)
                cpmodel.add(left != right).only_enforce_if(~eq_var)
                return eq_var

        elif node.is_lt() or node.is_le() or node.is_plus() or node.is_minus() or node.is_times():
            left = self._add_constraints(problem, node.arg(0), variables, expressions, cpmodel)
            right = self._add_constraints(problem, node.arg(1), variables, expressions, cpmodel)

            if node.is_lt():
                var = cpmodel.new_bool_var(f"lt_{id(node)}")
                cpmodel.add(left < right).only_enforce_if(var)
                cpmodel.add(left >= right).only_enforce_if(~var)
                return var
            if node.is_le():
                var = cpmodel.new_bool_var(f"le_{id(node)}")
                cpmodel.add(left <= right).only_enforce_if(var)
                cpmodel.add(left > right).only_enforce_if(~var)
                return var
            if node.is_plus():
                return left + right
            if node.is_minus():
                return left - right
            if node.is_times():
                return left * right

        else:
            raise NotImplementedError(f"Node type {node.node_type} not implemented")

    def _call_cp_solver(self, node, old_problem, new_problem):
        variables = {}
        expressions = {}
        cpmodel = cp_model.CpModel()

        # constraints
        constraints = self._add_constraints(old_problem, node, variables, expressions, cpmodel)
        if isinstance(constraints, cp_model.IntVar):
            cpmodel.add(constraints == 1)
        else:
            cpmodel.add(constraints)
        cp_variables = list(variables.values())

        solver = cp_model.CpSolver()
        solution_printer = VarArraySolutionPrinter(cp_variables)
        solver.parameters.enumerate_all_solutions = True
        status = solver.solve(cpmodel, solution_printer)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # possible solutions
            possible_solutions = []
            solutions = solution_printer.solutions
            if len(solutions) > 1:
                # Agrupar valors i descartar variables que agafen tot el domini - aconsegueixo reduir a 1/4 part
                solutions = self.simplify_solutions(old_problem, solution_printer.solutions)

            for s in solutions:
                assignations = []
                for f, v in s.items():
                    if new_problem.has_fluent(f):
                        fluent = new_problem.fluent(f)  # vigilar! com guardar parametres
                        if old_problem.fluent(f).type.is_int_type():
                            if isinstance(v, set):
                                or_e = [Equals(fluent, self._get_object_n(new_problem, value)) for value in v]
                                assignations.append(Or(or_e))
                            else:
                                value = self._get_object_n(new_problem, v)
                                assignations.append(Equals(fluent, value))
                        elif old_problem.fluent(f).type.is_bool_type():
                            value = True if v == 1 else False
                            assignations.append(Iff(fluent, value))
                        else:
                            raise UPProblemDefinitionError(f"Not implemented yet!")
                    else:
                        assignations.append(expressions[f] if v == 1 else Not(expressions[f]))

                possible_solutions.append(And(assignations).simplify())
            return Or(possible_solutions).simplify()
        else:
            return None


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
        if node.node_type == OperatorKind.LE:
            return GT(node.arg(0), node.arg(1)).simplify()
        elif node.node_type == OperatorKind.LT:
            return GE(node.arg(0), node.arg(1)).simplify()
        elif node.node_type == OperatorKind.AND:
            return Or(Not(a) for a in node.args).simplify()
        elif node.node_type == OperatorKind.OR:
            return And(Not(a) for a in node.args).simplify()
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
            if node.arg(0).node_type in constants:
                return Not(self._to_nnf(new_problem, node.arg(0)))
            elif node.arg(0).is_equals(): # deixem el not equals
                return node
            else:
                return self._to_nnf(new_problem, self._apply_negation(node.arg(0)))
        elif node.node_type == OperatorKind.IMPLIES:
            return Or(self._to_nnf(new_problem, Not(node.arg(0))), self._to_nnf(new_problem, node.arg(1)))
        elif node.node_type == OperatorKind.IFF:
            return And(Or(self._to_nnf(new_problem, Not(node.arg(0))), self._to_nnf(new_problem, node.arg(1))),
                       Or(self._to_nnf(new_problem, node.arg(0)), self._to_nnf(new_problem, Not(node.arg(1)))))
        else:
            em = new_problem.environment.expression_manager
            return em.create_node(node.node_type, tuple([self._to_nnf(new_problem, a) for a in node.args]))

    def _factorise_cases(self, cases):
        grouped = defaultdict(list)

        # 1. Agrupem segons totes les variables menys una
        for case in cases:
            for var in case:
                # clau = tot menys 'var'
                key = tuple(sorted((k, v) for k, v in case.items() if k != var))
                grouped[(key, var)].append(case[var])

        simplified = []
        used = set()

        # 2. Si una variable té diversos valors, compactem-ho
        for (key, var), vals in grouped.items():
            if len(vals) > 1:
                # conjunt de valors
                compact = dict(key)
                compact[var] = set(vals)
                simplified.append(compact)
                used.update((frozenset(case.items()) for case in cases
                             if all(case.get(k) == v for k, v in key)))

        # 3. Afegim els casos que no s’han pogut compactar
        for case in cases:
            if frozenset(case.items()) not in used:
                simplified.append({k: v for k, v in case.items()})

        return simplified

    def simplify_solutions(self, problem: AbstractProblem, solutions: list[dict[str, int]]):
        """
        Agrupa solucions que només difereixen en una o poques variables.
        Retorna solucions compactades on un valor pot ser un set i
        elimina variables que agafen tots els valors del seu domini.
        """
        if not solutions:
            return []

        # Obtenir totes les variables
        all_vars = list(solutions[0].keys())

        simplified = []
        used = set()

        # Intentar agrupar per cada variable
        for var_to_vary in all_vars:
            groups = {}

            for i, sol in enumerate(solutions):
                if i in used:
                    continue

                # Clau = tots els valors menys la variable que pot variar
                key = tuple((k, v) for k, v in sorted(sol.items()) if k != var_to_vary)

                if key not in groups:
                    groups[key] = []
                groups[key].append((i, sol[var_to_vary]))

            # Si un grup té múltiples valors per la variable, compactar
            for key, indices_and_vals in groups.items():
                if len(indices_and_vals) > 1:
                    # Marcar com usades
                    for idx, _ in indices_and_vals:
                        used.add(idx)

                    # Crear solució compactada
                    compact = dict(key)
                    values_set = set(val for _, val in indices_and_vals)

                    # Comprovar si té tots els valors del domini
                    min_val, max_val = problem.fluent(var_to_vary).type.lower_bound, problem.fluent(var_to_vary).type.upper_bound
                    expected_values = set(range(min_val, max_val + 1))

                    # Si NO té tots els valors, guardar el set
                    if values_set != expected_values:
                        compact[var_to_vary] = values_set
                    # Si té tots els valors, no afegir la variable (l'eliminem)

                    simplified.append(compact)

        # Afegir solucions que no s'han pogut agrupar
        for i, sol in enumerate(solutions):
            if i not in used:
                simplified.append(sol)

        return simplified

    def _get_new_node(
            self,
            old_problem: "up.model.AbstractProblem",
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
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

            if self._has_integers(node):
                node_nnf = self._to_nnf(new_problem, node).simplify()
                return self._call_cp_solver(node_nnf, old_problem, new_problem)
            else:
                # Other nodes
                new_args = [self._get_new_node(old_problem, new_problem, arg) for arg in node.args]
                if None in new_args:
                    return None
                if node.is_exists() or node.is_forall():
                    new_variables = [
                        Variable(v.name, tm.UserType('Number')) if v.type.is_int_type() else v
                        for v in node.variables()
                    ]
                    return em.create_node(node.node_type, tuple(new_args), payload=tuple(new_variables)).simplify()
                return em.create_node(node.node_type, tuple(new_args)).simplify()

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
        tm = problem.environment.type_manager
        em = problem.environment.expression_manager
        number_type = tm.UserType('Number')
        if not problem.has_object(f'n{n}'):
            ut_number = model.Object(f'n{n}', number_type)
            problem.add_object(ut_number)
        else:
            ut_number = problem.object(f'n{n}')
        return em.ObjectExp(ut_number)

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
        new_problem.clear_axioms()
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

            new_preconditions = self._get_new_node(problem, new_problem, And(action.preconditions))
            if new_preconditions is None or new_preconditions == FALSE():
                remove_action = True
                break
            new_action.add_precondition(new_preconditions)
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

        # Axioms
        for axiom in problem.axioms:
            remove_axiom = False
            new_axiom = axiom.clone()
            new_axiom.name = get_fresh_name(new_problem, new_axiom.name)
            new_axiom.clear_preconditions()
            new_axiom.clear_effects()
            for precondition in axiom.preconditions:
                new_precondition = self._get_new_node(problem, new_problem, precondition)
                if new_precondition is None or new_precondition == FALSE():
                    remove_axiom = True
                    break
                new_axiom.add_precondition(new_precondition)
            if not remove_axiom:
                for effect in axiom.effects:
                    new_node = self._get_new_node(problem, new_problem, effect.fluent)
                    new_condition = self._get_new_node(problem, new_problem, effect.condition)
                    new_value = self._get_new_node(problem, new_problem, effect.value)
                    new_axiom.add_effect(new_node, new_value, new_condition, effect.forall)

                new_to_old[new_axiom] = axiom
                new_problem.add_axiom(new_axiom)

        for goal in problem.goals:
            new_goal = self._get_new_node(problem, new_problem, goal)
            if new_goal is None:
                raise UPProblemDefinitionError("Goal cannot be translated!")
            new_problem.add_goal(new_goal)

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