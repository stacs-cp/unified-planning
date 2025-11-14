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
"""This module defines the integers remover class."""
import bisect
import operator
import unified_planning as up
import unified_planning.engines as engines
from ortools.sat.python import cp_model
from bidict import bidict
from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, Action, ProblemKind, Variable, Effect, EffectKind, Object, FNode, Fluent, \
    InstantaneousAction
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action, updated_minimize_action_costs
from typing import Dict, Optional, Iterator, OrderedDict, Tuple, Union, List
from functools import partial
from unified_planning.shortcuts import And, Or, Equals, Not, FALSE, Iff, UserType, TRUE, ObjectExp, Int


class CPSolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collects all unique solutions from CP-SAT solver."""

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solutions = []
        self.__seen = set()  # To detect duplicates

    def on_solution_callback(self):
        solution = {str(v): self.value(v) for v in self.__variables}
        sol_tuple = tuple(sorted(solution.items()))

        if sol_tuple not in self.__seen:
            self.__seen.add(sol_tuple)
            self.__solutions.append(solution)

    @property
    def solutions(self) -> list[dict[str, int]]:
        return self.__solutions

class IntegersRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes bounded integers from a planning problem.

    Converts integer fluents to object-typed fluents where objects represent numeric values (n0, n1, n2, ...).
    Integer arithmetic and comparisons are handled by enumerating possible value combinations.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGERS_REMOVING)
        self._domains: Dict[str, Tuple[int, int]] = {}
        self._number_objects_cache: Dict[int, FNode] = {}

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

    # Operators that can appear inside arithmetic expressions
    ARITHMETIC_OPS = {
        OperatorKind.PLUS: 'plus',
        OperatorKind.MINUS: 'minus',
        OperatorKind.DIV: 'div',
        OperatorKind.TIMES: 'mult',
    }

    # ==================== METHODS ====================

    def _get_number_object(self, problem: Problem, value: int) -> FNode:
        """Get or create object representing numeric value (e.g., n5 for 5)."""
        if value in self._number_objects_cache:
            return self._number_objects_cache[value]

        new_object = Object(f'n{value}', UserType('Number'))
        problem.add_object(new_object)
        new_object_expression = ObjectExp(new_object)
        self._number_objects_cache[value] = new_object_expression
        return new_object_expression

    def _is_value_in_bounds(self, fluent_name: str, value: int) -> bool:
        """Check if a value is within the bounds of a fluent's domain."""
        if fluent_name not in self._domains:
            return True
        lb, ub = self._domains[fluent_name]
        return lb <= value <= ub

    def _has_arithmetic(self, node: FNode) -> bool:
        """Check if expression contains arithmetic operations."""
        if node.node_type in self.ARITHMETIC_OPS or node.is_le() or node.is_lt():
            return True
        return any(self._has_arithmetic(arg) for arg in node.args)

    def _find_integer_fluents(self, node: FNode) -> dict[str, list[int]]:
        """Extract all integer fluents and their domains from expression."""
        fluents = {}
        if node.is_fluent_exp():
            if not node.fluent().type.is_int_type():
                return fluents

            fluent_type = node.fluent().type
            fluents[node.fluent().name] = list(range(
                fluent_type.lower_bound,
                fluent_type.upper_bound + 1
            ))
            return fluents
        for arg in node.args:
            fluents.update(self._find_integer_fluents(arg))
        return fluents

    # ==================== CP-SAT Constraint Solving ====================

    def _add_cp_constraints(self, problem: Problem, node: FNode, variables: bidict, model: cp_model.CpModel):
        """Recursively build CP-SAT constraints from expression tree."""
        # Constants
        if node.is_constant():
            return model.new_constant(node.constant_value())

        # Fluents
        if node.is_fluent_exp():
            if node in variables:
                return variables[node]
            fluent = node.fluent()
            if fluent.type.is_int_type():
                var = model.new_int_var(
                    fluent.type.lower_bound,
                    fluent.type.upper_bound,
                    str(node)
                )
            else:
                var = model.new_bool_var(str(node))
            variables[node] = var
            return var

        # AND
        if node.is_and():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            and_var = model.new_bool_var(f"and_{id(node)}")
            model.add(sum(child_vars) == len(child_vars)).only_enforce_if(and_var)
            model.add(sum(child_vars) < len(child_vars)).only_enforce_if(~and_var)
            return and_var

        # OR
        if node.is_or():
            child_vars = [self._add_cp_constraints(problem, arg, variables, model) for arg in node.args]
            or_var = model.new_bool_var(f"or_{id(node)}")
            model.add(sum(child_vars) >= 1).only_enforce_if(or_var)
            model.add(sum(child_vars) == 0).only_enforce_if(~or_var)
            return or_var

        # NOT
        if node.is_not():
            inner = self._add_cp_constraints(problem, node.arg(0), variables, model)
            not_var = model.new_bool_var(f"not_{id(node)}")
            model.add(inner == 0).only_enforce_if(not_var)
            model.add(inner == 1).only_enforce_if(~not_var)
            return not_var

        # Equality
        if node.is_equals():
            # User types treated as variables
            if node.arg(0).type.is_user_type():
                if node in variables:
                    return variables[node]
                var = model.new_bool_var(str(node))
                variables[node] = var
                return var

            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)
            eq_var = model.new_bool_var(f"eq_{id(node)}")
            model.add(left == right).only_enforce_if(eq_var)
            model.add(left != right).only_enforce_if(~eq_var)
            return eq_var

        # Comparisons and arithmetic
        if node.is_lt() or node.is_le():
            left = self._add_cp_constraints(problem, node.arg(0), variables, model)
            right = self._add_cp_constraints(problem, node.arg(1), variables, model)
            var = model.new_bool_var(f"{'lt' if node.is_lt() else 'le'}_{id(node)}")
            if node.is_lt():
                model.add(left < right).only_enforce_if(var)
                model.add(left >= right).only_enforce_if(~var)
            else:
                model.add(left <= right).only_enforce_if(var)
                model.add(left > right).only_enforce_if(~var)
            return var
        if node.is_plus() or node.is_minus() or node.is_times():
            # Process all arguments
            processed_args = [
                self._add_cp_constraints(problem, arg, variables, model)
                for arg in node.args
            ]

            if node.is_plus():
                return sum(processed_args)
            elif node.is_minus():
                if len(processed_args) == 1:
                    return -processed_args[0]
                else:
                    return processed_args[0] - sum(processed_args[1:])
            else:
                result = processed_args[0]
                for arg in processed_args[1:]:
                    result = result * arg
                return result

        raise NotImplementedError(f"Node type {node.node_type} not implemented in CP-SAT")

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

    def _get_fluents(self, node: FNode):
        """Returns a list of unique fluents in an expression."""
        fluents = set()
        def visit(n):
            if n.is_fluent_exp():
                fluents.add(n)
            else:
                for a in n.args:
                    visit(a)
        visit(node)
        return list(fluents)

    def _simplify_solutions(self, variables: bidict, solutions: list[dict[str, int]]) -> list[dict[str, int]]:
        """
        Compact solutions by grouping those differing in few variables.
        Remove variables that take all possible domain values.
        """
        if not solutions:
            return []
        all_vars = list(solutions[0].keys())
        if len(all_vars) == 1:
            return solutions
        simplified = []
        used = set()

        # Try grouping by each variable
        for var_name in all_vars:
            groups = {}

            for i, sol in enumerate(solutions):
                if i in used:
                    continue

                # Key = all values except the varying variable
                key = tuple((k, v) for k, v in sorted(sol.items()) if k != var_name)
                groups.setdefault(key, []).append((i, sol[var_name]))

            # Compact groups with multiple values
            for key, indices_vals in groups.items():
                if len(indices_vals) <= 1:
                    continue

                # Mark as used
                for idx, _ in indices_vals:
                    used.add(idx)

                # Find the fluent node
                fnode = None
                for node, var in variables.items():
                    if str(node) == var_name:
                        fnode = node
                        break
                if not fnode:
                    continue

                compact = dict(key)
                values_set = {val for _, val in indices_vals}

                # Check if covers entire domain
                if fnode.fluent().type.is_int_type():
                    lb = fnode.fluent().type.lower_bound
                    ub = fnode.fluent().type.upper_bound
                    domain = set(range(lb, ub + 1))

                    # Only include if not entire domain
                    if values_set != domain:
                        compact[var_name] = values_set
                elif fnode.fluent().type.is_bool_type():
                    if values_set != {0, 1}:
                        compact[var_name] = values_set
                else:
                    compact[var_name] = values_set

                simplified.append(compact)

        # Add ungrouped solutions
        for i, sol in enumerate(solutions):
            if i not in used:
                simplified.append(sol)

        return simplified

    def _solve_with_cp_sat(
            self, old_problem: Problem, new_problem: Problem, node: FNode
    ) -> Optional[FNode]:
        """Use CP-SAT solver to enumerate valid value assignments."""
        #print("====bounds", bounds)
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        # Build constraints from bounds (fluents that appear in node)
        #node_fluents = self._get_fluents(node)
        #and_expr = []
        #for f, bounds in bounds.items():
        #    if f in node_fluents:
        #        if f.type.is_bool_type():
        #            or_expr = [Iff(f, b) for b in bounds]
        #        else:
        #            or_expr = [Equals(f, b) for b in bounds]
        #        and_expr.append(Or(*or_expr))
        # Build constraints from node
        result = self._add_cp_constraints(old_problem, node, variables, cp_model_obj)

        # Ensure constraint is satisfied
        if isinstance(result, cp_model.IntVar):
            cp_model_obj.add(result == 1)
        else:
            cp_model_obj.add(result)

        # Solve
        solver = cp_model.CpSolver()
        collector = CPSolutionCollector(list(variables.values()))
        solver.parameters.enumerate_all_solutions = True
        status = solver.solve(cp_model_obj, collector)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        solutions = collector.solutions

        # Here - take preconditions into account to simplify ---
        solutions = self._simplify_solutions(variables, solutions) # quan 1 variable esta sola i te tots els valors la borra

        var_str_to_fnode = {str(node_key): node_key for node_key in variables.keys()}

        # Convert solutions to formula
        or_clauses = []
        for solution in solutions:
            and_clauses = []

            for var_str, value in solution.items():
                # Find corresponding FNode
                fnode = var_str_to_fnode.get(var_str)
                if not fnode:
                    continue
                if fnode.is_fluent_exp():
                    fluent = fnode.fluent()
                    new_fluent = new_problem.fluent(fluent.name)(*fnode.args)

                    if fluent.type.is_int_type():
                        if isinstance(value, set):
                            # Multiple values: (f = v1) OR (f = v2) OR ...
                            lb, ub = self._domains[new_fluent.fluent().name]
                            or_eq = [Equals(new_fluent, self._get_number_object(new_problem, v))
                                     for v in value if lb <= v <= ub]
                            if or_eq:
                                and_clauses.append(Or(or_eq))
                        else:
                            if self._is_value_in_bounds(new_fluent.fluent().name, value):
                                and_clauses.append(Equals(new_fluent, self._get_number_object(new_problem, value)))
                    elif fluent.type.is_bool_type():
                        bool_val = (value == 1)
                        and_clauses.append(Iff(fnode, bool_val))
                else:
                    # Boolean variable
                    and_clauses.append(fnode if value == 1 else Not(fnode))
            if and_clauses:
                or_clauses.append(And(and_clauses))
        return Or(or_clauses).simplify() if or_clauses else None

    # ==================== NODE TRANSFORMATION ====================

    def _transform_node(
            self, old_problem: Problem, new_problem: Problem, node: FNode
    ) -> Optional[FNode]:
        """Transform expression node to use Number objects instead of integers."""
        em = new_problem.environment.expression_manager

        # Integer constants become Number objects
        if node.is_int_constant():
            return self._get_number_object(new_problem, node.constant_value())

        # Integer fluents
        if node.is_fluent_exp():
            if node.fluent().type.is_int_type():
                return new_problem.fluent(node.fluent().name)(*node.args)
            return node

        # Other terminals
        if node.is_object_exp() or node.is_constant() or node.is_parameter_exp():
            return node

        # Check for arithmetic operations
        if node.node_type in self.ARITHMETIC_OPS:
            raise UPProblemDefinitionError(
                f"Arithmetic operation {self.ARITHMETIC_OPS[node.node_type]} "
                f"not supported as external expression"
            )

        # Expressions with integers need CP-SAT solving
        if self._has_arithmetic(node):
            return self._solve_with_cp_sat(old_problem, new_problem, node)

        # Recursively transform children
        new_args = []
        for arg in node.args:
            transformed = self._transform_node(old_problem, new_problem, arg)
            if transformed is None:
                return None
            new_args.append(transformed)

        # Handle quantifiers
        # aixo passara mai?
        if node.is_exists() or node.is_forall():
            new_vars = [
                Variable(v.name, UserType('Number')) if v.type.is_int_type() else v
                for v in node.variables()
            ]
            return em.create_node(node.node_type, tuple(new_args), payload=tuple(new_vars)).simplify()

        return em.create_node(node.node_type, tuple(new_args)).simplify()

    # ==================== EFFECT TRANSFORMATION ====================

    def _transform_increase_decrease_effect(
            self,
            effect: Effect,
            old_problem: Problem,
            new_problem: Problem
    ) -> Iterator[Effect]:
        """Convert increase/decrease effects to conditional assignments."""
        fluent = effect.fluent.fluent()
        lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
        new_condition = self._transform_node(old_problem, new_problem, effect.condition)
        new_fluent = new_problem.fluent(fluent.name)(*effect.fluent.args)

        # Calculate the valid bounds
        try:
            int_value = effect.value.constant_value()
        except:
            int_value = effect.value

        if effect.is_increase():
            # Per increase: valor final = i + delta, per tant i ha d'estar en [lb, ub-delta]
            valid_range = range(max(lb, lb), min(ub - int_value, ub) + 1) if isinstance(int_value, int) else range(lb, ub + 1)
        else:
            # Per decrease: valor final = i - delta, per tant i ha d'estar en [lb+delta, ub]
            valid_range = range(max(lb + int_value, lb), min(ub, ub) + 1) if isinstance(int_value, int) else range(lb, ub + 1)

        returned = set()

        for i in valid_range:
            next_val = (i + int_value) if effect.is_increase() else (i - int_value)
            try:
                next_val_int = next_val.simplify().constant_value() if hasattr(next_val, 'simplify') else next_val
            except:
                continue

            old_obj = self._get_number_object(new_problem, i)
            new_obj = self._get_number_object(new_problem, next_val_int)
            new_effect = Effect(
                new_fluent,
                new_obj,
                And(Equals(new_fluent, old_obj), new_condition).simplify(),
                EffectKind.ASSIGN,
                effect.forall
            )
            if new_effect not in returned:
                yield new_effect
                returned.add(new_effect)

    def _transform_arithmetic_assignment(
            self,
            effect: Effect,
            old_problem: Problem,
            new_problem: Problem,
    ) -> Iterator[Effect]:
        """Handle assignments with arithmetic expressions by enumerating combinations."""
        fluent_domains = self._find_integer_fluents(effect.value)
        if not fluent_domains:
            return

        lb, ub = self._domains[effect.fluent.fluent().name] # rang del fluent al que assignem

        # Per cada valor possible del fluent d'assignació, trobar combinacions vàlides
        for v in range(lb, ub + 1):
            combinations = self._solve_with_cp_sat(old_problem, new_problem, Equals(effect.value, v))
            if combinations is None:
                continue
            # Create effects
            new_fluent = new_problem.fluent(effect.fluent.fluent().name)
            new_base_condition = self._transform_node(old_problem, new_problem, effect.condition)
            full_condition = And(new_base_condition, combinations).simplify()

            yield Effect(
                new_fluent(*effect.fluent.args),
                self._get_number_object(new_problem, v),
                full_condition,
                EffectKind.ASSIGN,
                effect.forall
            )

    # ==================== ACTION TRANSFORMATION ====================

    def _transform_action_integers(
            self, problem: Problem, new_problem: Problem, old_action: Action
    ) -> Union[Action, None]:
        """
        First get the bounds of the fluents
        Then change all integers in the action for their new user-type fluent.
        """
        #bounds = self._get_bounds(new_problem, And(*old_action.preconditions))

        params = OrderedDict(((p.name, p.type) for p in old_action.parameters))
        new_action = InstantaneousAction(old_action.name, _parameters=params, _env=problem.environment)
        # Transform preconditions
        for precondition in old_action.preconditions:
            new_precondition = self._transform_node(problem, new_problem, precondition)
            if new_precondition is None or new_precondition == FALSE():
                # Impossible action
                return None
            new_action.add_precondition(new_precondition)

        # Transform effects
        for effect in old_action.effects:
            if effect.is_increase() or effect.is_decrease():
                # Increase/decrease effects
                for new_effect in self._transform_increase_decrease_effect(effect, problem, new_problem):
                    new_action.add_effect(
                        new_effect.fluent, new_effect.value, new_effect.condition, new_effect.forall
                    )

            elif effect.value.node_type in self.ARITHMETIC_OPS:
                # Assignment with arithmetic
                effects_generated = False
                for new_effect in self._transform_arithmetic_assignment(effect, problem, new_problem):
                    new_action.add_effect(
                        new_effect.fluent,
                        new_effect.value,
                        new_effect.condition,
                        new_effect.forall
                    )
                    effects_generated = True

                if not effects_generated:
                    # No valid assignments -> skip action
                    return None

            else:
                # Check bounds for assignments
                if effect.fluent.fluent().type.is_int_type() and effect.value.is_int_constant():
                    value = effect.value.constant_value()
                    if not self._is_value_in_bounds(effect.fluent.fluent().name, value):
                        if effect.condition == TRUE():
                            return None
                        continue

                new_fluent = self._transform_node(problem, new_problem, effect.fluent)
                new_condition = self._transform_node(problem, new_problem, effect.condition)
                new_value = self._transform_node(problem, new_problem, effect.value)

                if new_value is None or new_value is None:
                    return None
                if new_condition not in [None, FALSE()]:
                    new_action.add_effect(new_fluent, new_value, new_condition, effect.forall)
        return new_action

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions by grounding integer parameters."""
        new_to_old = {}
        for action in problem.actions:
            new_action = self._transform_action_integers(problem, new_problem, action)
            if new_action is not None:
                new_problem.add_action(new_action)
                new_to_old[new_action] = action
        return new_to_old


    # ==================== FLUENT TRANSFORMATION ====================

    def _transform_fluents(self, problem: Problem, new_problem: Problem):
        """Transform integer fluents -> user-type fluents."""
        number_ut = UserType('Number')

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            if fluent.type.is_int_type():
                # Integer fluent -> Number-typed fluent
                new_fluent = Fluent(fluent.name, number_ut, fluent.signature, new_problem.environment)
                lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
                assert lb is not None and ub is not None
                self._domains[fluent.name] = (lb, ub)

                if default_value is not None:
                    default_obj = self._get_number_object(new_problem, default_value.constant_value())
                    new_problem.add_fluent(new_fluent, default_initial_value=default_obj)
                else:
                    new_problem.add_fluent(new_fluent)

                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        new_problem.set_initial_value(
                            new_problem.fluent(fluent.name)(*f.args),
                            self._get_number_object(new_problem, v.constant_value())
                        )
            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        new_problem.set_initial_value(f, v)

    def _compile(
            self,
            problem: "up.model.AbstractProblem",
            compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)
        self._number_objects_cache.clear()
        self._domains.clear()

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_axioms()
        new_problem.initial_values.clear()
        new_problem.clear_quality_metrics()

        # Transform components
        self._transform_fluents(problem, new_problem)
        new_to_old = self._transform_actions(problem, new_problem)

        # Transform Axioms
        #for axiom in problem.axioms:
        #    new_axiom = axiom.clone()
        #    new_axiom.name = get_fresh_name(new_problem, axiom.name)
        #    new_axiom.clear_preconditions()
        #    new_axiom.clear_effects()
        #    # Transform preconditions
        #    skip_axiom = False
        #    for precondition in axiom.preconditions:
        #        new_precondition = self._transform_node(problem, new_problem, precondition)
        #        if new_precondition is None or new_precondition == FALSE():
        #            skip_axiom = True
        #            break
        #        new_axiom.add_precondition(new_precondition)
        #    if skip_axiom:
        #        continue
#
        #    # Transform effects
        #    for effect in axiom.effects:
        #        new_fluent = self._transform_node(problem, new_problem, effect.fluent)
        #        new_condition = self._transform_node(problem, new_problem, effect.condition)
        #        new_value = self._transform_node(problem, new_problem, effect.value)
        #        if new_fluent is None or new_condition is None or new_value is None:
        #            skip_axiom = True
        #            break
        #        new_axiom.add_effect(new_fluent, new_value, new_condition, effect.forall)
        #    if not skip_axiom:
        #        new_to_old[new_axiom] = axiom
        #        new_problem.add_axiom(new_axiom)

        # ========== Transform Goals ==========
        for goal in problem.goals:
            new_goal = self._transform_node(problem, new_problem, goal)
            if new_goal is None:
                raise UPProblemDefinitionError(
                    f"Goal cannot be translated after integer removal: {goal}"
                )
            new_problem.add_goal(new_goal)

        # ========== Transform Quality Metrics ==========
        for metric in problem.quality_metrics:
            if metric.is_minimize_action_costs():
                new_problem.add_quality_metric(
                    updated_minimize_action_costs(
                        metric,
                        new_to_old,
                        new_problem.environment
                    )
                )
            else:
                new_problem.add_quality_metric(metric)

        return CompilerResult(
            new_problem,
            partial(replace_action, map=new_to_old),
            self.name
        )
