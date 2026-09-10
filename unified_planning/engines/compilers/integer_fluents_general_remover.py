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
"""This module defines the integer fluents general remover class."""
import math
import unified_planning as up
import unified_planning.engines as engines
from bidict import bidict
from ortools.sat.python import cp_model
from unified_planning.engines.compilers.utils import (
    add_cp_constraints, add_effect_bounds_constraints, solve_with_cp_sat,
    get_fluent_exps_in_expression, get_params_in_expression, evaluate_with_solution,
    remove_write_only_fluents, requires_csp, is_complex_goal, wrap_as_derived_fluent_axiom
)
from typing import Any, List, Iterable, Tuple
from unified_planning.model.expression import ListExpression
from unified_planning.model.operators import OperatorKind
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import (
    Problem, Action, ProblemKind, Effect, EffectKind, Object, FNode, InstantaneousAction, Axiom, Fluent,
    MinimizeActionCosts, AbstractProblem
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import get_fresh_name, replace_action, updated_minimize_action_costs
from typing import Optional, OrderedDict, Union
from functools import partial
from unified_planning.shortcuts import And, Or, Equals, Not, FALSE, UserType, TRUE, ObjectExp, DerivedBoolType, Iff
from typing import Dict

class IntegerFluentsGeneralRemover(engines.engine.Engine, CompilerMixin):
    """
    Compiler that removes bounded integer fluents from a planning problem.

    Converts integer fluents to object-typed fluents where objects represent numeric values (n0, n1, n2, ...).
    Integer arithmetic and comparisons are handled by enumerating possible value combinations.
    """

    def __init__(self, representation: str = 'object'):
        assert representation in ('object', 'binary'), \
            f"representation must be 'object' or 'binary', got {representation}"
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INTEGER_FLUENTS_GENERAL_REMOVING)
        self._conditions: Dict[FNode, str] = {}
        self.representation = representation

        # Object representation state
        self._number_objects: Dict[int, Object] = {}

        # Binary representation state
        if representation == 'binary':
            self.n_bits: OrderedDict = OrderedDict()
            self.offsets: Dict[str, int] = {}

    @property
    def name(self):
        return "iofgr" if self.representation == 'object' else "ilfgr"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        #supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
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
        return problem_kind <= IntegerFluentsGeneralRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INTEGER_FLUENTS_GENERAL_REMOVING

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
        try:
            return ObjectExp(problem.object(f'n{value}'))
        except UPProblemDefinitionError:
            raise UPProblemDefinitionError(
                f"Number object 'n{value}' not found. "
                f"Ensure _compute_number_range covers all needed values."
            )

    def _convert_value(self, value: int, n_bits: int, offset: int = 0) -> List[bool]:
        """Convert integer value to binary list of n_bits, applying offset."""
        shifted = value - offset
        assert shifted >= 0, f"Value {value} - offset {offset} = {shifted} < 0"
        return [b == '1' for b in bin(shifted)[2:].zfill(n_bits)]

    def _get_bit_fluents(self, new_problem: Problem, fluent_exp: FNode) -> List[FNode]:
        """Get the bit fluents representing an integer fluent."""
        fluent = fluent_exp.fluent()

        # Only integer fluents are transformed to bits
        if not fluent.type.is_int_type():
            return [fluent_exp]

        name = fluent.name
        if name not in self.n_bits:
            # Not a transformed fluent, return as-is
            return [fluent_exp]

        n_bits = self.n_bits[name]
        return [
            new_problem.fluent(f"{name}_{i}")(*fluent_exp.args)
            for i in range(n_bits)
        ]

    def _get_fluent_domain(
            self,
            fluent: Fluent,
            save: bool = False
    ) -> Iterable[int]:
        """Calculate and cache the number of bits required for an integer fluent."""
        if not fluent.type.is_int_type():
            return []

        inner_fluent = fluent.type
        assert inner_fluent.is_int_type(), f"Fluent {fluent.name} must be integer type. Arrays should be removed beforehand."

        # Calculate and save number of bits required to encode the integer domain
        if save:
            lb = inner_fluent.lower_bound
            ub = inner_fluent.upper_bound
            self.offsets[fluent.name] = lb  # <-- afegir
            n_values = ub - lb + 1
            if n_values <= 1:
                self.n_bits[fluent.name] = 1
            else:
                self.n_bits[fluent.name] = math.ceil(math.log2(n_values))

        return [fluent.name]

    def _set_fluent_bits(self, problem, fluent, k_args, new_value, n_bits, object_ref: Optional[Object] = None):
        """Set initial values for all bit fluents representing one encoded integer value."""
        for bit_index in range(n_bits):
            this_fluent = problem.fluent(f"{fluent.name}_{bit_index}")(*k_args, *(object_ref,) if object_ref is not None else ())
            problem.set_initial_value(this_fluent, new_value[bit_index])

    # ==================== NODE TRANSFORMATION ====================

    def _transform_node_object(
            self, old_problem: Problem, new_problem: Problem, node: FNode
    ) -> Union[Union[None, str, FNode], Any]:
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
        if node.is_object_exp() or node.is_constant() or node.is_parameter_exp() or node.is_variable_exp():
            return node

        # Check for arithmetic operations
        if node.node_type in self.ARITHMETIC_OPS:
            raise UPProblemDefinitionError(
                f"Arithmetic operation {self.ARITHMETIC_OPS[node.node_type]} "
                f"not supported as external expression"
            )

        # Recursively transform children
        new_args = []
        for arg in node.args:
            transformed = self._transform_node_object(old_problem, new_problem, arg)
            if transformed is None:
                return None
            new_args.append(transformed)

        # Preserve payload (variables) for Exists/Forall nodes
        if node.is_exists() or node.is_forall():
            return em.create_node(node.node_type, tuple(new_args), tuple(node.variables())).simplify()
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _get_new_fluent(
            self,
            new_problem: "up.model.AbstractProblem",
            node: "up.model.fnode.FNode",
    ) -> List["up.model.fnode.FNode"]:
        """Return the bit-fluent expansion for an integer fluent expression."""
        assert node.is_fluent_exp()

        fluent = node.fluent()

        # Only integer fluents are transformed to bits
        if not fluent.type.is_int_type():
            return [node]

        name = fluent.name
        if name not in self.n_bits:
            # Not a transformed fluent, return as-is
            return [node]

        n_bits = self.n_bits[name]

        return [
            new_problem.fluent(f"{name}_{i}")(*node.args)
            for i in range(n_bits)
        ]


    def _convert_fluent_and_value(
            self,
            new_problem: AbstractProblem,
            fluent: FNode,
            value: FNode,
    ) -> Tuple[List[FNode], List[FNode]]:
        """Convert a fluent/value pair into aligned bit-level representations."""
        n_bits = self.n_bits[fluent.fluent().name]
        new_fluents = self._get_new_fluent(new_problem, fluent)
        if value.is_fluent_exp():
            new_values = self._get_new_fluent(new_problem, value)
        else:
            assert value.is_constant(), "Value must be a constant!"
            new_values = self._convert_value(value.constant_value(), n_bits, self.offsets.get(fluent.fluent().name, 0))
        return new_fluents, new_values


    def _get_new_expression(
            self,
            new_problem: AbstractProblem,
            node: FNode
    ) -> FNode:
        """
        Transform expressions over encoded fluents into equivalent Boolean formulas.
        """
        # Handle equality on integers
        if node.node_type == OperatorKind.EQUALS:
            left, right = node.arg(0), node.arg(1)

            # Check if one side is an integer fluent
            if left.is_fluent_exp() and left.fluent().type.is_int_type():
                fluent, value = left, right
            elif right.is_fluent_exp() and right.fluent().type.is_int_type():
                fluent, value = right, left
            else:
                # Not an integer equality, just recurse
                new_args = [self._get_new_expression(new_problem, arg) for arg in node.args]
                em = new_problem.environment.expression_manager
                # Preserve payload (variables) for Exists/Forall nodes
                if node.is_exists() or node.is_forall():
                    return em.create_node(node.node_type, tuple(new_args), tuple(node.variables())).simplify()
                return em.create_node(node.node_type, tuple(new_args)).simplify()

            # Get bit fluents and values
            new_fluents, new_values = self._convert_fluent_and_value(new_problem, fluent, value)

            # All bits must match
            and_clauses = []
            for f, v in zip(new_fluents, new_values):
                if value.is_fluent_exp():
                    and_clauses.append(Iff(f, v))
                else:
                    and_clauses.append(f if v else Not(f))
            return And(and_clauses) if len(and_clauses) > 1 else and_clauses[0]

        # For non-comparison nodes: recurse and transform recursively
        if node.is_constant() or node.is_parameter_exp() or node.is_timing_exp():
            return node
        elif node.is_fluent_exp():
            return node
        elif node.args:
            new_args = [self._get_new_expression(new_problem, arg) for arg in node.args]
            em = new_problem.environment.expression_manager
            # Preserve payload (variables) for Exists/Forall nodes
            if node.is_exists() or node.is_forall():
                return em.create_node(node.node_type, tuple(new_args), tuple(node.variables())).simplify()
            return em.create_node(node.node_type, tuple(new_args)).simplify()
        return node

    # ==================== ACTION TRANSFORMATION ====================

    def _transform_increase_decrease_effect(self, effect, problem, new_problem):
        """Convert increase/decrease effects to conditional assignments.

        For each valid current value of the fluent, creates a conditional effect
        that sets the fluent to (current ± delta). Only iterates over values
        that keep the result within bounds.
        """
        fluent = effect.fluent.fluent()
        lb, ub = fluent.type.lower_bound, fluent.type.upper_bound

        # Get delta (as int if constant)
        try:
            delta = effect.value.constant_value()
        except:
            delta = None

        # Compute valid range for current value
        if delta is None:
            valid_range = range(lb, ub + 1)
        elif effect.is_increase():
            # current + delta must be in [lb, ub]  =>  current in [lb, ub - delta]
            valid_range = range(lb, ub - delta + 1)
        else:  # decrease
            # current - delta must be in [lb, ub]  =>  current in [lb + delta, ub]
            valid_range = range(lb + delta, ub + 1)

        # Representation-specific setup
        if self.representation == 'object':
            new_fluent = new_problem.fluent(fluent.name)(*effect.fluent.args)
            transformed_condition = (
                self._transform_node_object(problem, new_problem, effect.condition)
                if not effect.condition.is_true() else TRUE()
            )
        else:  # binary
            name_fluent = fluent.name.split('[')[0]
            n_bits = self.n_bits[name_fluent]
            offset = self.offsets.get(name_fluent, 0)
            new_fluents = self._get_new_fluent(new_problem, effect.fluent)

        result = []
        for i in valid_range:
            if delta is None:
                continue
            next_val = i + delta if effect.is_increase() else i - delta

            if self.representation == 'object':
                old_obj = self._get_number_object(new_problem, i)
                new_obj = self._get_number_object(new_problem, next_val)
                value_cond = Equals(new_fluent, old_obj)
                full_condition = (
                    And(value_cond, transformed_condition).simplify()
                    if not transformed_condition.is_true() else value_cond.simplify()
                )
                result.append(Effect(
                    new_fluent, new_obj, full_condition,
                    EffectKind.ASSIGN, effect.forall
                ))
            else:  # binary
                current_bits = self._convert_value(i, n_bits, offset)
                next_bits = self._convert_value(next_val, n_bits, offset)

                cond_clauses = [
                    f if bit_val else Not(f)
                    for f, bit_val in zip(new_fluents, current_bits)
                ]
                value_condition = (
                    And(cond_clauses) if len(cond_clauses) > 1 else cond_clauses[0]
                )
                full_condition = (
                    And(value_condition, effect.condition).simplify()
                    if effect.condition != TRUE() else value_condition
                )

                for f, next_bit in zip(new_fluents, next_bits):
                    result.append(Effect(
                        f, TRUE() if next_bit else FALSE(),
                        full_condition, EffectKind.ASSIGN, effect.forall
                    ))

        return result

    def _create_precondition_from_variable(
            self,
            fnode: FNode,
            value: int,
            new_problem: Problem
    ) -> Optional[FNode]:
        """
        Create a precondition from a variable and its value.
        Handles int fluents, UserType fluents, bool fluents, and parameters.
        """
        if fnode.is_fluent_exp():
            fluent = fnode.fluent()
            new_fluent = new_problem.fluent(fluent.name)(*fnode.args)
            if fluent.type.is_int_type():
                num_obj = self._get_number_object(new_problem, value)
                return Equals(new_fluent, num_obj)
            elif fluent.type.is_user_type():
                obj = self._get_object_from_index(fluent.type, value)
                if obj:
                    return Equals(new_fluent, ObjectExp(obj))
            elif fluent.type.is_bool_type() or fluent.type.is_derived_bool_type():
                if value == 1:
                    return new_fluent
                else:
                    return Not(new_fluent)

        elif fnode.is_parameter_exp():
            param = fnode.parameter()
            assert param.type.is_user_type(), "param type is not UserType"
            obj = self._get_object_from_index(param.type, value)
            if obj:
                return Equals(fnode, ObjectExp(obj))

        return None

    def _create_multiple_actions(self, old_action, problem, new_problem, params, solutions, variables,
                                 dependent_effects=None, independent_effects=None):
        """Create one grounded action per CP-SAT solution.

        Each solution fixes some variables; those become preconditions of the
        resulting action. Dependent effects use the solution values;
        independent effects are transformed directly.
        """
        dependent_effects = dependent_effects if dependent_effects is not None else old_action.effects
        independent_effects = independent_effects or []

        # Precompute fluent strings used in preconditions
        prec_fluent_strs = set()
        for prec in old_action.preconditions:
            for f in get_fluent_exps_in_expression(prec):
                prec_fluent_strs.add(str(f))

        # Fluent strings modified by dependent effects (that shouldn't appear as preconditions)
        modified_fluent_strs = {
            str(effect.fluent) for effect in dependent_effects
            if str(effect.fluent) not in prec_fluent_strs
               and not effect.is_increase() and not effect.is_decrease()
               and effect.condition.is_true()
        }

        new_actions = []
        for idx, solution in enumerate(solutions):
            action_name = (old_action.name if len(solutions) == 1 else f"{old_action.name}_d{idx}")
            new_action = InstantaneousAction(
                action_name, _parameters=params, _env=problem.environment
            )

            # Preconditions fixed by CP-SAT solution
            object_solution_conds = []
            for fnode, var in variables.items():
                var_str = str(fnode)
                if var_str not in solution:
                    continue
                # Object: skip if modified unconditionally
                if self.representation == 'object' and var_str in modified_fluent_strs:
                    continue

                # Binary: only add preconditions for fluents that actually appear in the action's original preconditions
                if self.representation == 'binary' and var_str not in prec_fluent_strs:
                    continue

                value = solution[var_str]
                if self.representation == 'object':
                    cond = self._create_precondition_from_variable(fnode, value, new_problem)
                    if cond:
                        object_solution_conds.append(cond)
                else:  # binary
                    self._add_binary_solution_preconditions(new_action, fnode, value, new_problem)

            # Object: group into single And precondition (preserves original behavior)
            if object_solution_conds:
                new_action.add_precondition(
                    And(object_solution_conds) if len(object_solution_conds) > 1
                    else object_solution_conds[0]
                )

            # Dependent effects (use solution values)
            self._add_effects_for_solution(
                new_action, problem, new_problem, solution, dependent_effects
            )

            # Independent effects (transformed directly)
            for effect in independent_effects:
                self._add_independent_effect(
                    new_action, effect, problem, new_problem, solution, old_action
                )

            new_actions.append(new_action)

        return new_actions

    def _add_binary_solution_preconditions(self, new_action, fnode, value, new_problem):
        """For binary representation: add preconditions asserting fnode has the given value.

        Handles: integer fluents (bit-level), user-type fluents (Equals), and boolean fluents.
        """
        if not fnode.is_fluent_exp():
            return
        fluent = fnode.fluent()

        if fluent.type.is_int_type():
            name_fluent = fluent.name
            if name_fluent in self.n_bits:
                n_bits = self.n_bits[name_fluent]
                value_bits = self._convert_value(value, n_bits, self.offsets.get(name_fluent, 0))
                new_fluents = self._get_bit_fluents(new_problem, fnode)
                for f, bit_val in zip(new_fluents, value_bits):
                    new_action.add_precondition(f if bit_val else Not(f))
        elif fluent.type.is_user_type():
            # User-type fluent: use Equals with corresponding object
            obj = self._get_object_from_index(fluent.type, value)
            if obj is not None:
                new_action.add_precondition(Equals(fnode, ObjectExp(obj)))
        elif fluent.type.is_bool_type() or fnode.fluent().type.is_derived_bool_type():
            # Boolean fluent: direct or negated
            if value == 1:
                new_action.add_precondition(fnode)
            else:
                new_action.add_precondition(Not(fnode))
        else:
            raise UPProblemDefinitionError(
                f"Cannot generate solution precondition for fluent {fnode} of type {fluent.type}"
            )

    def _add_binary_int_effect(self, new_action, effect_fluent, effect_value,
                               effect_forall, base_cond, new_problem):
        """Add a binary integer effect (f := g or f := c) with bit-level handling.

        - f := g (fluent copy): emits conditional effects (one positive, one negative per bit)
        - f := c (constant or expression): emits direct bit assignments via _convert_fluent_and_value

        Assumes effect_fluent is an integer fluent expression.
        """
        # Detect f := g (fluent copy)
        if (effect_value.is_fluent_exp()
                and effect_value.fluent().type.is_int_type()):
            f_name = effect_fluent.fluent().name
            g_name = effect_value.fluent().name
            n_bits = self.n_bits[f_name]

            f_bits = [new_problem.fluent(f"{f_name}_{i}")(*effect_fluent.args)
                      for i in range(n_bits)]
            g_bits = [new_problem.fluent(f"{g_name}_{i}")(*effect_value.args)
                      for i in range(n_bits)]

            for f_bit, g_bit in zip(f_bits, g_bits):
                pos_cond = And(base_cond, g_bit).simplify() if base_cond != TRUE() else g_bit
                new_action.add_effect(f_bit, TRUE(), pos_cond, effect_forall)
                neg_cond = And(base_cond, Not(g_bit)).simplify() if base_cond != TRUE() else Not(g_bit)
                new_action.add_effect(f_bit, FALSE(), neg_cond, effect_forall)
        else:
            # f := constant/expression: direct bit assignment
            new_fluents, new_values = self._convert_fluent_and_value(
                new_problem, effect_fluent, effect_value
            )
            for f, v in zip(new_fluents, new_values):
                if isinstance(v, FNode):
                    v_fnode = v
                else:
                    v_fnode = TRUE() if v else FALSE()
                new_action.add_effect(f, v_fnode, base_cond, effect_forall)

    def _add_independent_effect(self, new_action, effect, problem, new_problem, solution, old_action):
        """Add a single independent effect, representation-specific."""
        # Increase/decrease
        if effect.is_increase() or effect.is_decrease():
            if requires_csp(effect.condition):
                raise NotImplementedError(
                    f"Independent increase/decrease with arithmetic condition not supported: "
                    f"{effect} in action {old_action.name}"
                )
            # Both object and binary: expand via _transform_increase_decrease_effect
            for new_eff in self._transform_increase_decrease_effect(effect, problem, new_problem):
                new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
            return

        # Binary integer effect (f := g fluent copy OR f := c constant/expression)
        if (self.representation == 'binary'
                and effect.fluent.type.is_int_type()):

            if effect.forall:
                # Forall variables don't go to CP translator
                base_cond = self._transform_node_object(problem, new_problem, effect.condition) or TRUE()
            elif effect.condition != TRUE() and requires_csp(effect.condition):
                base_cond = self._expand_condition_with_cp(problem, new_problem, effect.condition, solution)
            else:
                base_cond = self._transform_node_object(problem, new_problem, effect.condition) or TRUE()

            self._add_binary_int_effect(
                new_action, effect.fluent, effect.value,
                effect.forall, base_cond, new_problem
            )
            return

        # Non-integer effect: transform fluent and value
        if self.representation == 'object':
            new_fluent = self._transform_node_object(problem, new_problem, effect.fluent)
            new_value = self._transform_node_object(problem, new_problem, effect.value)
        else:  # binary
            new_fluent = self._get_new_expression(new_problem, effect.fluent)
            new_value = self._get_new_expression(new_problem, effect.value)

        if not new_fluent or not new_value:
            return

        # Handle condition
        if effect.forall:
            new_cond = self._transform_node_object(
                problem, new_problem, effect.condition
            ) or TRUE()
            new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)

        elif effect.condition != TRUE() and requires_csp(effect.condition):
            expansions = self._expand_condition_with_cp(
                problem, new_problem, effect.condition, solution
            )
            new_action.add_effect(new_fluent, new_value, expansions, effect.forall)

        else:
            new_cond = self._transform_node_object(
                problem, new_problem, effect.condition
            ) or TRUE()
            new_action.add_effect(new_fluent, new_value, new_cond, effect.forall)

    def _expand_condition_with_cp(self, problem, new_problem, condition, solution):
        """Expand a condition into a disjunction of concrete satisfying assignments via CP-SAT.

        Solves the CP-SAT constraints derived from `condition`, given a partial
        `solution` (which fixes some variables). Returns a disjunction of clauses,
        one per satisfying assignment of the remaining variables.
        """
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()
        result_var = add_cp_constraints(
            problem, condition, variables, cp_model_obj, self._object_to_index
        )

        # Fix variables to the given partial solution
        for fnode, var in list(variables.items()):
            if str(fnode) in solution:
                cp_model_obj.Add(var == solution[str(fnode)])

        cp_model_obj.Add(result_var == 1)
        true_solutions = solve_with_cp_sat(variables, cp_model_obj) or []

        unknown_vars = {
            str(fnode): fnode for fnode, var in variables.items()
            if str(fnode) not in solution
        }

        clauses = []
        for sol in true_solutions:
            sol_conds = []
            for var_str, fnode in unknown_vars.items():
                if var_str not in sol:
                    continue
                cond = self._condition_for_value(fnode, sol[var_str], new_problem)
                if cond is not None:
                    sol_conds.append(cond)
            if sol_conds:
                clauses.append(
                    And(sol_conds).simplify() if len(sol_conds) > 1 else sol_conds[0]
                )

        if not clauses:
            return FALSE()

        return Or(clauses).simplify() if len(clauses) > 1 else clauses[0]

    def _condition_for_value(self, fnode, value, new_problem):
        """Build the condition asserting that fnode has the given value.

        Representation-specific:
        - object: uses Number objects and Equals (via _create_precondition_from_variable)
        - binary: uses bit-level conditions for integers, Equals for user-types
        """
        if self.representation == 'object':
            return self._create_precondition_from_variable(fnode, value, new_problem)

        # binary
        if not fnode.is_fluent_exp():
            return None

        fluent = fnode.fluent()

        # Integer fluent: bit-level condition
        if fluent.type.is_int_type():
            if fluent.name not in self.n_bits:
                return None
            n_bits = self.n_bits[fluent.name]
            value_bits = self._convert_value(value, n_bits, self.offsets.get(fluent.name, 0))
            bit_fluents = self._get_bit_fluents(new_problem, fnode)
            bit_conds = [f if b else Not(f) for f, b in zip(bit_fluents, value_bits)]
            return And(bit_conds) if len(bit_conds) > 1 else bit_conds[0]

        # User-type fluent: Equals with the object at index `value`
        if fluent.type.is_user_type():
            obj = self._get_object_from_index(fluent.type, value)
            if obj is not None:
                return Equals(fnode, ObjectExp(obj))
            return None

        # Boolean fluent
        if fluent.type.is_bool_type():
            return fnode if value == 1 else Not(fnode)

        return None

    def _add_effects_for_solution(self, new_action, problem, new_problem, solution, old_effects):
        """Add effects to a new action, given a CP-SAT solution.

        Iterates over old_effects, evaluates each condition with the solution,
        and adds the corresponding representation-specific effects.
        """
        for old_effect in old_effects:
            # Evaluate condition
            if old_effect.condition.is_true():
                new_condition = TRUE()
            else:
                new_condition = evaluate_with_solution(new_problem, old_effect.condition, solution)
                if new_condition == FALSE():
                    continue

            needs_cp_expansion = not new_condition.is_true() and requires_csp(new_condition)

            # ========== Increase/Decrease ==========
            if old_effect.is_increase() or old_effect.is_decrease():
                fluent = old_effect.fluent.fluent()

                try:
                    delta = old_effect.value.constant_value()
                except:
                    # Non-constant delta: fall back to full expansion
                    for new_eff in self._transform_increase_decrease_effect(old_effect, problem, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                cur_val = solution.get(str(old_effect.fluent))
                if cur_val is None:
                    for new_eff in self._transform_increase_decrease_effect(old_effect, problem, new_problem):
                        new_action.add_effect(new_eff.fluent, new_eff.value, new_eff.condition, new_eff.forall)
                    continue

                next_val = (cur_val + delta) if old_effect.is_increase() else (cur_val - delta)

                if self.representation == 'object':
                    new_fluent = new_problem.fluent(fluent.name)(*old_effect.fluent.args)
                    new_obj = self._get_number_object(new_problem, next_val)
                    cond = self._maybe_expand_cond(new_condition, needs_cp_expansion, problem, new_problem, solution)
                    new_action.add_effect(new_fluent, new_obj, cond, old_effect.forall)
                else:  # binary
                    # NOTE: binary originally did NOT expand CSP conditions here — kept for parity
                    n_bits = self.n_bits[fluent.name]
                    next_bits = self._convert_value(next_val, n_bits, self.offsets.get(fluent.name, 0))
                    bit_fluents = self._get_bit_fluents(new_problem, old_effect.fluent)
                    for f, bit_val in zip(bit_fluents, next_bits):
                        new_action.add_effect(
                            f, TRUE() if bit_val else FALSE(), new_condition, old_effect.forall
                        )

            # ========== Integer assignment ==========
            elif old_effect.fluent.type.is_int_type():
                evaluated_val = evaluate_with_solution(new_problem, old_effect.value, solution)
                cond = self._maybe_expand_cond(new_condition, needs_cp_expansion, problem, new_problem, solution)

                if self.representation == 'object':
                    new_fluent = self._transform_node_object(problem, new_problem, old_effect.fluent)
                    new_value = self._transform_node_object(problem, new_problem, evaluated_val)
                    if not new_fluent or not new_value:
                        continue
                    new_action.add_effect(new_fluent, new_value, cond, old_effect.forall)
                else:  # binary
                    new_fluents, new_values = self._convert_fluent_and_value(
                        new_problem, old_effect.fluent, evaluated_val
                    )
                    for f, v in zip(new_fluents, new_values):
                        if isinstance(v, FNode):
                            v_fnode = v
                        else:
                            v_fnode = TRUE() if v else FALSE()
                        new_action.add_effect(f, v_fnode, cond, old_effect.forall)

            # ========== Non-integer assignment ==========
            else:
                # Evaluate value with the CP-SAT solution to propagate constants
                evaluated_val = evaluate_with_solution(new_problem, old_effect.value, solution)

                if self.representation == 'object':
                    new_fluent = self._transform_node_object(problem, new_problem, old_effect.fluent)
                    new_value = self._transform_node_object(problem, new_problem, evaluated_val)
                else:  # binary
                    new_fluent = self._get_new_expression(new_problem, old_effect.fluent)
                    new_value = self._get_new_expression(new_problem, evaluated_val)

                if not new_fluent or not new_value:
                    continue

                cond = self._maybe_expand_cond(new_condition, needs_cp_expansion, problem, new_problem, solution)
                if cond == FALSE():
                    continue
                new_action.add_effect(new_fluent, new_value, cond, old_effect.forall)

    def _maybe_expand_cond(self, new_condition, needs_cp_expansion, problem, new_problem, solution):
        """If the condition needs CP-SAT expansion, expand it; otherwise return as-is."""
        if needs_cp_expansion:
            return self._expand_condition_with_cp(problem, new_problem, new_condition, solution)
        return new_condition

    def _transform_action_integers(self, problem, new_problem, old_action):
        """Transform an action, dispatching internally based on self.representation.

        Handles both:
        - No-arithmetic case: direct expression transformation
        - Arithmetic case: CP-SAT expansion into multiple grounded actions
        """
        params = OrderedDict(((p.name, p.type) for p in old_action.parameters))

        # Helper for direct expression transformation (varies by representation)
        def transform_expr(expr):
            if self.representation == 'object':
                return self._transform_node_object(problem, new_problem, expr)
            else:  # binary
                return self._get_new_expression(new_problem, expr)

        # Classify effects: dependent (interact with preconditions) vs independent
        prec_vars = set()
        for prec in old_action.preconditions:
            for f in get_fluent_exps_in_expression(prec):
                prec_vars.add(str(f))
            # Add also the parameters that appear in preconditions
            for p in get_params_in_expression(prec):
                prec_vars.add(str(p))

        dependent_effects = []
        independent_effects = []

        for effect in old_action.effects:
            if effect.is_increase() or effect.is_decrease():
                effect_vars = get_fluent_exps_in_expression(effect.fluent)
                value_vars = get_fluent_exps_in_expression(effect.value)
                value_params = get_params_in_expression(effect.value)
                if (any(str(v) in prec_vars for v in effect_vars | value_vars) or
                        (any(str(p) in prec_vars for p in value_params))):
                    dependent_effects.append(effect)
                else:
                    independent_effects.append(effect)
            elif (self.representation == 'binary'
                  and effect.fluent.type.is_int_type()
                  and effect.value.is_fluent_exp()
                  and effect.value.fluent().type.is_int_type()):
                # Binary-specific: fluent copy always independent (handled via bit-level conditional effects)
                independent_effects.append(effect)
            else:
                # An effect containing arithmetic must go through CP-SAT
                if requires_csp(effect.value):
                    dependent_effects.append(effect)
                    continue
                # Otherwise, dependent iff shares fluent/parameter with preconditions
                value_vars = get_fluent_exps_in_expression(effect.value)
                value_params = get_params_in_expression(effect.value)
                if (any(str(v) in prec_vars for v in value_vars) or
                        any(str(p) in prec_vars for p in value_params)):
                    dependent_effects.append(effect)
                else:
                    independent_effects.append(effect)

        # All preconditions go through CP-SAT: this enables constant propagation
        # and consistent handling of boolean, integer, and Count expressions.
        cp_precs = list(old_action.preconditions)

        # Setup CP-SAT
        self._object_to_index = {}
        for ut in problem.user_types:
            objects = list(problem.objects(ut))
            for idx, obj in enumerate(objects):
                self._object_to_index[(ut, obj)] = idx
        self._index_to_object = {}
        variables = bidict({})
        cp_model_obj = cp_model.CpModel()

        if cp_precs:
            result_var = add_cp_constraints(
                problem, And(cp_precs), variables, cp_model_obj, self._object_to_index
            )
            cp_model_obj.Add(result_var == 1)

        # Bounds constraints (object: only if dependent_effects; binary: always)
        if self.representation == 'object':
            if dependent_effects:
                add_effect_bounds_constraints(
                    problem, variables, cp_model_obj, dependent_effects,
                    self._object_to_index, False
                )
        else:  # binary
            add_effect_bounds_constraints(
                problem, variables, cp_model_obj, dependent_effects,
                self._object_to_index, False
            )

        # Solve CP-SAT
        if self.representation == 'object' and not cp_precs and not dependent_effects:
            solutions = [{}]
        else:
            solutions = solve_with_cp_sat(variables, cp_model_obj)
            if not solutions:
                return []

        self._index_to_object = {
            (t, idx): obj for (t, obj), idx in self._object_to_index.items()
        }

        # Build actions from solutions (representation-specific)
        return self._create_multiple_actions(
            old_action, problem, new_problem, params, solutions, variables,
            dependent_effects=dependent_effects,
            independent_effects=independent_effects
        )

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Action]:
        """Transform all actions by grounding integer parameters into objects."""
        new_to_old = {}
        for old_action in problem.actions:
            new_actions = self._transform_action_integers(problem, new_problem, old_action)
            for new_action in new_actions:
                new_problem.add_action(new_action)
                new_to_old[new_action] = old_action
        return new_to_old

    # ==================== AXIOMS TRANSFORMATION ====================

    def _transform_axioms(self, problem: Problem, new_problem: Problem, new_to_old: Dict):
        """Transform axioms"""
        for axiom in problem.axioms:
            params = OrderedDict((p.name, p.type) for p in axiom.parameters)
            new_axiom_name = get_fresh_name(new_problem, axiom.name)
            new_axiom = Axiom(new_axiom_name, params, axiom.environment)

            skip_axiom = False
            new_axiom.set_head(axiom.head.fluent)
            for body in axiom.body:
                if self.representation == 'object':
                    new_body = self._transform_node_object(problem, new_problem, body)
                else:  # binary
                    new_body = self._get_new_expression(new_problem, body)

                if new_body is None:
                    skip_axiom = True
                    break
                new_axiom.add_body_condition(new_body)

            if skip_axiom:
                continue
            new_problem.add_axiom(new_axiom)
            new_to_old[new_axiom] = axiom

    # ==================== GOALS TRANSFORMATION ====================

    def _extract_objects(self, expr: FNode) -> set:
        """Recursively extract all Object instances appearing in an expression."""
        objects = set()
        if expr.is_object_exp():
            objects.add(expr.object())
        for arg in expr.args:
            objects.update(self._extract_objects(arg))
        return objects

    def _add_goal_as_axiom(self, problem, new_problem, goal_expr, i, arithmetic):
        self._object_to_index = {}

        if arithmetic:
            body = self._expand_condition_with_cp(problem, new_problem, goal_expr, {})
        else:
            if self.representation == 'object':
                body = self._transform_node_object(problem, new_problem, goal_expr)
            else:
                body = self._get_new_expression(new_problem, goal_expr)

        fluent_name = f"goal_{i}"
        goal_fluent_exp = wrap_as_derived_fluent_axiom(new_problem, body, fluent_name)
        new_problem.add_goal(goal_fluent_exp)

    def _transform_goals(self, problem: Problem, new_problem: Problem) -> None:
        """Transform goals: separate arithmetic and non-arithmetic."""
        csp_goals = []
        axiom_only_goals = []
        direct_goals = []
        goals = problem.goals
        if len(goals) == 1 and goals[0].is_and():
            goals = problem.goals[0].args

        for goal in goals:
            if requires_csp(goal):
                csp_goals.append(goal)
            elif is_complex_goal(goal):
                axiom_only_goals.append(goal)
            else:
                direct_goals.append(goal)

        # 1. Direct goals: translate and add directly
        for goal in direct_goals:
            if self.representation == 'object':
                translated_goal = self._transform_node_object(problem, new_problem, goal)
            else:  # binary
                translated_goal = self._get_new_expression(new_problem, goal)
            new_problem.add_goal(translated_goal)

        # 2. Axiom-only goals: wrap in axiom for structural simplification
        for i, goal in enumerate(axiom_only_goals):
            j = len(direct_goals) + i
            self._add_goal_as_axiom(problem, new_problem, goal, j, False)

        # 3. CSP goals: each becomes an axiom whose body is solved by CP-SAT
        for i, goal in enumerate(csp_goals):
            j = len(direct_goals) + len(axiom_only_goals) + i
            self._add_goal_as_axiom(problem, new_problem, goal, j, True)

    def _get_object_from_index(self, user_type, index):
        """
        Get object corresponding to an index for a UserType.

        Uses the internal index-to-object mapping created during CP-SAT constraint building.
        """
        if hasattr(self, '_index_to_object'):
            return self._index_to_object.get((user_type, index))
        return None

    def _transform_fluents_object(self, problem: Problem, new_problem: Problem):
        """
        Transform integer fluents to object-typed fluents with Number type.

        Each integer fluent becomes an object fluent where objects represent
        numeric values (n0, n1, n2, ...). Non-integer fluents are copied unchanged.
        """
        number_ut = UserType('Number')

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            if fluent.type.is_int_type():
                # Integer fluent -> Object fluent with Number type
                from unified_planning.model import Fluent
                new_fluent = Fluent(fluent.name, number_ut, fluent.signature, new_problem.environment)
                lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
                assert lb is not None and ub is not None

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

        #self._lt_fluent = setup_lt_predicate(new_problem)

    def _transform_fluents_binary(self, problem: Problem, new_problem: Problem):
        """
        Transform integer fluents into bit-level boolean fluents.

        Each integer fluent becomes n_bits boolean fluents (one per bit).
        Non-integer fluents are copied unchanged.
        """
        for fluent in problem.fluents:
            if fluent.type.is_int_type():
                # Calculate bits needed for this integer fluent
                self._get_fluent_domain(fluent, save=True)
                n_bits = self.n_bits[fluent.name]

                # Default initial values
                default_value = problem.fluents_defaults.get(fluent)
                if default_value is not None:
                    dv = default_value.constant_value()
                    lb = self.offsets.get(fluent.name, 0)
                    ub = fluent.type.upper_bound
                    if lb <= dv <= ub:
                        default_bits = self._convert_value(dv, n_bits, lb)
                    else:
                        default_bits = [False] * n_bits
                else:
                    default_bits = [False] * n_bits

                # Create bit fluents
                for i in range(n_bits):
                    bit_fluent = Fluent(f"{fluent.name}_{i}", _signature=fluent.signature, environment=new_problem.environment)
                    new_problem.add_fluent(bit_fluent, default_initial_value=default_bits[i])

                # Set initial values from the original problem
                for k, v in problem.explicit_initial_values.items():
                    if k.fluent() == fluent:
                        new_value = self._convert_value(v.constant_value(), n_bits, self.offsets.get(fluent.name, 0))
                        self._set_fluent_bits(new_problem, fluent, k.args, new_value, n_bits)
            else:
                # Non-integer fluent: copy as-is
                default_value = problem.fluents_defaults.get(fluent)
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for k, v in problem.explicit_initial_values.items():
                    if k.fluent() == fluent:
                        new_problem.set_initial_value(k, v)

    def _compute_needed_values(self, problem: Problem) -> set[int]:
        """Compute the set of integer values that actually need Number objects."""
        needed = set()

        for fluent in problem.fluents:
            if not fluent.type.is_int_type():
                continue
            lb, ub = fluent.type.lower_bound, fluent.type.upper_bound
            needed.update(range(lb, ub + 1))
        return needed

    def _extract_int_constants(self, expr: FNode) -> set[int]:
        found = set()
        if expr.is_int_constant():
            found.add(expr.constant_value())
        for arg in expr.args:
            found.update(self._extract_int_constants(arg))
        return found

    def _compile(
            self,
            problem: "up.model.AbstractProblem",
            compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)

        original_problem = problem
        cleaned_problem = remove_write_only_fluents(problem)

        # Mapping name between cleaned and original actions
        name_to_original = {a.name: a for a in original_problem.actions}

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_goals()
        new_problem.clear_axioms()
        new_problem.explicit_initial_values.clear()
        new_problem.clear_quality_metrics()

        if self.representation == 'object':
            # Create Number objects for all needed values
            needed_values = self._compute_needed_values(problem)
            ut_number = UserType('Number')
            for v in sorted(needed_values):
                new_problem.add_object(Object(f'n{v}', ut_number))

        # ========== Transform Fluents ==========
        if self.representation == 'object':
            self._transform_fluents_object(cleaned_problem, new_problem)
        else:  # binary
            self._transform_fluents_binary(cleaned_problem, new_problem)

        # ========== Transform Actions ==========
        new_to_old_cleaned = self._transform_actions(cleaned_problem, new_problem)

        # ========== Transform Axioms ==========
        self._transform_axioms(cleaned_problem, new_problem, new_to_old_cleaned)

        # Remap
        new_to_old = {}
        for new_action, cleaned_action in new_to_old_cleaned.items():
            original = name_to_original.get(cleaned_action.name, cleaned_action)
            new_to_old[new_action] = original

        # ========== Transform Goals ==========
        self._transform_goals(cleaned_problem, new_problem)

        # ========== Transform Quality Metrics ==========
        for metric in problem.quality_metrics:
            if metric.is_minimize_action_costs():
                updated = updated_minimize_action_costs(
                    metric,
                    new_to_old,
                    new_problem.environment
                )
                # Dummy goal-achievement actions have cost 0
                em = new_problem.environment.expression_manager
                new_costs = dict(updated.costs)
                for action in new_problem.actions:
                    if action.name == "achieve_all_compound_goals":
                        new_costs[action] = em.Int(0)
                new_problem.add_quality_metric(
                    MinimizeActionCosts(new_costs, default=updated.default, environment=new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(metric)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
