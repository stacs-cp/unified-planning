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
"""This module defines the Integer Parameters in Actions, Arrays and Range Variables remover class."""
import re
from itertools import product

from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model.fnode import FNode
import unified_planning.engines as engines
from collections import OrderedDict
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import (
    Problem,
    InstantaneousAction,
    Action,
    ProblemKind,
    Fluent,
    MinimizeActionCosts,
    RangeVariable,
    Type,
    OperatorKind,
    Parameter,
    Object, Effect,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    lift_action_instance,
)
from typing import Dict, List, Optional, Tuple, OrderedDict, Union
from functools import partial

from unified_planning.shortcuts import Int, FALSE, TRUE, And, UserType

class IntParameterActionsRemover(engines.engine.Engine, CompilerMixin):
    """
    Removes integer parameters from actions by instantiating them.
    Also transforms array fluents into indexed fluents.

    Transforms:
    1. Array fluents -> indexed fluents with Index type (UserType)
    2. Integer action parameters -> grounded actions
    3. Range variables -> expanded quantifiers
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INT_PARAMETER_ACTIONS_REMOVING)
        self.domains: Dict[str, List[Tuple[int, ...]]] = {}

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
        supported_kind.set_fluents_type("DERIVED_FLUENTS")
        supported_kind.set_fluents_type("SET_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_conditions_kind("COUNTING")
        supported_kind.set_conditions_kind("MEMBERING")
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

    # ==================== FLUENT TRANSFORMATION ====================

    def _get_array_domain_and_type(self, fluent: Fluent) -> Tuple[List[int], Type]:
        """
        Extract domain and element type from array fluent.
        Stores valid positions in self.domains.
        """
        current_type = fluent.type
        domain_ranges = []
        dimensions = []

        while current_type.is_array_type():
            dimensions.append(current_type.size)
            domain_ranges.append(range(current_type.size))
            current_type = current_type.elements_type

        # All possible positions
        all_positions = list(product(*domain_ranges))

        # Filter out undefined positions
        if fluent.undefined_positions is not None:
            valid_positions = [
                pos for pos in all_positions
                if pos not in fluent.undefined_positions
            ]
        else:
            valid_positions = all_positions

        self.domains[fluent.name] = valid_positions
        return dimensions, current_type

    def _get_param_combinations(self, problem: Problem, signature):
        """Get all combinations of parameter values."""
        param_values = [problem.objects(param.type) for param in signature]
        return list(product(*param_values))

    def _add_array_as_indexed_fluent(self, problem, new_problem, fluent, default_value, index_ut):
        """
        Transform array fluent into indexed fluent.
        Example: board[3][3]: int → board(i1: Index, i2: Index): int with (i1, i2, i3): Index new objects
        """
        # Get domain and element type
        n_elements, element_type = self._get_array_domain_and_type(fluent)
        max_index = max(n_elements)

        # Add index objects if not present
        for i in range(max_index):
            if not new_problem.has_object(f'i{i}'):
                new_problem.add_object(Object(f'i{i}', index_ut))

        # Create new signature with Index parameters
        new_signature = [
            Parameter(f'i_{dim + 1}', index_ut)
            for dim in range(len(n_elements))
        ] + list(fluent.signature)

        new_fluent = Fluent(fluent.name, element_type, new_signature, fluent.environment)
        new_problem.add_fluent(new_fluent, default_initial_value=default_value)

        # Set initial values
        for f, v in problem.explicit_initial_values.items():
            fluent_name = f.fluent().name.split('[')[0]

            if f.fluent() == fluent or fluent_name == fluent.name:
                new_equalities = self._get_new_fluent_value(
                    new_problem, new_fluent, f, v
                )
                for nf, nv in new_equalities.items():
                    new_problem.set_initial_value(nf, nv)

    def _get_element_value(self, array_value: FNode, indices: Tuple[int, ...]) -> FNode:
        """Extract element value from nested array constant."""
        element = array_value
        for idx in indices:
            element = element.constant_value()[idx]
        return element

    def _get_index_object(self, problem: Problem, n: int) -> Object:
        """Get or create index object i{n}."""
        obj_name = f'i{n}'
        if not problem.has_object(obj_name):
            obj = Object(obj_name, UserType('Index'))
            problem.add_object(obj)
            return obj
        return problem.object(obj_name)

    def _get_new_fluent_value(self, new_problem: Problem, new_fluent: Fluent, f: FNode, v: FNode ) -> Dict[FNode, FNode]:
        """Convert array (partial and full) assignment to indexed fluent assignments."""
        # Extract pre-indices from fluent name (e.g., board[2][3] → (2, 3))
        pre_indices = tuple(int(i) for i in re.findall(r'\[([0-9]+)\]', f.fluent().name))

        new_equalities = {}
        old_params = list(f.args)

        if pre_indices:
            # Partial array assignment: board[2] = value
            new_params = [self._get_index_object(new_problem, i) for i in pre_indices] + old_params

            if not f.fluent().type.is_array_type():
                # Single element: board[2][3] = 5
                new_equalities[new_fluent(*new_params)] = v
            else:
                # Sub-array: board[2] = [1,2,3]
                post_indices = [
                    pos[len(pre_indices):]
                    for pos in self.domains[new_fluent.name]
                    if pos[:len(pre_indices)] == pre_indices
                ]

                for post_idx in post_indices:
                    full_params = new_params + [
                        self._get_index_object(new_problem, i)
                        for i in post_idx
                    ]
                    element_value = self._get_element_value(v, post_idx)
                    new_equalities[new_fluent(*full_params)] = element_value
        else:
            # Full array assignment: board = [[1,2],[3,4]]
            for pos in self.domains[new_fluent.name]:
                element_value = self._get_element_value(v, pos)
                params = [self._get_index_object(new_problem, i) for i in pos] + old_params
                new_equalities[new_fluent(*params)] = element_value

        return new_equalities

    def _transform_fluents(self, problem: Problem, new_problem: Problem):
        """Transform array fluents -> indexed fluents."""
        index_ut = UserType('Index')

        for fluent in problem.fluents:
            default_value = problem.fluents_defaults.get(fluent)

            if fluent.type.is_array_type():
                self._add_array_as_indexed_fluent(
                    problem, new_problem, fluent, default_value, index_ut
                )
            else:
                new_problem.add_fluent(fluent, default_initial_value=default_value)
                for f, v in problem.explicit_initial_values.items():
                    if f.fluent() == fluent:
                        new_problem.set_initial_value(fluent(*f.args), v)

    # ==================== EXPRESSION TRANSFORMATION ====================

    def _extract_array_indices(
            self,
            new_problem: Problem,
            fluent_name: str,
            int_params: Optional[Dict[str, int]] = None,
            instantiations: Optional[Tuple[int, ...]] = None
    ) -> Union[List[Object], None]:
        """Extract and evaluate array indices from fluent name."""
        pattern = r'\[(.*?)\]'
        indices = []
        for access_expr in re.findall(pattern, fluent_name):
            if access_expr.isdigit():
                # Constant index
                index_value = int(access_expr)
            else:
                # Expression with parameters
                evaluated_expr = access_expr
                for param_name, param_idx in int_params.items():
                    if param_name in evaluated_expr:
                        evaluated_expr = evaluated_expr.replace(
                            param_name,
                            str(instantiations[param_idx])
                        )
                try:
                    index_value = eval(evaluated_expr)
                except:
                    return None
            indices.append(index_value)
        valid_accesses = self.domains[fluent_name.split('[')[0]]
        if tuple(indices) in valid_accesses:
            object_indices = [self._get_index_object(new_problem, i) for i in indices]
            return object_indices
        else:
            return None

    def _transform_fluent_exp(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Dict[str, int],
            instantiations: Tuple[int, ...]
    ) -> Union[FNode, None]:
        """Transform fluent expression, handling arrays."""
        # Transform arguments first
        new_args = []
        for arg in node.args:
            transformed = self._transform_expression(old_problem, new_problem, arg, int_params, instantiations)
            if transformed is None:
                return None
            new_args.append(transformed)

        fluent_base_name = node.fluent().name.split('[')[0]
        old_fluent = old_problem.fluent(fluent_base_name)

        # Array fluent: extract indices from name
        if old_fluent.type.is_array_type():
            assert new_args == [], "Array fluents with parameters not supported"

            new_fluent = new_problem.fluent(fluent_base_name)
            index_params = self._extract_array_indices(
                new_problem, node.fluent().name, int_params, instantiations
            )
            if index_params is None:
                return None
            return new_fluent(*index_params)

        # Regular fluent
        return node.fluent()(*new_args)

    def _extract_variables(self, variables: List) -> Tuple[Tuple, Dict[str, Tuple[int, int]]]:
        """Separate regular variables from range variables."""
        regular_vars = []
        range_vars = {}
        for var in variables:
            if isinstance(var, RangeVariable):
                range_vars[var.name] = (var.initial, var.last)
            else:
                regular_vars.append(var)
        return tuple(regular_vars), range_vars

    def _update_range_vars(
            self, range_vars: Dict[str, Tuple[int, int]], int_params: Dict[str, int], instantiations: Tuple[int, ...]
    ) -> Dict[str, Tuple[int, int]]:
        """Evaluate range expressions with current parameter values."""
        updated = {}
        for var_name, (initial, last) in range_vars.items():
            # Convert to strings for replacement
            initial_str = str(initial)
            last_str = str(last)
            # Replace parameters with their values
            for param_name, param_idx in int_params.items():
                param_value = str(instantiations[param_idx])
                initial_str = initial_str.replace(param_name, param_value)
                last_str = last_str.replace(param_name, param_value)

            # Evaluate expressions
            updated[var_name] = (eval(initial_str), eval(last_str))
        return updated

    def _get_range_instantiations(self, ranges: Dict[str, Tuple[int, int]]) -> List[Tuple[int, ...]]:
        """Generate all combinations of values for range variables."""
        if not ranges:
            return [()]
        range_iterables = [
            range(start, end + 1)
            for start, end in ranges.values()
        ]
        return list(product(*range_iterables))

    def _transform_quantifier(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Dict[str, int],
            instantiations: Tuple[int, ...],
            bounds: Dict
    ) -> FNode:
        """Transform forall/exists by expanding range variables."""
        regular_vars, range_vars = self._extract_variables(node.variables())

        if not range_vars:
            # No range variables: keep quantifier
            new_args = [
                self._transform_expression(old_problem, new_problem, arg, int_params, instantiations, bounds)
                for arg in node.args
            ]
            new_args = self._handle_none_args(node.node_type, new_args)
            if new_args is None:
                return None
            em = old_problem.environment.expression_manager
            return em.create_node(node.node_type, tuple(new_args), regular_vars).simplify()

        # Update ranges with current parameter values
        updated_ranges = self._update_range_vars(range_vars, int_params, instantiations)

        # ho he girat d'ordre

        # Expand range variables
        expanded_int_params = int_params.copy()
        for var_name in range_vars.keys():
            expanded_int_params[var_name] = len(expanded_int_params)

        # Get all instantiations for range variables
        range_instantiations = self._get_range_instantiations(updated_ranges)
        # Expand quantifier body for each instantiation
        expanded_args = []
        for range_inst in range_instantiations:
            full_inst = instantiations + range_inst
            for arg in node.args:
                transformed = self._transform_expression(
                    old_problem, new_problem, arg, expanded_int_params, full_inst, bounds
                )
                if transformed is not None:
                    expanded_args.append(transformed)
        if not expanded_args:
            return None

        # Combine with appropriate operator
        em = new_problem.environment.expression_manager
        new_op = OperatorKind.AND if node.is_forall() else OperatorKind.OR
        return em.create_node(new_op, tuple(expanded_args)).simplify()

    def _transform_array_comparison(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Dict[str, int],
            instantiations: Tuple[int, ...]
    ) -> FNode:
        """
        Transform comparison with array fluent.
        Example: board == [[1,2],[3,4]] → multiple comparisons
        """
        assert all(arg.type.is_array_type() for arg in node.args), \
            "All arguments must be array type"

        # Identify fluent and value
        fluent_arg, value_arg = (
            (node.arg(0), node.arg(1))
            if node.arg(1).is_constant()
            else (node.arg(1), node.arg(0))
        )
        fluent_base_name = fluent_arg.fluent().name.split('[')[0]
        new_fluent = new_problem.fluent(fluent_base_name)

        # Generate comparison for each position
        em = new_problem.environment.expression_manager
        comparisons = []
        for pos, val in self._get_new_fluent_value(new_problem, new_fluent, fluent_arg, value_arg).items():
            comparison = em.create_node(node.node_type, (pos, val))
            comparisons.append(comparison.simplify())
        return And(comparisons)

    def _handle_none_args(self, node_type: OperatorKind, args: List) -> Union[List[FNode], None]:
        """
        Handle undefined (None) values in arguments based on operator semantics.
        - OR/EXISTS/COUNT: ignore None
        - IMPLIES: special handling
        - AND/FORALL/...: None propagates
        """
        if None not in args:
            return args
        if node_type in {OperatorKind.OR, OperatorKind.COUNT, OperatorKind.EXISTS}:
            return [arg for arg in args if arg is not None]
        elif node_type == OperatorKind.IMPLIES:
            if args[1] is None and args[0] is not None:
                return [args[0], FALSE()]
            return args[1]
        else:
            return None

    def _transform_generic(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Dict[str, int],
            instantiations: Tuple[int, ...],
            bounds: Dict
    ) -> Union[FNode, None]:
        """Generic recursive transformation."""
        em = old_problem.environment.expression_manager

        new_args = [
            self._transform_expression(old_problem, new_problem, arg, int_params, instantiations, bounds)
            for arg in node.args
        ]
        new_args = self._handle_none_args(node.node_type, new_args)
        if new_args is None or new_args == []:
            return None
        return em.create_node(node.node_type, tuple(new_args)).simplify()

    def _transform_expression(
            self,
            old_problem: Problem,
            new_problem: Problem,
            node: FNode,
            int_params: Optional[Dict[str, int]] = None,
            instantiations: Optional[Tuple[int, ...]] = None,
            bounds: Optional[Dict[FNode, List]] = None
    ) -> Union[FNode, None]:
        """
        Transform expression by substituting integer parameters.
        :param int_params: Map from parameter name to its position
        :param instantiations: Tuple of integer values for parameters
        :param bounds: Bounds analysis for optimization
        """
        if int_params is None:
            int_params = {}
        if instantiations is None:
            instantiations = ()
        if bounds is None:
            bounds = {}

        # Base cases
        if node.is_constant() or node.is_variable_exp() or node.is_timing_exp():
            return node

        if node.is_parameter_exp():
            # Replace integer parameters with constants
            param_name = node.parameter().name
            if param_name in int_params:
                param_index = int_params[param_name]
                return Int(instantiations[param_index])
            return node
        if node.is_fluent_exp():
            return self._transform_fluent_exp(old_problem, new_problem, node, int_params, instantiations)
        if node.is_forall() or node.is_exists():
            return self._transform_quantifier(old_problem, new_problem, node, int_params, instantiations, bounds)
        # Special case: array fluent comparisons
        if node.arg(0).type.is_array_type():
            return self._transform_array_comparison(old_problem, new_problem, node, int_params, instantiations)
        # Generic recursive transformation
        return self._transform_generic(old_problem, new_problem, node, int_params, instantiations, bounds)

    # ==================== ACTION TRANSFORMATION ====================

    def _transform_actions(self, problem: Problem, new_problem: Problem) -> Dict[Action, Tuple[Action, Tuple[int, ...]]]:
        """Transform all actions by grounding integer parameters."""
        trace_back_map = {}
        for action in problem.actions:
            instantiated_actions = self._instantiate_action(problem, new_problem, action)
            for new_action, instantiation in instantiated_actions:
                new_problem.add_action(new_action)
                trace_back_map[new_action] = (action, instantiation)
        return trace_back_map

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

    def _add_single_effect(
            self,
            action: Action,
            effect_type: str,
            fluent: FNode,
            value: FNode,
            condition: FNode,
            original_condition: FNode,
            forall: Tuple
    ) -> bool:
        """Add single effect to action. Returns False if action should be pruned."""
        # Check for invalid values (out of bounds for integers)
        if fluent is not None and value is not None:
            if fluent.type.is_int_type():
                lower = fluent.type.lower_bound
                upper = fluent.type.upper_bound
                # Check if value is provably out of bounds
                if (value < lower).simplify() == TRUE() or (value > upper).simplify() == TRUE():
                    value = None
        # Handle unconditional effects
        if original_condition == TRUE():
            if fluent is None or value is None:
                # Invalid unconditional effect -> prune action
                return False
            self._add_effect_to_action( action, effect_type, fluent, value, condition, forall)
        # Handle conditional effects
        else:
            if condition not in (None, FALSE()) and fluent is not None and value is not None:
                self._add_effect_to_action(action, effect_type, fluent, value, condition, forall)
        return True

    def _add_instantiated_effect(
            self,
            problem: Problem,
            new_problem: Problem,
            effect: Effect,
            new_action: Action,
            int_param_map: Dict[str, int],
            instantiation: Tuple[int, ...]
    ) -> bool:
        """Add single effect to action. Returns False if action should be pruned."""
        # Determine effect type
        if effect.is_increase():
            effect_type = 'increase'
        elif effect.is_decrease():
            effect_type = 'decrease'
        else:
            effect_type = 'none'

        # Handle forall effects
        regular_forall, range_vars = self._extract_variables(effect.forall)
        if not range_vars:
            # No range variables in forall
            new_fluent = self._transform_expression(problem, new_problem, effect.fluent, int_param_map, instantiation)
            new_value = self._transform_expression(problem, new_problem, effect.value, int_param_map, instantiation)
            new_condition = self._transform_expression(problem, new_problem, effect.condition, int_param_map, instantiation)

            return self._add_single_effect(
                new_action, effect_type, new_fluent, new_value, new_condition, effect.condition, regular_forall
            )

        # ho he girat
        updated_ranges = self._update_range_vars(range_vars, int_param_map, instantiation)

        # Expand forall with range variables
        expanded_int_params = int_param_map.copy()
        for var_name in range_vars.keys():
            expanded_int_params[var_name] = len(expanded_int_params)

        range_insts = self._get_range_instantiations(updated_ranges)
        for range_inst in range_insts:
            full_inst = instantiation + range_inst
            new_fluent = self._transform_expression(problem, new_problem, effect.fluent, expanded_int_params, full_inst)
            new_value = self._transform_expression(problem, new_problem, effect.value, expanded_int_params, full_inst)
            new_condition = self._transform_expression(problem, new_problem, effect.condition, expanded_int_params, full_inst)
            success = self._add_single_effect(
                new_action, effect_type, new_fluent, new_value, new_condition, effect.condition, regular_forall
            )
            if not success:
                return False
        return True

    def _add_instantiated_effects(
            self,
            problem: Problem,
            new_problem: Problem,
            old_action: Action,
            new_action: Action,
            int_param_map: Dict[str, int],
            instantiation: Tuple[int, ...]
    ) -> bool:
        """Add all effects to instantiated action. Returns True if any effects added."""
        for effect in old_action.effects:
            success = self._add_instantiated_effect(
                problem, new_problem, effect, new_action, int_param_map, instantiation
            )
            if not success:
                return False
        return len(new_action.effects) > 0

    def _create_instantiated_action(
            self,
            problem: Problem,
            new_problem: Problem,
            action: Action,
            regular_params: OrderedDict,
            int_param_map: Dict[str, int],
            instantiation: Tuple[int, ...]
    ) -> Union[Action, None]:
        """Create single instantiated action."""
        # Generate unique name
        action_name = get_fresh_name(new_problem, action.name, list(map(str, instantiation)))

        # Create action with only regular parameters
        assert isinstance(action, InstantaneousAction), "Only InstantaneousActions are supported"
        new_action = InstantaneousAction(action_name, regular_params, action.environment)
        # Transform preconditions
        for precondition in action.preconditions:
            new_precondition = self._transform_expression(
                problem, new_problem, precondition, int_param_map, instantiation
            )
            if new_precondition is None or new_precondition == FALSE():
                # Impossible action
                return None
            new_action.add_precondition(new_precondition)

        # Transform effects
        has_valid_effects = self._add_instantiated_effects(
            problem, new_problem, action, new_action, int_param_map, instantiation
        )
        if not has_valid_effects:
            return None
        return new_action

    def _instantiate_action(
            self, problem: Problem, new_problem: Problem, action: Action
    ) -> List[Tuple[Action, Tuple[int, ...]]]:
        """
        Create all instantiations of an action for integer parameters.
        Returns list of (new_action, instantiation) pairs.
        """
        # Separate regular and integer parameters
        regular_params = OrderedDict()
        int_param_map = {}
        int_param_ranges = []

        for param in action.parameters:
            if param.type.is_user_type():
                regular_params[param.name] = param.type
            elif param.type.is_int_type():
                int_param_map[param.name] = len(int_param_map)
                int_param_ranges.append((param.type.lower_bound, param.type.upper_bound))
            else:
                raise UPProblemDefinitionError(f"Parameter type {param.type} not supported")

        # Generate all instantiations
        instantiations = self._get_range_instantiations(
            {f"p{i}": r for i, r in enumerate(int_param_ranges)}
        ) if int_param_ranges else [()]

        result = []
        for inst in instantiations:
            new_action = self._create_instantiated_action(
                problem, new_problem, action, regular_params, int_param_map, inst
            )
            if new_action is not None:
                result.append((new_action, inst))
        return result

    # ==================== GOAL TRANSFORMATION ====================

    def _transform_goals(self, problem: Problem, new_problem: Problem):
        """Transform all goals."""
        for goal in problem.goals:
            new_goal = self._transform_expression(problem, new_problem, goal)
            if new_goal is not None:
                new_problem.add_goal(new_goal)

    # ==================== AXIOMS TRANSFORMATION ====================

    def _transform_axioms(self, problem: Problem, new_problem: Problem, trace_back_map: Dict):
        """Transform axioms"""
        for axiom in problem.axioms:
            # Check for integer parameters
            for param in axiom.parameters:
                if param.type.is_int_type():
                    raise NotImplementedError(
                        "Integer parameters in axioms are not supported!"
                    )

            # Clone and transform
            new_axiom = axiom.clone()
            new_axiom.name = get_fresh_name(new_problem, new_axiom.name)
            new_axiom.clear_preconditions()
            new_axiom.clear_effects()

            for precondition in axiom.preconditions:
                new_precondition = self._transform_expression(problem, new_problem, precondition)
                if new_precondition is not None:
                    new_axiom.add_precondition(new_precondition)

            for effect in axiom.effects:
                new_fluent = self._transform_expression(problem, new_problem, effect.fluent)
                new_value = self._transform_expression(problem, new_problem, effect.value)
                new_condition = self._transform_expression(problem, new_problem, effect.condition)

                if not new_condition.is_false() and new_fluent is not None:
                    if effect.is_increase():
                        new_axiom.add_increase_effect(new_fluent, new_value, new_condition, effect.forall)
                    elif effect.is_decrease():
                        new_axiom.add_decrease_effect(new_fluent, new_value, new_condition, effect.forall)
                    else:
                        new_axiom.add_effect(new_fluent, new_value, new_condition, effect.forall)

            new_problem.add_axiom(new_axiom)
            trace_back_map[new_axiom] = axiom

    # ==================== QUALITY METRICS TRANSFORMATION ====================

    def _transform_quality_metrics(
            self, problem: Problem, new_problem: Problem, trace_back_map: Dict[Action, Tuple[Action, Tuple[int, ...]]]
    ):
        """Transform quality metrics, handling action costs specially."""
        for qm in problem.quality_metrics:
            if qm.is_minimize_sequential_plan_length() or qm.is_minimize_makespan():
                new_problem.add_quality_metric(qm)
            elif qm.is_minimize_action_costs():
                assert isinstance(qm, MinimizeActionCosts)
                new_costs = self._transform_action_costs(qm, trace_back_map)
                new_problem.add_quality_metric(
                    MinimizeActionCosts(new_costs, environment=new_problem.environment)
                )
            else:
                new_problem.add_quality_metric(qm)

    def _transform_action_costs(
            self,
            qm: MinimizeActionCosts,
            trace_back_map: Dict[Action, Tuple[Action, Tuple[int, ...]]]
    ) -> Dict[Action, "up.model.Expression"]:
        """Transform action costs, substituting integer parameters."""
        new_costs = {}
        for new_action, (old_action, instantiation) in trace_back_map.items():
            if old_action is None:
                continue
            old_cost = qm.get_action_cost(old_action)
            # If cost is a parameter, substitute its value
            if old_cost.is_parameter_exp():
                # Find which integer parameter this is
                param_idx = 0
                for param in old_action.parameters:
                    if param.name == str(old_cost):
                        break
                    if param.type.is_int_type():
                        param_idx += 1
                new_costs[new_action] = Int(instantiation[param_idx])
            else:
                new_costs[new_action] = old_cost
        return new_costs

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        """Main compilation"""
        assert isinstance(problem, Problem)

        # Create new problem
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_fluents()
        new_problem.clear_actions()
        new_problem.clear_axioms()
        new_problem.clear_goals()
        new_problem.clear_quality_metrics()
        new_problem.initial_values.clear()

        self.domains.clear()

        # Transform components
        self._transform_fluents(problem, new_problem)
        trace_back_map = self._transform_actions(problem, new_problem)
        self._transform_goals(problem, new_problem)
        self._transform_quality_metrics(problem, new_problem, trace_back_map)
        self._transform_axioms(problem, new_problem, trace_back_map)

        return CompilerResult(
            new_problem,
            partial(lift_action_instance, map=trace_back_map),
            self.name,
        )