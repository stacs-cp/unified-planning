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
    Effect,
)
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.model.walkers import ExpressionQuantifiersRemover
from unified_planning.engines.compilers.utils import (
    get_fresh_name,
    replace_action,
    updated_minimize_action_costs,
)
from typing import Dict, List, Optional, Tuple
from functools import partial


class ArraysRemover(engines.engine.Engine, CompilerMixin):
    """
    Arrays remover class: ...
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.ARRAYS_REMOVING)

    @property
    def name(self):
        return "qurm"

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
        return problem_kind <= ArraysRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.ARRAYS_REMOVING

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
        # new_problem.clear_timed_goals()
        # new_problem.clear_goals()
        # new_problem.clear_quality_metrics()
        new_problem.clear_fluents()
        new_problem.clear_actions()
        # transformar fluents i valors inicials
        for fluent in problem.fluents:
            this_fluent = fluent.type
            domain = []
            while this_fluent.is_array_type():
                domain_in = []
                for i in range(0, this_fluent.n_elements):
                    domain_in.append(i)
                domain.append(domain_in)
                this_fluent = this_fluent.elements_type

            combinations = list(product(*domain))
            for combination in combinations:
                print(combination)
                new_type = this_fluent.elements_type
                new_name = fluent.name + f'[{combination}]'
                print(new_name,new_type)


                new_name = fluent.name + f'[{i}]'
                new_default_value = list(problem.fluents_defaults.get(fluent).constant_value())[i]
                while new_type.is_array_type():
                    print("nt: ", new_type)
                    new_type = new_type.elements_type
                    for j in range(new_type.n_elements):
                        new_name = new_name + f'[{j}]'
                        new_default_value = list(new_default_value)[i]

                new_problem.add_fluent(model.Fluent(new_name, new_type, fluent.signature, fluent.environment),
                                                   default_initial_value=0)

        print("old: ", problem.fluents)
        print("new: ", new_problem.fluents)
        print(new_problem.fluents_defaults)

        # transformar valors inicials fluents
        for fluent_default in problem.fluents_defaults:
            pass

        # transformar accions
        for action in problem.actions:
            if isinstance(action, InstantaneousAction):
                pass

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
