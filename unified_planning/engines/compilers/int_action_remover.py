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
"""This module defines the int action remover class."""


import unified_planning as up
import unified_planning.engines as engines
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model import Problem, ProblemKind, Fluent, FNode, Action
from unified_planning.model.fluent import get_all_fluent_exp
from unified_planning.model.types import _RealType, _IntType
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.model.walkers import FluentsSubstituter
from unified_planning.engines.compilers.utils import (
    add_invariant_condition_apply_function_to_problem_expressions,
    replace_action,
)
from typing import List, Dict, OrderedDict, Optional, Union, cast
from functools import partial


class IntActionRemover(engines.engine.Engine, CompilerMixin):
    """
    Bounded types remover class: this class offers the capability
    to transform a :class:`~unified_planning.model.Problem` with Bounded :class:`Types <unified_planning.model.Type>`
    into a `Problem` without bounded `Types` (only IntType and RealType can be bounded).
    This capability is offered by the :meth:`~unified_planning.engines.compilers.BoundedTypesRemover.compile`
    method, that returns a :class:`~unified_planning.engines.CompilerResult` in which the :meth:`problem <unified_planning.engines.CompilerResult.problem>` field
    is the compiled Problem.

    This is done by changing the type of the fluents to unbounded types, and adding to every action's condition and
    every goal of the problem the artificial condition that emulates the typing bound.

    For example, if we have a fluent `F` of type `int[0, 5]`, the added condition would be `0 <= F <= 5`.

    This `Compiler` supports only the the `BOUNDED_TYPES_REMOVING` :class:`~unified_planning.engines.CompilationKind`.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.INT_ACTION_REMOVING)

    @property
    def name(self):
        return "iarac"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        # canviar!!!!!!!
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= IntActionRemover.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.INT_ACTION_REMOVING

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        return problem_kind.clone()

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
        tm = env.type_manager
        new_problem = Problem(f"{problem.name}_{self.name}", env)
        new_problem.add_objects(problem.all_objects)

        int_type = tm.IntType()
        real_type = tm.RealType()
        conditions: List[FNode] = []

        new_fluents: Dict[Fluent, Fluent] = {}
        for old_fluent in problem.fluents:
            print(old_fluent)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
