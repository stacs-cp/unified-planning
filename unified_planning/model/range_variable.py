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
"""
This module defines the RangeVariable class.
A RangeVariable has a name and a type.
"""


from typing import List, Optional, Union, FrozenSet

from unified_planning.environment import Environment, get_environment
from unified_planning.model import Parameter
from unified_planning.model.fnode import FNode
from unified_planning.model.operators import OperatorKind
import unified_planning
import unified_planning.model.walkers as walkers
import unified_planning.model.operators as op


class RangeVariable:
    """Represents a variable; a `RangeVariable` has a name and a type."""

    def __init__(
        self,
        name: str,
        initial: Union[int, Parameter],
        last: Union[int, Parameter],
        environment: Optional[Environment] = None,
    ):
        self._name = name
        self._initial = initial
        self._last = last
        self._env = get_environment(environment)
        if type(initial) == int:
            low = initial
        else:
            low = initial.type.lower_bound
        if type(last) == int:
            high = last
        else:
            high = last.type.upper_bound
        self._type_int = self._env.type_manager.IntType(low, high)

    def __repr__(self) -> str:
        return f"integer[{str(self.initial)}, {str(self.last)}] {self.name}"

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, RangeVariable):
            return (
                self._name == oth._name
                and self._initial == self._initial
                and self._last == self._last
                and self._type_int == self._type_int
                and self._env == oth._env
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash(self._name) + hash(self._type_int)

    @property
    def name(self) -> str:
        """Returns the `Variable` name."""
        return self._name

    @property
    def initial(self) -> Union[str, int]:
        """Returns the `Variable` `Initial`."""
        if type(self._initial) is Parameter:
            return self._initial.name
        else:
            return self._initial

    @property
    def last(self) -> Union[str, int]:
        """Returns the `Variable` `Last`."""
        if type(self._last) is Parameter:
            return self._last.name
        else:
            return self._last

    @property
    def type(self) -> "unified_planning.model.types.Type":
        """Returns the `Variable` `Type`."""
        return self._type_int

    @property
    def environment(self) -> "Environment":
        """Return the `RangeVariable` `Environment`."""
        return self._env

    #
    # Infix operators
    #

    def __add__(self, right):
        return self._env.expression_manager.Plus(self, right)

    def __radd__(self, left):
        return self._env.expression_manager.Plus(left, self)

    def __sub__(self, right):
        return self._env.expression_manager.Minus(self, right)

    def __rsub__(self, left):
        return self._env.expression_manager.Minus(left, self)

    def __mul__(self, right):
        return self._env.expression_manager.Times(self, right)

    def __rmul__(self, left):
        return self._env.expression_manager.Times(left, self)

    def __truediv__(self, right):
        return self._env.expression_manager.Div(self, right)

    def __rtruediv__(self, left):
        return self._env.expression_manager.Div(left, self)

    def __floordiv__(self, right):
        return self._env.expression_manager.Div(self, right)

    def __rfloordiv__(self, left):
        return self._env.expression_manager.Div(left, self)

    def __gt__(self, right):
        return self._env.expression_manager.GT(self, right)

    def __ge__(self, right):
        return self._env.expression_manager.GE(self, right)

    def __lt__(self, right):
        return self._env.expression_manager.LT(self, right)

    def __le__(self, right):
        return self._env.expression_manager.LE(self, right)

    def __pos__(self):
        return self._env.expression_manager.Plus(0, self)

    def __neg__(self):
        return self._env.expression_manager.Minus(0, self)

    def Equals(self, right):
        return self._env.expression_manager.Equals(self, right)

    def And(self, *other):
        return self._env.expression_manager.And(self, *other)

    def __and__(self, *other):
        return self._env.expression_manager.And(self, *other)

    def __rand__(self, *other):
        return self._env.expression_manager.And(*other, self)

    def Or(self, *other):
        return self._env.expression_manager.Or(self, *other)

    def __or__(self, *other):
        return self._env.expression_manager.Or(self, *other)

    def __ror__(self, *other):
        return self._env.expression_manager.Or(*other, self)

    def Not(self):
        return self._env.expression_manager.Not(self)

    def __invert__(self):
        return self._env.expression_manager.Not(self)

    def Xor(self, *other):
        em = self._env.expression_manager
        return em.And(em.Or(self, *other), em.Not(em.And(self, *other)))

    def __xor__(self, *other):
        em = self._env.expression_manager
        return em.And(em.Or(self, *other), em.Not(em.And(self, *other)))

    def __rxor__(self, other):
        em = self._env.expression_manager
        return em.And(em.Or(*other, self), em.Not(em.And(*other, self)))

    def Implies(self, right):
        return self._env.expression_manager.Implies(self, right)

    def Iff(self, right):
        return self._env.expression_manager.Iff(self, right)