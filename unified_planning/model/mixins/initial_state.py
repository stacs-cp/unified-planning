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
from collections import Counter
from itertools import product
from typing import Union, Dict, Any, List, Optional

import unified_planning as up
from unified_planning.exceptions import (
    UPTypeError,
    UPExpressionDefinitionError,
)
from unified_planning.model.fluent import get_all_fluent_exp
from unified_planning.model.mixins import ObjectsSetMixin, FluentsSetMixin
from unified_planning.model.types import domain_size


class InitialStateMixin:
    """A Problem mixin that allows setting and infering the value of fluents in the initial state."""

    def __init__(
        self,
        object_set: ObjectsSetMixin,
        fluent_set: FluentsSetMixin,
        environment: "up.environment.Environment",
    ):
        self._object_set = object_set
        self._fluent_set = fluent_set
        self._env = environment
        self._initial_value: Dict["up.model.fnode.FNode", "up.model.fnode.FNode"] = {}

    def set_initial_value(
        self,
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: Union[
            "up.model.expression.NumericExpression",
            "up.model.fluent.Fluent",
            "up.model.object.Object",
            bool,
        ],
    ):
        """
        Sets the initial value for the given `Fluent`. The given `Fluent` must be grounded, therefore if
        it's :func:`arity <unified_planning.model.Fluent.arity>` is `> 0`, the `fluent` parameter must be
        an `FNode` and the method :func:`~unified_planning.model.FNode.is_fluent_exp` must return `True`.

        :param fluent: The grounded `Fluent` of which the initial value must be set.
        :param value: The `value` assigned in the initial state to the given `fluent`.
        """
        if fluent.type.is_array_type() and type(value) is list:
            value = [value]
        fluent_exp, value_exp = self._env.expression_manager.auto_promote(fluent, value)
        assert fluent_exp.is_fluent_exp(), "fluent field must be a fluent"
        if fluent.type.is_derived_bool_type():
            raise UPTypeError("You cannot set the initial value of a derived fluent!")
        if not fluent_exp.type.is_compatible(value_exp.type):
            raise UPTypeError("Initial value assignment has not compatible types!")
        self._initial_value[fluent_exp] = value_exp

    def create_multidimensional_array(self, dimensions, elements_default):
        # Crear un array multidimensional usando recursión
        if len(dimensions) == 1:
            return [elements_default] * dimensions[0]
        return [self.create_multidimensional_array(dimensions[1:], elements_default) for _ in range(dimensions[0])]

    def initial_value(
        self, fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"]
    ) -> Optional["up.model.fnode.FNode"]:
        """
        Retrieves the initial value assigned to the given `fluent`.

        :param fluent: The target `fluent` of which the `value` in the initial state must be retrieved.
        :return: The `value` expression assigned to the given `fluent` in the initial state.
        """
        (fluent_exp,) = self._env.expression_manager.auto_promote(fluent)
        for a in fluent_exp.args:
            if not a.is_constant():
                raise UPExpressionDefinitionError(
                    f"Impossible to return the initial value of a fluent expression with no constant arguments: {fluent_exp}."
                )
        if fluent_exp in self._initial_value:
            return self._initial_value[fluent_exp]
        elif fluent_exp.fluent().type.is_derived_bool_type():
            return self._env.expression_manager.FALSE()
        elif fluent_exp.fluent() in self._fluent_set.fluents_defaults:
            if fluent.type.is_array_type():
                elements_default = self._fluent_set.fluents_defaults[fluent_exp.fluent()]
                this_fluent = fluent.type
                dimensions = []
                while this_fluent.is_array_type():
                    dimensions.append(this_fluent.size)
                    this_fluent = this_fluent.elements_type
                new_default = self.create_multidimensional_array(dimensions, elements_default)
                (new_expression,) = self._env.expression_manager.auto_promote([new_default])
                return new_expression
            else:
                return self._fluent_set.fluents_defaults[fluent_exp.fluent()]
        else:
            return None

    @property
    def initial_values(self) -> Dict["up.model.fnode.FNode", "up.model.fnode.FNode"]:
        """
        Gets the initial value of all the grounded fluents present in the `Problem`.

        IMPORTANT NOTE: this property does a lot of computation, so it should be called as
        seldom as possible.
        """
        res = self._initial_value
        for f in self._fluent_set.fluents:
            for f_exp in get_all_fluent_exp(self._object_set, f):
                value = self.initial_value(f_exp)
                if not f.type.is_derived_bool_type() and value is not None:
                    res[f_exp] = value
        return res

    @property
    def explicit_initial_values(
        self,
    ) -> Dict["up.model.fnode.FNode", "up.model.fnode.FNode"]:
        """
        Returns the problem's defined initial values; those are only the initial values set with the
        :func:`~unified_planning.model.Problem.set_initial_value` method.

        IMPORTANT NOTE: For all the initial values of the problem use :func:`initial_values <unified_planning.model.Problem.initial_values>`.
        """
        return self._initial_value

    def __eq__(self, oth: Any) -> bool:
        """Returns true iff the two initial states are equivalent."""
        if not isinstance(oth, InitialStateMixin):
            return False
        oth_initial_values = oth.initial_values
        initial_values = self.initial_values
        if len(initial_values) != len(oth_initial_values):
            return False
        for fluent, value in initial_values.items():
            oth_value = oth_initial_values.get(fluent, None)
            if oth_value is None:
                return False
            elif value != oth_value:
                return False
        return True

    def __hash__(self):
        return sum(map(hash, self.initial_values.items()))

    def _clone_to(self, other: "InitialStateMixin"):
        other._initial_value = self._initial_value.copy()

    def _fluents_with_undefined_values(self) -> List["up.model.fluent.Fluent"]:
        """Returns a list of fluents that have at least one undefined value in the initial state"""
        undef_fluents = []
        # gather a count of all explicit initial values for each fluent
        inits = Counter([x.fluent() for x in self.explicit_initial_values])
        for fluent in self._fluent_set.fluents:
            if fluent in self._fluent_set.fluents_defaults:
                continue  # fluent has a default values and thus can not be undefined

            # count the number of state variables associated to the fluent
            ground_size = 1
            for p in fluent.signature:
                ds = domain_size(self._object_set, p.type)
                ground_size *= ds

            assert (
                inits.get(fluent, 0) <= ground_size
            ), "Invariant broken: more initial values than state variables"
            if ground_size != inits.get(fluent, 0):
                undef_fluents.append(
                    fluent
                )  # at least one state variable has no initial values
        return undef_fluents
