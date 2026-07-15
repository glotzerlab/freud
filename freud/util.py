# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt

import freud.box
from freud._typing import ArrayLike, ShapeLike

C = TypeVar("C", bound="_Compute")
R = TypeVar("R")
RequirementFlag: TypeAlias = Literal[
    "C",
    "C_CONTIGUOUS",
    "CONTIGUOUS",
    "F",
    "F_CONTIGUOUS",
    "FORTRAN",
    "A",
    "ALIGNED",
    "W",
    "WRITEABLE",
    "O",
    "OWNDATA",
]


def _as_concrete_shape(shape: ShapeLike | None) -> tuple[int, ...] | None:
    if shape is None:
        return None

    concrete_shape = []
    for dim in shape:
        if dim is None:
            msg = "shape must be fully specified when initializing a new array."
            raise ValueError(msg)
        concrete_shape.append(dim)
    return tuple(concrete_shape)


class _Compute:
    r"""Parent class for all compute classes in freud.

    The primary purpose of this class is to prevent access of uncomputed
    values. This is accomplished by maintaining a boolean flag to track whether
    the compute method in a class has been called and decorating class
    properties that rely on compute having been called.

    To use this class, one would write, for example,

    .. code-block:: python
        class Cluster(_Compute):

            def compute(...)
                ...

            @_Compute._computed_property
            def cluster_idx(self):
                return ...

    Attributes:
        _called_compute (bool):
            Flag representing whether the compute method has been called.
    """

    def __init__(self) -> None:
        self._called_compute = False

    def __getattribute__(self, attr: str):
        """Compute methods set a flag to indicate that quantities have been
        computed. Compute must be called before plotting."""
        attribute = object.__getattribute__(self, attr)
        if attr == "compute":
            # Set the attribute *after* computing. This enables
            # self._called_compute to be used in the compute method itself.
            compute = attribute

            @wraps(compute)
            def compute_wrapper(*args: object, **kwargs: object) -> object:
                return_value = compute(*args, **kwargs)
                self._called_compute = True
                return return_value

            return compute_wrapper
        if attr == "plot":
            if not self._called_compute:
                msg = "The compute method must be called before calling plot."
                raise AttributeError(msg)
        return attribute

    @staticmethod
    def _computed_property(prop: Callable[[C], R]) -> property:
        r"""Decorator that makes a class method to be a property with limited access.

        Args:
            prop (callable): The property function.

        Returns:
            Decorator decorating appropriate property method.
        """

        @wraps(prop)
        def wrapper(self: C) -> R:
            if not self._called_compute:
                msg = "Property not computed. Call compute first."
                raise AttributeError(msg)
            return prop(self)

        return property(wrapper)

    def __str__(self) -> str:
        return repr(self)


@overload
def _convert_array(
    array: ArrayLike | None,
    shape: ShapeLike | None = None,
    dtype: type[np.float32] | np.dtype[np.float32] = np.float32,
    requirements: Sequence[RequirementFlag] = ("C",),
    allow_copy: bool = True,
) -> npt.NDArray[np.float32]: ...


@overload
def _convert_array(
    array: ArrayLike | None,
    shape: ShapeLike | None = None,
    dtype: type[np.float64] | np.dtype[np.float64] = ...,
    requirements: Sequence[RequirementFlag] = ("C",),
    allow_copy: bool = True,
) -> npt.NDArray[np.float64]: ...


@overload
def _convert_array(
    array: ArrayLike | None,
    shape: ShapeLike | None = None,
    dtype: type[np.int32] | np.dtype[np.int32] = ...,
    requirements: Sequence[RequirementFlag] = ("C",),
    allow_copy: bool = True,
) -> npt.NDArray[np.int32]: ...


@overload
def _convert_array(
    array: ArrayLike | None,
    shape: ShapeLike | None = None,
    dtype: type[np.uint32] | np.dtype[np.uint32] = ...,
    requirements: Sequence[RequirementFlag] = ("C",),
    allow_copy: bool = True,
) -> npt.NDArray[np.uint32]: ...


@overload
def _convert_array(
    array: ArrayLike | None,
    shape: ShapeLike | None = None,
    dtype: type[bool] | type[np.bool_] | np.dtype[np.bool_] = ...,
    requirements: Sequence[RequirementFlag] = ("C",),
    allow_copy: bool = True,
) -> npt.NDArray[np.bool_]: ...


def _convert_array(
    array: ArrayLike | None,
    shape: ShapeLike | None = None,
    dtype: npt.DTypeLike = np.float32,
    requirements: Sequence[RequirementFlag] = ("C",),
    allow_copy: bool = True,
) -> npt.NDArray[np.generic]:
    """Function which takes a given array, checks the dimensions and shape,
    and converts to a supplied dtype.

    Args:
        array (:class:`numpy.ndarray` or :code:`None`): Array to check and convert.
            If :code:`None`, an empty array of given shape and type will be initialized
            (Default value: :code:`None`).
        shape: (tuple of int and :code:`None`): Expected shape of the array.
            Only the dimensions that are not :code:`None` are checked.
            (Default value = :code:`None`).
        dtype: :code:`dtype` to convert the array to if :code:`array.dtype`
            is different. If :code:`None`, :code:`dtype` will not be changed
            (Default value = :attr:`numpy.float32`).
        requirements (Sequence[str]): A sequence of string flags to be passed to
            :func:`numpy.require`.
        allow_copy (bool): If :code:`False` and the input array does not already
            conform to the required dtype and other requirements, this function
            will raise an error rather than coercing the array into a copy that
            does satisfy the requirements (Default value = :code:`True`).

    Returns:
        :class:`numpy.ndarray`: Array.
    """
    if array is None:
        concrete_shape = _as_concrete_shape(shape)
        if concrete_shape is None:
            msg = "shape must be provided when initializing a new array."
            raise ValueError(msg)
        return np.empty(concrete_shape, dtype=dtype)

    array = np.asarray(array)
    return_arr = np.require(array, dtype=dtype, requirements=requirements)

    if not allow_copy and return_arr is not array:
        msg = (
            "The provided output array must have dtype "
            f"{dtype}, and have the following array flags: "
            f"{', '.join(requirements)}."
        )
        raise ValueError(msg)

    if shape is not None:
        if return_arr.ndim != len(shape):
            msg = f"array.ndim = {return_arr.ndim}; expected ndim = {len(shape)}"
            raise ValueError(msg)

        for i, s in enumerate(shape):
            if s is not None and return_arr.shape[i] != s:
                shape_str = (
                    "("
                    + ", ".join(str(i) if i is not None else "..." for i in shape)
                    + ")"
                )
                msg = f"array.shape= {return_arr.shape}; expected shape = {shape_str}"
                raise ValueError(msg)

    return return_arr


def _convert_box(
    box: freud.box.BoxLike, dimensions: int | None = None
) -> freud.box.Box:
    """Function which takes a box-like object and attempts to convert it to
    :class:`freud.box.Box`. Existing :class:`freud.box.Box` objects are
    used directly.

    Args:
        box (box-like object (see :meth:`freud.box.Box.from_box`)): Box to
            check and convert if needed.
        dimensions (int): Number of dimensions the box should be. If not None,
            used to verify the box dimensions (Default value = :code:`None`).

    Returns:
        :class:`freud.box.Box`: freud box.
    """
    if not isinstance(box, freud.box.Box):
        try:
            box = freud.box.Box.from_box(box)
        except ValueError:
            raise

    if dimensions is not None and box.dimensions != dimensions:
        msg = f"The box must be {dimensions}-dimensional."
        raise ValueError(msg)

    return box
