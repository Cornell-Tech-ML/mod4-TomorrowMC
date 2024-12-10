from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np  # type: ignore
from numba import prange  # type: ignore
from numba import njit as _njit  # type: ignore

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT compile a function."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Maps a function across elements in a tensor.

    Args:
    ----
        fn: Function to apply to each element

    Returns:
    -------
        Compiled function that applies fn to each element in the tensor

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        size = 1
        for i in range(len(out_shape)):
            size *= out_shape[i]

        # Main parallel loop
        for i in prange(size):
            # Calculate output index
            out_index = np.empty(MAX_DIMS, np.int32)
            to_index(i, out_shape, out_index)

            # Calculate input index
            in_index = np.empty(MAX_DIMS, np.int32)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Get positions
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

            # Apply function
            out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Applies a function to pairs of elements from two tensors.

    Args:
    ----
        fn: Function to apply to pairs of elements

    Returns:
    -------
        Compiled function that applies fn to pairs of elements from the input tensors

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        size = 1
        for i in range(len(out_shape)):
            size *= out_shape[i]

        # Main parallel loop
        for i in prange(size):
            out_index = np.empty(MAX_DIMS, np.int32)
            to_index(i, out_shape, out_index)

            # Handle broadcasting for both inputs
            a_index = np.empty(MAX_DIMS, np.int32)
            b_index = np.empty(MAX_DIMS, np.int32)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Get positions
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply function
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Reduces a tensor along a dimension by applying a function to pairs of elements.

    Args:
    ----
        fn: Function to apply in the reduction operation

    Returns:
    -------
        Compiled function that performs the reduction operation on the input tensor

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        size = 1
        for i in range(len(out_shape)):
            size *= out_shape[i]
        reduce_size = a_shape[reduce_dim]

        for i in prange(size):
            # Get output index
            out_index = np.empty(MAX_DIMS, np.int32)
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            # Set up input index
            a_index = np.empty(MAX_DIMS, np.int32)
            for j in range(len(out_shape)):
                a_index[j] = out_index[j]

            # First value
            a_index[reduce_dim] = 0
            acc = a_storage[index_to_position(a_index, a_strides)]

            # Reduce remaining values
            for j in range(1, reduce_size):
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index, a_strides)
                acc = fn(acc, a_storage[a_pos])

            out[out_pos] = acc

    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    # Get dimensions
    n_batches = np.int32(np.maximum(a_shape[0], b_shape[0]))
    n_rows = a_shape[1]
    n_cols = b_shape[2]
    n_inner = a_shape[2]

    # Get batch strides, handling broadcasting
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Parallel loop over batches
    for batch in prange(n_batches):
        for i in range(n_rows):
            for j in range(n_cols):
                acc = 0.0

                # Calculate base positions for this output element
                out_pos = (
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                )
                a_base = batch * a_batch_stride + i * a_strides[1]
                b_base = batch * b_batch_stride + j * b_strides[2]

                # Inner product
                for k in range(n_inner):
                    a_pos = a_base + k * a_strides[2]
                    b_pos = b_base + k * b_strides[1]
                    acc += a_storage[a_pos] * b_storage[b_pos]

                out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
