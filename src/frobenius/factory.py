from typing import Collection, List

from frobenius import validate
from frobenius.matrix import MatrixType, Shape
from frobenius.numbers import N, Number

__all__ = ["matrix"]


def singleton(value: Number) -> MatrixType:
    return MatrixType.singleton(value)


def eye(nrows: int = None, ncols: int = None, shape: Shape = None):
    return MatrixType.eye(_shape(nrows, ncols, shape))


def ones(nrows: int = None, ncols: int = None, shape: Shape = None):
    return MatrixType.ones(_shape(nrows, ncols, shape))


def zeros(nrows: int = None, ncols: int = None, shape: Shape = None):
    return MatrixType.zeros(_shape(nrows, ncols, shape))


def _shape(nrows: int = None, ncols: int = None, shape: Shape = None) -> Shape:
    if isinstance(shape, tuple):
        return Shape(*shape)
    elif isinstance(shape, Shape):
        return shape
    elif isinstance(nrows, int) and ncols is None:
        return Shape(nrows, nrows)
    elif isinstance(nrows, int) and isinstance(ncols, int):
        return Shape(nrows, ncols)
    else:
        raise ValueError("Invalid arguments")


def matrix(data: Collection[Collection[N]]):
    return shaped(_flatten(data), _shape_of(data))


def vector(data: Collection[N]):
    return shaped([float(d) for d in data], Shape(len(data), 1))


def row_vector(data: Collection[N]):
    return shaped([float(d) for d in data], Shape(1, len(data)))


def shaped(data: Collection[N], shape: Shape):
    return MatrixType.from_flat_collection(data, shape)


def _flatten(data: Collection[Collection[N]]) -> List[float]:
    flat_list: List[float] = []
    for r in data:
        for c in r:
            flat_list.append(float(c))
    return flat_list


def _shape_of(data: Collection[Collection[N]]) -> Shape:
    return Shape(_count_rows(data), _count_cols(data))


def _count_rows(data: Collection[Collection[N]]) -> int:
    validate.is_collection("data", data)
    return len(data)


def _count_cols(data: Collection[Collection[N]]) -> int:
    validate.is_collection("data", data)
    ncols = None
    for row_idx, row in enumerate(data):
        validate.is_collection(f"data.row[{row_idx}]", row)
        row_length = len(row)
        if ncols is not None and ncols != row_length:
            raise ValueError(
                f"Row {row_idx} has an incorrect number of columns. "
                f"Expected {ncols} but got {row_length}"
            )
        else:
            ncols = row_length
    if ncols is None:
        raise ValueError(
            "Impossible to determine column size because no rows has columns"
        )
    return ncols
