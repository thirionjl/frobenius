from typing import Collection, List

from frobenius import validate
from frobenius.matrix import MatrixType, Shape, Vector
from frobenius.numbers import N

__all__ = ['matrix']


def eye(shape: Shape):
    return MatrixType.eye(shape)


def ones(shape: Shape):
    return MatrixType.ones(shape)


def zeros(shape: Shape):
    return MatrixType.zeros(shape)


def matrix(data: Collection[Collection[N]]):
    return shaped(_flatten(data), _shape_of(data))


def vector(data: Collection[N]):
    return Vector.from_collection([float(c) for c in data])


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
    validate.is_collection('data', data)
    return len(data)


def _count_cols(data: Collection[Collection[N]]) -> int:
    validate.is_collection('data', data)
    ncols = None
    row_idx = 0
    for row in data:
        validate.is_collection(f'data.row[{row_idx}]', row)
        row_length = len(row)
        if ncols is not None and ncols != row_length:
            raise ValueError(
                f"Row {row_idx} has an incorrect number of columns. Expected {ncols} but got {row_length}")
        else:
            ncols = row_length
        row_idx += 1
    if ncols is None:
        raise ValueError(f"Impossible to determine column size because no rows has columns")
    return ncols
