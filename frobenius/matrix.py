import math
from array import array
from itertools import repeat
from typing import (
    Iterable,
    Union,
    overload,
    Tuple,
    List,
    Collection,
    Callable,
    Iterator,
)

from frobenius import numbers
from frobenius.numbers import N, Number
from frobenius.shape import Shape
from frobenius.utils_subscript import (
    Coordinates,
    Slices,
    Subscript,
    normalize_subscript,
)

Operand = Union[Number, "MatrixType"]

DISPLAY_DIGITS = 6


class MatrixType:
    __slots__ = ["_data", "_shift", "_nrows", "_row_stride", "_ncols", "_col_stride"]

    ###############
    # Constructors
    ###############
    def __init__(
        self,
        data: array,
        nrows: int,
        ncols: int,
        shift: int = 0,
        row_stride: int = None,
        col_stride: int = 1,
    ):
        self._data: array = data
        self._nrows: int = nrows
        self._ncols: int = ncols
        self._shift: int = shift
        self._row_stride = ncols if row_stride is None else row_stride
        self._col_stride = col_stride

    @classmethod
    def from_flat_collection(cls, data: Collection[N], shape: Shape) -> "MatrixType":
        return cls(array("f", [float(c) for c in data]), shape.nrows, shape.ncols)

    @classmethod
    def singleton(cls, f: Number) -> "MatrixType":
        return cls(array("f", [f]), nrows=1, ncols=1)

    @classmethod
    def zeros(cls, shape: Shape) -> "MatrixType":
        return cls(
            array("f", (0 for _ in range(shape.ncols * shape.nrows))),
            shape.nrows,
            shape.ncols,
        )

    @classmethod
    def ones(cls, shape: Shape) -> "MatrixType":
        return cls(
            array("f", (1 for _ in range(shape.ncols * shape.nrows))),
            shape.nrows,
            shape.ncols,
        )

    @classmethod
    def eye(cls, shape: Shape) -> "MatrixType":
        return cls(
            array(
                "f",
                (
                    (1 if i == j else 0)
                    for i in range(shape.ncols)
                    for j in range(shape.nrows)
                ),
            ),
            shape.nrows,
            shape.ncols,
        )

    ###############
    # Iteration
    ###############
    def to_list(self):
        rows: List = list()
        for row_idx in range(self.nrows):
            row_start_idx = self._shift + row_idx * self._row_stride
            row = [
                self._data[row_start_idx + col_idx * self._col_stride]
                for col_idx in range(self._ncols)
            ]
            rows.append(row)
        return rows

    def _row_first_iter(self) -> Iterable[int]:
        for row_idx in range(self._nrows):
            row_start_idx = self._shift + row_idx * self._row_stride
            for col_idx in range(self._ncols):
                yield row_start_idx + col_idx * self._col_stride

    def _col_first_iter(self) -> Iterable[int]:
        for col_idx in range(self._ncols):
            col_start_idx = self._shift + col_idx * self._col_stride
            for row_idx in range(self._nrows):
                yield col_start_idx + row_idx * self._row_stride

    def _broadcast_row_iter(self, repeats: int) -> Iterable[int]:
        assert self.nrows == 1
        for _ in range(repeats):
            for col_idx in range(self._ncols):
                yield self._shift + col_idx * self._col_stride

    def _broadcast_col_iter(self, repeats: int) -> Iterable[int]:
        assert self.ncols == 1
        for _ in range(repeats):
            for row_idx in range(self._nrows):
                yield self._shift + row_idx * self._row_stride

    def _broadcast_row_iterx(self, inner_repeats: int) -> Iterable[int]:
        assert self.nrows == 1
        for col_idx in range(self._ncols):
            idx = self._shift + col_idx * self._col_stride
            for _ in range(inner_repeats):
                yield idx

    def _broadcast_col_iterx(self, inner_repeats: int) -> Iterable[int]:
        assert self.ncols == 1
        for row_idx in range(self._nrows):
            idx = self._shift + row_idx * self._row_stride
            for _ in range(inner_repeats):
                yield idx

    def _broadcast_singleton_iter(self, repeats: int) -> Iterable[int]:
        assert self.ncols == self.nrows
        assert self.ncols == 1
        return repeat(self._shift, times=repeats)

    # Refactor and test !
    def _broadcast_iters(
        self, other: "MatrixType"
    ) -> Tuple[Shape, bool, Iterable[int], Iterable[int]]:
        if isinstance(other, MatrixType):
            # No broadcast
            if other.shape == self.shape:
                return self.shape, True, self._row_first_iter(), other._row_first_iter()

            # other side broadcasts
            elif other.shape == (1, self._ncols):
                return (
                    self.shape,
                    True,
                    self._row_first_iter(),
                    other._broadcast_row_iter(self.nrows),
                )
            elif other.shape == (self._nrows, 1):
                return (
                    self.shape,
                    False,
                    self._col_first_iter(),
                    other._broadcast_col_iter(self.ncols),
                )
            elif other.shape == (1, 1):
                return (
                    self.shape,
                    True,
                    self._row_first_iter(),
                    other._broadcast_singleton_iter(self.ncols * self.nrows),
                )

            # self side broadcast
            elif self.shape == (1, other.ncols):
                return (
                    other.shape,
                    True,
                    self._broadcast_row_iter(other.nrows),
                    other._row_first_iter(),
                )
            elif self.shape == (other.nrows, 1):
                return (
                    other.shape,
                    False,
                    self._broadcast_col_iter(other.ncols),
                    other._col_first_iter(),
                )
            elif self.shape == (1, 1):
                return (
                    other.shape,
                    True,
                    self._broadcast_singleton_iter(other.ncols * other.nrows),
                    other._row_first_iter(),
                )

            # 2 sided broadcasts
            elif self.ncols == 1 and other.nrows == 1:
                return (
                    Shape(self.nrows, other.ncols),
                    False,
                    self._broadcast_col_iter(other.ncols),
                    other._broadcast_row_iterx(self.nrows),
                )
            elif self.nrows == 1 and other.ncols == 1:
                return (
                    Shape(other.nrows, self.ncols),
                    True,
                    self._broadcast_row_iter(other.nrows),
                    other._broadcast_col_iterx(self.ncols),
                )
            else:
                raise ValueError(
                    f"Incompatible operands shapes: {self.shape} " f"and {other.shape}"
                )
        else:
            raise ValueError(f"Incompatible operands: {type(other)}")

    ###############
    # Get / Set
    ###############

    @overload
    def __getitem__(self, subscript: Coordinates) -> Number:
        ...

    @overload
    def __getitem__(self, subscript: Slices) -> "MatrixType":
        ...

    def __getitem__(self, subscript: Subscript) -> Union["MatrixType", float]:
        is_coordinates, r, c = normalize_subscript(subscript, self.nrows, self.ncols)
        return self._get_cell_at(r, c) if is_coordinates else self._get_sub_matrix(r, c)

    def __setitem__(self, subscript: Subscript, value: Operand) -> None:
        is_coordinates, r, c = normalize_subscript(subscript, self.nrows, self.ncols)
        v = MatrixType._as_matrix(value)

        if is_coordinates:
            self._set_cell_at(r, c, v)
        else:
            self._set_sub_matrix(r, c, v)

    def _get_cell_at(self, r: int, c: int) -> float:
        self._check_row_index(r)
        self._check_col_index(c)
        idx = self._shift + r * self._row_stride + c * self._col_stride
        return self._data[idx]

    def _set_cell_at(self, r: int, c: int, value: "MatrixType") -> None:
        self._check_row_index(r)
        self._check_col_index(c)
        idx = self._shift + r * self._row_stride + c * self._col_stride
        if value.shape != (1, 1):
            raise ValueError(
                f"To set a single cell matrix, submitted value to set should be of shape (1,1)"
            )

        self._data[idx] = value._get_cell_at(0, 0)

    def _get_sub_matrix(self, row_slice: slice, col_slice: slice) -> "MatrixType":
        row_start, row_end, row_step = row_slice.indices(self._nrows)
        col_start, col_end, col_step = col_slice.indices(self._ncols)

        return MatrixType(
            self._data,
            nrows=math.ceil((row_end - row_start) / row_step),
            ncols=math.ceil((col_end - col_start) / col_step),
            shift=self._shift
            + row_start * self._row_stride
            + col_start * self._col_stride,
            row_stride=self._row_stride * row_step,
            col_stride=self._col_stride * col_step,
        )

    def _set_sub_matrix(
        self, row_slice: slice, col_slice: slice, value: "MatrixType"
    ) -> "MatrixType":
        destination = self._get_sub_matrix(row_slice, col_slice)

        target_shape, _, dst_it, src_it = destination._broadcast_iters(value)
        if target_shape != destination.shape:
            raise ValueError(
                f"Cannot fit value with shape {target_shape} into destination shape {destination.shape}"
            )

        for dst_idx, src_idx in zip(dst_it, src_it):
            self._data[dst_idx] = value._data[src_idx]
        return self

    @staticmethod
    def _as_matrix(value: Union[Number, "MatrixType"]) -> "MatrixType":
        if numbers.is_number(value):
            return MatrixType.singleton(value)
        elif isinstance(value, MatrixType):
            return value
        else:
            raise ValueError(f"Values of type {type(value)} cannot be set into matrix")

    def _check_row_index(self, idx: int):
        fixed_idx = idx if idx >= 0 else self._nrows - idx
        if not (0 <= fixed_idx < self._nrows):
            raise ValueError(
                f"Row index is out of bound got {idx} size is {self._nrows}"
            )

    def _check_col_index(self, idx: int):
        fixed_idx = idx if idx >= 0 else self._ncols - idx
        if not (0 <= fixed_idx < self._ncols):
            raise ValueError(
                f"Column index is out of bound got {idx} size is {self._ncols}"
            )

    ###############
    # Element-Wise Binary operators
    ###############

    def __add__(self, other: Operand) -> "MatrixType":
        return self._apply_binary_op_element_wise(other, lambda x, y: x + y)

    def __radd__(self, other: Number) -> "MatrixType":
        if numbers.is_number(other):
            return self.__add__(other)
        else:
            raise ValueError(f"Unsupported type:" + type(other))

    def __mul__(self, other: Operand) -> "MatrixType":
        return self._apply_binary_op_element_wise(other, lambda x, y: x * y)

    def __rmul__(self, other: Number) -> "MatrixType":
        if numbers.is_number(other):
            return self.__mul__(other)
        else:
            raise ValueError(f"Unsupported type:" + type(other))

    def __truediv__(self, other: Operand) -> "MatrixType":
        return self._apply_binary_op_element_wise(other, lambda x, y: x / y)

    def __floordiv__(self, other: Operand) -> "MatrixType":
        return self._apply_binary_op_element_wise(other, lambda x, y: x // y)

    def __mod__(self, other: Operand) -> "MatrixType":
        return self._apply_binary_op_element_wise(other, lambda x, y: x % y)

    def __sub__(self, other: Operand) -> "MatrixType":
        return self._apply_binary_op_element_wise(other, lambda x, y: x - y)

    def _apply_binary_op_element_wise(
        self, o: Operand, binary_op: Callable
    ) -> "MatrixType":
        other = MatrixType._as_matrix(o)
        target_shape, row_order, self_it, other_it = self._broadcast_iters(other)
        target = self.zeros(target_shape)
        target_it = target._row_first_iter() if row_order else target._col_first_iter()
        for t_idx, s_idx, o_idx in zip(target_it, self_it, other_it):
            target._data[t_idx] = binary_op(self._data[s_idx], other._data[o_idx])
        return target

    ###############
    # Unary operators
    ###############

    @property
    def T(self) -> "MatrixType":
        return MatrixType(
            self._data,
            nrows=self.ncols,
            ncols=self.nrows,
            shift=self._shift,
            row_stride=self._col_stride,
            col_stride=self._row_stride,
        )

    def __abs__(self) -> "MatrixType":
        return self._apply_unary_op_element_wise(abs)

    def __pow__(self, power, modulo=None) -> "MatrixType":
        return self._apply_unary_op_element_wise(lambda x: x.__pow__(power, modulo))

    def __neg__(self) -> "MatrixType":
        return self._apply_unary_op_element_wise(lambda x: x.__neg__())

    def __floor__(self) -> "MatrixType":
        return self._apply_unary_op_element_wise(math.floor)

    def __ceil__(self) -> "MatrixType":
        return self._apply_unary_op_element_wise(math.ceil)

    def __round__(self, n=None):
        return self._apply_unary_op_element_wise(lambda x: round(x, n))

    def __trunc__(self):
        return self._apply_unary_op_element_wise(math.trunc)

    def _apply_unary_op_element_wise(self, unary_op: Callable) -> "MatrixType":
        target = self.zeros(self.shape)
        for target_idx, self_idx in zip(
            target._row_first_iter(), self._row_first_iter()
        ):
            target._data[target_idx] = unary_op(self._data[self_idx])
        return target

    ###############
    # Matrix multiplication
    ###############

    def __matmul__(self, other: "MatrixType") -> "MatrixType":
        if isinstance(other, MatrixType):
            if other.nrows != self.ncols:
                raise ValueError(
                    f"Incompatible shapes for matrix multiplication: "
                    f"{self.shape} and {other.shape}"
                )

            res = self.zeros(Shape(self._nrows, other._ncols))
            idx = 0
            for i in range(self.nrows):
                for j in range(other.ncols):
                    idx_self = self._shift + i * self._row_stride
                    idx_other = other._shift + j * other._col_stride
                    dot_product = 0.0
                    for k in range(self.ncols):
                        dot_product += self._data[idx_self] * other._data[idx_other]
                        idx_self += self._col_stride
                        idx_other += other._row_stride
                    res._data[idx] = dot_product
                    idx += 1

            return res

    ###############
    # Utility props and methods
    ###############
    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    @property
    def shape(self) -> Shape:
        return Shape(self.nrows, self.ncols)

    @property
    def size(self) -> int:
        return self._nrows * self._ncols

    def __len__(self) -> int:
        return self.size

    def __copy__(self):
        return self.copy()

    def copy(self) -> "MatrixType":
        return MatrixType(data=array("f", self), nrows=self.nrows, ncols=self.ncols)

    def __contains__(self, item: Number) -> bool:
        return item in iter(self)

    def __iter__(self) -> Iterator[Number]:
        return (self._data[idx] for idx in self._row_first_iter())

    ###############
    # Comparison operators
    ###############
    def __eq__(self, other):
        if isinstance(other, MatrixType):
            return other.to_list() == self.to_list()
        else:
            return False

    def __hash__(self):
        return hash(self.to_list())

    ###############
    # Display
    ###############
    def __str__(self):
        rows_ = self._rows_to_str_list()
        repr_str = "[" + ",\n ".join(rows_) + "]"
        return f"{repr_str}"

    def __repr__(self):
        rows_ = self._rows_to_str_list()
        repr_str = "[" + ",\n        ".join(rows_) + "]"
        return f"matrix({repr_str})"

    def _rows_to_str_list(self) -> Iterable[str]:
        pattern = "{0:." + str(DISPLAY_DIGITS) + "f}"
        col_width = max(
            (len(pattern.format(e).rstrip("0")) for e in self._data), default=0
        )
        rows = self.to_list()
        return (MatrixType._row_to_str(r, col_width) for r in rows)

    @staticmethod
    def _row_to_str(row: Iterable[Number], col_width: int) -> str:
        digits = min(DISPLAY_DIGITS, col_width - 1)
        ff = "{0:" + str(col_width) + "." + str(digits) + "f}"
        return (
            "["
            + ", ".join(
                MatrixType._replace_trailing_zeros(ff.format(elt)) for elt in row
            )
            + "]"
        )

    @staticmethod
    def _replace_trailing_zeros(s: str):
        length = len(s)
        stripped = s.rstrip("0")
        pad = " " * (length - len(stripped))
        return stripped + pad

    def is_square(self):
        return self.ncols == self.nrows
