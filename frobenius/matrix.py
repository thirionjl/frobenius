from array import array, ArrayType
from typing import Iterable, Union, TypeVar, overload, Tuple, List, Collection

from frobenius.numbers import N

Number = Union[int, float, bool]
T = TypeVar('T', int, float, bool)

DISPLAY_DIGITS = 6


class Shape(tuple):

    def __new__(cls, *args):
        if len(args) != 2:
            raise TypeError("Shape should have to int arguments")
        return tuple.__new__(cls, (args[0], args[1]))

    @property
    def nrows(self):
        return self[0]

    @property
    def ncols(self):
        return self[1]


class Vector:
    __slots__ = ['_data', '_start_idx', '_stop_idx', '_step']

    def __init__(self, data: ArrayType, start_idx: int = 0, stop_idx: int = None, step: int = 1):
        self._data: ArrayType = data
        self._start_idx: int = start_idx
        self._stop_idx: int = len(data) if stop_idx is None else stop_idx
        self._step: int = step

    @classmethod
    def from_collection(cls, data: Collection[N], start_idx: int = 0, stop_idx: int = None, step: int = 1):
        return cls(array('f', [float(c) for c in data]), start_idx, stop_idx, step)

    def __len__(self):
        return (self._stop_idx - self._start_idx) // abs(self._step)

    @overload
    def __getitem__(self, subscript: int) -> float:
        ...

    @overload
    def __getitem__(self, subscript: slice) -> 'Vector':
        ...

    def __getitem__(self, subscript: Union[slice, int]):
        if isinstance(subscript, int):
            return self._data.__getitem__(subscript)
        elif isinstance(subscript, slice):
            fr, to, step = subscript.indices(len(self))
            if step <= 0:
                raise ValueError(f'Unsupported non positive step length {step} in slice')
            if to > len(self):
                raise ValueError(f'Index {to} out of bound')
            return Vector(self._data, start_idx=self._start_idx + (fr * self._step),
                          stop_idx=self._start_idx + (to * self._step),
                          step=self._step * step)
        else:
            raise ValueError(f'Unsupported subscript type: ' + type(subscript))

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            self._data[self._start_idx + idx] = value
        elif isinstance(idx, slice) and isinstance(value, Vector):
            fr, to, step = idx.indices(len(self))
            if step <= 0:
                raise ValueError(f'Unsupported non positive step length {step} in slice')
            if to > len(self):
                raise ValueError(f'Index {to} out of bound')
            seq_len = len(value)
            slice_len = (to - fr) // step
            if slice_len != seq_len:
                raise ValueError(f'Invalid sequence length {seq_len} regarding slice length {slice_len}')
            shifted_slice = slice(self._start_idx + (fr * self._step), self._start_idx + (to * self._step),
                                  self._step * step)
            self._data[shifted_slice] = value._data

    def __iter__(self):
        return iter(self._data)

    def __str__(self):
        return f"{self._to_str()}"

    def __repr__(self):
        return f"matrix({self._to_str()})"

    def _to_str(self) -> str:
        pattern = '{0:.' + str(DISPLAY_DIGITS) + 'f}'
        real_data = self.to_list()
        return '[' + ', '.join(pattern.format(elt).rstrip('0') for elt in real_data) + ']'

    def to_list(self) -> List[float]:
        real_data = self._data[self._start_idx: self._stop_idx: self._step]
        return list(real_data)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return other.to_list() == self.to_list()
        else:
            return False

    def __hash__(self):
        return hash(self.to_list())


class MatrixType:
    __slots__ = ['_data', '_row_count', '_col_count']

    def __init__(self, data: array, row_count: int, col_count: int):
        self._row_count: int = row_count
        self._col_count: int = col_count
        self._data: array = data

    @classmethod
    def from_flat_collection(cls, data: Collection[N], shape: Shape):
        return cls(array('f', [float(c) for c in data]), shape.nrows, shape.ncols)

    @classmethod
    def zeros(cls, shape: Shape):
        return cls(array('f', (0 for i in range(shape.ncols * shape.nrows))), shape.nrows, shape.ncols)

    @classmethod
    def ones(cls, shape: Shape):
        return cls(array('f', (1 for i in range(shape.ncols * shape.nrows))), shape.nrows, shape.ncols)

    @classmethod
    def eye(cls, shape: Shape):
        return cls(array('f', ((1 if i == j else 0) for i in range(shape.ncols) for j in range(shape.nrows))),
                   shape.nrows, shape.ncols)

    def to_list(self):
        rows: List = list()
        for i in range(self._row_count):
            row: List = list(self._data[i * self.ncols: (i + 1) * self.ncols])
            rows.append(row)
        return rows

    @overload
    def __getitem__(self, subscript: Tuple[int, int]) -> float:
        ...

    @overload
    def __getitem__(self, subscript: Tuple[int, slice]) -> 'MatrixType':
        ...

    @overload
    def __getitem__(self, subscript: Tuple[slice, int]) -> 'MatrixType':
        ...

    @overload
    def __getitem__(self, subscript: Tuple[slice, slice]) -> 'MatrixType':
        ...

    @overload
    def __getitem__(self, subscript: slice) -> 'MatrixType':
        ...

    @overload
    def __getitem__(self, subscript: int) -> Vector:
        ...

    def __getitem__(self, given: Union[int, tuple, slice]):
        if isinstance(given, slice):
            return self._get_sub_matrix(given, slice(None, None, None))
        elif isinstance(given, tuple):
            if len(given) != 2:
                raise TypeError("Invalid number of arguments for tuple indexing")

            rows, cols = given

            if isinstance(rows, int) and isinstance(cols, int):
                return self._get_cell(rows, cols)
            else:
                return self._get_sub_matrix(rows, cols)

        elif isinstance(given, int):
            return self._get_row(given)
        else:
            raise ValueError("Unsupported index type: " + type(given))

    def _get_sub_matrix(self, rows: slice, cols: slice) -> 'MatrixType':
        row_start, row_end, row_step = self._as_slice(rows).indices(self._row_count)
        col_start, col_end, col_step = self._as_slice(cols).indices(self._col_count)

        self._check_row_index(row_start)
        self._check_row_index(row_end)
        self._check_col_index(col_start)
        self._check_col_index(col_step)

        res: array = array('f', [])
        new_row_count = (row_end - row_start) // row_step
        new_col_count = (col_end - col_start) // col_step
        for row_idx in range(row_start, row_end, row_step):
            idx_start = row_idx * self._col_count
            idx_end = idx_start + self._col_count
            row = self._data[idx_start:idx_end]
            sl = row[col_start:col_end:col_step]
            res.extend(sl)

        return MatrixType(res, new_row_count, new_col_count)

    def _get_cell(self, r: int, c: int) -> float:
        self._check_row_index(r)
        self._check_col_index(c)
        fr = r * self._col_count + c
        return self._data[fr]

    def _get_row(self, i: int):
        self._check_row_index(i)
        fr = i * self._col_count
        to = fr + self._col_count
        if self._col_count == 1:
            return self._data[fr]
        else:
            return Vector(self._data[fr:to])

    def _check_row_index(self, value: int):
        if not (0 <= value < self._row_count):
            raise ValueError(f"Row index is out of bound got {value} to be between 0 and {self._row_count - 1}")

    def _check_col_index(self, value: int):
        if not (0 <= value < self._col_count):
            raise ValueError(f"Column index is out of bound got {value} to be between 0 and {self._col_count - 1}")

    @staticmethod
    def _as_slice(idx: Union[int, slice]) -> slice:
        if isinstance(idx, int):
            return slice(idx, idx + 1, 1)
        elif isinstance(idx, slice):
            return idx

    @property
    def nrows(self):
        return self._row_count

    @property
    def ncols(self):
        return self._col_count

    @property
    def size(self):
        return len(self._data)

    def __len__(self):
        return len(self._data)

    def __matmul__(self, other: 'MatrixType') -> 'MatrixType':

        if isinstance(other, MatrixType):
            if other.nrows != self.ncols:
                raise ValueError(f'Incompatible shapes for matrix multiplication: {self.shape()} and {other.shape()}')

            res: array = array('f', [0.0]) * (self.nrows * other.ncols)
            idx = 0
            for i in range(self.nrows):
                for j in range(other.ncols):
                    idx_left = i * self.ncols
                    idx_right = j
                    cell_value = 0.0
                    for k in range(self.ncols):
                        cell_value += self._data[idx_left] * other._data[idx_right]
                        idx_left += 1
                        idx_right += other.ncols
                    res[idx] = cell_value
                    idx += 1

            return MatrixType(res, self.nrows, other.ncols)

    def rows(self):
        return (self.row(i) for i in range(self.nrows))

    def row(self, i: int):
        row_start = i * self.ncols
        return (self._data[row_start + j] for j in range(self.ncols))

    def columns(self):
        return (self.column(j) for j in range(self.ncols))

    def column(self, j: int):
        row_size = self.ncols
        return (self._data[idx] for idx in range(j, self.size, row_size))

    def __add__(self, other: 'MatrixType') -> 'MatrixType':
        if isinstance(other, MatrixType):
            if other.shape() == self.shape():
                return self._add_same_size(other)
            elif other.shape() == (1, self.ncols):
                return self._add_horizontal(other)
            elif other.shape() == (self.ncols, 1):
                return self._add_vertical(other)
            elif other.shape() == (1, 1):
                return self._add_everywhere(other)
            elif self.shape() == (1, other.ncols):
                return other._add_horizontal(self)
            elif self.shape() == (other.ncols, 1):
                return other._add_vertical(self)
            elif self.shape() == (1, 1):
                return other._add_everywhere(self)
            else:
                raise ValueError(f'Incompatible shapes for matrix addition: {self.shape()} and {other.shape()}')
        else:
            raise ValueError(f'Incompatible types for matrix addition: {type(other)}')

    def _add_vertical(self, other):
        res: array = array('f', [0.0]) * (self.nrows * self.ncols)
        i: int = 0
        for r in range(self.nrows):
            for c in range(self.ncols):
                res[i] = self._data[r * self.ncols + c] + other._data[r]
                i += 1
        return MatrixType(res, self.nrows, self.ncols)

    def _add_horizontal(self, other):
        res: array = array('f', [0.0]) * (self.nrows * self.ncols)
        i: int = 0
        for r in range(self.nrows):
            for c in range(self.ncols):
                res[i] = self._data[r * self.ncols + c] + other._data[c]
                i += 1
        return MatrixType(res, self.nrows, self.ncols)

    def _add_same_size(self, other):
        res: array = array('f', [0.0]) * (self.nrows * self.ncols)
        for i in range(self.size):
            res[i] = self._data[i] + other._data[i]
        return MatrixType(res, self.nrows, self.ncols)

    def _add_everywhere(self, other):
        res: array = array('f', [0.0]) * (self.nrows * self.ncols)
        for i in range(self.size):
            res[i] = self._data[i] + other._data[0]
        return MatrixType(res, self.nrows, self.ncols)

    def __eq__(self, other):
        if isinstance(other, MatrixType):
            return other.to_list() == self.to_list()
        else:
            return False

    def __hash__(self):
        return hash(self.to_list())

    def shape(self) -> Tuple[int, int]:
        return self.nrows, self.ncols

    def __str__(self):
        rows_ = self._rows_to_str_list()
        repr_str = '[' + ',\n '.join(rows_) + ']'
        return f"{repr_str}"

    def __repr__(self):
        rows_ = self._rows_to_str_list()
        repr_str = '[' + ',\n        '.join(rows_) + ']'
        return f"matrix({repr_str})"

    def _rows_to_str_list(self) -> Iterable[str]:
        pattern = '{0:.' + str(DISPLAY_DIGITS) + 'f}'
        col_width = max((len(pattern.format(e).rstrip('0')) for e in self._data))
        rows = self.to_list()
        return (MatrixType._row_to_str(r, col_width) for r in rows)

    @staticmethod
    def _row_to_str(row: Iterable[float], col_width: int) -> str:
        digits = min(DISPLAY_DIGITS, col_width - 1)
        ff = '{0:' + str(col_width) + '.' + str(digits) + 'f}'
        return '[' + ', '.join(MatrixType._replace_trailing_zeros(ff.format(elt)) for elt in row) + ']'

    @staticmethod
    def _replace_trailing_zeros(s: str):
        length = len(s)
        stripped = s.rstrip('0')
        pad = ' ' * (length - len(stripped))
        return stripped + pad
