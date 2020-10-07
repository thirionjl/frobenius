from typing import Tuple, Union, overload

Coordinates = Tuple[int, int]
Slices = Union[int, slice, Tuple[slice, slice], Tuple[slice, int], Tuple[int, slice]]
Subscript = Union[Slices, Coordinates]


@overload
def normalize_subscript(
    subscript: Coordinates, rows: int, cols: int
) -> Tuple[bool, int, int]:
    ...


@overload
def normalize_subscript(
    subscript: Slices, rows: int, cols: int
) -> Tuple[bool, slice, slice]:
    ...


def normalize_subscript(
    subscript: Subscript, rows: int, cols: int
) -> Union[Tuple[bool, slice, slice], Tuple[bool, int, int]]:
    if isinstance(subscript, int):
        return False, _as_slice(subscript, rows), slice(None, None, None)  # Row
    elif isinstance(subscript, slice):
        return False, subscript, slice(None, None, None)
    elif isinstance(subscript, tuple):
        if len(subscript) != 2:
            raise TypeError("Invalid number of arguments for tuple indexing")

        r, c = subscript
        if isinstance(r, int) and isinstance(c, int):
            return True, r, c  # Cell
        else:
            return False, _as_slice(r, rows), _as_slice(c, cols)  # Matrix
    else:
        raise ValueError("Unsupported index type: " + type(subscript))


def _as_slice(idx: Union[int, slice], size: int) -> slice:
    if isinstance(idx, int):
        return slice(idx, idx + 1, 1) if idx != -1 else slice(idx, size, 1)
    elif isinstance(idx, slice):
        return idx
