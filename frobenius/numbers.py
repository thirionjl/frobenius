from typing import Union, TypeVar

Number = Union[int, float, bool]
N = TypeVar('N', int, float, bool)


def is_number(x: Number):
    return isinstance(x, Number.__args__)
