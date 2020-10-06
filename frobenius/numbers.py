from typing import Union, TypeVar, Any

Number = Union[int, float, bool]
N = TypeVar("N", int, float, bool)


def is_number(x: Any):
    return isinstance(x, Number.__args__)
