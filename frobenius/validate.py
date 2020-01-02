from typing import Any, Collection

from frobenius import numbers


def is_number(name: str, value: object) -> None:
    if not numbers.is_number(name):
        raise ValueError(f"Parameter {name} must be numeric, but got {type(value)}")


def is_collection(name: str, value: Collection[Any]) -> None:
    valid_collection = hasattr(value, '__iter__') and hasattr(value, '__len__') and hasattr(value, '__contains__')
    if not valid_collection:
        raise ValueError(f"Parameter {name} must be a collection type, but got {type(value)}")
