import math

from frobenius.matrix import MatrixType


def norm(m: MatrixType, exponent: int = 2) -> float:
    assert 2 <= exponent <= 100
    total = sum(map(lambda v: math.pow(v, exponent), m))
    return math.pow(total, 1 / exponent)
