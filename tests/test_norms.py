from frobenius import factory as f
from frobenius import norms as n


def test_norm2():
    a = f.matrix([[-1, 0, 0], [4, 0, 3], [-3, 1, 0]])
    assert n.norm(a) == 6


def test_norm3():
    a = f.matrix([[2, 0, 2], [2, 0, 0], [1, 1, 1]])
    assert n.norm(a, 3) == 3.0
