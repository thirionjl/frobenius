from pytest import approx

from frobenius import factory, determinants


def test_det():
    a = factory.matrix([[1, 3, 2], [-3, -1, -3], [2, 3, 1]])
    assert determinants.det(a) == approx(-15)
