import math

from frobenius import factory as f


def test_matrix_constructor():
    assert f.matrix([[]]).shape == (1, 0)
    assert f.matrix([[1, 2]]).shape == (1, 2)
    assert f.matrix([[1], [2]]).shape == (2, 1)
    assert f.matrix([[1, 2, 3], [4, 5, 6]]).shape == (2, 3)
    assert f.vector([1, 2, 3]).shape == (3, 1)
    assert f.row_vector([1, 2, 3]).shape == (1, 3)


def test_factory_methods_and_equals():
    assert f.eye(3) == f.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert f.ones(2, 3) == f.matrix([[1, 1, 1], [1, 1, 1]])
    assert f.zeros(shape=(1, 4)) == f.matrix([[0, 0, 0, 0]])
    assert f.singleton(18) == f.matrix([[18]])


def test_matrix_get_item():
    m = f.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert m[:] == m
    assert m[:, :] == m
    assert m[2, 1] == 8.0
    assert m[1] == f.row_vector([4, 5, 6])
    assert m[:, -1] == f.vector([3, 6, 9])
    assert m[:, 1] == f.vector([2, 5, 8])
    assert m[:-1, :-1] == f.matrix([[1, 2], [4, 5]])
    assert m[::2, ::2] == f.matrix([[1, 3], [7, 9]])
    assert m[::-1, 1] == f.vector([8, 5, 2])


def test_matrix_set_item():
    m = f.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    n = m[:-1, :-1]
    o = m[:, ::2]
    n[0, :] = f.matrix([[10, 20]])
    assert n == f.matrix([[10, 20], [4, 5]])

    o[1:, :] = f.matrix([[1000, 2000]])
    assert o == f.matrix([[10, 3], [1000, 2000], [1000, 2000]])
    assert n == f.matrix([[10, 20], [1000, 5]])
    assert m == f.matrix([[10, 20, 3], [1000, 5, 2000], [1000, 8, 2000]])

    m[:-1, :-1] = 0
    assert m == f.matrix([[0, 0, 3], [0, 0, 2000], [1000, 8, 2000]])


def test_add():
    m = f.matrix([[1, 2], [3, 4]])
    r = f.matrix([[100, 200]])
    c = f.matrix([[100], [200]])
    e = f.matrix([[1000]])
    assert (1 + m) == f.matrix([[2, 3], [4, 5]])
    assert (m + m) == f.matrix([[2, 4], [6, 8]])
    assert (m + r) == f.matrix([[101, 202], [103, 204]])
    assert (m + c) == f.matrix([[101, 102], [203, 204]])
    assert (m + e) == f.matrix([[1001, 1002], [1003, 1004]])


def test_mul():
    m = f.matrix([[1, 2], [3, 4]])
    r = f.matrix([[100, 200]])
    assert (m * r) == f.matrix([[100, 400], [300, 800]])
    assert 10 * f.matrix([[1, 2], [3, 4]]) == f.matrix([[10, 20], [30, 40]])


def test_true_div():
    m = f.matrix([[2, 4], [6, 8]])
    r = f.matrix([[1, 2]])
    assert (m / r) == f.matrix([[2, 2], [6, 4]])


def test_floor_div():
    m = f.matrix([[5, 3], [6, 8]])
    assert (m // 2) == f.matrix([[2, 1], [3, 4]])


def test_mod():
    m = f.matrix([[5, 3], [6, 8]])
    assert (m % 2) == f.matrix([[1, 1], [0, 0]])


def test_sub():
    m = f.matrix([[5, 3], [6, 8]])
    assert (m - 1) == f.matrix([[4, 2], [5, 7]])


def test_transpose():
    m = f.matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    assert m.T == f.matrix([[1, 5, 9],
                            [2, 6, 10],
                            [3, 7, 11],
                            [4, 8, 12]])

    assert m[:-1, 1:].T == f.matrix([[2, 6],
                                     [3, 7],
                                     [4, 8]])

    assert m.T.T == m


def test_abs():
    assert abs(f.vector([-1, 2])) == f.vector([1, 2])


def test_neg():
    assert - f.vector([-1, 2]) == f.vector([1, -2])


def test_pow():
    assert f.vector([-1, 2, -3]) ** 3 == f.vector([-1, 8, -27])


def test_floor():
    assert math.floor(f.vector([-1.3, 2.8, 3])) == f.vector([-2, 2, 3])


def test_ceil():
    assert math.ceil(f.vector([-1.3, 2.8, 3])) == f.vector([-1, 3, 3])


def test_round():
    assert round(f.vector([-1.3, 2.8, 3])) == f.vector([-1, 3, 3])


def test_trunc():
    assert math.trunc(f.vector([-1.3, 2.8, 3])) == f.vector([-1, 2, 3])


def test_matmul():
    m = f.matrix([[1, 0, 2], [3, -1, -1]])
    n = f.matrix([[1, 2], [-1, 1], [3, -1]])
    assert m @ n == f.matrix([[7.0, 0.0], [1.0, 6.0]])

    o = m[:, :-1]
    p = m[:, 1:]
    assert o @ p == f.matrix([[0, 2], [1, 7]])

    q = m[::2, :]
    r = n[:, 0]
    assert q @ r == f.matrix([[7]])


def test_copy():
    m = f.matrix([[1, 0, 2], [3, -1, -1]])
    m_copy = m.copy()
    assert m_copy is not m
    assert m_copy == m


def test_iter():
    assert list(f.matrix([[1, 2, 3], [4, 5, 6]])) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_contains():
    m = f.matrix([[1, 0, 2], [3, -1, -1]])
    assert 3 in m
    assert 4 not in m


def test_str():
    assert str(f.matrix([[1, 0, 2], [3, -1, -1]])) == '[[1.  , 0.  , 2.  ],\n [3.  , -1.  , -1.  ]]'


def test_repr():
    assert repr(f.matrix([[1, 0, 2], [3, -1, -1]])) == 'matrix([[1.  , 0.  , 2.  ],\n        [3.  , -1.  , -1.  ]])'
