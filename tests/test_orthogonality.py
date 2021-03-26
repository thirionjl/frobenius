from frobenius import factory as f
from frobenius import orthogonality as o


def test_gram_schmidt():
    a = f.matrix([[1, 0, 0], [2, 0, 3], [4, 5, 6]]).T
    q = o.gram_schmidt(a)

    assert q == f.matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])


def test_qr_decompose():
    a = f.matrix([[1, 0, 0], [2, 0, 3], [4, 5, 6]]).T
    q, r = o.qr_decompose(a)

    assert q == f.matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert r == f.matrix([[1, 2, 4], [0, 3, 6], [0, 0, 5]])
