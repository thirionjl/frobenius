from frobenius import factory as f
from frobenius import orthogonality as o


def test_gram_schmidt():
    a = f.matrix([[4, 5, 6], [0, 15, 0], [12, 0, 0]]).T
    q, perm = o.gram_schmidt(a)

    assert q == f.matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert perm == [1, 2, 0]


def test_qr_decompose():
    a = f.matrix([[0, 15, 0], [12, 0, 0], [4, 5, 6]]).T
    q, r = o.qr_decompose(a)

    print(q)
    print(r)

    assert q == f.matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert r == f.matrix([[15.0, 0.0, 5.0], [0.0, 12.0, 4.0], [0.0, 0.0, 6.0]])
