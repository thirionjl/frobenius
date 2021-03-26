import math

from frobenius import factory as f

epsilon = 1e-8


def test_gram():
    v1 = f.row_vector([1, 0, 0])
    v2 = f.row_vector([2, 0, 3])
    v3 = f.row_vector([4, 5, 6])
    a = f.stack(v1, v2, v3, vertical=False)
    a = f.stack(v1, v2, v3, vertical=True)

    a = f.matrix([[1, 0, 0], [2, 0, 3], [4, 5, 6]]).T
    a = f.matrix([[1, 0, 1], [1, 2, 1], [5, 7, 7]]).T

    q = f.zeros(a.nrows, a.ncols)
    for i in range(a.ncols):
        u = a[:, i]
        qs = q[:, :i]
        u = u - qs @ (qs.T @ u)
        # for j in range(i):
        #     coord = q[:, :i].T @ u
        #     u -= u * coord
        norm = math.sqrt(float(u.T @ u))
        q[:, i] = u / norm

    r = q.T @ a

    print("---Q---")
    print(q)
    print("---R---")
    print(r)

    print("---check---")
    print(q @ r)
