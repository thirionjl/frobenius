from frobenius import factory as f
from frobenius.matrix import Vector, MatrixType, Shape


def test_vector_creation():
    f.vector([1, 2, 3, 4])


def test_vector_indexing():
    v = f.vector([1, 2, 3, 4])

    assert v[0] == 1.0
    assert v[1] == 2.0
    assert v[2] == 3.0
    assert v[3] == 4.0

    sub_vector = v[2:]
    assert isinstance(sub_vector, Vector)
    assert len(sub_vector) == 2


def test_vector_set():
    v = f.vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = v[::3]
    x = w[::2]

    assert w == f.vector([1, 4, 7, 10])
    assert x == f.vector([1, 7])

    v[0:-2:2] = f.vector([100, 300, 500, 700, 900])
    assert v == f.vector([100, 2, 300, 4, 500, 6, 700, 8, 900, 10, 11, 12])
    assert w == f.vector([100, 4, 700, 10])
    assert x == f.vector([100, 700])


def test_matrix_creation():
    m = f.matrix([[1], [2]])
    n = f.matrix([[1, 2], [3, 4]])
    repr(m)


def test_matrix_basic_indexing():
    m = f.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    m[:-1, :-1] = f.matrix([[10, 20], [40, 50]])
    assert m == f.matrix([[10, 20, 3], [40, 50, 6], [7, 8, 9]])


def test_matrix_basic_indexing2():
    a = f.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a[0] += a[1]
    assert a == f.matrix([[5, 7, 9], [4, 5, 6], [7, 8, 9]])


def test_matrix_sliced_indexing():
    m: MatrixType = f.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    sm: MatrixType = m[:-1, :-1]
    assert isinstance(sm, MatrixType)
    assert sm.nrows == 2
    assert sm.ncols == 2
    assert sm.size == 4
    assert sm[0, 0] == 1.0
    assert sm[0, 1] == 2.0
    assert sm[1, 0] == 4.0
    assert sm[1, 1] == 5.0

    sm2: MatrixType = m[1:2, 1]
    assert sm2.size == 1


def test_matmul():
    m = f.matrix([[1, 0, 2], [3, -1, -1]])
    n = f.matrix([[1, 2], [-1, 1], [3, -1]])
    assert m @ n == f.matrix([[7.0, 0.0], [1.0, 6.0]])


def test_add():
    m = f.matrix([[1, 2], [3, 4]])
    r = f.matrix([[100, 200]])
    c = f.matrix([[100], [200]])
    assert (m + m) == f.matrix([[2, 4], [6, 8]])
    assert (m + r) == f.matrix([[101, 202], [103, 204]])
    assert (m + c) == f.matrix([[101, 102], [203, 204]])


def test_walk():
    m: MatrixType = f.matrix([[1, 2], [4, 5], [7, 8]])
    for row_idx, row in enumerate(m.rows()):
        print(f'\nRow {row_idx}: ')
        for col_idx, col in enumerate(row):
            print(col, end=' ')

    for col_idx, col in enumerate(m.columns()):
        print(f'\nColumn {col_idx}: ')
        for row_idx, row in enumerate(col):
            print(row, end=' ')


def test_reduce():
    m: MatrixType = f.matrix([[2, -1, 0], [-1, 2, -1], [0, -3, 4]])

    # Max in col 0
    col_idx = 0
    l = sorted(enumerate(m.row(col_idx)), key=lambda t: t[1][col_idx], reverse=True)

    row1 = f.shaped(m.row(1), Shape(1, 3))
    print()
    print(m + row1)
