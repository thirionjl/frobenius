from frobenius import factory as f, elimination

epsilon = 1e-8


def test_gauss1():
    a = f.matrix([[10, -7, 0], [-3, 2, 6], [5, -1, 5]])  # carrée !
    assert a.nrows == a.ncols
    # n = a.nrows
    #
    # # Init
    # lu = a.copy()
    # perm = list(range(a.nrows))
    #
    # for i in range(n):
    #     # All work will be done on submatrix lu[i:, i:]
    #     pivot_row_idx, pivot_value = max(enumerate(lu[i:, i]),
    #                                      key=lambda x: x[1])
    #
    #     if pivot_value < epsilon:  # TODO track dependant rows !!
    #         raise ValueError(f'Matrix has not independent rows (or columns) '
    #                          f'and hence cannot be LU decomposed')
    #
    #     if pivot_row_idx != i:
    #         # Execute row exchange on perm
    #         temp = perm[pivot_row_idx]
    #         perm[pivot_row_idx] = perm[i]
    #         perm[i] = temp
    #
    #         # Execute Row exchange on LU
    #         temp = lu[pivot_row_idx].copy()
    #         lu[pivot_row_idx] = lu[i]
    #         lu[i] = temp
    #
    #     # Perform elimination
    #     factors_column = lu[i + 1:, :] / pivot_value
    #     pivot_row = lu[i, i + 1:]
    #
    #     lu[i + 1:] = factors_column
    #     lu[i + 1:, i + 1:] -= factors_column * pivot_row
    #


def test_gauss2():
    a = f.matrix([[1, 2, 2, 2], [2, 4, 6, 8], [3, 6, 8, 10]])
    echelon = elimination.echelon(a)

    print(f"rref=\n{echelon.rref()}")

    # print(f"rowspace=\n{echelon.rowspace()}")
    # print(f"colspace=\n{echelon.colspace()}")
    print(f"nullspace=\n{echelon.nullspace()}")
    # print(f"left nullspace=\n{echelon.left_nullspace()}")

    print(f"x_particular=\n{echelon.solve(f.vector([1, 4, 50]))}")
