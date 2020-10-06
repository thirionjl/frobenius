from frobenius import factory as f

epsilon = 1e-6


def test_gauss1():
    a = f.matrix([[10, -7, 0], [-3, 2, 6], [5, -1, 5]])  # carrée !
    assert a.nrows == a.ncols
    n = a.nrows
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
    a = f.matrix([[1, 2, 1, 1, 0, 0], [3, 8, 1, 0, 1, 0], [0, 4, 1, 0, 0, 1]])
    l = f.eye(a.nrows)
    u = a.copy()

    # 1 Clean col 1
    # Find pivot
    pivot_col_idx = 0

    # Descent
    for current_col_idx in range(u.nrows):
        pivot_row_idx = current_col_idx
        pivot_col_idx = pivot_row_idx  # TODO
        pivot = u[pivot_row_idx, pivot_col_idx]

        for other_row_idx in range(pivot_row_idx + 1, u.nrows):
            val = u[other_row_idx, current_col_idx]
            if val != 0.0:
                factor = val / pivot
                u[other_row_idx] -= factor * u[pivot_row_idx]
                l[other_row_idx, current_col_idx] = factor

    print("===Initial===")
    print(a)

    print("===L matrix ===")
    print(l)

    print("===U matrix ===")
    print(u)

    # RR
    r = u.copy()
    for current_row_idx in reversed(range(r.nrows)):
        current_col_idx = current_row_idx
        pivot_row_idx = current_row_idx
        pivot = r[current_row_idx, current_col_idx]

        for other_row_idx in range(0, current_row_idx):
            val = r[other_row_idx, current_col_idx]
            if val != 0.0:
                factor = val / pivot
                r[other_row_idx] -= factor * r[pivot_row_idx]

        r[pivot_row_idx] /= pivot

    print("===Inverse matrix ===")
    print(r[:, 3:])
