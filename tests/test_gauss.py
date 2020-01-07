from frobenius import factory as f


def test_palu():
    a = f.matrix([[10, -7, 0], [-3, 2, 6], [5, -1, 5]])  # carrée !
    assert a.nrows == a.ncols
    n = a.nrows

    # Init
    l = f.eye(a.nrows)
    u = a.copy()
    p = list(range(a.nrows))

    for i in range(n):
        max_pivot_row_idx = max(enumerate(abs(u[:, i])), key=lambda x: x[1])[0]

        if max_pivot_row_idx != i:
            # swaps
            pivot_row = u[max_pivot_row_idx].copy()
            u[max_pivot_row_idx, :] = u[i, :]
            u[i, :] = pivot_row

            # l



        max(enumerate(abs(u[:, i])), key=lambda x:x[1])
        max(enumerate(abs(u[:, i])), key=lambda x: x[1])
        column = enumerate(u[:, i])
        pivot = u[i, i]


def test_gauss():
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
