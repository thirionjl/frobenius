from typing import Tuple, Collection

from frobenius import validate, factory
from frobenius.matrix import MatrixType

epsilon = 1e-6


class LuDecomposition:

    def __init__(self, lu_matrix: MatrixType, permutation: Tuple[int],
                 cnt_row_exchanges: int):
        self._lu_matrix = lu_matrix
        self._perm = permutation
        self._cnt_row_exchanges = cnt_row_exchanges
        self._dim = lu_matrix.nrows

    def l_matrix(self) -> MatrixType:
        mat = factory.eye(self._dim, self._dim)
        for row_idx in range(self._lu_matrix.nrows):
            for col_idx in range(self._lu_matrix.ncols):
                if row_idx < col_idx:
                    mat[row_idx, col_idx] = self._lu_matrix[row_idx, col_idx]
        return mat

    def u_matrix(self) -> MatrixType:
        mat = factory.zeros(self._dim, self._dim)
        for row_idx in range(self._lu_matrix.nrows):
            for col_idx in range(self._lu_matrix.ncols):
                if row_idx >= col_idx:
                    mat[row_idx, col_idx] = self._lu_matrix[row_idx, col_idx]
        return mat

    def __str__(self):
        return f"L = \n{self.l_matrix()}\n,\n" \
               f"U = \n{self.u_matrix()}\n,\n" \
               f"Perm = {self._perm}"

    def solve(self, b: MatrixType) -> MatrixType:
        # Reapply elimination to right-hand sides
        n = self._dim
        x = factory.zeros(n, b.ncols)
        lu = self._lu_matrix

        # Permute rows
        for i in range(n):
            x[i] = b[self._perm[i]]

        # Elimination
        for i in range(n):
            factors_column = lu[i + 1:, i]
            pivot_row = x[i, :]
            x[i + 1:] -= factors_column * pivot_row

        # Solve triangular system
        for i in reversed(range(n)):
            for k in range(i + 1, n):
                x[i] -= lu[i, k] * x[k]
            x[i] /= lu[i, i]

        return x


class RowReducedEchelonForm:

    def __init__(self, reduced_form: MatrixType,
                 permutation: Tuple[int],
                 m: int,
                 n: int,
                 pivot_cols: Collection[int]):
        self.reduced_form = reduced_form
        self.perm = permutation
        self.m = m
        self.n = n
        self.pivot_cols = pivot_cols
        pivot_cols_set = set(pivot_cols)
        self.free_cols = [i for i in range(self.n) if i not in pivot_cols_set]

    def nullspace(self) -> MatrixType:
        ns = factory.zeros(self.n, len(self.free_cols))

        for i, f in enumerate(self.free_cols):
            ns[f, i] = 1
            for j, p in enumerate(self.pivot_cols):
                ns[p, i] = - self.reduced_form[j, f]

        return ns


def rref(a: MatrixType) -> RowReducedEchelonForm:
    m, n = a.shape

    # Init
    lu = a.copy()
    perm = list(range(m))
    cnt_row_exchanges = 0

    j = 0
    i = 0
    pivot_cols = []
    while i < m and j < n:
        # All work will be done on sub-matrix lu[i:, j:]
        pivot_row_idx, pivot_value = max(enumerate(lu[i:, j], start=i),
                                         key=lambda x: abs(x[1]))

        if abs(pivot_value) < epsilon:
            j += 1
            continue

        if pivot_row_idx != i:
            # Execute row exchange on perm
            temp = perm[pivot_row_idx]
            perm[pivot_row_idx] = perm[i]
            perm[i] = temp

            # Execute Row exchange on LU
            temp = lu[pivot_row_idx].copy()
            lu[pivot_row_idx] = lu[i]
            lu[i] = temp

            cnt_row_exchanges += 1

        # Perform elimination
        pivot_cols.append(j)
        factors_column = lu[i + 1:, j] / pivot_value
        pivot_row = lu[i, j + 1:]

        lu[i + 1:, j] = 0
        lu[i + 1:, j + 1:] -= factors_column * pivot_row
        i += 1
        j += 1

    # UP
    # Solve system
    for i in reversed(range(i)):
        j = pivot_cols[i]
        pivot_value = lu[i, j]
        factors_column = lu[:i, j] / pivot_value
        pivot_row = lu[i, j + 1:]
        lu[:i, j + 1:] -= factors_column * pivot_row
        lu[:i, j] = 0
        lu[i, j] = 1
        lu[i, j + 1:] = pivot_row / pivot_value

    return RowReducedEchelonForm(lu, perm, m, n, pivot_cols)


def lu_decompose(a: MatrixType) -> LuDecomposition:
    validate.is_true("matrix", "must be square", a.is_square())
    n = a.nrows

    # Init
    lu = a.copy()
    perm = list(range(a.nrows))
    cnt_row_exchanges = 0

    for i in range(n):
        # All work will be done on sub-matrix lu[i:, i:]
        pivot_row_idx, pivot_value = max(enumerate(lu[i:, i], start=i),
                                         key=lambda x: abs(x[1]))

        if abs(pivot_value) < epsilon:
            raise ValueError(f'Matrix has not independent rows (or columns) '
                             f'and hence cannot be LU decomposed')

        if pivot_row_idx != i:
            # Execute row exchange on perm
            temp = perm[pivot_row_idx]
            perm[pivot_row_idx] = perm[i]
            perm[i] = temp

            # Execute Row exchange on LU
            temp = lu[pivot_row_idx].copy()
            lu[pivot_row_idx] = lu[i]
            lu[i] = temp

            cnt_row_exchanges += 1

        # Perform elimination
        factors_column = lu[i + 1:, i] / pivot_value
        pivot_row = lu[i, i + 1:]

        lu[i + 1:, i] = factors_column
        lu[i + 1:, i + 1:] -= factors_column * pivot_row

    return LuDecomposition(lu, perm, cnt_row_exchanges)
