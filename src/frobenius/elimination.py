import math
from typing import Sequence, List, NamedTuple

from frobenius import validate, factory
from frobenius.matrix import MatrixType

epsilon = 1e-6


class Solution(NamedTuple):
    count: int
    solution: MatrixType
    nullspace: MatrixType


def rref(a: MatrixType) -> MatrixType:
    return echelon(a).rref()


def solve(a: MatrixType, b: MatrixType) -> Solution:
    m, n = a.shape
    validate.is_true("b", f"needs to be have {m} rows and at least 1 column ", b.nrows == m and b.ncols >= 1)
    e = echelon(a)
    s = e.solve(b)

    has_solution = next(filter(math.isnan, s), None) is None

    if not has_solution:
        return Solution(0, None, None)
    elif e.rank >= n:
        return Solution(1, s, None)
    elif e.rank < n:
        return Solution(math.inf, s, e.nullspace())


class LuDecomposition:
    def __init__(
            self,
            lu_matrix: MatrixType,
            permutation: Sequence[int],
            cnt_row_exchanges: int,
    ):
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
        return (
            f"L = \n{self.l_matrix()}\n,\n"
            f"U = \n{self.u_matrix()}\n,\n"
            f"Perm = {self._perm}"
        )

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


def lu_decompose(a: MatrixType) -> LuDecomposition:
    validate.is_true("matrix", "must be square", a.is_square())
    n = a.nrows

    # Init
    lu = a.copy()
    perm: List[int] = list(range(a.nrows))
    cnt_row_exchanges = 0

    for i in range(n):
        # All work will be done on sub-matrix lu[i:, i:]
        pivot_row_idx, pivot_value = max(
            enumerate(lu[i:, i], start=i), key=lambda x: abs(x[1])
        )

        if abs(pivot_value) < epsilon:
            raise ValueError(
                "Matrix has not independent rows (or columns) "
                "and hence cannot be LU decomposed"
            )

        if pivot_row_idx != i:
            # Execute row exchange on perm
            temp_idx = perm[pivot_row_idx]
            perm[pivot_row_idx] = perm[i]
            perm[i] = temp_idx

            # Execute Row exchange on LU
            temp_row = lu[pivot_row_idx].copy()
            lu[pivot_row_idx] = lu[i]
            lu[i] = temp_row

            cnt_row_exchanges += 1

        # Perform elimination
        factors_column = lu[i + 1:, i] / pivot_value
        pivot_row = lu[i, i + 1:]

        lu[i + 1:, i] = factors_column
        lu[i + 1:, i + 1:] -= factors_column * pivot_row

    return LuDecomposition(lu, perm, cnt_row_exchanges)


class EchelonForm:
    def __init__(
            self,
            matrix: MatrixType,
            lu_echelon: MatrixType,
            permutation: Sequence[int],
            pivot_cols: Sequence[int],
    ):
        self._matrix = matrix
        self._lu_echelon = lu_echelon
        self._perm = permutation
        self.m, self.n = self._matrix.shape
        self._pivot_cols = pivot_cols
        pivot_cols_set = set(pivot_cols)
        self._free_cols = [i for i in range(self.n) if i not in pivot_cols_set]
        self.rank = len(self._pivot_cols)
        self._rref = None

    def rref(self):
        if self._rref is None:
            # Echelon to Rref
            lu = self._lu_echelon.copy()
            for i in reversed(range(self.rank)):
                j = self._pivot_cols[i]
                pivot_value = lu[i, j]
                factors_column = lu[:i, j] / pivot_value
                pivot_row = lu[i, j + 1:]
                lu[:i, j + 1:] -= factors_column * pivot_row
                lu[:i, j] = 0
                if i + 1 < self.m:
                    lu[i + 1:, j] = 0
                lu[i, j] = 1
                lu[i, j + 1:] = pivot_row / pivot_value

            self._rref = lu

        return self._rref

    def colspace(self) -> MatrixType:
        cs = factory.zeros(self.m, self.rank)
        for i, c in enumerate(self._pivot_cols):
            cs[:, i] = self._matrix[:, c]
        return cs

    def rowspace(self) -> MatrixType:
        return self.rref()[: self.rank, :].T

    def nullspace(self) -> MatrixType:
        ns = factory.zeros(self.n, len(self._free_cols))

        for i, f in enumerate(self._free_cols):
            ns[f, i] = 1
            for j, p in enumerate(self._pivot_cols):
                ns[p, i] = -self.rref()[j, f]

        return ns

    def left_nullspace(self) -> MatrixType:
        bs = self._apply_elimination(factory.eye(self.m, self.m))
        return bs[self.rank:].T

    def solve(self, b: MatrixType) -> MatrixType:
        return self._find_single_solution(self._apply_elimination(b))

    def _find_single_solution(self, b):
        lu = self._lu_echelon
        r = self.rank

        # Take out non solvable rhs
        unsolvable_cols = set()
        for i in range(r, self.m):
            for j in range(b.ncols):
                sq = b[i, j] * b[i, j]
                if abs(sq) > epsilon:
                    unsolvable_cols.add(j)

        # Solve
        x = factory.zeros(self.n, b.ncols)
        for i in reversed(range(r)):
            j = self._pivot_cols[i]
            x[j] = b[i]
            for kk in range(i + 1, r):
                k = self._pivot_cols[kk]
                x[j] = x[j] - lu[i, k] * x[k]
            x[j] = x[j] / lu[i, j]

        # Set to NaN non solvable rhs
        for uc in unsolvable_cols:
            x[:, uc] = float("nan")

        return x

    def _apply_elimination(self, b):
        assert b.nrows == self.m
        # Reapply elimination to right-hand sides
        b_elim = factory.zeros(shape=b.shape)
        lu = self._lu_echelon
        # Permute rows
        for i in range(self.m):
            b_elim[i] = b[self._perm[i]]
        # Elimination on b matrix
        for i, j in enumerate(self._pivot_cols):
            factors_column = lu[i + 1:, j]
            b_elim[i + 1:, :] -= factors_column * b_elim[i]
        return b_elim


def echelon(a: MatrixType) -> EchelonForm:
    m, n = a.shape

    # Init
    lu = a.copy()
    perm: List[int] = list(range(m))
    cnt_row_exchanges = 0

    j = 0
    i = 0
    pivot_cols = []
    while i < m and j < n:
        # All work will be done on sub-matrix lu[i:, j:]
        pivot_row_idx, pivot_value = max(
            enumerate(lu[i:, j], start=i), key=lambda x: abs(x[1])
        )

        if abs(pivot_value) < epsilon:
            j += 1
            continue

        if pivot_row_idx != i:
            # Execute row exchange on perm
            temp_idx = perm[pivot_row_idx]
            perm[pivot_row_idx] = perm[i]
            perm[i] = temp_idx

            # Execute Row exchange on LU
            temp_row = lu[pivot_row_idx].copy()
            lu[pivot_row_idx] = lu[i]
            lu[i] = temp_row

            cnt_row_exchanges += 1

        # Perform elimination
        pivot_cols.append(j)
        factors_column = lu[i + 1:, j] / pivot_value
        pivot_row = lu[i, j + 1:]

        lu[i + 1:, j] = factors_column
        lu[i + 1:, j + 1:] -= factors_column * pivot_row
        i += 1
        j += 1

    return EchelonForm(a, lu, perm, pivot_cols)
