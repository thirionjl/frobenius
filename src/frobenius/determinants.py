from frobenius import elimination
from frobenius.matrix import MatrixType


def det(m: MatrixType) -> float:
    if not m.is_square():
        raise ValueError("Determinants only make sense for square matrices")
    lu = elimination.lu_decompose(m)
    return lu.determinant()
