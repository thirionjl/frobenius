from typing import Optional, NamedTuple

from frobenius import factory as f
from frobenius.matrix import MatrixType
from frobenius.norms import norm


class QR(NamedTuple):
    q: MatrixType
    r: MatrixType


def gram_schmidt(a: MatrixType, n: Optional[int] = None) -> MatrixType:
    cnt_vectors = min(a.ncols, n) if n else a.ncols
    q = f.zeros(a.nrows, cnt_vectors)
    for i in range(cnt_vectors):
        u = a[:, i]
        qs = q[:, :i]
        u = u - qs @ (qs.T @ u)
        q[:, i] = u / norm(u)
    return q


def qr_decompose(a: MatrixType) -> QR:
    q = gram_schmidt(a)
    r = q.T @ a
    return QR(q, r)
