from typing import Optional, NamedTuple, Tuple, List

from frobenius import factory as f
from frobenius.matrix import MatrixType
from frobenius.norms import norm


class QR(NamedTuple):
    q: MatrixType
    r: MatrixType


def simple_gram_schmidt(a: MatrixType, n: Optional[int] = None) -> MatrixType:
    cnt_vectors = min(a.ncols, n) if n else a.ncols
    q = f.zeros(a.nrows, cnt_vectors)
    for i in range(cnt_vectors):
        u = a[:, i]
        qs = q[:, :i]
        u = u - qs @ (qs.T @ u)
        q[:, i] = u / norm(u)
    return q


def gram_schmidt(
    a: MatrixType, n: Optional[int] = None
) -> Tuple[MatrixType, List[int]]:
    cnt_vectors = min(a.ncols, n) if n else a.ncols
    q = a.copy()
    perm = list(range(cnt_vectors))

    for j in range(0, cnt_vectors):
        pivot_col_idx, pivot_norm = max(
            ((c, norm(q[:, c])) for c in range(j, cnt_vectors)), key=lambda x: x[1]
        )

        # Swap columns
        if j != pivot_col_idx:
            temp = q[:, j].copy()
            q[:, j] = q[:, pivot_col_idx]
            q[:, pivot_col_idx] = temp
            perm[pivot_col_idx], perm[j] = perm[j], perm[pivot_col_idx]

        # Normalize
        q[:, j] /= pivot_norm
        u = q[:, j]

        # Remove U component everywhere
        q[:, j + 1 :] -= u @ (u.T @ q[:, j + 1 :])

    return q, perm


def qr_decompose(a: MatrixType) -> QR:
    q, perm = gram_schmidt(a)

    aa = f.zeros(shape=a.shape)
    qq = f.zeros(shape=a.shape)
    for i in range(aa.ncols):
        aa[:, perm[i]] = a[:, i]
        qq[:, perm[i]] = q[:, i]

    return QR(qq, qq.T @ aa)
