
import numpy as np
from scipy import linalg
import circ
import pytest


def get_col(dtype):
    if dtype is float:
        return np.asarray((2,5,4,0,1), dtype=dtype)
    elif dtype is complex:
        return np.asarray((2+1j,5-3j,4,0,1j), dtype=dtype)


@pytest.mark.parametrize("dtype", [float, complex])
def test_toeplitz(dtype):
    np.random.seed(12345)

    col = get_col(dtype)
    row = np.zeros_like(col)
    row[0] = col[0]
    row[1] = 1
    mat = (col, row)

    tmat = circ.ToeplitzLinearOperator(*mat)
    tinv = circ.ToeplitzLinearOperator(*mat, inverse=True)

    b = np.random.randn(len(col))

    x = tinv.matvec(b)

    assert np.allclose(x, linalg.solve_toeplitz(mat, b))
    assert np.allclose(tmat.matvec(x), b)


@pytest.mark.parametrize("dtype", [float, complex])
def test_circulant(dtype):
    np.random.seed(12345)

    col = get_col(dtype)

    C = circ.circulant(col)

    assert np.allclose(C, linalg.circulant(col))

    col = np.asarray((2,5,4,0,0), dtype=float)
    row = np.zeros_like(col)
    row[0] = col[0]

    C = circ.circulant(col, alpha=0)

    assert np.allclose(C, linalg.toeplitz(col,row))


@pytest.mark.parametrize("dtype", [float, complex])
def test_circulant_operator(dtype):
    np.random.seed(12345)

    col = get_col(dtype)

    cmat = circ.CirculantLinearOperator(col)
    cinv = circ.CirculantLinearOperator(col, inverse=True)

    b = np.random.randn(len(col))

    x = cinv.matvec(b)

    assert np.allclose(x, linalg.solve_circulant(col, b))
    assert np.allclose(cmat.matvec(x), b)


@pytest.mark.parametrize("alpha", [0.001, 0.5, 1.])
@pytest.mark.parametrize("dtype", [float, complex])
def test_alpha_circulant_operator(alpha, dtype):
    np.random.seed(12345)

    col = get_col(dtype)
    n = len(col)

    cmat = circ.AlphaCirculantLinearOperator(col, alpha=alpha)
    cinv = circ.AlphaCirculantLinearOperator(col, alpha=alpha, inverse=True)

    b = np.random.randn(n)

    C = circ.circulant(col, alpha=alpha)

    x = cinv.matvec(b)
    if dtype is float: x = x.real

    assert np.allclose(x, linalg.solve(C, b))
    assert np.allclose(cmat.matvec(x).real, b)
