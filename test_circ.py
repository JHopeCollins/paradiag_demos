
import numpy as np
from scipy import linalg
import circ
import pytest


@pytest.fixture(params=[float, complex])
def col(request):
    dtype = request.param
    if dtype is float:
        return np.asarray((2,5,4,0,1), dtype=dtype)
    elif dtype is complex:
        return np.asarray((2+1j,5-3j,4,0,1j), dtype=dtype)


def test_toeplitz(col):
    np.random.seed(12345)

    #col = get_col(dtype)
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


def test_circulant(col):
    np.random.seed(12345)

    C = circ.circulant(col)

    assert np.allclose(C, linalg.circulant(col))

    col = np.asarray((2,5,4,0,0), dtype=float)
    row = np.zeros_like(col)
    row[0] = col[0]

    C = circ.circulant(col, alpha=0)

    assert np.allclose(C, linalg.toeplitz(col,row))


def test_circulant_operator(col):
    np.random.seed(12345)

    cmat = circ.CirculantLinearOperator(col)
    cinv = circ.CirculantLinearOperator(col, inverse=True)

    b = np.random.randn(len(col))

    x = cinv.matvec(b)

    assert np.allclose(x, linalg.solve_circulant(col, b))
    assert np.allclose(cmat.matvec(x), b)


@pytest.mark.parametrize("alpha", [0.001, 0.5, 1.])
def test_alpha_circulant_operator(alpha, col):
    np.random.seed(12345)

    n = len(col)

    cmat = circ.AlphaCirculantLinearOperator(col, alpha=alpha)
    cinv = circ.AlphaCirculantLinearOperator(col, alpha=alpha, inverse=True)

    b = np.random.randn(n)

    C = circ.circulant(col, alpha=alpha)

    x = cinv.matvec(b)

    assert np.allclose(x, linalg.solve(C, b))
    assert np.allclose(cmat.matvec(x).real, b)
