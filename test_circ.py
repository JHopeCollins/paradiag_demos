
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


def test_block_toeplitz_operator():
    np.random.seed(12345)
    nb = 5
    col = np.asarray((2,5,4,0,1), dtype=float)

    col0 = col.copy()
    col1 = col.copy()

    col0[0]-=1
    col1[0]+=1

    block0 = circ.CirculantLinearOperator(col0)
    block1 = circ.CirculantLinearOperator(col1)

    tbmat = circ.BlockToeplitzLinearOperator(nb, block0, block1)

    bmat0 = circ.circulant(col0)
    bmat1 = circ.circulant(col1)

    Tb =  np.kron(np.eye(nb, k=-1), bmat0) + np.kron(np.eye(nb, k=0), bmat1)

    b = np.random.randn(Tb.shape[0])

    x = tbmat.matvec(b)

    assert np.allclose(x, np.matmul(Tb, b))


def test_block_circulant_operator():
    np.random.seed(12345)

    block_col0 = np.asarray((2,5,4,0,1), dtype=float)
    block_col1 = np.asarray((-1,3,0,-2,0), dtype=float)

    def block_op(l0, l1):
        col = l0*block_col0 + l1*block_col1
        return circ.CirculantLinearOperator(col)

    block = circ.circulant(block_col0 + block_col1)

    col0 = np.zeros(4)
    col0[0] = 1
    col0[1] = -1

    col1 = np.zeros(4)
    col1[0] = 0.75
    col1[1] = 0.25

    Cop = circ.BlockCirculantLinearOperator(col0, col1, block_op, len(block_col0))

    Cmat = np.kron(circ.circulant(col0), circ.circulant(block_col0)) \
          +np.kron(circ.circulant(col1), circ.circulant(block_col1))

    b = np.random.randn(Cmat.shape[0])

    x = Cop.matvec(b)

    assert np.allclose(x, np.matmul(Cmat, b))


def test_block_alpha_circulant_operator():
    np.random.seed(12345)

    alpha = 0.1
    block_col0 = np.asarray((2,5,0,0,0), dtype=float)
    block_col1 = np.asarray((-1,3,0,0,0), dtype=float)
    #block_col0 = np.asarray((2,5), dtype=float)
    #block_col1 = np.asarray((-1,3), dtype=float)

    def block_op(l0, l1):
        col = l0*block_col0 + l1*block_col1
        return circ.CirculantLinearOperator(col)

    block = circ.circulant(block_col0 + block_col1)

    col0 = np.zeros(4)
    col0[0] = 1
    col0[1] = -1

    col1 = np.zeros(4)
    col1[0] = 0.75
    col1[1] = 0.25

    Cop = circ.BlockAlphaCirculantLinearOperator(col0, col1, block_op, len(block_col0), alpha)

    Cmat = np.kron(circ.circulant(col0, alpha), circ.circulant(block_col0)) \
          +np.kron(circ.circulant(col1, alpha), circ.circulant(block_col1))

    n = Cmat.shape[0]

    b = np.random.randn(n)
    xt = Cop.matvec(b)
    xc = np.matmul(Cmat, b)

    # for i in range(n):
    #     b = np.zeros(n)
    #     b[i] = 1

    #     xt = Cop.matvec(b)
    #     xc = np.matmul(Cmat, b)

    #     print(b)
    #     print(xt)
    #     print(xc)
    #     print("")
    assert np.allclose(xt, xc)
