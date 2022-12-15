
import numpy as np
from scipy import linalg
from scipy.fft import fft, ifft
import scipy.sparse.linalg as spla
from operator import add
from functools import reduce


def circulant(c, alpha=1):
    C = linalg.circulant(c)
    n = len(c)
    for i in range(n-1):
        for j in range(i+1, n):
            C[i,j]*=alpha
    return C


class ToeplitzLinearOperator(spla.LinearOperator):
    def __init__(self, col, row, inverse=False):
        self.col = col
        self.row = row
        self.mat = tuple((col, row))

        self.shape = tuple((len(col), len(row)))
        self.dtype = col.dtype

        if inverse:
            self.op = self._solve
        else:
            self.op = self._matmul

    def _matmul(self, v):
        return linalg.matmul_toeplitz(self.mat, v)

    def _solve(self, v):
        return linalg.solve_toeplitz(self.mat, v)

    def _matvec(self, v):
        return self.op(v)


class CirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, inverse=False):
        self.col = col

        self.is_complex = np.iscomplexobj(col)
        self.dtype = col.dtype
        self.shape = tuple((len(col), len(col)))

        if inverse: self.op = self._solve
        else: self.op = self._matmul

        self.eigvals = fft(col, norm='backward')

    def _matmul(self, v):
        return ifft(fft(v)*self.eigvals)

    def _solve(self, v):
        return ifft(fft(v)/self.eigvals)

    def _matvec(self, v):
        y = self.op(v)
        if self.is_complex: return y
        else: return y.real


class AlphaCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, alpha=1, inverse=False):
        self.alpha = alpha
        self.col = col
        n = len(col)

        self.is_complex = np.iscomplexobj(col)
        self.dtype = col.dtype
        self.shape = tuple((n, n))

        if inverse:
            self.op = self._solve
        else:
            self.op = self._matmul

        # fft weighting
        self.gamma = alpha**(np.arange(n)/n)

        # eigenvalues
        self.eigvals = fft(col*self.gamma, norm='backward')

    def _matmul(self, v):
        return self._from_eigvecs(self._to_eigvecs(v)*self.eigvals)

    def _solve(self, v):
        return self._from_eigvecs(self._to_eigvecs(v)/self.eigvals)

    def _matvec(self, v):
        y = self.op(v)
        if self.is_complex:
            return y
        else:
            return y.real

    def _to_eigvecs(self, v):
        return fft(v*self.gamma)

    def _from_eigvecs(self, v):
        return ifft(v)/self.gamma


class BlockToeplitzLinearOperator(spla.LinearOperator):
    def __init__(self, nblocks, block0, block1):
        self.block0 = block0
        self.block1 = block1

        self.nblocks = nblocks
        self.block_dim = block0.shape[0]

        self.dim = self.nblocks*self.block_dim
        self.shape = tuple((self.dim, self.dim))

    def _matvec(self, v):
        v = v.reshape((self.nblocks, self.block_dim))
        w = np.zeros_like(v)

        w[0]+= self.block1.matvec(v[0])
        for b in range(1, self.nblocks):
            w[b]+= self.block0.matvec(v[b-1])
            w[b]+= self.block1.matvec(v[b])

        return w.reshape(self.dim)


class BlockLinearOperator(spla.LinearOperator):
    def __init__(self, coeffs, blocks):
        assert len(blocks) == len(coeffs)
        assert len(set((b.dtype for b in blocks)))
        assert len(set((b.shape for b in blocks)))
        self.coeffs = coeffs
        self.blocks = blocks
        self.shape = blocks[0].shape
        self.dtype = blocks[0].dtype
        self.A = reduce(add, (c*b for c,b in zip(self.coeffs, self.blocks)))

    def update_coeffs(self, coeffs):
        self.coeffs = coeffs
        self.A = reduce(add, (c*b for c,b in zip(self.coeffs, self.blocks)))

    def _matvec(self, v):
        return linalg.solve(self.A, v)


class BlockCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col0, col1, block_op, block_dim):
        self.nblocks = len(col0)
        self.block_dim = block_dim
        self.dim = self.nblocks*self.block_dim
        self.shape = tuple((self.dim, self.dim))
        self.dtype = col0.dtype

        eigvals0 = fft(col0, norm='backward')
        eigvals1 = fft(col1, norm='backward')
        eigvals = zip(eigvals0, eigvals1)

        self.blocks = tuple((block_op(l1, l2)
                             for l1, l2 in eigvals))

    def _to_eigvecs(self, v):
        return fft(v, axis=0)

    def _from_eigvecs(self, v):
        return ifft(v, axis=0)

    def _block_solve(self, v):
        for i in range(self.nblocks):
            v[i] = self.blocks[i].matvec(v[i])
        return v

    def _matvec(self, v):
        y = v.reshape((self.nblocks, self.block_dim))
        y = self._to_eigvecs(y)
        y = self._block_solve(y)
        y = self._from_eigvecs(y)
        return y.reshape(self.dim).real


class BlockAlphaCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col0, col1, block_op, block_dim, alpha):
        self.nblocks = len(col0)
        self.block_dim = block_dim
        self.dim = self.nblocks*self.block_dim
        self.shape = tuple((self.dim, self.dim))
        self.dtype = col0.dtype

        self.gamma = alpha**(np.arange(self.nblocks)/self.nblocks)

        eigvals0 = fft(col0*self.gamma, norm='backward')
        eigvals1 = fft(col1*self.gamma, norm='backward')
        eigvals = zip(eigvals0, eigvals1)

        self.blocks = tuple((block_op(l1, l2)
                             for l1, l2 in eigvals))

    def _to_eigvecs(self, v):
        y = np.matmul(np.diag(self.gamma), v)
        return fft(y, axis=0)

    def _from_eigvecs(self, v):
        y = ifft(v, axis=0)
        return np.matmul(np.diag(1/self.gamma), y)

    def _block_solve(self, v):
        for i in range(self.nblocks):
            v[i] = self.blocks[i].matvec(v[i])
        return v

    def _matvec(self, v):
        y = v.reshape((self.nblocks, self.block_dim))
        y = self._to_eigvecs(y)
        y = self._block_solve(y)
        y = self._from_eigvecs(y)
        return y.reshape(self.dim).real
