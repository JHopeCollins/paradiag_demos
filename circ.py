
import numpy as np
from scipy import linalg
from scipy.fft import fft, ifft
import scipy.sparse.linalg as spla


def circulant(c, alpha=1):
    C = linalg.circulant(c)
    n = len(c)
    for i in range(n-1):
        for j in range(i+1, n):
            C[i,j]*=alpha
    return C


class ToeplitzLinearOperator(spla.LinearOperator):
    def __init__(self, col, row, dtype=None, inverse=False):
        self.col = col
        self.row = row
        self.mat = tuple((col, row))
        shape = tuple((len(col), len(row)))
        super().__init__(shape=shape, dtype=dtype)
        if inverse:
            self.op = linalg.solve_toeplitz
        else:
            self.op = linalg.matmul_toeplitz

    def _matvec(self, v):
        return self.op(self.mat, v)


class CirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, dtype=None, inverse=False):
        self.col = col
        shape = tuple((len(col), len(col)))
        super().__init__(shape=shape, dtype=dtype)
        if inverse:
            self.op = self._solve
        else:
            self.op = self._matmul

        self.eigvals = fft(col, norm='backward')

    def _matmul(self, v):
        return ifft(fft(v)*self.eigvals)

    def _solve(self, v):
        return ifft(fft(v)/self.eigvals)

    def _matvec(self, v):
        return self.op(v)


class AlphaCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, alpha=1, dtype=None, inverse=False):
        self.alpha = alpha
        self.col = col
        n = len(col)
        shape = tuple((n, n))
        super().__init__(shape=shape, dtype=dtype)
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
        return self.op(v)

    def _to_eigvecs(self, v):
        return fft(v*self.gamma, norm='ortho')

    def _from_eigvecs(self, v):
        return ifft(v, norm='ortho')/self.gamma
