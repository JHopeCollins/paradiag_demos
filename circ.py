
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
    def __init__(self, col, row, inverse=False):
        self.col = col
        self.row = row
        self.mat = tuple((col, row))

        self.shape = tuple((len(col), len(row)))
        self.dtype = col.dtype

        if inverse:
            self.op = linalg.solve_toeplitz
        else:
            self.op = linalg.matmul_toeplitz

    def _matvec(self, v):
        return self.op(self.mat, v)


class CirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, inverse=False):
        self.col = col

        self.is_complex = np.iscomplexobj(col)
        self.dtype = col.dtype
        self.shape = tuple((len(col), len(col)))

        if inverse:
            self.op = self._solve
        else:
            self.op = self._matmul

        self.eigvals = fft(col, norm='backward')

    def _matmul(self, v):
        y = ifft(fft(v)*self.eigvals)
        if self.is_complex:
            return y
        else:
            return y.real

    def _solve(self, v):
        y = ifft(fft(v)/self.eigvals)
        if self.is_complex:
            return y
        else:
            return y.real

    def _matvec(self, v):
        return self.op(v)


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
        y = self._from_eigvecs(self._to_eigvecs(v)*self.eigvals)
        if self.is_complex:
            return y
        else:
            return y.real

    def _solve(self, v):
        y = self._from_eigvecs(self._to_eigvecs(v)/self.eigvals)
        if self.is_complex:
            return y
        else:
            return y.real

    def _matvec(self, v):
        return self.op(v)

    def _to_eigvecs(self, v):
        return fft(v*self.gamma)

    def _from_eigvecs(self, v):
        return ifft(v)/self.gamma
