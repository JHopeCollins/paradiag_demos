
import numpy as np
from scipy import linalg
from scipy.fft import fft, ifft
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt

### === --- --- === ###
#
# Solve Dalhquist's ODE using implicit theta method
# solving the all-at-once system with ParaDiag
#
# y = e^{lamda*t}
# dydt = lambda*e^{i*lamda*t}
# dydt = lambda*y
#
# dy/dt = lambda*y
#
### === --- --- === ###

# parameters

verbose = False

nt = 1024
dt = 0.25
theta = 0.5

y0 = 1

lamda = -0.01 + 1.0j

dtype = complex

### timestepping toeplitz matrices

## mass toeplitz

# first column
b1 = np.zeros(nt, dtype=dtype)
b1[0] = 1/dt
b1[1] = -1/dt

# first row
r1 = np.zeros_like(b1)
r1[0] = b1[0]

## function toeplitz

# first column
b2 = np.zeros(nt, dtype=dtype)
b2[0] = -lamda*theta
b2[1] = -lamda*(1-theta)

# first row
r2 = np.zeros_like(b2)
r2[0] = b2[0]

### all-at-once system

## Jacobian

acol = b1 + b2
arow = r1 + r2
a = tuple((acol, arow))


class ToeplitzLinearOperator(spla.LinearOperator):
    def __init__(self, col, row, dtype=None, inverse=False):
        self.col = col
        self.row = row
        shape = tuple((len(col), len(row)))
        super().__init__(shape=shape, dtype=dtype)
        if inverse:
            self.op = linalg.solve_toeplitz
        else:
            self.op = linalg.matmul_toeplitz

    def _matvec(self, v):
        return self.op((self.col, self.row), v)


A = ToeplitzLinearOperator(acol, arow, dtype=dtype)

### paradiag preconditioner


class CirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, dtype=None, inverse=False):
        self.col = col
        shape = tuple((len(col), len(col)))
        super().__init__(shape=shape, dtype=dtype)
        if inverse:
            self.op = self._solve
        else:
            self.op = self._matmul

    def _matmul(self, v):
        return ifft(fft(v)*fft(self.col))

    def _solve(self, v):
        return ifft(fft(v)/fft(self.col))

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


#P = CirculantLinearOperator(acol, dtype=dtype, inverse=True)
P = AlphaCirculantLinearOperator(acol, alpha=1, dtype=dtype, inverse=True)

### right hand side

# initial condition
rhs = np.zeros(nt, dtype=dtype)
rhs[0] = -(b1[1] + b2[1])*y0

### solve all-at-once system

niterations = 0


def gmres_callback(pr_norm):
    global niterations
    print(f"niterations: {niterations} | residual: {pr_norm}")
    niterations += 1
    return


y, exit_code = spla.gmres(A, rhs, M=P,
                          tol=1e-14, atol=1e-14,
                          callback=gmres_callback,
                          callback_type='pr_norm')
print(f"gmres exit code: {exit_code}")
print(f"gmres iterations: {niterations}")

if verbose:
    B1 = linalg.toeplitz(b1, r1)
    B2 = linalg.toeplitz(b2, r2)
    A = B1 + B2
    print("B1")
    print(B1)
    print("B2")
    print(B2)
    print("A")
    print(A)
    print("rhs")
    print(rhs)
    print("y")
    print(y)
else:
    # plt.plot(y.real, y.imag)
    plt.plot(y.real)
    plt.show()
