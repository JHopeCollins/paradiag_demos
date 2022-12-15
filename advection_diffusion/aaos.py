
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.sparse import linalg as spla
from scipy import linalg

from math import pi

### === --- --- === ###
#
# Solve the linear advection diffusion equation
#   using sequential timestepping
#
# ADE solved on a periodic mesh using:
#   time discretisation: implicit theta-method
#   space discretisation: second order central differences
#
### === --- --- === ###

# parameters

alpha = 0.001

# number of time and space points
nt = 256
nx = 512

# size of domain
lx = float(nx)
dx = lx/nx

# velocity and reynolds number
u = 1
re = 10
cfl = 0.7

# timestep
nu = 2*u/re
dt = cfl*dx/u

# parameter for theta timestepping
theta=0.5

# width of initial profile
width = lx/4

cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

# mesh
x = np.linspace( start=-lx/2, stop=lx/2, num=nx, endpoint=False )

# identity mass matrix
M = np.zeros_like(x)
M[0] = 1

# gradient matrix
Ka = np.zeros_like(x)
Ka[-1] = 1/(2*dx)
Ka[1] = -1/(2*dx)

# laplacian matrix
Kd = np.zeros_like(x)
Kd[-1] = 1/dx**2
Kd[0] = -2/dx**2
Kd[1] = 1/dx**2

Ka = Ka
Kd = Kd

K = u*Ka - nu*Kd

un0_col = -M + dt*(1 - theta)*K
un1_col = M + dt*theta*K


# linear operators
class CirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col, inverse=False):
        self.col = col
        self.shape = tuple((len(col), len(col)))
        self.dtype = col.dtype
        self.is_complex = np.iscomplexobj(col)

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

A0 = CirculantLinearOperator(un0_col, inverse=False)
A1 = CirculantLinearOperator(un1_col, inverse=False)

# initial conditions
qinit = np.zeros_like(x)
qinit[:] = 1 + np.cos(np.minimum(2*pi*np.abs(x+lx/4)/width, pi))

q = np.zeros((nt, nx))


class BlockToeplitzLinearOperator(spla.LinearOperator):
    def __init__(self, nblocks, block0, block1):
        self.block0 = block0
        self.block1 = block1

        self.nblocks = nblocks
        self.block_dim = block0.shape[0]

        self.dim = self.nblocks*self.block_dim
        self.shape = tuple((self.dim, self.dim))
        self.dtype = block0.dtype

    def _matvec(self, v):
        v = v.reshape((self.nblocks, self.block_dim))
        w = np.zeros_like(v)

        w[0]+= self.block1.matvec(v[0])
        for b in range(1, self.nblocks):
            w[b]+= self.block0.matvec(v[b-1])
            w[b]+= self.block1.matvec(v[b])

        return w.reshape(self.dim)


class BlockCirculantLinearOperator(spla.LinearOperator):
    def __init__(self, col0, col1, block_op, block_dim, alpha=1):
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


A = BlockToeplitzLinearOperator(nt, A0, A1)

b1 = np.zeros(nt)
b1[0] = 1/dt
b1[1] = -1/dt

b2 = np.zeros(nt)
b2[0] = theta
b2[1] = 1-theta

def block_op(l1, l2):
    col = l1*M + l2*K
    assert np.iscomplexobj(col)
    return CirculantLinearOperator(col, inverse=True)

P = BlockCirculantLinearOperator(b1, b2, block_op, nx, alpha)


rhs = np.zeros_like(q)
rhs[0]-= A0.matvec(qinit)

q = q.reshape(nt*nx)
rhs = rhs.reshape(nt*nx)

niterations = 0


def gmres_callback(y):
    global niterations
    print(f"niterations: {str(niterations).rjust(3,' ')}  |  residual:  {y}")
    niterations += 1
    return


q, exit_code = spla.gmres(A, rhs, M=P,
                          tol=1e-12, atol=1e-12,
                          callback=gmres_callback,
                          callback_type='pr_norm')

print(f"gmres exit code: {exit_code}")
print(f"gmres iterations: {niterations}")
print(f"residual: {linalg.norm(rhs-A.matvec(q))}")

# plotting
q = q.reshape((nt,nx))
plt.plot(x,qinit,label='i')
for i in range(20,nt,20):
    plt.plot(x,q[i],label=str(i))
#plt.legend(loc='center left')
plt.grid()
plt.show()
