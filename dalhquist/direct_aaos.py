
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt

from sys import exit

### === --- --- === ###
#
# Solve Dalhquist's ODE using implicit theta method
# and a direct solve of the all-at-once system
#
### === --- --- === ###

# y = e^{i*lamda*t}
# dydt = i*lambda*e^{i*lamda*t}
# dydt = i*lambda*y

# parameters

verbose = False

nt = 256
dt = 0.2
theta = 0.5

y0 = 1

#lamda = -0.1
lamda = -0.05 + 0.5j

dtype = complex

# timestepping toeplitz matrices
zeros = np.zeros(nt, dtype=dtype)

# mass toeplitz
b1 = np.zeros(nt, dtype=dtype)
b1[0] = 1
b1[1] = -1
B1 = linalg.toeplitz(b1, zeros)

if verbose:
    print("B1")
    print(B1)

# function toeplitz
b2 = np.zeros(nt, dtype=dtype)
b2[0] = theta
b2[1] = 1-theta
B2 = linalg.toeplitz(b2, zeros)

if verbose:
    print("B2")
    print(B2)

# all-at-once system
A = (1/dt)*B1 - lamda*B2

if verbose:
    print("A")
    print(A)

# rhs
rhs = np.zeros(nt, dtype=dtype)

# initial condition
rhs[0] = -(b1[1]/dt + b2[1]*lamda)*y0

if verbose:
    print("rhs")
    print(rhs)

# dy/dt = lambda*y

y = linalg.solve(A, rhs)

if verbose:
    print("y")
    print(y)

if not verbose:
    # plt.plot(y.real, y.imag)
    plt.plot(y.real)
    plt.show()
