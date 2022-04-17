
import numpy as np
import scipy.linalg as linalg

import matplotlib.pyplot as plt

# change this to use fft
def circ_mul( c, x ):
    return np.matmul(linalg.circulant(c),x)

# newton iterations
def newton_solve( func,         # function to solve
                  fprime,       # return derivative of function at x
                  fprime_solve, # solve func(x)/fprime(x)
                  x,            # initial guess
                  tol=1e-7 ):   # tolerance to test convergence
    res = tol+1
    its=0
    xnew = x.copy()
    while res > tol:
        dx = fprime_solve( fprime(xnew), -func(xnew) )
        xnew += dx
        res = np.sum( np.square( dx ) )
        its+=1
    return xnew, its, res

# parameters

# number of time and space points
nt = 10
nx = 64

# velocity and reynolds number
u0 = 1
re = 5e1

# timestep
dt = 0.04

# coefficient for theta-method timestepping
#   0,1 for forwards,backwards euler and 0.5 for trapezium
theta = 0.5

# viscosity and mesh spacing
nu = 2*u0/re
dx = 2/(nx-1)

cfl_v = nu*dt/dx**2
cfl_u = u0*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

# mesh
x = np.linspace(start=-1,stop=1,num=nx)

# initial conditions

# solution at each timestep
q = np.zeros( (nt, x.shape[0]) )
q[0,:] = np.exp( -10*x**2 )

# jacobian elements for spatial terms

# A_{i}
diag = -2*nu/dx**2

# A_{i+1}
upper = nu/dx**2 - u0/(2*dx)

# A_{i-1}
lower = nu/dx**2 + u0/(2*dx)

# circulant matrix for spatial operator
a = np.zeros_like(x)
a[0] =    diag
a[1] =    lower
a[nx-1] = upper

jac = -theta*a.copy()
jac[0]+= (1/dt)

def spatial_residual( u ):
    return circ_mul(a,u)

def full_residual( u0, u1 ):
    rn0 = spatial_residual(u0)
    rn1 = spatial_residual(u1)
    return (u1-u0)/dt - (1-theta)*rn0 - theta*rn1

plt.plot(x,q[0])
for i in range (0,nt-1):

    func = lambda u: full_residual(q[i],u)
    fprime = lambda u: jac
    fprime_solve = lambda A, b: linalg.solve_circulant( A, b )

    q[i+1],its,res = newton_solve( func, fprime, fprime_solve, q[i] )

    print( f"{i} | {its} | {res}" )
    plt.plot(x,q[i+1])

plt.show()

