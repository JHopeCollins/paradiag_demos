
import numpy as np
import scipy.linalg as linalg

import circulant as circ
import mesh

import matplotlib.pyplot as plt

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
x = mesh.periodic_mesh( xl=-1, xr=1, nx=nx )

# initial conditions

# solution at each timestep
q = np.zeros( (nt, x.shape[0]) )
q[0,:] = np.exp( -10*x**2 )

# circulant matrices for gradient and laplacian
ddx  = mesh.gradient_matrix(  x, form='circ' )
ddx2 = mesh.laplacian_matrix( x, form='circ' )

# spatial jacobian
a = nu*ddx2 - u0*ddx

# full jacobian for update
jac = -theta*a.copy()
jac[0]+= (1/dt)

def spatial_residual( u ):
    return circ.matmul(a,u)

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

