
import numpy as np
import scipy.linalg as linalg

import circulant as circ
import mesh

import matplotlib.pyplot as plt

def advection_diffusion_matrices( x, u, nu, form='circ' ):

    if form!='circ' and form!='full':
        raise ValueError( "form must be 'circ' or 'full'" )

    # circulant matrices for gradient and laplacian
    ddx  = mesh.gradient_matrix(  x, form=form )
    ddx2 = mesh.laplacian_matrix( x, form=form )

    # spatial jacobian
    A = u*ddx - nu*ddx2

    # mass matrix
    M = np.zeros(nx)
    M[0]=1

    if form=='full': M=circ.circulant(M)

    return M,A


def linear_solve( M, A, b, theta, dt, q0 ):

    # circulant linear jacobian
    jac_l = M/dt + theta*A
    jac_r = M/dt - (1.0-theta)*A

    # forcing function and solution from previous timestep
    rhs = b + circ.vecmul(jac_r,q0)

    return circ.solve( jac_l,rhs )

# parameters

# number of time and space points
nt = 10
nx = 32

# velocity and reynolds number
u = 1
re = 1e2

# timestep
dt = 0.08

# parameter for theta timestepping
theta=1

# viscosity and mesh spacing
nu = 2*u/re
dx = 2/(nx-1)

cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

# mesh
x = mesh.periodic_mesh( xl=-1, xr=1, nx=nx )

# solution at each timestep
q = np.zeros( (nt, x.shape[0]) )

# initial conditions
q[0,:] = np.exp( -10*x**2 )

# mass and stiffness matrices
M,A = advection_diffusion_matrices(x,u,nu)

# forcing function
b = np.zeros(nx)

for i in range (1,nt):
    q[i] = linear_solve( M, A, b, theta, dt, q[i-1] )

# plotting
for i in range(0,nt): plt.plot(x,q[i])
plt.grid()
plt.show()

