
import numpy as np
import scipy.linalg as linalg

import circulant as circ
import mesh

import matplotlib.pyplot as plt

# mass and stiffness matrices for linear advection-diffusion on periodic mesh
def advection_diffusion_matrices( x, u, nu, form='circ' ):

    if form!='circ' and form!='full':
        raise ValueError( "form must be 'circ' or 'full'" )

    # (circulant) matrices for gradient and laplacian
    ddx  = mesh.gradient_matrix(  x, form=form )
    ddx2 = mesh.laplacian_matrix( x, form=form )

    # spatial jacobian
    K = u*ddx - nu*ddx2

    # mass matrix
    M = np.zeros(nx)
    M[0]=1

    if form=='full': M=circ.circulant(M)

    return M,K

# solve linear ODE for next timestep using theta-method
# M*dq/dt + K*q = b
def linear_solve( M, K, b, theta, dt, q0 ):

    # circulant linear jacobians for implicit/explicit operators
    jac_l = M/dt + theta*K
    jac_r = M/dt - (1.0-theta)*K

    # forcing function and explicit operator from previous timestep
    rhs = b + circ.vecmul( jac_r, q0 )

    return circ.solve( jac_l, rhs )

# parameters

# number of time and space points
nt = 3
nx = 32

# length of domain
lx = nx
dx = lx/nx

# velocity and reynolds number
u = 1
re = 1e2
nu = 2*u/re

# timestep
dt = 1.5

# parameter for theta timestepping
theta=1

# sharpness of initial profile
sharp = 6/lx

cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

# mesh
x = mesh.periodic_mesh( xl=-lx/2, xr=lx/2, nx=nx )

# solution at each timestep
q = np.zeros( (nt, x.shape[0]) )

# initial conditions
q[0,:] = np.exp( -(sharp*x)**2 )

# mass and stiffness matrices
M,K = advection_diffusion_matrices(x,u,nu)

# forcing function
b = np.zeros(nx)

for i in range (1,nt):
    q[i] = linear_solve( M, K, b, theta, dt, q[i-1] ).real

# plotting
for i in range(0,nt): plt.plot(x,q[i])
plt.grid()
plt.show()

