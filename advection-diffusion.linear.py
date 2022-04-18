
import numpy as np
import scipy.linalg as linalg

import circulant as circ
import mesh

import matplotlib.pyplot as plt

# parameters

# number of time and space points
nt = 10
nx = 32

# velocity and reynolds number
u = 1
re = 1e2

# timestep
dt = 0.08

# coefficient for theta-method timestepping
#   0,1 for forwards,backwards euler and 0.5 for trapezium
theta = 0.5

# viscosity and mesh spacing
nu = 2*u/re
dx = 2/(nx-1)

cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

# mesh
x = mesh.periodic_mesh( xl=-1, xr=1, nx=nx )

# initial conditions

q0 = np.exp( -10*x**2 )

# circulant matrices for gradient and laplacian
ddx  = mesh.gradient_matrix(  x, form='circ' )
ddx2 = mesh.laplacian_matrix( x, form='circ' )

# spatial jacobian
jac = nu*ddx2 - u*ddx

# circulant matrices for explicit and implicit operators
cimp = -theta*jac
cimp[0]+=1/dt

cexp = (1-theta)*jac
cexp[0]+=1/dt

# solution at each timestep
q = np.zeros( (nt, x.shape[0]) )
q[0,:] = q0

plt.plot(x,q[0])
for i in range (1,nt):
    b = circ.matmul(cexp,q[i-1])
    q[i] = linalg.solve_circulant(cimp,b)
    plt.plot(x,q[i])

plt.show()

