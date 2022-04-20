
import numpy as np
import matplotlib.pyplot as plt

import mesh
import ade_utils as ade

# parameters

# number of time and space points
nt = 5
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

# mass and stiffness matrices
M = ade.mass_matrix( x )
K = ade.stiffness_matrix( x,u,nu )

# forcing function
b = np.zeros_like(x)

# initial conditions
qinit = np.exp( -(sharp*x)**2 )

# solution at each timestep
q = ade.solve_timeseries( M,K,b, nt,dt,theta, qinit )

# plotting
plt.plot(x,qinit)
for i in range(0,nt): plt.plot(x,q[i])
plt.grid()
plt.show()

