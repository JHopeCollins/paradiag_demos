
import numpy as np
import scipy.linalg as linalg

import circulant as circ
import mesh
import newton

import burgers_utils as bge

import matplotlib.pyplot as plt

### === --- --- === ###
#
# Solve the nonlinear burgers equation
#   using sequential timestepping
#
# BGE solved on a periodic mesh using:
#   time discretisation: implicit theta-method
#   space discretisation: second order central differences
#
### === --- --- === ###

# parameters

# number of time and space points
nt = 64
nx = 64
save_freq=16

lx = 8

# bump sharpness
sharp=2

# velocity and reynolds number
u0 = 1
du = 0.5
re = 1e2

# timestep
dt = 0.05

# coefficient for theta-method timestepping
#   0,1 for forwards,backwards euler and 0.5 for trapezium
theta = 0.5

# viscosity and mesh spacing
nu = 2*u0/re
dx = lx/(nx-1)

cfl_v = nu*dt/dx**2
cfl_u = u0*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

# mesh
x = mesh.periodic_mesh( xl=-lx/2, xr=lx/2, nx=nx )

# initial conditions
qinit = u0 + du*np.exp( -sharp*(x+lx/4)**2 )

q = bge.solve_timeseries(x,nu,nt,dt,theta,qinit,jac_form='nl',out_freq=save_freq)

for i in range(0,nt,save_freq):
    plt.plot(x,q[i])
plt.plot(x,q[-1,:])

plt.grid()
plt.show()

