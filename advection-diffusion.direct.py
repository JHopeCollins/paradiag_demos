
import numpy as np
import scipy.linalg as linalg

import circulant as circ
import mesh
import ade_utils as ade

import matplotlib.pyplot as plt

### === --- --- === ###
#
# Solve the linear advection diffusion equation
#   using parallel timestepping
#   and solving the all-at-once system directly
#
# ADE solved on a periodic mesh using:
#   time discretisation: implicit theta-method
#   space discretisation: second order central differences
#
### === --- --- === ###

### === --- parameters --- === ###

# number of time and space points
nt = 8
nx = 32

# length of domain
lx = nx
dx = lx/nx

# velocity and reynolds number
u = 1
re = 5
nu = 2*u/re

# timestep
dt = 1.5

# parameter for theta timestepping
theta=0.5

# alpha circulant parameter
alpha = 0.01

# sharpness of initial profile
sharp = 6/lx

cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

### === --- set up --- === ###

# mesh
x = mesh.periodic_mesh( xl=-lx/2, xr=lx/2, nx=nx )

# mass and stiffness matrices
M = ade.mass_matrix( x )
K = ade.stiffness_matrix( x,u,nu )

# forcing function
forcing = np.zeros(nx)

# initial conditions are guess for all timesteps
qinit = np.exp( -(sharp*x)**2 )

### === --- serial solution --- === ###

b = forcing.copy()
qserial = ade.solve_timeseries( M,K,b, nt,dt,theta, qinit )

### === --- parallel solution --- === ###

# full mass and stiffness matrices
Mfull = circ.circulant(M)
Kfull = circ.circulant(K)

# timestepping matrices
zeros = np.zeros(nt)

b1 = np.zeros(nt)
b1[0] =  1/dt
b1[1] = -1/dt
B1 = linalg.toeplitz(b1,zeros)

b2 = np.zeros(nt)
b2[0] =   theta
b2[1] = 1-theta
B2 = linalg.toeplitz(b2,zeros)

# all-at-once system

B1M = np.kron(B1,Mfull)
B2K = np.kron(B2,Kfull)

A = B1M + B2K

# right-hand-side

# forcing function
b = np.zeros((nt,nx))
for i in range(0,nt): b[i,:]=forcing[:]

# initial condition of ODE becomes boundary condition for all-at-once system
b[0,:]+=circ.vecmul( M/dt - (1-theta)*K, qinit)

b = b.reshape(nt*nx)

# direct all-at-once solve

qparallel = linalg.solve(A,b)
qparallel = qparallel.reshape(nt,nx)

### === --- plotting --- === ###

# initial condition
plt.plot(x,qinit,color='black',label='i')

# serial solution
plt.gca().set_prop_cycle(None)
for i in range(0,nt): plt.plot(x,qserial[i], linestyle=":", label="s"+str(i))

# parallel solution
plt.gca().set_prop_cycle(None)
for i in range(0,nt): plt.plot(x,qparallel[i],linestyle='--', label="p"+str(i))

plt.grid()
plt.legend(loc='center left')

plt.show()

