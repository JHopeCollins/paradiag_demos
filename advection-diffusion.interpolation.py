
from math import pi
from cmath import exp

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
#   by interpolating P_{alpha}u=b at alpha=0 values found from alpha = roots-of-unity
#       Kressner et al. 2022, Improved parallel-in-time integration via low-rank updates and interpolation
#
#   each Pu=b system is solved using the paradiag 3-step
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
re = 5e1
nu = 2*u/re

# timestep
dt = 2

# parameter for theta timestepping
theta=0.5

# sharpness of initial profile
sharp = 6/lx

cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )
print()

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

### === --- set up parallel solution --- === ###

# timestepping matrices
zeros = np.zeros(nt)

b1 = np.zeros(nt)
b1[0] =  1/dt
b1[1] = -1/dt

b2 = np.zeros(nt)
b2[0] =   theta
b2[1] = 1-theta

# right-hand-side

# forcing function
b = np.zeros((nt,nx))
for i in range(0,nt): b[i,:]=forcing[:]

# initial condition of ODE becomes boundary condition for all-at-once system
b[0,:]+=circ.vecmul( M/dt - (1-theta)*K, qinit)

### === --- parallel solve --- === ###

rho = 1e-2
d = 3

# solution evaluations for interpolation
qinterp = np.zeros( (d,nt,nx),dtype=complex )

roots=np.zeros(d,dtype=complex)
for i in range(0,d):
    roots[i] = exp(2j*pi*i/d)

# multiple solves of Pu=b with alpha equal to roots-of-unity

print( "evaluation interpolation with paradiag solve:" )
for j in range(0,d):

    # solve P*q_{k} = b with paradiag 3-step

    qinterp[j,:] = circ.paradiag_solve( M,K,b,
                                        b1,b2,
                                        nt,nx,
                                        alpha=rho*roots[j],
                                        linear_solver=circ.solve )

# interpolate to alpha=0
qparallel = np.zeros_like(qserial)
for j in range(0,d):
    qparallel[:] += qinterp[j,:].real
qparallel/=d

total_error = np.sum( np.sum( (qparallel-qserial)*(qparallel-qserial) ) )/(nx*nt)
print( "total error: ", total_error )

# plotting
plt.plot(x,qinit,color='black',label='i')

# serial solution
plt.gca().set_prop_cycle(None)
for i in range(0,nt): plt.plot(x,qserial[i], linestyle=":", label="s"+str(i))

# parallel solution
plt.gca().set_prop_cycle(None)
for i in range(0,nt): plt.plot(x,qparallel[i],linestyle='-.',  label="p"+str(i))

plt.grid()
plt.legend(loc='center left')
plt.show()

