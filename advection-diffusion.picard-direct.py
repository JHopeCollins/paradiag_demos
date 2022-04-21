
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
#   and solving the stationary iteration P*du=b-Au (equation 1.4)
#   each iteration is solved directly
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
re = 1e1
nu = 2*u/re

# timestep
dt = 1

# parameter for theta timestepping
theta=0.5

# alpha circulant parameter
alpha = 1e-2

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
forcing = np.zeros(nx)

# initial conditions are guess for all timesteps
qinit = np.exp( -(sharp*x)**2 )

### === --- serial solution --- === ###

b = forcing.copy()
qserial = ade.solve_timeseries( M,K,b, nt,dt,theta, qinit )

### === --- set up parallel solution --- === ###

# full mass and stiffness matrices
Mfull = circ.circulant(M)
Kfull = circ.circulant(K)

# timestepping matrices
zeros = np.zeros(nt)

b1 = np.zeros(nt)
b1[0] =  1/dt
b1[1] = -1/dt
B1 = linalg.toeplitz(b1,zeros)
C1 = circ.circulant(b1,alpha=alpha)

b2 = np.zeros(nt)
b2[0] =   theta
b2[1] = 1-theta
B2 = linalg.toeplitz(b2,zeros)
C2 = circ.circulant(b2,alpha=alpha)

# all-at-once system

# actual operators
B1M = np.kron(B1,Mfull)
B2K = np.kron(B2,Kfull)

# approximate operators
C1M = np.kron(C1,Mfull)
C2K = np.kron(C2,Kfull)

A = B1M + B2K
P = C1M + C2K

dA = P - A

# right-hand-side

# forcing function
b = np.zeros((nt,nx))
for i in range(0,nt): b[i,:]=forcing[:]

# initial condition of ODE becomes boundary condition for all-at-once system
b[0,:]+=circ.vecmul( M/dt - (1-theta)*K, qinit)

### === --- parallel solve --- === ###

# current guess for solution at each timestep
qcurr = np.zeros( (nt, nx) )

for i in range(0,nt):
    qcurr[i,:] = qinit[:]

# new guess for solution at each timestep
qnext = qcurr.copy()

# stationary iteration (eq 1.4) with direct Pu=b solve

b = b.reshape(nt*nx)
qcurr = qcurr.reshape(nt*nx)
qnext = qcurr.reshape(nt*nx)

rhs = np.zeros_like(b)

niters=10
print( "stationary iteration with direct solve:" )
for j in range(0,niters):

    # rhs = (P-A)qk + b
    rhs = np.matmul(dA,qcurr) + b
    qnext = linalg.solve(P,rhs)

    # test convergence
    res = np.sum( (qnext-qcurr)*(qnext-qcurr) )/(nx*nt)
    print( f" iteration: {j} | residual: {res}" )

    # copy over guess for next iteration
    qcurr[:] = qnext[:]

rhs   = rhs.reshape(nt,nx)
qcurr = qcurr.reshape(nt,nx)
qnext = qnext.reshape(nt,nx)
qparrallel = qnext.copy()

# plotting
plt.plot(x,qinit,color='black',label='i')

# serial solution
plt.gca().set_prop_cycle(None)
for i in range(0,nt): plt.plot(x,qserial[i], linestyle=":", label="s"+str(i))

# parallel solution
plt.gca().set_prop_cycle(None)
for i in range(0,nt): plt.plot(x,qparrallel[i],linestyle='-.',  label="p"+str(i))

plt.grid()
plt.legend(loc='center left')
plt.show()

