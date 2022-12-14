
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
#   using parallel timestepping
#   and solving the newon iteration P*du=b-A(u) (equation 1.9)
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
theta = 1

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

### === --- serial solution --- === ###

qserial = bge.solve_timeseries(x,nu,nt,dt,theta,qinit,jac_form='nl',out_freq=save_freq)

### === --- set up parallel solution --- === ###

M = bge.mass_matrix( x, form='full' )

spatial_residual = bge.spatial_operator( x,nu )
full_residual = bge.full_residual( x,nu,dt,theta )

jacobian = bge.spatial_jacobian( x,nu, form='nl' )

# initial condition becomes boundary condition on rhs of all-at-once system
b = np.zeros((nt,nx))
b[0,:]+= np.matmul(M/dt,qinit) - (1-theta)*spatial_residual(qinit)
b = b.reshape(nt*nx)

rhs = np.zeros((nt,nx))

# timestepping matrices for theta-stepping

zeros = np.zeros(nt)

b1 = np.zeros(nt)
b1[0] =  1/dt
b1[1] = -1/dt

b2 = np.zeros(nt)
b2[0] =   theta
b2[1] = 1-theta

# solution at every timestep
q = np.zeros_like(qserial)
for i in range(0,nt):
    q[i,:] = qinit[:]

alpha=0.001

niters = 3
for i in range(0,niters):

    # set up rhs (nonlinear residual + initial condition?)
    rhs[0,:] = full_residual( q[0], qinit )
    for j in range(1,nt):
        rhs[j,:] = full_residual(q[j],q[j-1])
    #rhs+= b

    # time average of solution
    qav = np.zeros_like(qinit)
    for j in range(0,nt):
        qav+=q[j]/nt

    # spatial jacobian for paradiag matrix
    K = jacobian( qav )

    # solve for update using paradiag jacobian
    dq = circ.paradiag_solve( M,K,rhs, b1,b2, nt,nx, alpha )

    # apply update
    q+=dq

    # print residual
    res = np.sum( np.sum( dq*dq ) )/(nx*nt)
    print( f" iteration: {i} | residual: {res}" )

qparallel = q

# plotting
plt.plot(x,qinit,color='black',label='i')

# serial solution
plt.gca().set_prop_cycle(None)
j=0
for i in range(0,nt,save_freq):
    plt.plot(x,qserial[j], linestyle=":", label="s"+str(i))
    j+=1

# parallel solution
plt.gca().set_prop_cycle(None)
j=0
for i in range(0,nt,save_freq):
    plt.plot(x,qparallel[j],linestyle='-.',  label="p"+str(i))
    j+=1

plt.grid()
plt.legend(loc='center left')
plt.show()

#for i in range(0,nt,save_freq):
#    plt.plot(x,qserial[i])
#plt.plot(x,qserial[-1,:])
#
#plt.grid()
#plt.show()
#
