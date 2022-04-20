
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
def linear_solve( M, K, b, theta, dt, qcurr ):

    # circulant linear jacobians for implicit/explicit operators
    jac_l = M/dt + theta*K
    jac_r = M/dt - (1.0-theta)*K

    # forcing function and explicit operator from previous timestep
    rhs = b + circ.vecmul( jac_r, qcurr )

    return circ.solve( jac_l, rhs )

# parameters

# number of time and space points
nt = 8
nx = 64

# length of domain
lx = nx
dx = lx/nx

# velocity and reynolds number
u = 1
re = 1e1
nu = 2*u/re

# timestep
dt = 3

# parameter for theta timestepping
theta=0.5

# alpha circulant parameter
alpha = 0.01

# sharpness of initial profile
sharp = 8/lx

cfl_v = nu*dt/dx**2
cfl_u = u*dt/dx

print( "nu, dt, cfl_v, cfl_u" )
print(  nu, dt, cfl_v, cfl_u  )

# mesh
x = mesh.periodic_mesh( xl=-lx/2, xr=lx/2, nx=nx )

# mass and stiffness matrices
M,K = advection_diffusion_matrices(x,u,nu)

# initial conditions are guess for all timesteps
qinit = np.exp( -(sharp*x)**2 )

# forcing function
forcing = np.zeros(nx)

# serial solution
qser = np.zeros( (nt, nx) )
bser = forcing.copy()

qser[0] = linear_solve( M,K,bser, theta, dt, qinit ).real
for i in range(1,nt):
    qser[i] = linear_solve( M, K, bser, theta, dt, qser[i-1] ).real

# parallel solution

# current guess for solution at each timestep
qcurr = np.zeros( (nt, nx) )

for i in range(0,nt):
    qcurr[i,:] = qinit[:]

# new guess for solution at each timestep
qnext = qcurr.copy()

# full mass and stiffness matrices
Mfull = circ.circulant(M)
Kfull = circ.circulant(K)

# forcing function
b = np.zeros_like(qcurr)
for i in range(0,nt): b[i,:]=forcing[:]

# initial condition of ODE becomes boundary condition for all-at-once system
b[0,:]+=circ.vecmul(M/dt,qinit) - (1-theta)*circ.vecmul(K,qinit)

# b only used for whole-system assigns
b = b.reshape(nt*nx)

# timestepping matrices
zeros = np.zeros(nt)

b1 = np.zeros(nt)
b1[0] =  1/dt
b1[1] = -1/dt
B1 = linalg.toeplitz(b1,zeros)
C1 = circ.circulant(b1,alpha=alpha)
D1 = circ.eigenvalues(b1,alpha=alpha)

b2 = np.zeros(nt)
b2[0] =   theta
b2[1] = 1-theta
B2 = linalg.toeplitz(b2,zeros)
C2 = circ.circulant(b2,alpha=alpha)
D2 = circ.eigenvalues(b2,alpha=alpha)

# time-space all-at-once operators

B1M = np.kron(B1,Mfull)
B2K = np.kron(B2,Kfull)

C1M = np.kron(C1,Mfull)
C2K = np.kron(C2,Kfull)

A = B1M + B2K # actual operator
P = C1M + C2K # alpha-approximation

dA = P-A

# right hand side of all-at-once system
rhs = np.zeros_like(qcurr)

# direct all-at-once solve
rhs = rhs.reshape(nt*nx)
rhs[:] = b[:]

qaao = linalg.solve(A,rhs)
qaao = qaao.reshape(nt,nx)

# stationary iteration (eq 1.4) with direct Pu=b solve

rhs = rhs.reshape(nt*nx)
qcurr = qcurr.reshape(nt*nx)

niters=4
print( "stationary iteration with direct solve:" )
for j in range(0,niters):

    # rhs = (P-A)qk + b
    rhs = np.matmul(dA,qcurr) + b
    qnext = linalg.solve(P,rhs)

    # copy over guess for next iteration
    res = np.sum( (qnext-qcurr)*(qnext-qcurr) )/(nx*nt)
    print( f" iteration: {j} | residual: {res}" )
    qcurr[:] = qnext[:]

print()

rhs   = rhs.reshape(nt,nx)
qcurr = qcurr.reshape(nt,nx)
qnext = qnext.reshape(nt,nx)
qpard = qnext.copy()

for i in range(0,nt):
    qcurr[i,:] = qinit[:]
    qnext[i,:] = qinit[:]

# intermediate solution arrays
s1 = np.zeros_like(qcurr,dtype=complex)
s2 = np.zeros_like(qcurr,dtype=complex)

niters=4
print( "stationary iteration with paradiag solve:" )
for j in range(0,niters):

    # set up right hand side of iterative system
    # rhs = (P-A)q_{k-1} + b
    rhs = rhs.reshape(nt*nx)
    qcurr = qcurr.reshape(nt*nx)

    rhs = np.matmul(dA,qcurr) + b

    rhs = rhs.reshape(nt,nx)
    qcurr = qcurr.reshape(nt,nx)

    # solve P*q_{k} = rhs with paradiag 3-step

    # step-(a): weighted fft on each time-pencil
    for i in range(0,nx):
        s1[:,i] = circ.to_eigenbasis( nt, rhs[:,i], alpha=alpha )

    # step-(b): weighted linear solve in space
    for i in range (0,nt):
        s2[i,:] = circ.solve( D1[i]*M + D2[i]*K, s1[i,:] )

    # step-(c): weighted ifft on each time-pencil
    for i in range(0,nx):
        qnext[:,i] = circ.from_eigenbasis( nt, s2[:,i], alpha=alpha ).real

    # copy over guess for next iteration
    res = np.sum( np.sum( (qnext-qcurr)*(qnext-qcurr) ) )/(nx*nt)
    print( f" iteration: {j} | residual: {res}" )
    qcurr[:] = qnext[:]
print()

qpari = qnext.copy()

# plotting
plt.plot(x,qinit,color='black',label='i')
for i in range(0,nt): plt.plot(x,qser[i], linestyle=":", label="s"+str(i))  # serial
plt.gca().set_prop_cycle(None)
#for i in range(0,nt): plt.plot(x,qaao[i],linestyle='--',  label="a"+str(i)) # direct solve of all-at-once
#plt.gca().set_prop_cycle(None)
#for i in range(0,nt): plt.plot(x,qpard[i],linestyle='-.',  label="d"+str(i)) # iterative with direct solve
#plt.gca().set_prop_cycle(None)
for i in range(0,nt): plt.plot(x,qpari[i],linestyle='--',  label="p"+str(i)) # iterative with iterative solve
plt.grid()
plt.legend()
plt.show()

