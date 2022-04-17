
import numpy as np
import scipy.linalg as linalg

import matplotlib.pyplot as plt

# change this to use fft
def circ_mul( c, x ):
    return np.matmul(linalg.circulant(c),x)

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
x = np.linspace(start=-1,stop=1,num=nx)

# initial conditions

q0 = np.exp( -10*x**2 )

# jacobian elements for spatial terms

# A_{i}
diag = -2*nu/dx**2

# A_{i+1}
upper = nu/dx**2 - u/(2*dx)

# A_{i-1}
lower = nu/dx**2 + u/(2*dx)

# circulant matrices for explicit and implicit operators
cimp = np.zeros_like(x)
cimp[0] = (1/dt)-theta*diag
cimp[1] =       -theta*lower
cimp[nx-1] =    -theta*upper

cexp = np.zeros_like(x)
cexp[0] = (1/dt)+(1-theta)*diag
cexp[1] =        (1-theta)*lower
cexp[nx-1] =     (1-theta)*upper

# solution at each timestep
q = np.zeros( (nt, x.shape[0]) )
q[0,:] = q0

plt.plot(x,q[0])
for i in range (1,nt):
    b = circ_mul(cexp,q[i-1])
    q[i] = linalg.solve_circulant(cimp,b)
    plt.plot(x,q[i])

plt.show()

