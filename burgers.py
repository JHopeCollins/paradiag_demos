
import numpy as np
import scipy.linalg as linalg
import scipy.fft as fft

import matplotlib.pyplot as plt

# C*x where C is circulant matrix with first column c
def circ_mul( c, x ):
    n = c.shape[0]
    eigvals = fft.fft(c)

    xnew = fft.fft(x)
    xnew*=eigvals
    xnew = fft.ifft(xnew)
    return xnew.real

# newton iterations
def newton_solve( func,         # function to solve
                  fprime,       # return derivative of function at x
                  fprime_solve, # solve func(x)/fprime(x)
                  x,            # initial guess
                  tol=1e-7 ):   # tolerance to test convergence
    res = tol+1
    its=0
    xnew = x.copy()
    while res > tol:
        dx = fprime_solve( fprime(xnew), -func(xnew) )
        xnew += dx
        res = np.sum( np.square( dx ) )
        its+=1
    return xnew, its, res

def gradient_matrix( x, form='full' ):
    # circulant gradient matrix for periodic mesh
    ddx = np.zeros_like(x)
    ddx[1]    = -1/(2*dx)
    ddx[nx-1] =  1/(2*dx)

    if   form=='circ': return ddx
    elif form=='full': return linalg.circulant(ddx)
    else: raise ValueError( "form must be 'circ' or 'full'" )

def laplacian_matrix( x, form='full' ):
    # circulant laplacian matrix for periodic mesh
    ddx2 = np.zeros_like(x)
    ddx2[1]    =  1/dx**2
    ddx2[0]    = -2/dx**2
    ddx2[nx-1] =  1/dx**2

    if   form=='circ': return ddx2
    elif form=='full': return linalg.circulant(ddx2)
    else: raise ValueError( "form must be 'circ' or 'full'" )

def gradient_operator( x ):
    ddx = gradient_matrix(x,form='circ')
    return lambda u: circ_mul(ddx,u)

def laplacian_operator( x ):
    ddx2 = laplacian_matrix(x,form='circ')
    return lambda u: circ_mul(ddx2,u)

# parameters

# number of time and space points
nt = 64
nx = 256
save_freq=16

lx = 8

# bump sharpness
sharp=2

# velocity and reynolds number
u0 = 1
du = 0.5
re = 2e2

# timestep
dt = 0.04

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
x = np.linspace(start=-lx/2,stop=lx/2,num=nx)

# solution at each timestep
q = np.zeros( (nt, x.shape[0]) )

# initial conditions
q[0,:] = u0 + du*np.exp( -sharp*(x+2)**2 )

# circulant gradient and laplacian matrices
ddx  =  gradient_matrix(x,form='circ')
ddx2 = laplacian_matrix(x,form='circ')

# gradient and laplacian operators
grad =  gradient_operator(x)
lapl = laplacian_operator(x)

# spatial operator
def spatial_residual( u ):
    return nu*lapl(u) - u*grad(u)

def full_residual( u0, u1 ):
    rn0 = spatial_residual(u0)
    rn1 = spatial_residual(u1)
    return (u1-u0)/dt - (1-theta)*rn0 - theta*rn1

def spatial_jacobian( u, form='nl' ):

    # nonlinear jacobian
    if form=='nl':
        jac_v = linalg.circulant( nu*ddx2 )
        jac_u = np.matmul(np.diagflat(u),linalg.circulant(ddx))
        return jac_v - jac_u

    else: # linear jacobian
        jac_c = nu*ddx2 - u0*ddx

        # circulant form linear jacobian
        if form=='lc':
            return jac_c

        # linear jacobian
        elif form=='l':
            return linalg.circulant(jac_c)

        else: raise ValueError( "form must be 'nl', 'l', or 'lc'" )

def full_jacobian( u, form='nl' ):

    # nonlinear jacobian
    if form=='nl':
        return np.identity(nx)/dt - theta*spatial_jacobian(u,form='nl')

    else: # linear jacobian
        jac_c = -theta*spatial_jacobian(u,form='lc')
        jac_c[0]+=1/dt

        # circulant linear jacobian
        if form=='lc':
            return jac_c

        # full linear jacobian
        elif form=='l':
            return linalg.circulant(jac_c)

        else: raise ValueError( "form must be 'nl', 'l', or 'lc'" )

def jacobian_solve( A, b, jac_type='full', P=None ):

    # standard solve
    if jac_type=='full':
        return linalg.solve( A, b )

    # circulant linearied jacobian
    elif jac_type=='circ':
        return linalg.solve_circulant(A,b)

    # preconditioned with circulant linear jacobian
    elif jac_type=='prec':
        raise NotImplemented( "preconditioning with circulant linear jacobian not implemented" )

    else: raise ValueError( "jac_type must be 'full' or 'circ'" )

for i in range (0,nt-1):
    func = lambda u: full_residual(q[i],u)
    fprime = lambda u: full_jacobian(u,form='lc')
    fprime_solve = lambda A,b: jacobian_solve(A,b,jac_type='circ')

    q[i+1],its,res = newton_solve( func, fprime, fprime_solve, q[i] )

    if i%save_freq==0:
        print( f"{i} | {its} | {res}" )
        plt.plot(x,q[i])

plt.plot(x,q[nt-1])
plt.grid()

plt.show()

