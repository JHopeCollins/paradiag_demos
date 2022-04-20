
import numpy as np
import circulant as circ
import mesh

# mass matrix for linear advection-diffusion on periodic mesh
def mass_matrix( x, form='circ' ):
    if form!='circ' and form!='full':
        raise ValueError( "form must be 'circ' or 'full'" )

    M = np.zeros_like(x)
    M[0]=1

    if form=='full': M=circ.circulant(M)

    return M


# stiffness matrix for linear advection-diffusion on periodic mesh
def stiffness_matrix( x, u, nu, form='circ' ):
    if form!='circ' and form!='full':
        raise ValueError( "form must be 'circ' or 'full'" )

    # (circulant) matrices for gradient and laplacian
    ddx  = mesh.gradient_matrix(  x, form=form )
    ddx2 = mesh.laplacian_matrix( x, form=form )

    # spatial jacobian
    K = u*ddx - nu*ddx2

    return K

# solve linear ODE for next timestep using theta-method
# M*dq/dt + K*q = b
def linear_solve( M, K, b, dt, theta, q0 ):

    # circulant linear jacobians for implicit/explicit operators
    jac_l = M/dt + theta*K
    jac_r = M/dt - (1.0-theta)*K

    # forcing function and explicit operator from previous timestep
    rhs = b + circ.vecmul( jac_r, q0 )

    return circ.solve( jac_l, rhs )

# solve linear ODE for next nt timesteps using theta-method
# uses qinit as initial conditions
# M*dq/dt + K*q = b
def solve_timeseries( M, K, b, nt, dt, theta, qinit ):

    # initial condition and solution at each timestep
    q = np.zeros( (nt+1, qinit.shape[0]), dtype=qinit.dtype  )
    q[0,:] = qinit[:]

    # integrate in time
    for i in range(1,nt+1):
        q[i] = linear_solve( M,K,b, dt,theta, q[i-1] )

    # do not return initial conditions
    return q[1:,:]

