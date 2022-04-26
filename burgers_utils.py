
import numpy as np
import circulant as circ
import mesh
import newton

# mass matrix for burgers equation on a periodic mesh
def mass_matrix( x, form='circ' ):
    if form!='circ' and form!='full':
        raise ValueError( "form must be 'circ' or 'full'" )

    M = np.zeros_like(x)
    M[0]=1

    if form=='full': return circ.circulant(M)
    else: return M

# returns function to evaluate convection and diffusion terms
def spatial_operator( x, nu ):
    grad =  mesh.gradient_operator(x)
    lapl = mesh.laplacian_operator(x)
    return lambda u: nu*lapl(u) - u*grad(u)

# returns function to evaluate the residual of the full equation du/dt-R(u)
def full_residual( x, nu, dt, theta ):
    R = spatial_operator( x,nu )
    M = mass_matrix(x,form='circ')/dt
    return lambda u0, u1: circ.vecmul(M,u1-u0) - (1-theta)*R(u0) - theta*R(u1)

# returns function to evaluate jacobian of spatial operator
def spatial_jacobian( x, nu, form='lc', u0=None ):

    # circulant form assuming periodic mesh
    ddx =   mesh.gradient_matrix( x, form='circ' )
    ddx2 = mesh.laplacian_matrix( x, form='circ' )

    # full form of jacobian requested
    if form!='lc':
        ddx  = circ.circulant(ddx)
        ddx2 = circ.circulant(ddx2)

    # nonlinear jacobian
    if form == 'nl':
        return lambda u: nu*ddx2 - np.matmul(np.diagflat(u),ddx)

    # linearised jacobian
    elif (form=='l') or (form=='lc'):

        # linearised velocity is given
        if u0!=None:
            return lambda u: nu*ddx2 - u0*ddx

        # linearised velocity found from solution
        else:
            n = x.shape[0]
            return lambda u: nu*ddx - (np.sum(u)/n)*ddx

    else:
        raise ValueError( "form must be 'nl', 'l', or 'lc'" )

# returns function to evaluate jacobian of full equation du/dt-R(u)
def full_jacobian( x, nu, dt, theta, form='lc', u0=None ):

    J = spatial_jacobian( x,nu,form,u0 )

    if form=='lc': M = mass_matrix( x, form='circ' )
    else: M = mass_matrix( x, form='full' )
    M/=dt

    return lambda u: M - theta*J(u)

def solve_timeseries( x, nu, nt, dt, theta, uinit, jac_form='nl', u0=None, out_freq=10 ):

    # initial condition and solution at each timestep
    u = np.zeros( (nt+1, x.shape[0]) )
    u[0,:] = uinit[:]

    # build function objects
    residual = full_residual(x,nu,dt,theta)
    jacobian = full_jacobian(x,nu,dt,theta,form=jac_form,u0=u0)

    if jac_form=='lc': jac_type='circ'
    else: jac_type='full'

    jacobian_solve = lambda A,b: newton.jacobian_solve(A,b,jac_type=jac_type)

    # integrate in time
    for i in range(0,nt):
        u[i+1],its,res = newton.solve( lambda q: residual(u[i],q),
                                       jacobian,
                                       jacobian_solve,
                                       uinit )

        if i%out_freq==0:
            print( f"{i} | {its} | {res}" )

    # do not return initial conditions
    return u[1:,:]
