
import numpy as np
import scipy.linalg as linalg

# newton iterations
def solve( func,         # function to solve
           fprime,       # return derivative of function at x
           fprime_solve, # solve func(x)/fprime(x)
           x,            # initial guess
           tol=1e-7,     # tolerance to test convergence
           maxits=20 ):  # maximum number of iterations
    res = tol+1
    its=0
    xnew = x.copy()
    while (res > tol) and (its<20):
        dx = fprime_solve( fprime(xnew), -func(xnew) )
        xnew += dx
        res = np.sum( np.square( dx ) )
        its+=1
    return xnew, its, res


# how to solve Ax=b
def jacobian_solve( A, b, jac_type='full', P=None ):

    # standard solve
    if jac_type=='full':
        return linalg.solve( A, b )

    # circulant linearied jacobian
    elif jac_type=='circ':
        return circ.solve(A,b)

    # preconditioned with circulant linear jacobian
    elif jac_type=='prec':
        raise NotImplemented( "preconditioning with circulant linear jacobian not implemented" )

    else: raise ValueError( "jac_type must be 'full' or 'circ'" )

