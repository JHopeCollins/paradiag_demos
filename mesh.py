
from math import isclose

import numpy as np
import circulant as circ

def periodic_mesh( xl, xr, nx, endpoint=False ):
    return np.linspace( start=xl, stop=xr, num=nx, endpoint=endpoint )

def gradient_matrix( x, form='full', order=2 ):
    if order!=2:
        raise NotImplemented( "only second order gradient implemented" )

    # circulant gradient matrix for periodic mesh
    dx = x[1]-x[0]
    ddx = np.zeros_like(x)
    ddx[1]  = -1/(2*dx)
    ddx[-1] =  1/(2*dx)

    if   form=='circ': return ddx
    elif form=='full': return linalg.circulant(ddx)
    else: raise ValueError( "form must be 'circ' or 'full'" )

def laplacian_matrix( x, form='full', order=2 ):
    if order!=2:
        raise NotImplemented( "only second order laplacian implemented" )

    # circulant laplacian matrix for periodic mesh
    dx = x[1]-x[0]
    ddx2 = np.zeros_like(x)
    ddx2[1]  =  1/dx**2
    ddx2[0]  = -2/dx**2
    ddx2[-1] =  1/dx**2

    if   form=='circ': return ddx2
    elif form=='full': return linalg.circulant(ddx2)
    else: raise ValueError( "form must be 'circ' or 'full'" )

def gradient_operator( x, order=2 ):
    if order!=2:
        raise NotImplemented( "only second order gradient implemented" )

    ddx = gradient_matrix(x,form='circ',order=order)
    return lambda u: circ.matmul(ddx,u)

def laplacian_operator( x, order=2 ):
    if order!=2:
        raise NotImplemented( "only second order laplacian implemented" )

    ddx2 = laplacian_matrix(x,form='circ',order=order)
    return lambda u: circ.matmul(ddx2,u)

