
from math  import pi
from cmath import exp

import numpy as np
import scipy.fft as fft
import scipy.linalg as linalg

# construct (alpha-)circulant matrix
def circulant( c, alpha=None ):
    C = linalg.circulant(c)
    # scale upper triangle of alpha-circulant matrix
    if alpha!=None:
        n = c.shape[0]
        for i in range(0,n-1):
            for j in range(i+1,n):
                C[i,j]*=alpha
    return C

# diagonal matrix Gamma_alpha for alpha-circulant diagonalisation
def gamalph( n, alpha ):
    if alpha==None: return np.ones(n)
    return alpha**(np.arange(n)/n)

# eigenvalues of (alpha-)circulant matrix
def eigenvalues( c, alpha=None ):
    norm='backward' # 1/n scaling is in eigenvectors
    n = c.shape[0]

    eigval = np.asarray(c,dtype=complex)

    if alpha!=None: # alpha-circulant matrix
        eigval*= gamalph(n,alpha)

    return fft.fft( eigval, norm=norm )

# eigenvectors of (alpha-)circulant matrix
def eigenvectors( n, inverse=False, alpha=None ):
    norm='sqrtn' # 1/sqrt(n) scaling on eigenvectors
    F = linalg.dft(n,scale=norm)

    if alpha==None: # standard circulant matrix
        if inverse:
            return F
        else:
            return F.conj().T

    else: # alpha-circulant matrix
        gamma = gamalph(n,alpha)
        if inverse:
            return np.matmul( F, np.diag(gamma) )
        else:
            return np.matmul( np.diag(1./gamma), F.conj().T )

# transform vector x to eigenbasis of (alpha-)circulant matrix with rank n
def to_eigenbasis( n, x, alpha=None ):
    norm='ortho' # 1/sqrt(n) scaling on eigenvectors

    b = np.asarray(x,dtype=complex)

    if alpha!=None: # alpha-circulant matrix
        b*= gamalph( n, alpha )

    b = fft.fft( b, norm=norm )

    return b

# transform vector x from eigenbasis of (alpha-)circulant matrix with rank n
def from_eigenbasis( n, x, alpha=None ):
    norm='ortho' # 1/sqrt(n) scaling on eigenvectors

    b = fft.ifft( x, norm=norm )

    if alpha!=None: # alpha-circulant matrix
        b/= gamalph( n, alpha )

    return b

# multiplication Cx=b of vector by (alpha-)circulant matrix
def vecmul( c, x, alpha=None ):
    n = c.shape[0]
    b = to_eigenbasis( n, x, alpha=alpha )
    b*= eigenvalues( c, alpha=alpha)
    b = from_eigenbasis( n, b, alpha=alpha )

    is_real = (c.dtype == float)

    if is_real: return b.real
    else: return b

# solve Cx=b for (alpha-)circulant matrix
def solve( c, b, alpha=None ):
    n = c.shape[0]
    b = to_eigenbasis( n, b, alpha=alpha )
    b/= eigenvalues( c, alpha=alpha)
    b = from_eigenbasis( n, b, alpha=alpha )

    is_real = (c.dtype == float)

    if is_real: return b.real
    else: return b

# solve Px=r with the ParaDiag type alpha-circulant matrix P using the ParaDiag 3-step
def paradiag_solve( M,K,r, b1,b2, nt,nx, alpha, linear_solver=linalg.solve ):

    # eigenvalues of alpha-circulant timestepping matrices
    D1 = eigenvalues(b1,alpha=alpha)
    D2 = eigenvalues(b2,alpha=alpha)

    # intermediate arrays
    s1 = np.zeros_like(r,dtype=complex)
    s2 = np.zeros_like(r,dtype=complex)

    # value type
    is_real =    (    M.dtype == float) \
             and (    K.dtype == float) \
             and (    r.dtype == float) \
             and (   b1.dtype == float) \
             and (   b2.dtype == float) \
             and (type(alpha) == float) \

    if is_real: dtype=float
    else:       dtype=complex

    # solution array
    x = np.zeros_like(r,dtype=dtype)

    # step-(a): weighted fft on each time-pencil
    for i in range(0,nx):
        s1[:,i] = to_eigenbasis( nt, r[:,i], alpha=alpha )

    # step-(b): weighted linear solve in space
    for i in range (0,nt):
        s2[i,:] = linear_solver( D1[i]*M + D2[i]*K, s1[i,:] )

    # step-(c): weighted ifft on each time-pencil
    for i in range(0,nx):
        if is_real:
            x[:,i] = from_eigenbasis( nt, s2[:,i], alpha=alpha ).real
        else:
            x[:,i] = from_eigenbasis( nt, s2[:,i], alpha=alpha )

    return x

# solve Ax=r by interpolation of paradiag Px=r at alpha=rho*(dth-roots-of-unity)
def paradiag_interp( M,K,r, b1,b2, nt,nx, rho, d, linear_solver=linalg.solve ):

    qinterp = np.zeros( (nt,nx),dtype=complex )

    # evaluate Px=r at alpha = rho*(dth-roots-of-unity)
    for i in range(0,d):
        qinterp[:]+= paradiag_solve( M,K,r,
                                     b1,b2,
                                     nt,nx,
                                     alpha=rho*exp(2j*pi*i/d),
                                     linear_solver=linear_solver )
    # interpolate to alpha=0
    return qinterp.real/d

