
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
    eigval = c.copy()

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

    b = x.copy()

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
    return b.real

# solve Cx=b for (alpha-)circulant matrix
def solve( c, b, alpha=None ):
    n = c.shape[0]
    b = to_eigenbasis( n, b, alpha=alpha )
    b/= eigenvalues( c, alpha=alpha)
    b = from_eigenbasis( n, b, alpha=alpha )
    return b

