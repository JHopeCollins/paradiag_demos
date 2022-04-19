
import numpy as np
import scipy.fft as fft
import scipy.linalg as linalg

# construct (alpha-)circulant matrix
def circulant( c, alpha=None ):
    if alpha==None: # standard circulant matrix
        return linalg.circulant(c)

    else: # alpha-circulant matrix
        raise NotImplemented( "construction of alpha-circulant matrix not implemented" )

# solve Cx=b for (alpha-)circulant matrix
def solve( c, b, alpha=None ):
    if alpha==None: # standard circulant matrix
        return linalg.solve_circulant(c,b)

    else: # alpha-circulant matrix
        raise NotImplemented( "solve of alpha-circulant matrix not implemented" )

# diagonal matrix Gamma_alpha for alpha-circulant diagonalisation
def gamalph( n, alpha ):
    return alpha**(np.arange(n)/n)

# eigenvalues of (alpha-)circulant matrix
def eigenvalues( c, alpha=None ):
    n = c.shape[0]
    norm='backward' # 1/n scaling is in eigenvectors

    if alpha==None: # standard circulant matrix
        return fft.fft( c, norm=norm )

    else: # alpha-circulant matrix
        gamma = gamalph(n,alpha)
        return fft.fft( gamma*c, norm=norm )

# eigenvectors of (alpha-)circulant matrix
def eigenvectors( c, inverse=False, alpha=None ):
    n = c.shape[0]
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

# multiplication of vector by (alpha-)circulant matrix
def vecmul( c, r, alpha=None ):
    if alpha!=None:
        raise NotImplemented( "vector multiplication of alpha-circulant matrix not implemented yet" )

    norm='ortho' # 1/sqrt(n) scaling on eigenvectors

    rnew = fft.fft(r,norm=norm)
    rnew*= eigenvalues(c)
    rnew = fft.ifft(rnew,norm=norm)
    return rnew.real

