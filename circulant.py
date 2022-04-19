
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

# multiplication Cx=b of vector by (alpha-)circulant matrix
def vecmul( c, x, alpha=None ):
    norm='ortho' # 1/sqrt(n) scaling on eigenvectors

    if alpha==None: # standard circulant matrix
        # to eigenbasis
        b = fft.fft(x,norm=norm)
        # scale by eigenvalues
        b*= eigenvalues(c)
        # from eigenbasis
        b = fft.ifft(b,norm=norm)
        return b.real

    else: # alpha-circulant matrix
        n = c.shape[0]
        gamma = gamalph(n,alpha)

        # to eigenbasis
        b = gamma*x
        b = fft.fft(b,norm=norm)
        # scale by eigenvalues
        b*= eigenvalues(c,alpha=alpha)
        # from eigenbasis
        b = fft.ifft(b,norm=norm)
        b/= gamma
        return b.real

# solve Cx=b for (alpha-)circulant matrix
def solve( c, b, alpha=None ):
    if alpha==None: # standard circulant matrix
        return linalg.solve_circulant(c,b)

    else: # alpha-circulant matrix
        norm='ortho' # 1/sqrt(n) scaling on eigenvectors
        n = c.shape[0]
        gamma = gamalph(n,alpha)

        # to eigenbasis
        x = gamma*b
        x = fft.fft(x,norm=norm)
        # scale by eigenvalues
        x/= eigenvalues(c,alpha=alpha)
        # from eigenbasis
        x = fft.ifft(x,norm=norm)
        x/= gamma
        return x.real

