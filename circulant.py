
import numpy as np
import scipy.fft as fft
import scipy.linalg as linalg

def eigenvalues( c ):
    return fft.fft(c,norm='backward')/c.shape[-1]

def eigenvectors( c ):
    return linalg.dft(c.shape[-1])

def matmul( c, r ):
    rnew = fft.fft(r,norm='backward')
    rnew*= eigenvalues(c)
    rnew = fft.ifft(rnew,norm='forward')
    return rnew.real

