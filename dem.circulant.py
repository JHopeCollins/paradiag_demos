
from math import pi, sqrt

import numpy as np

import scipy.fft as fft
import scipy.linalg as linalg

import matplotlib.pyplot as plt

nx = 256
dx = 2*pi/32

x = np.linspace( 0, 2*pi, nx, endpoint=False )

y = np.sin( x ) + np.sin( 7*x ) - np.cos( 20*x )

def circ_eigvals( c ):
    return fft.fft(c,norm='backward')/c.shape[-1]

def circ_eigvecs( c ):
    return linalg.dft(c.shape[-1])

def circ_mul_brute( c, r ):
    eigvecs = circ_eigvecs(c)
    eigvals = circ_eigvals(c)

    rnew = np.matmul(eigvecs.T,r)
    rnew*= eigvals
    rnew = np.matmul(eigvecs,rnew)
    rnew[1:]=rnew[:0:-1]
    return rnew.real

def circ_mul( c, r ):
    eigvals = circ_eigvals(c)
    rnew = fft.fft(r,norm='backward')
    rnew*= eigvals
    rnew = fft.ifft(rnew,norm='forward')
    return rnew.real

# circulant matrix
c = np.zeros((nx))
for i in range(1,nx+1): c[i-1]=sqrt(2*i+0.6)-0.2

A = linalg.circulant(c)
Ay = np.matmul(A,y)

ynew = circ_mul_brute(c,y)

error = ynew - Ay

print( np.sum( circ_eigvecs(c)-circ_eigvecs(c).T ) )

print( np.sum(error*error) )

