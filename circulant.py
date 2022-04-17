
from math import pi, sqrt

import numpy as np

import scipy.fft as fft
import scipy.linalg as linalg

import matplotlib.pyplot as plt

nx = 256
dx = 2*pi/32

x = np.linspace( 0, 2*pi, nx, endpoint=False )

y = np.sin( x ) + np.sin( 7*x ) - np.cos( 20*x )

def circ_mul_brute( c, r ):
    n = c.shape[0]
    # eigenvectors are fourier matrix
    F = linalg.dft(n)
    eigvals = fft.fft(c)/n

    rnew = np.matmul(F,r)
    rnew*= eigvals
    rnew = np.matmul(F.T,rnew)
    rnew[1:]=rnew[:0:-1]
    return rnew.real

def circ_mul( c, r ):
    n = c.shape[0]
    eigvals = fft.fft(c)

    rnew = fft.fft(r)
    rnew*=eigvals
    rnew = fft.ifft(rnew)
    return rnew.real

# circulant matrix
c = np.zeros((nx))
for i in range(1,nx+1): c[i-1]=sqrt(2*i+0.6)-0.2

A = linalg.circulant(c)
Ay = np.matmul(A,y)

ynew = circ_mul(c,y)

error = ynew - Ay

print( np.sum(error*error) )

