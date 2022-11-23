
from math import pi, sqrt, isclose

import numpy as np

import scipy.fft as fft
import scipy.linalg as linalg

import matplotlib.pyplot as plt

import circulant as circ

def blank_zero( A ):
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            if isclose(A[i,j].real,0):
                A[i,j]=0+A[i,j].imag
            if isclose(A[i,j].imag,0):
                A[i,j]=A[i,j].real + 0.j

n = 8
dx = 2*pi/32

x = np.linspace( 0, 2*pi, n, endpoint=False )

y = np.sin( x ) + np.sin( 7*x ) - np.cos( 20*x )

# circulant matrix
c = np.zeros((n))
for i in range(1,n+1): c[i-1]=sqrt(2*i+0.6)-0.2

A = linalg.circulant(c)

alpha = 0.25
gamma = circ.gamalph(n,alpha)

# temporal mass matrix
b1 = np.zeros(n)
b1[0] =  1
b1[1] = -1
B1 = linalg.circulant(b1)

# temporal stiffness matrix
theta = 0.5
b2 = np.zeros(n)
b2[0]=theta
b2[1]=1-theta
B2 = linalg.circulant(b2)

# strang alpha-circulant matrices
C1 = linalg.circulant(b1)
C1[0,-1] = -alpha

C2 = linalg.circulant(b2)
C2[0,-1] = (1-theta)*alpha

diag1 = fft.fft( gamma*b1, norm='backward' )
diag2 = fft.fft( gamma*b2, norm='backward' )

F = linalg.dft(n,scale='sqrtn')

V  = circ.eigenvectors( b1, inverse=False, alpha=alpha )
V1 = circ.eigenvectors( b1, inverse=True,  alpha=alpha )

idt = np.matmul(V,V1)

D = circ.eigenvalues( b1, alpha=alpha )
Dtest = linalg.eigvals( B1 )

C1test = np.matmul( V, np.matmul( np.diag(D), V1 ) )


