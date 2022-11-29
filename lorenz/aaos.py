
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import newton_krylov

# ## === --- --- === ###
#
# Solve the Lorenz '63 model using implicit theta method
# solving the all-at-once system with ParaDiag
#
# dx/dt = sigma*(y - x)
# dy/dt = x*(rho - z) - y
# dz/dt = x*y - beta*z
#
# ## === --- --- === ###

# parameters

T = 0.2
nt = 500
dt = T/nt
theta = 0.5

x0 = 5
y0 = -5
z0 = 20

# setup timeseries

xyz = np.zeros((nt+1, 3))

xyz[:, 0] = x0
xyz[:, 1] = y0
xyz[:, 2] = z0


def lorentz(q, sigma=10., beta=8./3, rho=28.):
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return np.asarray([dxdt, dydt, dzdt])


def theta_method(q1, q0):
    return q1 - q0 - dt*(theta*lorentz(q1) + (1 - theta)*lorentz(q0))


def aaosfunc(q):
    r = np.zeros_like(q)
    for i in range(nt):
        q0 = q[i, :]
        q1 = q[i+1, :]
        r[i+1, :] = theta_method(q1, q0)
    return r


xyz = newton_krylov(aaosfunc, xyz,
                    verbose=True, maxiter=100)

plt.plot(xyz[:, 0], xyz[:, 2])
plt.show()
