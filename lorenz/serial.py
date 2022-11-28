
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov

# ## === --- --- === ###
#
# Solve the Lorenz '63 model using implicit theta method
# and serial timestepping
#
# dx/dt = sigma*(y - x)
# dy/dt = x*(rho - z) - y
# dz/dt = x*y - beta*z
#
# ## === --- --- === ###

# parameters

T = 10.
nt = 5000
dt = T/nt
theta = 0.5

x0 = 5
y0 = -5
z0 = 20

# setup timeseries

xyz = np.zeros((nt+1, 3))

xyz[0, 0] = x0
xyz[0, 1] = y0
xyz[0, 2] = z0

# timestepping loop


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


for i in range(nt):
    q0 = xyz[i, :]

    def f(q1):
        return theta_method(q1, q0)

    xyz[i+1, :] = newton_krylov(f, q0)

# plot

plt.plot(xyz[:, 0], xyz[:, 2])
plt.show()
