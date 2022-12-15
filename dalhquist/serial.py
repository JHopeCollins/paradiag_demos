
import numpy as np
import matplotlib.pyplot as plt

# ## === --- --- === ###
#
# Solve Dalhquist's ODE using implicit theta method
# and serial timestepping
#
# y = e^{lamda*t}
# dydt = lambda*e^{lamda*t}
# dydt = lambda*y
#
# dy/dt = lambda*y
#
# ## === --- --- === ###

# parameters

nt = 256
dt = 0.2
theta = 0.5

q0 = 1

lamda = -0.05 + 0.5j

# setup timeseries

q = np.zeros(nt+1, dtype=complex)

q[0] = q0

# timestepping loop

for i in range(nt):
    rhs = (1 + dt*(1-theta)*lamda)*q[i]
    jac = (1 - dt*theta*lamda)

    q[i+1] = rhs/jac
    #print(q[i])

# plot

plt.plot(q.real)
plt.show()
