
import numpy as np
import matplotlib.pyplot as plt

### === --- --- === ###
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
### === --- --- === ###

# parameters

nt = 256
dt = 0.2
theta = 0.5

y0 = 1

lamda = -0.05 + 0.5j

dtype = complex

y = np.zeros(nt, dtype=dtype)

y[0] = y0

print(y[0])
for i in range(1, nt):
    # (y[i] - y[i-1])/dt = theta*lamda*y[i] + (1-theta)*lambda*y[i-1]
    # y[i]*(1 - dt*theta*lamda) = y[i-1] + dt*(1-theta)*lambda*y[i-1]

    rhs = (y[i-1] + dt*(1-theta)*lamda*y[i-1])
    jac = (1 - dt*theta*lamda)

    y[i] = rhs/jac
    print(y[i])

plt.plot(y.real)
plt.show()
