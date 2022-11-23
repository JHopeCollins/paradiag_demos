
import numpy as np
import matplotlib.pyplot as plt

### === --- --- === ###
#
# Solve Dalhquist's ODE using implicit theta method
# and serial timestepping
#
### === --- --- === ###

# parameters

nt = 128
dt = 0.2
theta = 1.0

y0 = 1

lamda = -0.1

# y = e^{i*lamda*t}
# dydt = i*lambda*e^{i*lamda*t}
# dydt = i*lambda*y

# dy/dt = lambda*y

y = np.zeros(nt)

y[0] = y0

print(y[0])
for i in range(1,nt):
    # (y[i] - y[i-1])/dt = theta*lamda*y[i] + (1-theta)*lambda*y[i-1]
    # y[i]*(1 - dt*theta*lamda) = y[i-1] + (1-theta)*lambda*y[i-1]

    y[i] = (y[i-1] + dt*(1-theta)*lamda*y[i-1])/(1 - dt*theta*lamda)
    print(y[i])

plt.plot(y)
plt.show()
