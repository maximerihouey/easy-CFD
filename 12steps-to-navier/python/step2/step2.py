# 1d convection
# Inviscid Burgers equation

import numpy as np
import matplotlib.pyplot as plt

#
# Simulation parameters
#
# Spatial space
nx = 41 # number of points in the space discretization
abscissa = np.linspace(0, 2, nx) # space discretization
dx = abscissa[1] - abscissa[0] # space step size

# Temporal space
nt = 30 # number of time steps
dt = .015 # dt is the amount of time each timestep covers (delta t)

# Physical parameter
c = 1 # wavespeed

# initial condition
u_0 = np.ones(nx)
u_0[int(.5 / dx):int(1 / dx + 1)] = 2

#
# Simulation
#
# result vector
u_n = np.zeros((nx, nt+1))
u_n[:,0] = u_0.copy()

# executing the convection one step at a time
for n in range(nt):
    u_n[:,n+1] = u_n[:,n] - u_n[:,n] * (dt / dx) * np.insert(np.diff(u_n[:,n]), 0, u_n[0,n] - u_n[-1,n])

#
# Plotting result
#
plt.plot(abscissa, u_0, 'r', label="initial condition")
for n in range(1, nt-1):
    if n % 5 == 0:
        plt.plot(abscissa, u_n[:,n], 'g--')
plt.plot(abscissa, u_n[:,nt], 'b', label="state after %d steps" % nt)
ylim = plt.ylim(); plt.ylim((ylim[0], ylim[1]*1.2))
plt.legend(loc=0)
plt.title("1d convection (inviscid Burgers)")
plt.show()
