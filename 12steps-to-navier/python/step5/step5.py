# 2d linear convection

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#
# Simulation parameters
#
# Spatial space
nx = 41 # number of points in the space discretization
abscissa = np.linspace(0, 2, nx) # space discretization
dx = abscissa[1] - abscissa[0] # space step size

# Temporal space
nt = 30 # number of time steps
dt = .01 # dt is the amount of time each timestep covers (delta t)

# Physical parameter
c = 1.0

# initial condition
u_0 = np.ones((nx, nx))
#u_0[int(.5 / dx):int(1 / dx + 1),int(.5 / dx):int(1 / dx + 1)] = 2
normal_distrib = multivariate_normal(mean=[0.8,0.8], cov=[[0.2,0],[0,0.2]])
for i in range(nx):
    for j in range(nx):
        x, y = abscissa[i], abscissa[j]
        u_0[i,j] = normal_distrib.pdf((x, y))

#
# Simulation
#
# result vector
u_n = np.zeros((nx, nx, nt+1))
u_n[:,:,0] = u_0.copy()

# executing the convection one step at a time
for n in range(nt):
    for i in range(nx):
        for j in range(nx):
            u_n[i,j,n+1] = u_n[i,j,n] - c * (dt/dx) * (u_n[i,j,n] - u_n[(i-1) % nx,j,n]) - c * (dt/dx) * (u_n[i,j,n] - u_n[i,(j-1) % nx,n])

#
# Plotting result
#
plt.figure(1)
plt.imshow(u_n[:,:,0])

plt.figure(2)
plt.imshow(u_n[:,:,-1])

plt.show()
