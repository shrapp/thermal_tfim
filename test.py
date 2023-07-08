from library import *
import numpy as np

tau = 0.4    # in natural units
dt = 0.00001    # time step
t_max = 100 # total simulation time
psi0 = np.array([0.0, 1.0], dtype=complex)
ks = np.linspace(np.pi/2, np.pi,10)
plot_k2(ks,t_max, dt, psi0, tau)