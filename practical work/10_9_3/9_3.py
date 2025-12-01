import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def rayleigh_function(y, mu):
    return np.array([y[1], mu * (1 - y[1]**2) * y[1] - y[0]])

@jit(nopython=True)
def rayleigh_equation_numba(y0, mu, t_k, h):
    n_steps = int(t_k // h) + 1
    y_values = np.zeros((n_steps, 2))
    y_values[0] = y0
    for i in range(1, n_steps):
        y = y_values[i-1]
        k1 = rayleigh_function(y, mu)
        k2 = rayleigh_function(y + (h / 2) * k1, mu)
        k3 = rayleigh_function(y + (h / 2) * k2, mu)
        k4 = rayleigh_function(y + h * k3, mu)
        y_values[i] = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return y_values
y_traj = rayleigh_equation_numba(np.array([0.0, 0.001]), 1000, 1000, 0.001)
t = np.linspace(0, 1000, len(y_traj))

plt.plot(t, y_traj[:, 0], linewidth=1, color='black', label='x(t)')
plt.xlabel('Время, t')
plt.ylabel('Координата, x')
plt.title('Решение уравнения Релея')
plt.legend()
plt.grid(True)
plt.show()