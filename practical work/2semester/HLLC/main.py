# main.py
import numpy as np
import matplotlib.pyplot as plt
from params import *
from HLLC import HLLC

dx = (x_end - x_start) / N
x = np.linspace(x_start + dx/2, x_end - dx/2, N)

U = np.zeros((5, N))
for i in range(N):
    if x[i] < 0:
        ro = ro_L_init
        u  = u_L_init
        p  = p_L_init
    else:
        ro = ro_R_init
        u  = u_R_init
        p  = p_R_init
    e = p / ((gamma - 1) * ro)
    E = ro * e + 0.5 * ro * u**2
    U[0, i] = ro
    U[1, i] = ro * u
    U[2, i] = 0.0
    U[3, i] = 0.0
    U[4, i] = E

t = 0.0
while t < t_end:
    ro = U[0, :]
    u  = U[1, :] / ro
    p  = (gamma - 1) * (U[4, :] - 0.5 * ro * u**2)
    a  = np.sqrt(gamma * p / ro)
    dt = CFL * dx / np.max(np.abs(u) + a)
    dt = min(dt, t_end - t)

    F = np.zeros((5, N + 1))
    for i in range(1, N):
        F[:, i] = HLLC(U[:, i-1], U[:, i])

    F[:, 0] = HLLC(U[:, 0], U[:, 0])
    F[:, N] = HLLC(U[:, N-1], U[:, N-1])
    U = U - dt / dx * (F[:, 1:] - F[:, :-1])

    t += dt

ro = U[0, :]
u  = U[1, :] / ro
p  = (gamma - 1) * (U[4, :] - 0.5 * ro * u**2)
e  = p / ((gamma - 1) * ro)

fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].plot(x, ro, 'b.')
axs[0].set_ylabel('ρ, кг/м³')
axs[0].grid(True)

axs[1].plot(x, u, 'b.')
axs[1].set_ylabel('u, м/с')
axs[1].grid(True)

axs[2].plot(x, p / 101325, 'b.')
axs[2].set_ylabel('p, атм')
axs[2].grid(True)

axs[3].plot(x, e / 1000, 'b.')
axs[3].set_ylabel('e, кДж/кг')
axs[3].grid(True)
axs[3].set_xlabel('x, м')

plt.tight_layout()
plt.savefig('result.png', dpi=150)
plt.show()