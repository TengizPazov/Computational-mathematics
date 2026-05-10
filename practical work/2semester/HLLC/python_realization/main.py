# main.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

history = []
times = []
step = 0

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
    step += 1

    if step % 1 == 0:
        ro_ = U[0, :]
        u_  = U[1, :] / ro_
        p_  = (gamma - 1) * (U[4, :] - 0.5 * ro_ * u_**2)
        e_  = p_ / ((gamma - 1) * ro_)
        history.append((ro_.copy(), u_.copy(), p_.copy() / 101325, e_.copy() / 1000))
        times.append(t)

fig, axs = plt.subplots(4, 1, figsize=(10, 12))
labels = ['ρ, кг/м³', 'u, м/с', 'p, атм', 'e, кДж/кг']

lines = []
for ax, label in zip(axs, labels):
    line, = ax.plot([], [], 'b.')
    ax.set_xlim(x_start, x_end)
    ax.set_ylabel(label)
    ax.grid(True)
    lines.append(line)
axs[-1].set_xlabel('x, м')
title = fig.suptitle('')

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    ro_, u_, p_, e_ = history[frame]
    for line, d, ax in zip(lines, [ro_, u_, p_, e_], axs):
        line.set_data(x, d)
        ax.relim()
        ax.autoscale_view()
    title.set_text(f't = {times[frame]:.5f} с')
    return lines + [title]

ani = animation.FuncAnimation(fig, update, frames=len(history),
                               init_func=init, interval=50, blit=True)

ani.save('hllc.gif', writer='pillow', fps=20)