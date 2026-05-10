import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

df = pd.read_csv('output.csv')
times = sorted(df['t'].unique())
x = df[df['t'] == times[0]]['x'].values

fig, axs = plt.subplots(4, 1, figsize=(10, 12))
labels = ['ρ, кг/м³', 'u, м/с', 'p, атм', 'e, кДж/кг']
keys   = ['ro', 'u', 'p', 'e']

lines = []
for ax, label in zip(axs, labels):
    line, = ax.plot([], [], 'b.')
    ax.set_xlim(x.min(), x.max())
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
    t = times[frame]
    chunk = df[df['t'] == t]
    for line, key, ax in zip(lines, keys, axs):
        line.set_data(chunk['x'].values, chunk[key].values)
        ax.relim()
        ax.autoscale_view()
    title.set_text(f't = {t:.5f} с')
    return lines + [title]

ani = animation.FuncAnimation(fig, update, frames=len(times),
                               init_func=init, interval=50, blit=True)

ani.save('hllc.gif', writer='pillow', fps=30)
print('hllc.gif сохранён')