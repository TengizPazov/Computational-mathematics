import numpy as np
import matplotlib.pyplot as plt
from thomas import thomas_algorithm
from parameters import *
from rho import *
from PIL import Image, ImageDraw
import io
# Шаг сетки
h = L / (NX - 1)
images = []
p = np.full(NX, float(p_init))
t = 0
time_next_output = 0.1 * 86400
dt_out = 0.1 * 86400

while t < T_max:
    a = np.zeros(NX)
    b = np.zeros(NX)
    c = np.zeros(NX)
    d = np.zeros(NX)

    # Внутренние узлы
    for i in range(1, NX - 1):
        if p[i] >= p[i+1]:
            rho_right = rho(p[i])
        else:
            rho_right = rho(p[i+1])

        if p[i-1] >= p[i]:
            rho_left = rho(p[i-1])
        else:
            rho_left = rho(p[i])

        c[i] = k * rho_left  / (mu * h**2)
        b[i] = k * rho_right / (mu * h**2)
        a[i] = -c[i] - b[i] - phi * cf * ro0 / tau
        d[i] = -(phi * cf * ro0 / tau) * p[i]

    # Левая граница
    a[0]  = 1.0
    b[0]  = 0.0
    c[0]  = 0.0
    d[0]  = p_inj

    # Правая граница
    a[-1] = 1.0
    b[-1] = 0.0
    c[-1] = 0.0
    d[-1] = p_prod

    p = thomas_algorithm(a=a, b=b, c=c, d=d)

    t += tau
    if t >= time_next_output:
        x = np.linspace(0, L, NX)
        fig, ax = plt.subplots()
        ax.plot(x, p / 101325)
        ax.set_xlabel('x, м')
        ax.set_ylabel('P, атм')
        ax.set_title(f't = {t/86400:.2f} дней')
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        images.append(Image.open(buf).copy())

        plt.close(fig)
        time_next_output += dt_out
images[0].save(
    'pressure.gif',
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=400,
    loop=0
)