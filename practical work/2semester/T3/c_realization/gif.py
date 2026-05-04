import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import glob

times_days = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

files = sorted(glob.glob("pressure_*.dat"))
images = []

for idx, fname in enumerate(files):
    data = np.loadtxt(fname, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    x = data[:, 0]
    p = data[:, 2]

    t_days = times_days[idx] if idx < len(times_days) else 0.0

    fig, ax = plt.subplots()
    ax.plot(x, p)
    ax.set_xlabel("x, м")
    ax.set_ylabel("P, атм")
    ax.set_title(f"t = {t_days:.2f} дней")
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    images.append(img)

    plt.close(fig)
images[0].save(
    "pressure.gif",
    save_all=True,
    append_images=images[1:],
    optimize=False,
    duration=500,
    loop=0
)
