import subprocess
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from io import BytesIO


def make_gif(frames, name):
    images = []
    for x, y, t in frames:
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, lw=2)
        plt.ylim(-1.2, 1.2)
        plt.title(f"{name}, t={t:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        plt.close()
        buf.seek(0)
        images.append(imageio.imread(buf))

    imageio.mimsave(f"{name}.gif", images, duration=0.05)
    print("GIF создан:", f"{name}.gif")


def main():
    proc = subprocess.Popen(["./solver"], stdout=subprocess.PIPE, text=True)

    frames = []
    scheme = None
    CFL = None

    x = []
    y = []
    t = 0

    for line in proc.stdout:
        line = line.strip()

        if line.startswith("SCHEME"):
            if frames:
                make_gif(frames, f"{scheme}_CFL{CFL}")
                frames = []

            _, scheme, _, CFL = line.split()
            CFL = float(CFL)

        elif line.startswith("FRAME"):
            if x:
                frames.append((np.array(x), np.array(y), t))
            x, y = [], []
            t = float(line.split("=")[1])

        elif line == "END":
            pass

        elif line == "DONE":
            make_gif(frames, f"{scheme}_CFL{CFL}")
            frames = []

        else:
            xi, yi = map(float, line.split())
            x.append(xi)
            y.append(yi)


if __name__ == "__main__":
    main()
