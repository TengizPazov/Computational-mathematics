import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

CSV_FILE   = "output/solution.csv"
OUTPUT_DIR = "output"
ATM        = 101325.0

N_FRAMES = 60
df = pd.read_csv(CSV_FILE)
df["P_atm"]  = df["P"] / ATM
df["e_kJkg"] = df["e"] / 1e3

all_times = sorted(df["t"].unique())

idx_sel   = np.linspace(0, len(all_times) - 1, N_FRAMES, dtype=int)
times     = [all_times[i] for i in idx_sel]
df = df[df["t"].isin(times)].reset_index(drop=True)

# ping-pong: вперёд + назад = плавная петля
loop_idx = list(range(len(times))) + list(range(len(times) - 2, 0, -1))

VARS = [
    ("rho",    r"$\rho$, кг/м³",  "royalblue"),
    ("u",      r"$u$, м/с",       "tomato"),
    ("e_kJkg", r"$e$, кДж/кг",    "seagreen"),
    ("P_atm",  r"$P$, атм",       "darkorchid"),
]

ranges = {}
for col, _, _ in VARS:
    lo, hi = df[col].min(), df[col].max()
    pad = 0.05 * (hi - lo) if hi != lo else 1.0
    ranges[col] = (lo - pad, hi + pad)
cache = []
for t_s in times:
    sub = df[df["t"] == t_s]
    cache.append((t_s, sub["x"].values, sub))
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Задача Римана — распад разрыва", fontsize=13)
time_text = fig.text(0.5, 0.965, "", ha="center", fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow"))

lines = []
for ax, (col, label, color) in zip(axes.flat, VARS):
    (ln,) = ax.plot([], [], color=color, linewidth=2)
    ax.set_xlim(df["x"].min(), df["x"].max())
    ax.set_ylim(*ranges[col])
    ax.set_xlabel("x, м")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    lines.append((ln, col))

plt.tight_layout(rect=[0, 0, 1, 0.95])

def update_all(fi):
    idx = loop_idx[fi]
    t_s, x_vals, sub = cache[idx]
    for ln, col in lines:
        ln.set_data(x_vals, sub[col].values)
    time_text.set_text(f"t = {t_s:.5f} с")
    return [ln for ln, _ in lines] + [time_text]

ani = animation.FuncAnimation(fig, update_all,
                               frames=len(loop_idx), interval=60, blit=True)
path = os.path.join(OUTPUT_DIR, "animation_all.gif")
ani.save(path, writer="pillow", fps=20, dpi=85)
plt.close()

for step_num, (col, label, color) in enumerate(VARS, start=2):
    varname = col.replace("_", "")
    print(f"\n[{step_num}/5] {label} ...")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.set_xlim(df["x"].min(), df["x"].max())
    ax2.set_ylim(*ranges[col])
    ax2.set_xlabel("x, м", fontsize=11)
    ax2.set_ylabel(label, fontsize=11)
    ax2.grid(True, alpha=0.3)
    (ln2,) = ax2.plot([], [], color=color, linewidth=2.2)
    ttl = ax2.set_title("", fontsize=11)
    plt.tight_layout()

    def make_update(ln, c, title_obj):
        def _update(fi):
            idx = loop_idx[fi]
            t_s, x_vals, sub = cache[idx]
            ln.set_data(x_vals, sub[c].values)
            title_obj.set_text(f"t = {t_s:.5f} с")
            return ln, title_obj
        return _update

    ani2 = animation.FuncAnimation(fig2, make_update(ln2, col, ttl),
                                    frames=len(loop_idx), interval=60, blit=True)
    path2 = os.path.join(OUTPUT_DIR, f"anim_{varname}.gif")
    ani2.save(path2, writer="pillow", fps=20, dpi=100)
    plt.close()
for col, _, _ in VARS:
    print(f"   anim_{col.replace('_','')}.gif")