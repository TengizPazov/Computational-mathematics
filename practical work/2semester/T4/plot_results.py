import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import os

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1.  Convergence plot ────────────────────────────────────────────────────
conv = pd.read_csv(os.path.join(OUT_DIR, "convergence.csv"))
h    = conv["h"].values
err  = conv["error_max"].values
N    = conv["N"].values

# Fit slope in log–log space to find empirical convergence order
coeffs = np.polyfit(np.log(h), np.log(err), 1)
order  = coeffs[0]
print(f"Empirical convergence order: {order:.3f}  (expected ≈ 2.0 for O(h²) scheme)")

# O(h²) reference line scaled to pass through the last data point
ref_h2 = h**2 * (err[-1] / h[-1]**2)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(h, err,    "o-",  color="#1f77b4", lw=2, ms=7, label=r"$\|\varphi_{num} - \varphi_{an}\|_\infty$")
ax.loglog(h, ref_h2, "--",  color="#d62728", lw=1.5, label=r"$O(h^2)$ reference")

# Annotate grid sizes
for hi, ei, ni in zip(h, err, N):
    ax.annotate(f"N={ni}", xy=(hi, ei),
                xytext=(5, 4), textcoords="offset points", fontsize=7)

ax.set_xlabel("Grid spacing  $h$",       fontsize=12)
ax.set_ylabel(r"Max-norm error",          fontsize=12)
ax.set_title(r"Convergence: 2-D heat equation"
             "\n"
             rf"Empirical order $\approx {order:.2f}$",
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which="both", ls=":", alpha=0.5)
fig.tight_layout()
conv_path = os.path.join(OUT_DIR, "convergence.png")
fig.savefig(conv_path, dpi=150)
print(f"Saved {conv_path}")

# ─── 2.  Field comparison for N=50 ──────────────────────────────────────────
snap_path = os.path.join(OUT_DIR, "snapshot_N50.csv")
if os.path.exists(snap_path):
    snap = pd.read_csv(snap_path)
    x_pts = snap["x"].values
    y_pts = snap["y"].values
    num   = snap["numerical"].values
    ana   = snap["analytical"].values
    diff  = np.abs(num - ana)

    # Interpolate onto a regular 200×200 grid for smooth imshow
    xi = np.linspace(x_pts.min(), x_pts.max(), 200)
    yi = np.linspace(y_pts.min(), y_pts.max(), 200)
    XI, YI = np.meshgrid(xi, yi)
    pts = np.column_stack([x_pts, y_pts])

    Z_num  = griddata(pts, num,  (XI, YI), method="linear")
    Z_ana  = griddata(pts, ana,  (XI, YI), method="linear")
    Z_diff = griddata(pts, diff, (XI, YI), method="linear")

    fig2 = plt.figure(figsize=(14, 4.5))
    gs   = GridSpec(1, 3, figure=fig2, wspace=0.35)

    def add_panel(gs_slot, Z, title, cmap="RdBu_r"):
        ax = fig2.add_subplot(gs_slot)
        im = ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1],
                       aspect="equal", cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return ax

    add_panel(gs[0], Z_num,  r"Numerical $\varphi_{num}$  (N=50, t=0.01)")
    add_panel(gs[1], Z_ana,  r"Analytical $\varphi_{an}$  (t=0.01)")
    add_panel(gs[2], Z_diff, r"$|\varphi_{num}-\varphi_{an}|$", cmap="hot_r")

    field_path = os.path.join(OUT_DIR, "field_N50.png")
    fig2.savefig(field_path, dpi=150, bbox_inches="tight")
    print(f"Saved {field_path}")

# ─── 3.  Convergence order bar chart ────────────────────────────────────────
# local slopes between consecutive points
local_orders = (np.diff(np.log(err)) / np.diff(np.log(h)))
labels = [f"N={n1}→{n2}" for n1, n2 in zip(N[:-1], N[1:])]

fig3, ax3 = plt.subplots(figsize=(8, 4))
bars = ax3.bar(labels, local_orders, color="#2ca02c", edgecolor="black", alpha=0.8)
ax3.axhline(2.0, color="red", lw=1.5, ls="--", label="Theoretical order 2")
ax3.set_ylabel("Local convergence order", fontsize=11)
ax3.set_title("Local convergence order between consecutive grid sizes", fontsize=11)
ax3.legend(fontsize=10)
ax3.set_ylim(0, 3)
for bar, val in zip(bars, local_orders):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)
fig3.tight_layout()
order_path = os.path.join(OUT_DIR, "local_order.png")
fig3.savefig(order_path, dpi=150)
print(f"Saved {order_path}")

plt.close("all")
print("\nAll plots saved to results/")
