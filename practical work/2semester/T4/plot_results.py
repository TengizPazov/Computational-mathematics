import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

conv = pd.read_csv('convergence.csv')
h   = conv["h"].values
err = conv["error_max"].values
N   = conv["N"].values

# порядок сходимости
order = np.polyfit(np.log(h), np.log(err), 1)[0]
print(f"Порядок сходимости: {order:.3f} (ожидается 2.0)")

fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(h, err, "o-", color="steelblue", lw=2, ms=7, label="Численная ошибка")
ax.loglog(h, h**2 * err[-1] / h[-1]**2, "--", color="tomato", lw=1.5, label="O(h²)")

for hi, ei, ni in zip(h, err, N):
    ax.annotate(f"N={ni}", xy=(hi, ei), xytext=(5, 4),
                textcoords="offset points", fontsize=8)

ax.set_xlabel("Шаг сетки h", fontsize=12)
ax.set_ylabel("Максимальная ошибка", fontsize=12)
ax.set_title(f"Сходимость схемы\nПорядок ≈ {order:.2f}", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig('convergence.png', dpi=150)
print("Сохранено: convergence.png")