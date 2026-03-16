📘 Upwind and Lax–Wendroff Schemes with GIF Visualization
This repository contains an implementation of two finite‑difference schemes for the one‑dimensional linear advection equation:

𝑢
𝑡
+
𝑎
𝑢
𝑥
=
0
,
𝑎
=
1
,
with periodic boundary conditions.
The numerical solution is computed in C++ and visualized as GIF animations using Python.

Implemented schemes:

Upwind (first‑order)

Lax–Wendroff (second‑order)

Simulations are performed for Courant numbers:

𝜏
ℎ
=
1.0
,
  
0.6
,
  
0.3.
📁 Repository Structure
Код
T1/
├── solver.cpp              # C++ implementation of both numerical schemes
├── solver                  # Compiled executable
├── graph.py                # Python script that runs solver and generates GIFs
├── upwind_CFL0.3.gif
├── upwind_CFL0.6.gif
├── upwind_CFL1.0.gif
├── laxwendroff_CFL0.3.gif
├── laxwendroff_CFL0.6.gif
└── laxwendroff_CFL1.0.gif
🎞 GIF Animations
Upwind Scheme
CFL = 1.0
[Похоже, результат оказался небезопасным для отображения. Давайте внесем изменения и попробуем что-нибудь другое!]

CFL = 0.6
[Похоже, результат оказался небезопасным для отображения. Давайте внесем изменения и попробуем что-нибудь другое!]

CFL = 0.3
[Похоже, результат оказался небезопасным для отображения. Давайте внесем изменения и попробуем что-нибудь другое!]

Lax–Wendroff Scheme
CFL = 1.0
[Похоже, результат оказался небезопасным для отображения. Давайте внесем изменения и попробуем что-нибудь другое!]

CFL = 0.6
[Похоже, результат оказался небезопасным для отображения. Давайте внесем изменения и попробуем что-нибудь другое!]

CFL = 0.3
[Похоже, результат оказался небезопасным для отображения. Давайте внесем изменения и попробуем что-нибудь другое!]

📐 Numerical Method Overview
Grid
Uniform spatial grid:

𝑥
𝑖
=
𝑖
ℎ
,
𝑖
=
0
,
…
,
𝑁
.
Initial condition:

𝑢
(
𝑥
,
0
)
=
sin
⁡
(
4
𝜋
𝑥
𝐿
)
.
Periodic boundary conditions:

𝑢
(
0
,
𝑡
)
=
𝑢
(
𝐿
,
𝑡
)
.
🔸 Upwind Scheme
𝑢
𝑖
𝑛
+
1
=
𝑢
𝑖
𝑛
−
𝜆
(
𝑢
𝑖
𝑛
−
𝑢
𝑖
−
1
𝑛
)
.
First‑order accurate

Introduces numerical diffusion for 
𝜆
<
1

🔸 Lax–Wendroff Scheme
𝑢
𝑖
𝑛
+
1
=
𝑢
𝑖
𝑛
−
𝜆
2
(
𝑢
𝑖
+
1
𝑛
−
𝑢
𝑖
−
1
𝑛
)
+
𝜆
2
2
(
𝑢
𝑖
+
1
𝑛
−
2
𝑢
𝑖
𝑛
+
𝑢
𝑖
−
1
𝑛
)
.
Second‑order accurate

Dispersive for small CFL

🚀 Execution Pipeline
1. Compile the solver
bash
g++ solver.cpp -o solver -O3 -std=c++17
2. Generate GIF animations
bash
python3 graph.py
The Python script:

runs the solver as a subprocess

reads solution frames from standard output

converts them into images in memory

assembles them into GIF files

No intermediate files are created.

📦 Resulting Files
After running graph.py, the following GIFs appear:

upwind_CFL0.3.gif

upwind_CFL0.6.gif

upwind_CFL1.0.gif

laxwendroff_CFL0.3.gif

laxwendroff_CFL0.6.gif

