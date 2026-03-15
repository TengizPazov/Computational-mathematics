# Upwind and Lax–Wendroff Schemes with GIF Visualization

This repository contains an implementation of two finite‑difference schemes for the one‑dimensional linear advection equation



\[
u_t + a u_x = 0,\qquad a = 1,
\]



with periodic boundary conditions. The numerical solution is computed in C++ and visualized as GIF animations using Python.  
The schemes implemented:

- Upwind (first‑order)
- Lax–Wendroff (second‑order)

Simulations are performed for Courant numbers



\[
\frac{\tau}{h} = 1.0,\; 0.6,\; 0.3.
\]



---

## Repository Structure

T1/
│
├── solver.cpp
│   C++ implementation of both numerical schemes.
│
├── solver
│   Compiled executable produced from solver.cpp.
│
├── graph.py
│   Python script that runs the solver, reads streamed output,
│   and generates GIF animations without intermediate files.
│
├── upwind_CFL0.3.gif
├── upwind_CFL0.6.gif
├── upwind_CFL1.0.gif
│   GIF animations for the Upwind scheme.
│
├── laxwendroff_CFL0.3.gif
├── laxwendroff_CFL0.6.gif
└── laxwendroff_CFL1.0.gif
GIF animations for the Lax–Wendroff scheme.


---

## GIF Animations

### Upwind Scheme

**CFL = 1.0**  
![Upwind CFL 1.0](upwind_CFL1.0.gif)

**CFL = 0.6**  
![Upwind CFL 0.6](upwind_CFL0.6.gif)

**CFL = 0.3**  
![Upwind CFL 0.3](upwind_CFL0.3.gif)

---

### Lax–Wendroff Scheme

**CFL = 1.0**  
![Lax–Wendroff CFL 1.0](laxwendroff_CFL1.0.gif)

**CFL = 0.6**  
![Lax–Wendroff CFL 0.6](laxwendroff_CFL0.6.gif)

**CFL = 0.3**  
![Lax–Wendroff CFL 0.3](laxwendroff_CFL0.3.gif)

---

## Numerical Method Overview

### Grid
Uniform spatial grid:


\[
x_i = ih,\quad i=0,\dots,N.
\]



Initial condition:


\[
u(x,0) = \sin\left(\frac{4\pi x}{L}\right).
\]



Periodic boundary conditions:


\[
u(0,t) = u(L,t).
\]



### Upwind Scheme


\[
u_i^{n+1} = u_i^n - \lambda (u_i^n - u_{i-1}^n).
\]



First‑order accurate.  
Introduces numerical diffusion for \(\lambda < 1\).

### Lax–Wendroff Scheme


\[
u_i^{n+1} = u_i^n
- \frac{\lambda}{2}(u_{i+1}^n - u_{i-1}^n)
+ \frac{\lambda^2}{2}(u_{i+1}^n - 2u_i^n + u_{i-1}^n).
\]



Second‑order accurate.  
Dispersive for small CFL.

---

## Execution Pipeline

### 1. Compile the solver
```bash
g++ solver.cpp -o solver -O3 -std=c++17

2. Generate GIF animations

python3 graph.py
The Python script:

runs the solver as a subprocess,

reads solution frames from standard output,

converts them into images in memory,

assembles them into GIF files.

No intermediate text or image files are created.
Resulting Files
After running graph.py, the following GIFs appear in the directory:
upwind_CFL0.3.gif
upwind_CFL0.6.gif
upwind_CFL1.0.gif
laxwendroff_CFL0.3.gif
laxwendroff_CFL0.6.gif
laxwendroff_CFL1.0.gif

