import numpy as np
import matplotlib.pyplot as plt

def adams_bashforth_2(f, y0, x0, x_end, h):
    n_steps = int((x_end - x0) / h) + 1

    x_arr = np.linspace(x0, x_end, n_steps)
    y_arr = np.zeros(n_steps)
    y_arr[0] = y0
    
    # Первый шаг
    k1 = h * f(x_arr[0], y_arr[0])
    k2 = h * f(x_arr[0] + h/2, y_arr[0] + k1/2)
    k3 = h * f(x_arr[0] + h/2, y_arr[0] + k2/2)
    k4 = h * f(x_arr[0] + h, y_arr[0] + k3)
    y_arr[1] = y_arr[0] + (k1 + 2*k2 + 2*k3 + k4) / 6

    for i in range(1, n_steps - 1):
        f_current = f(x_arr[i], y_arr[i])
        f_prev = f(x_arr[i-1], y_arr[i-1])
        
        # Формула Адамса
        y_arr[i+1] = y_arr[i] + h * (1.5 * f_current - 0.5 * f_prev)
    
    return x_arr, y_arr

def f(x, y):
    return -2 * y

# Начальные условия
x0 = 0.0
y0 = -2.0
x_end = 100.0
h = 0.3

x_arr, y_arr = adams_bashforth_2(f, y0, x0, x_end, h)

def exact_solution(x):
    return -2 * np.exp(-2 * x)

x_exact = np.linspace(x0, x_end, 1000)
y_exact = exact_solution(x_exact)

plt.figure(figsize=(12, 6))
plt.plot(x_arr, y_arr, 'b-', linewidth=2, label='Численное решение (Адамс 2-го порядка)')
plt.plot(x_exact, y_exact, 'r--', linewidth=1, alpha=0.7, label='Точное решение')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Решение y\' = -2y, y(0) = -2\nПолный график (x от 0 до 100)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

