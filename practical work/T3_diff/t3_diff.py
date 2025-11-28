import numpy as np
import matplotlib.pyplot as plt

def thomas_algorithm(a, b, c, d):
    """
    Универсальный метод прогонки для решения трехдиагональной системы:
    a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]
    
    Parameters:
    a - нижняя диагональ (a[0] не используется)
    b - главная диагональ  
    c - верхняя диагональ (c[-1] не используется)
    d - правая часть
    
    Returns:
    x - решение системы
    """
    n = len(d)
    
    # Прогоночные коэффициенты
    alpha = np.zeros(n)
    beta = np.zeros(n)
    x = np.zeros(n)
    
    # Прямой ход
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denominator = b[i] + a[i] * alpha[i-1]
        alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i] * beta[i-1]) / denominator
    
    # Обратный ход
    x[n-1] = (d[n-1] - a[n-1] * beta[n-2]) / (b[n-1] + a[n-1] * alpha[n-2])
    
    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]
    
    return x

def f(x: np.ndarray) -> np.ndarray:
    """Правая часть дифференциального уравнения"""
    return 2 - 6*x + 2*x**3 + (x**2 - 3)*np.exp(x)*np.sin(x)*(1 + np.cos(x)) + np.cos(x)*(np.exp(x) + (x**2 - 1) + x**4 - 3*x**2)

def g(x: np.ndarray) -> np.ndarray:
    """Коэффициент при первой производной"""
    return x**2 - 3

def h_func(x: np.ndarray) -> np.ndarray:
    """Коэффициент при функции"""
    return (x**2 - 3) * np.cos(x)

def solve_bvp():
    """
    Решение краевой задачи:
    y'' + g(x)y' + h(x)y = f(x)
    y(0) = 0, y(π) = 0
    """
    #Params
    a_boundary = 0.0
    b_boundary = np.pi
    N = 30
    h = (b_boundary - a_boundary) / (N + 1)
    
    x = np.linspace(a_boundary, b_boundary, N + 2)
    print(f"Шаг сетки: {h:.4f}")
    print(f"Количество точек: {len(x)}")
    print(f"Границы: [{x[0]}, {x[-1]}]")

    y0 = 0
    yN = 9.6

    x_internal = x[1:-1]
   
    a_coeff = np.ones(N)
    b_coeff = -2 - h * g(x_internal) + h**2 * h_func(x_internal)
    c_coeff = np.ones(N) * (1 + h * g(x_internal))
    d_coeff = h**2 * f(x_internal)
    d_coeff[0] -= a_coeff[0] * y0 
    d_coeff[-1] -= c_coeff[-1] * yN

    print("\nРешение системы методом прогонки...")
    y_internal = thomas_algorithm(a_coeff, b_coeff, c_coeff, d_coeff)

    y_full = np.zeros(N + 2)
    y_full[0] = y0
    y_full[1:-1] = y_internal
    y_full[-1] = yN
    
    return x, y_full

def print_solution_at_target_points(x, y, target_points):
    """Вывод решения в целевых точках"""
    print("\n" + "="*50)
    print("Решение в целевых точках:")
    print("="*50)
    
    for point in target_points:
        idx = np.argmin(np.abs(x - point))
        print(f"x = {point:4.1f} | y = {y[idx]:12.6f} | (точка сетки: {x[idx]:6.4f})")

if __name__ == "__main__":
    x, y = solve_bvp()
    
    # Целевые точки для вывода
    target_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    print_solution_at_target_points(x, y, target_points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Решение')
    plt.plot(target_points, [y[np.argmin(np.abs(x - p))] for p in target_points], 
             'ro', markersize=8, label='Целевые точки')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Решение краевой задачи методом прогонки')
    plt.legend()
    plt.show()
