import numpy as np
import matplotlib.pyplot as plt

# Метод прогонки (Thomas algorithm)
def thomas_algorithm(a, b, c, d):
    n = len(d)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    x = np.zeros(n)

    # Прямой ход
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] + a[i] * alpha[i - 1]
        alpha[i] = -c[i] / denom
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

    # Обратный ход
    x[n - 1] = (d[n - 1] - a[n - 1] * beta[n - 2]) / (b[n - 1] + a[n - 1] * alpha[n - 2])
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x

def safe_exp(x, max_val=50.0):
    return np.exp(np.clip(x, -max_val, max_val))

def f_original(x, y, y_prime):
    eps = 1e-12
    log_x = np.where(np.abs(np.log(x)) < eps, eps, np.log(x))
    exp_term = safe_exp(y_prime)
    under_sqrt = 1.0 / x**2 + np.e * y**2 / log_x - y * exp_term
    return np.sqrt(np.maximum(under_sqrt, eps))

def f_y(x, y, y_prime):
    eps = 1e-12
    log_x = np.where(np.abs(np.log(x)) < eps, eps, np.log(x))
    exp_term = safe_exp(y_prime)
    under_sqrt = 1.0 / x**2 + np.e * y**2 / log_x - y * exp_term
    denominator = 2.0 * np.sqrt(np.maximum(under_sqrt, eps))
    numerator = 2.0 * np.e * y / log_x - exp_term
    return np.clip(numerator / denominator, -1e10, 1e10)

def f_y_prime(x, y, y_prime):
    eps = 1e-12
    log_x = np.where(np.abs(np.log(x)) < eps, eps, np.log(x))
    exp_term = safe_exp(y_prime)
    under_sqrt = 1.0 / x**2 + np.e * y**2 / log_x - y * exp_term
    denominator = 2.0 * np.sqrt(np.maximum(under_sqrt, eps))
    numerator = -y * exp_term
    return np.clip(numerator / denominator, -1e10, 1e10)

def g_n(x, y, y_prime):
    return f_original(x, y, y_prime) - f_y(x, y, y_prime) * y - f_y_prime(x, y, y_prime) * y_prime

# Решение линейной задачи методом прогонки
def linear_bvp_solve_sweep(p, q, r, x, y_left, y_right):
    N = len(x)
    h = x[1] - x[0]
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    a[0], b[0], c[0], d[0] = 0, 1, 0, y_left
    a[-1], b[-1], c[-1], d[-1] = 0, 1, 0, y_right

    for i in range(1, N - 1):
        a[i] = 1.0 / h**2 - p[i] / (2.0 * h)
        b[i] = -2.0 / h**2 + q[i]
        c[i] = 1.0 / h**2 + p[i] / (2.0 * h)
        d[i] = r[i]

    return thomas_algorithm(a, b, c, d)

# Метод квазилинеаризации
def quasilinear_solve(N=200, tol=1e-10, max_iter=10000):
    a, b = np.e, np.e**2
    y_a, y_b = np.e, 2 * np.e**2
    x = np.linspace(a, b, N)
    h = x[1] - x[0]

    # Начальное приближение
    y = x * np.log(x)

    for it in range(max_iter):
        y_prime = np.gradient(y, h)
        p = -f_y_prime(x, y, y_prime)
        q = -f_y(x, y, y_prime)
        r = g_n(x, y, y_prime)

        y_new = linear_bvp_solve_sweep(p, q, r, x, y_a, y_b)
        diff_norm = np.linalg.norm(y_new - y, np.inf)

        print(f"Итерация {it+1}: {diff_norm:.3e}")
        if diff_norm < tol:
            y = y_new
            print("Сходимость достигнута")
            break
        y = y_new

    return x, y

if __name__ == "__main__":
    x_result, y_result = quasilinear_solve(N=200, tol=1e-6)

    plt.figure(figsize=(10, 6))
    plt.plot(x_result, y_result, 'b-', linewidth=2, label='Решение')
    plt.plot(x_result, x_result * np.log(x_result), 'r--', linewidth=1, label='Начальное приближение')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Квазилинеаризация')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
