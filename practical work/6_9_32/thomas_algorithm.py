import numpy as np
import matplotlib.pyplot as plt

def thomas_algorithm(h, u):
    """
    Метод прогонки для решения трёхдиагональной системы сплайнов
    """
    N = len(h)  # количество отрезков
    n_unknown = N - 1  # количество неизвестных
    
    xi = np.zeros(n_unknown)
    eta = np.zeros(n_unknown)
    c = np.zeros(N + 1)
    
    # Прямой ход
    if n_unknown >= 1:
        beta1 = 2.0
        gamma1 = h[1] / (h[0] + h[1])
        xi[0] = -gamma1 / beta1
        eta[0] = (6 * u[0]) / beta1
    
    # Внутренние уравнения
    for i in range(1, n_unknown - 1):
        alpha = h[i] / (h[i] + h[i+1])
        beta = 2.0
        gamma = h[i+1] / (h[i] + h[i+1])
        
        denominator = beta + alpha * xi[i-1]
        xi[i] = -gamma / denominator
        eta[i] = (6 * u[i] - alpha * eta[i-1]) / denominator
    
    # Последнее уравнение
    if n_unknown >= 2:
        i = n_unknown - 1
        alpha = h[N-1] / (h[N-2] + h[N-1])
        beta = 2.0
        
        denominator = beta + alpha * xi[i-1]
        eta[i] = (6 * u[i] - alpha * eta[i-1]) / denominator
    
    # Обратный ход
    if n_unknown >= 1:
        c[N-1] = eta[n_unknown - 1]
        for i in range(n_unknown - 2, -1, -1):
            c[i+1] = xi[i] * c[i+2] + eta[i]
    c[0] = 0.0
    c[N] = 0.0
    
    return c
