import numpy as np
def thomas_algorithm(a, b, c, d):
    """
    Решение трёхдиагональной системы:
    c[i]*x[i-1] + a[i]*x[i] + b[i]*x[i+1] = d[i]
    
    a - главная диагональ
    b - верхняя диагональ  
    c - нижняя диагональ
    d - правая часть
    """
    N = len(a)
    a = a.copy()
    d = d.copy()
    x = np.zeros(N)

    # Прямой ход
    for i in range(1, N):
        m = c[i] / a[i-1]
        a[i] -= m * b[i-1]
        d[i] -= m * d[i-1]

    # Обратный ход
    x[-1] = d[-1] / a[-1]
    for i in range(N-2, -1, -1):
        x[i] = (d[i] - b[i] * x[i+1]) / a[i]

    return x