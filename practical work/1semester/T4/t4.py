import numpy as np
import matplotlib.pyplot as plt

# y[0] = y, y[1] = v
def t4_problem_function(x, y, p):
    '''
    The function returns the right-hand sides of two Cauchy problems
    '''
    if x <= 1 or np.log(x) <= 0:
        sqrt_val = 0
        df_dv = 0
        df_dy = 0
    else:
        arg = (np.e / np.log(x)) * y[0]**2 + 1 / x**2 - (np.e**y[1]) * y[0]
        if arg < 0:
            sqrt_val = 0
            df_dv = 0
            df_dy = 0
        else:
            sqrt_val = np.sqrt(arg)
            df_dv = (np.e**y[1] * y[0] * 0.5) / sqrt_val
            df_dy = -((2 * np.e * y[0] / np.log(x) - np.e**y[1]) * 0.5) / sqrt_val
    y_system = np.array([y[1], sqrt_val])
    p_system = np.array([p[1], -df_dv * p[1] - df_dy * p[0]])
    
    return y_system, p_system

def t4_y_solution(y0, p0, x_start, x_end, h):
    n_steps = int((x_end - x_start) // h) + 1
    x_points = np.linspace(x_start, x_end, n_steps)
    y_values = np.zeros((n_steps, 2))
    y_values[0] = y0
    for i in range(1, n_steps):
        x_current = x_points[i-1]
        y = y_values[i-1]
        k1 = t4_problem_function(x_current, y, p0)[0]
        k2 = t4_problem_function(x_current + h/2, y + (h/2)*k1, p0)[0]
        k3 = t4_problem_function(x_current + h/2, y + (h/2)*k2, p0)[0]
        k4 = t4_problem_function(x_current + h, y + h*k3, p0)[0]
        y_values[i] = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_points, y_values

def t4_p_solution(y0, p0, x_start, x_end, h):
    n_steps = int((x_end - x_start) // h) + 1
    x_points = np.linspace(x_start, x_end, n_steps)
    p_values = np.zeros((n_steps, 2))
    p_values[0] = p0
    y_temp = np.array(y0)
    
    for i in range(1, n_steps):
        x_current = x_points[i-1]
        p = p_values[i-1]
        y_rhs, p_rhs = t4_problem_function(x_current, y_temp, p)
        
        k1 = p_rhs
        k2 = t4_problem_function(x_current + h/2, y_temp, p + (h/2)*k1)[1]
        k3 = t4_problem_function(x_current + h/2, y_temp, p + (h/2)*k2)[1]
        k4 = t4_problem_function(x_current + h, y_temp, p + h*k3)[1]
        
        p_values[i] = p + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        y_temp = y_temp + h * y_rhs
    
    return p_values[-1, 0]
x_start = 2.718
x_end = 7.389
h = 0.001

r0 = 1e-6
a0 = 1.0
y0 = [np.e, a0]
p0 = [0, 1]

i = 0
max_iter = 1000
while i < max_iter:
    x_points, y_full_solution = t4_y_solution(y0, p0, x_start, x_end, h)
    y_end = y_full_solution[-1, 0]
    residual = y_end - 2 * np.e**2
    if abs(residual) < r0:
        print(f"Сошлось на итерации: {i}")
        print(f"α = {round(a0, 5)}")
        final_alpha = a0
        final_y_solution = y_full_solution
        break
    p_end = t4_p_solution(y0, p0, x_start, x_end, h)
    if abs(p_end) > 1e-8:
        a0 = a0 - residual / p_end
    else:
        a0 = a0 - 0.5 * residual
    y0 = [np.e, a0]
    i += 1
plt.plot(x_points, y_full_solution[:, 0])
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Решение с помощью метода пристрелки')
plt.show()
#result = [y_full_solution[i] for i in range(len(y_full_solution[:, 0])) if x_points[i] == 0.5 or x_points[i] == 1.0 or x_points[i] == 1.5 or x_points[i] == 2 or x_points[i] == 2.5]
