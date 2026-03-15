import numpy as np
import matplotlib.pyplot as plt
def shooting_system(x, z, params):
    y, v, p, q = z[0], z[1], z[2], z[3]
    if y < 0:
        sqrt_y = 0
        dsqrt_dy = 0
    else:
        sqrt_y = np.sqrt(y)
        if y > 1e-12:
            dsqrt_dy = 1.0 / (2.0 * sqrt_y)
        else:
            dsqrt_dy = 0
    
    y_prime = v
    v_prime = x * sqrt_y
    df_dv = 0
    df_dy = -x * dsqrt_dy
    
    p_prime = q
    q_prime = -df_dv * q - df_dy * p
    
    return np.array([y_prime, v_prime, p_prime, q_prime])

def runge_kutta_step(f, x, z, h, params):
    k1 = f(x, z, params)
    k2 = f(x + h/2, z + h*k1/2, params)
    k3 = f(x + h/2, z + h*k2/2, params)
    k4 = f(x + h, z + h*k3, params)
    
    return z + h * (k1 + 2*k2 + 2*k3 + k4) / 6


def solve_shooting(alpha, n_steps=100):
    x = 0
    h = 1.0 / n_steps
    z = np.array([0.0, alpha, 0.0, 1.0])
    
    for i in range(n_steps):
        z = runge_kutta_step(shooting_system, x, z, h, None)
        x += h

    return z[0], z[2]


def shooting_method(tol=1e-9, max_iter=100):
    alpha = 1.0
    
    for i in range(max_iter):
        y_end, p_end = solve_shooting(alpha)
        
        r = y_end - 2.0
        
        if abs(r) < tol:
            print(f"alpha = {alpha}")
            return alpha
            
        alpha = alpha - r / p_end
    return alpha

def get_solution(alpha, n_steps=1000):
    x = 0
    h = 1.0 / n_steps
    xs = np.linspace(0, 1, n_steps + 1)
    ys = np.zeros(n_steps + 1)
    z = np.array([0.0, alpha, 0.0, 1.0])
    
    ys[0] = z[0]
    for i in range(n_steps):
        z = runge_kutta_step(shooting_system, x, z, h, None)
        x += h
        ys[i+1] = z[0]
    
    return xs, ys

if __name__ == "__main__":
    optimal_alpha = shooting_method()
    xs, ys = get_solution(optimal_alpha, n_steps=1000)
    print(f"y values array:\n{ys}")
    print(f"\nx values array:\n{xs}")
    plt.plot(xs, ys)
    plt.show()