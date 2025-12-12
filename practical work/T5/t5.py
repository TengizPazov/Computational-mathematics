import numpy as np
from typing import List, Tuple, Callable, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class State:
    """Состояние системы"""
    x: np.ndarray
    t: float

class RHS:
    """Правая часть системы ОДУ для ограниченной задачи трех тел"""
    def __init__(self, mu: float):
        self.mu = float(mu)
        self.eta = 1.0 - self.mu
    
    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Вычисление правой части системы ОДУ
        x = [x, u, y, v]
        где: u = dx/dt, v = dy/dt
        """
        x0, x1, x2, x3 = x  # x0 = x, x1 = u, x2 = y, x3 = v
        
        # Вычисление A и B
        a_sq = (x0 + self.mu)**2 + x2**2
        b_sq = (x0 - self.eta)**2 + x2**2
        
        A = np.sqrt(a_sq * a_sq * a_sq)
        B = np.sqrt(b_sq * b_sq * b_sq)  
        
        # Правые части уравнений
        dx0_dt = x1
        dx1_dt = x0 + 2*x3 - self.eta*(x0 + self.mu)/A - self.mu*(x0 - self.eta)/B
        dx2_dt = x3
        dx3_dt = x2 - 2*x1 - self.eta*x2/A - self.mu*x2/B
        
        return np.array([dx0_dt, dx1_dt, dx2_dt, dx3_dt], dtype=np.float64)

class DP5:
    """Метод Дормана-Принса 5-го порядка с автоматическим выбором шага"""
    
    # Коэффициенты метода Дормана-Принса 5-го порядка
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ], dtype=np.float64)
    
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=np.float64)
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=np.float64)
    
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=np.float64)
    
    def __init__(self, eps: float = 1e-14, safety_factor: float = 0.9, 
                 min_step: float = 1e-10, max_step: float = 1.0):
        """
        Инициализация метода DP5
        
        Parameters:
        eps : float
            Заданная точность
        safety_factor : float
            Коэффициент безопасности для выбора шага
        min_step : float
            Минимальный допустимый шаг
        max_step : float
            Максимальный допустимый шаг
        """
        self.eps = eps
        self.safety_factor = safety_factor
        self.min_step = min_step
        self.max_step = max_step
    
    def step(self, rhs: Callable, t: float, x: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Один шаг метода DP5
        
        Returns:
        x_new : np.ndarray
            Новое приближение 5-го порядка
            #x_err : np.ndarray
            Разность между приближениями 5-го и 4-го порядка -- ошибка шага
        err : float
            Оценка ошибки
        """
        # Вычисление k_i
        k = np.zeros((7, len(x)), dtype=np.float64)
        
        for i in range(7):
            sum_a = np.zeros_like(x)
            for j in range(i):
                sum_a += self.a[i, j] * k[j]
            
            k[i] = rhs(t + self.c[i] * h, x + h * sum_a)
        
        # Приближение 5-го порядка
        sum_b5 = np.zeros_like(x)
        for i in range(7):
            sum_b5 += self.b5[i] * k[i]
        x5 = x + h * sum_b5
        
        # Приближение 4-го порядка
        sum_b4 = np.zeros_like(x)
        for i in range(7):
            sum_b4 += self.b4[i] * k[i]
        x4 = x + h * sum_b4
        
        # Оценка ошибки
        error = np.linalg.norm(x5 - x4, np.inf)
        
        return x5, error
    
    def solve(self, rhs: RHS, initial_state: State, end_time: float, 
              initial_step: float = 1e-3) -> List[State]:
        """
        Решение системы ОДУ методом DP5
        
        Parameters:
        rhs : RHS
            Правая часть системы ОДУ
        initial_state : State
            Начальное состояние
        end_time : float
            Конечное время
        initial_step : float
            Начальный шаг интегрирования
            
        Returns:
        solution : List[State]
            Список состояний системы
        """
        solution = [initial_state]
        t = initial_state.t
        x = initial_state.x.copy()
        h = initial_step
        
        # Статистика
        steps_accepted = 0
        steps_rejected = 0
        
        while t < end_time:
            # Коррекция шага
            if t + h > end_time:
                h = end_time - t
            
            # шаг
            x_new, error = self.step(rhs, t, x, h)
            
            # точность
            if error <= self.eps:
                t += h
                x = x_new
                solution.append(State(x.copy(), t))
                steps_accepted += 1
                
                # Адаптация шага
                if error > 0:
                    h_new = h * self.safety_factor * (self.eps / error)**0.2
                else:
                    h_new = h * 2.0  # если ошибка нулевая, увеличиваем шаг
                
                h = np.clip(h_new, self.min_step, self.max_step)
            else:
                h_new = h * self.safety_factor * (self.eps / error)**0.25
                h = max(h_new, self.min_step)
                steps_rejected += 1
        
        print(f"Статистика решения:")
        print(f"  Принято шагов: {steps_accepted}")
        print(f"  Отклонено шагов: {steps_rejected}")
        print(f"  Всего шагов: {steps_accepted + steps_rejected}")
        print(f"  Минимальный шаг: {self.min_step:.2e}")
        print(f"  Максимальный шаг: {self.max_step:.2e}")
        
        return solution

def main():
    """Основная функция"""
    # Параметры задачи
    mu = 0.012277471
    T = 17.0652165601579625588917206249
    
    # Начальные условия
    initial_state = State(
        x=np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224], dtype=np.float64),
        t=0.0
    )
    
    # Параметры интегрирования
    end_time = 6 * T
    initial_step = 1e-6
    eps = 1e-14
    
    print(f"Параметры:")
    print(f"  μ = {mu}")
    print(f"  η = {1 - mu}")
    print(f"  Период T = {T}")
    print(f"  Время интегрирования = {end_time}")
    print(f"  Начальный шаг = {initial_step}")
    print(f"  Требуемая точность ε = {eps}")
    print(f"  Начальное состояние: x={initial_state.x[0]}, y={initial_state.x[2]}")
    print(f"  Начальные скорости: u={initial_state.x[1]}, v={initial_state.x[3]}")
    
    # Создаем правую часть
    rhs = RHS(mu)
    
    # Создаем решатель DP5
    dp5 = DP5(eps=eps, min_step=1e-10, max_step=0.1*T)
    
    # Решаем систему
    solution = dp5.solve(rhs, initial_state, end_time, initial_step)
    print(f"  Количество точек решения: {len(solution)}")
    print(f"  Время начала: {solution[0].t}")
    print(f"  Время конца: {solution[-1].t}")
    
    t_vals = [s.t for s in solution]
    x_vals = [s.x[0] for s in solution]
    y_vals = [s.x[2] for s in solution]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_vals, y_vals, 'b-', linewidth=1)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Траектория в системе Земля-Луна\nВремя интегрирования: {end_time:.1f} (5 периодов)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    ax.plot(-mu, 0, 'bo', markersize=12, label='Земля')
    ax.plot(1-mu, 0, 'mo', markersize=8, label='Луна')
    
    ax.plot(x_vals[0], y_vals[0], 'ro', markersize=8, label='Начало')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()