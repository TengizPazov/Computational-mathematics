#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <functional>

using namespace std;

// Вектор 4D-состояния
struct Vector4d {
    double data[4];
    
    Vector4d() { data[0] = data[1] = data[2] = data[3] = 0.0; }
    Vector4d(double x, double y, double z, double w) {
        data[0] = x; data[1] = y; data[2] = z; data[3] = w;
    }
    
    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
    
    Vector4d operator+(const Vector4d& other) const {
        return Vector4d(data[0] + other[0], data[1] + other[1], 
                       data[2] + other[2], data[3] + other[3]);
    }
    
    Vector4d operator-(const Vector4d& other) const {
        return Vector4d(data[0] - other[0], data[1] - other[1], 
                       data[2] - other[2], data[3] - other[3]);
    }
    
    Vector4d operator*(double scalar) const {
        return Vector4d(data[0] * scalar, data[1] * scalar, 
                       data[2] * scalar, data[3] * scalar);
    }
    
    Vector4d& operator+=(const Vector4d& other) {
        data[0] += other[0]; data[1] += other[1];
        data[2] += other[2]; data[3] += other[3];
        return *this;
    }
    
    friend Vector4d operator*(double scalar, const Vector4d& vec) {
        return vec * scalar;
    }
    
    // Норма
    double norm_inf() const {
        double max_val = fabs(data[0]);
        for (int i = 1; i < 4; ++i) {
            max_val = max(max_val, fabs(data[i]));
        }
        return max_val;
    }
};

// Структура
struct State {
    Vector4d x;  // [x, u, y, v]
    double t;
    
    State(const Vector4d& x_val, double t_val) : x(x_val), t(t_val) {}
    State() : x(Vector4d()), t(0.0) {}
};

// Класс для вычисления правой части ОДУ
class RHS {
private:
    double mu;
    double eta;
    
public:
    RHS(double mu_val) : mu(mu_val), eta(1.0 - mu_val) {}
    
    Vector4d operator()(double t, const Vector4d& x) const {
        double x0 = x[0];  // x
        double x1 = x[1];  // u = dx/dt
        double x2 = x[2];  // y
        double x3 = x[3];  // v = dy/dt
        
        // Вычисление A и B
        double a_sq = pow(x0 + mu, 2) + x2 * x2;
        double b_sq = pow(x0 - eta, 2) + x2 * x2;
        
        double A = sqrt(a_sq * a_sq * a_sq);
        double B = sqrt(b_sq * b_sq * b_sq);
        
        // Правые части уравнений
        Vector4d dxdt;
        dxdt[0] = x1;  // dx/dt = u
        dxdt[1] = x0 + 2.0 * x3 - eta * (x0 + mu) / A - mu * (x0 - eta) / B;  // du/dt
        dxdt[2] = x3;  // dy/dt = v
        dxdt[3] = x2 - 2.0 * x1 - eta * x2 / A - mu * x2 / B;  // dv/dt
        
        return dxdt;
    }
};

// Метод Дормана-Принса 5-го порядка с автоматическим выбором шага
class DP5 {
private:
    // Коэффициенты метода Дормана-Принса
    double a[7][7];
    double b5[7];
    double b4[7];
    double c[7];
    
    double eps;
    double safety_factor;
    double min_step;
    double max_step;
    
    // Статистика
    int steps_accepted;
    int steps_rejected;
    
    void initialize_coefficients() {
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                a[i][j] = 0.0;
            }
        }
        a[1][0] = 1.0/5.0;
        
        a[2][0] = 3.0/40.0;
        a[2][1] = 9.0/40.0;
        
        a[3][0] = 44.0/45.0;
        a[3][1] = -56.0/15.0;
        a[3][2] = 32.0/9.0;
        
        a[4][0] = 19372.0/6561.0;
        a[4][1] = -25360.0/2187.0;
        a[4][2] = 64448.0/6561.0;
        a[4][3] = -212.0/729.0;
        
        a[5][0] = 9017.0/3168.0;
        a[5][1] = -355.0/33.0;
        a[5][2] = 46732.0/5247.0;
        a[5][3] = 49.0/176.0;
        a[5][4] = -5103.0/18656.0;
        
        a[6][0] = 35.0/384.0;
        a[6][2] = 500.0/1113.0;
        a[6][3] = 125.0/192.0;
        a[6][4] = -2187.0/6784.0;
        a[6][5] = 11.0/84.0;
        
        // Коэффициенты b для 5-го порядка
        for (int i = 0; i < 7; ++i) b5[i] = 0.0;
        b5[0] = 35.0/384.0;
        b5[2] = 500.0/1113.0;
        b5[3] = 125.0/192.0;
        b5[4] = -2187.0/6784.0;
        b5[5] = 11.0/84.0;
        
        // Коэффициенты b для 4-го порядка
        for (int i = 0; i < 7; ++i) b4[i] = 0.0;
        b4[0] = 5179.0/57600.0;
        b4[2] = 7571.0/16695.0;
        b4[3] = 393.0/640.0;
        b4[4] = -92097.0/339200.0;
        b4[5] = 187.0/2100.0;
        b4[6] = 1.0/40.0;
        
        // Коэффициенты c
        for (int i = 0; i < 7; ++i) c[i] = 0.0;
        c[1] = 1.0/5.0;
        c[2] = 3.0/10.0;
        c[3] = 4.0/5.0;
        c[4] = 8.0/9.0;
        c[5] = 1.0;
        c[6] = 1.0;
    }
    
public:
    DP5(double eps_val = 1e-14, double safety_factor_val = 0.9,
        double min_step_val = 1e-10, double max_step_val = 1.0)
        : eps(eps_val), safety_factor(safety_factor_val),
          min_step(min_step_val), max_step(max_step_val),
          steps_accepted(0), steps_rejected(0) {
        initialize_coefficients();
    }
    
    // Один шаг метода DP5
    pair<Vector4d, double> step(const RHS& rhs, double t, const Vector4d& x, double h) {
        vector<Vector4d> k(7, Vector4d());
        
        // Вычисление k_i
        for (int i = 0; i < 7; ++i) {
            Vector4d sum_a;
            for (int j = 0; j < i; ++j) {
                sum_a += a[i][j] * k[j];
            }
            k[i] = rhs(t + c[i] * h, x + h * sum_a);
        }
        
        // Приближение 5-го порядка
        Vector4d sum_b5;
        for (int i = 0; i < 7; ++i) {
            sum_b5 += b5[i] * k[i];
        }
        Vector4d x5 = x + h * sum_b5;
        
        // Приближение 4-го порядка
        Vector4d sum_b4;
        for (int i = 0; i < 7; ++i) {
            sum_b4 += b4[i] * k[i];
        }
        Vector4d x4 = x + h * sum_b4;
        
        // норма ошибки
        Vector4d error_vec = x5 - x4;
        double error = error_vec.norm_inf();
        
        return make_pair(x5, error);
    }
    
    // Решение системы ОДУ
    vector<State> solve(const RHS& rhs, const State& initial_state, 
                       double end_time, double initial_step) {
        vector<State> solution;
        solution.push_back(initial_state);
        
        double t = initial_state.t;
        Vector4d x = initial_state.x;
        double h = initial_step;
        
        steps_accepted = 0;
        steps_rejected = 0;
        
        while (t < end_time) {
            // Коррекция шага
            if (t + h > end_time) {
                h = end_time - t;
            }
            
            // Выполнение шага
            auto result = step(rhs, t, x, h);
            Vector4d x_new = result.first;
            double error = result.second;
            
            // Проверка точности
            if (error <= eps) {
                t += h;
                x = x_new;
                solution.emplace_back(x, t);
                steps_accepted++;
                
                // Адаптация шага
                double h_new;
                if (error > 0) {
                    h_new = h * safety_factor * pow(eps / error, 0.2);
                } else {
                    h_new = h * 2.0;  // если ошибка нулевая, увеличиваем шаг
                }
                
                // Ограничение шага
                if (h_new < min_step) h_new = min_step;
                if (h_new > max_step) h_new = max_step;
                h = h_new;
            } else {
                double h_new = h * safety_factor * pow(eps / error, 0.25);
                if (h_new < min_step) h_new = min_step;
                h = h_new;
                steps_rejected++;
            }
        }
        
        return solution;
    }
    
    void print_statistics() const {
        cout << "Статистика решения:" << endl;
        cout << "  Принято шагов: " << steps_accepted << endl;
        cout << "  Отклонено шагов: " << steps_rejected << endl;
        cout << "  Всего шагов: " << steps_accepted + steps_rejected << endl;
        cout << "  Минимальный шаг: " << scientific << min_step << endl;
        cout << "  Максимальный шаг: " << scientific << max_step << endl;
    }
};

// Запись в файл
void write_solution_to_file(const vector<State>& solution, const string& filename) {
    ofstream file(filename);

    file << scientific << setprecision(15);
    file << "t,x,u,y,v\n";
    
    for (const auto& state : solution) {
        file << state.t << "," 
             << state.x[0] << ","
             << state.x[1] << ","
             << state.x[2] << ","
             << state.x[3] << "\n";
    }
    
    file.close();
    cout << "Результаты записаны в файл: " << filename << endl;
}

int main() {
    // Параметры задачи
    double mu = 0.012277471;
    double T = 17.0652165601579625588917206249;
    
    // Начальные условия
    Vector4d x0(0.994, 0.0, 0.0, -2.00158510637908252240537862224);
    
    State initial_state(x0, 0.0);
    
    // Параметры интегрирования
    double end_time = 5 * T;
    double initial_step = 1e-3;
    double eps = 1e-14;
    // Создаем правую часть
    RHS rhs(mu);
    
    // Создаем решатель DP5
    DP5 dp5(eps, 0.9, 1e-10, 0.1 * T);
    
    // Решаем систему
    auto solution = dp5.solve(rhs, initial_state, end_time, initial_step);
    dp5.print_statistics();
    write_solution_to_file(solution, "trajectory.csv");
    return 0;
}