#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// Функция для решения трехдиагональной системы методом прогонки
vector<double> thomas_algorithm(const vector<double>& a, const vector<double>& b, 
                               const vector<double>& c, const vector<double>& d) {
    int n = d.size();
    
    // Прогоночные коэффициенты
    vector<double> alpha(n);
    vector<double> beta(n);
    vector<double> x(n);
    
    // Прямой ход
    alpha[0] = -c[0] / b[0];
    beta[0] = d[0] / b[0];
    
    for (int i = 1; i < n-1; i++) {
        double denominator = b[i] + a[i] * alpha[i-1];
        alpha[i] = -c[i] / denominator;
        beta[i] = (d[i] - a[i] * beta[i-1]) / denominator;
    }
    
    // Обратный ход
    x[n-1] = (d[n-1] - a[n-1] * beta[n-2]) / (b[n-1] + a[n-1] * alpha[n-2]);
    
    for (int i = n-2; i >= 0; i--) {
        x[i] = alpha[i] * x[i+1] + beta[i];
    }
    
    return x;
}

// Правая часть дифференциального уравнения
double f(double x) {
    return 2.0 - 6.0*x + 2.0*pow(x,3) + 
           (pow(x,2) - 3.0)*exp(x)*sin(x)*(1.0 + cos(x)) + 
           cos(x)*(exp(x) + (pow(x,2) - 1.0) + pow(x,4) - 3.0*pow(x,2));
}

// Коэффициент при первой производной
double g(double x) {
    return pow(x,2) - 3.0;
}

// Коэффициент при функции
double h_func(double x) {
    return (pow(x,2) - 3.0) * cos(x);
}

// Решение краевой задачи
pair<vector<double>, vector<double>> solve_bvp() {
    // Параметры
    double a_boundary = 0.0;
    double b_boundary = M_PI;
    int N = 300;
    double h = (b_boundary - a_boundary) / (N + 1);
    
    cout << fixed << setprecision(4);
    cout << "Шаг сетки: " << h << endl;
    cout << "Количество точек: " << (N + 2) << endl;
    cout << "Границы: [" << a_boundary << ", " << b_boundary << "]" << endl;
    
    double y0 = 0.0;
    double yN = 9.6;
    
    // Создание сетки
    vector<double> x(N + 2);
    for (int i = 0; i < N + 2; i++) {
        x[i] = a_boundary + i * h;
    }
    
    // Внутренние точки
    vector<double> x_internal(N);
    for (int i = 0; i < N; i++) {
        x_internal[i] = x[i + 1];
    }
    
    // Коэффициенты трехдиагональной системы
    vector<double> a_coeff(N, 1.0);
    vector<double> b_coeff(N);
    vector<double> c_coeff(N);
    vector<double> d_coeff(N);
    
    for (int i = 0; i < N; i++) {
        double xi = x_internal[i];
        b_coeff[i] = -2.0 - h * g(xi) + h * h * h_func(xi);
        c_coeff[i] = 1.0 + h * g(xi);
        d_coeff[i] = h * h * f(xi);
    }
    
    // Учет граничных условий
    d_coeff[0] -= a_coeff[0] * y0;
    d_coeff[N-1] -= c_coeff[N-1] * yN;
    vector<double> y_internal = thomas_algorithm(a_coeff, b_coeff, c_coeff, d_coeff);
    vector<double> y_full(N + 2);
    y_full[0] = y0;
    for (int i = 0; i < N; i++) {
        y_full[i + 1] = y_internal[i];
    }
    y_full[N + 1] = yN;
    
    return make_pair(x, y_full);
}
void print_solution_at_target_points(const vector<double>& x, const vector<double>& y, 
                                   const vector<double>& target_points) {
    cout << "\n" << string(50, '=') << endl;
    cout << "Решение в целевых точках:" << endl;
    cout << string(50, '=') << endl;
    
    cout << fixed << setprecision(6);
    for (double point : target_points) {
        int closest_idx = 0;
        double min_diff = fabs(x[0] - point);
        
        for (int i = 1; i < x.size(); i++) {
            double diff = fabs(x[i] - point);
            if (diff < min_diff) {
                min_diff = diff;
                closest_idx = i;
            }
        }
        
        cout << "x = " << setw(4) << setprecision(1) << point 
             << " | y = " << setw(12) << setprecision(6) << y[closest_idx] 
             << " | (точка сетки: " << setw(6) << setprecision(4) << x[closest_idx] << ")" << endl;
    }
}

int main() {
    auto [x, y] = solve_bvp();
    
    // Целевые точки для вывода
    vector<double> target_points = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    print_solution_at_target_points(x, y, target_points);
    
    return 0;
}