#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

// Функция f(x) = sin(100x) * e^(-x²) * cos(2x)
double f_x(double x) {
    return sin(100 * x) * exp(-x * x) * cos(2 * x);
}

// Метод трапеций
double trapezoid_method(const std::vector<double>& x) {
    double result = 0.0;
    double h = x[1] - x[0];
    
    for (size_t i = 0; i < x.size() - 1; ++i) {
        result += h * (f_x(x[i + 1]) + f_x(x[i])) / 2.0;
    }
    
    return result;
}

// Метод Симпсона
double simpson(const std::vector<double>& x) {
    if (x.size() % 2 == 0) {
        throw std::invalid_argument("Для метода Симпсона нужно нечетное количество точек");
    }
    
    double result = 0.0;
    double h = x[1] - x[0];
    
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double midpoint = (x[i] + x[i + 1]) / 2.0;
        result += (h / 6.0) * (f_x(x[i]) + 4.0 * f_x(midpoint) + f_x(x[i + 1]));
    }
    
    return result;
}

// Правило 3/8
double three_eighths_rule(const std::vector<double>& x) {
    if ((x.size() - 1) % 3 != 0) {
        throw std::invalid_argument("Количество интервалов должно быть кратно 3");
    }
    
    double result = 0.0;
    double h = x[1] - x[0];
    
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double point1 = (2.0 * x[i] + x[i + 1]) / 3.0;
        double point2 = (x[i] + 2.0 * x[i + 1]) / 3.0;
        result += (h / 8.0) * (f_x(x[i]) + 3.0 * f_x(point1) + 3.0 * f_x(point2) + f_x(x[i + 1]));
    }
    
    return result;
}

int main() {
    std::vector<double> x;
    double a = 0.0;
    double b = 3.0;
    double step = 0.00001;
    
    for (double val = a; val <= b + 1e-10; val += step) {
        x.push_back(val);
    }
    
    std::cout << "Количество точек: " << x.size() << std::endl;
    
    // Применение метода трапеций
    std::cout << "Метод трапеций: " << trapezoid_method(x) << std::endl;
    
    // Применение метода Симпсона
    std::cout << "Метод Симпсона: " << simpson(x) << std::endl;
    
    // Создание сетки для правила 3/8
    std::vector<double> x_for_three_eights;
    double step_three_eights = 0.0000001;
    int num_intervals = static_cast<int>((b - a) / step_three_eights);
    
    num_intervals = (num_intervals / 3) * 3;
    double adjusted_step = (b - a) / num_intervals;
    
    for (int i = 0; i <= num_intervals; ++i) {
        x_for_three_eights.push_back(a + i * adjusted_step);
    }
    
    std::cout << "Количество точек для правила 3/8: " << x_for_three_eights.size() << std::endl;
    std::cout << "Правило 3/8: " << three_eighths_rule(x_for_three_eights) << std::endl;
    
    return 0;
}