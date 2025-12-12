#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

// Правая часть: dy/dx = -2y
double f(double x, double y) {
    return -2.0 * y;
}

// Рунге-Кутта 4-го порядка
double rk4(double x, double y, double h) {
    double k1 = h * f(x, y);
    double k2 = h * f(x + h/2.0, y + k1/2.0);
    double k3 = h * f(x + h/2.0, y + k2/2.0);
    double k4 = h * f(x + h, y + k3);
    return y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}

// Адамс 2-го порядка
pair<vector<double>, vector<double>> adams_bashforth_2(double y0, double x0, 
                                                        double x_end, double h) {
    int n = (x_end - x0) / h + 1;
    
    vector<double> x(n), y(n);
    x[0] = x0;
    y[0] = y0;
    
    // Первый шаг Рунге-Кутта
    x[1] = x0 + h;
    y[1] = rk4(x[0], y[0], h);
    
    // Адамс-Башфорт
    for (int i = 1; i < n - 1; i++) {
        x[i+1] = x[i] + h;
        double f_curr = f(x[i], y[i]);
        double f_prev = f(x[i-1], y[i-1]);
        y[i+1] = y[i] + h * (1.5*f_curr - 0.5*f_prev);
    }
    
    return {x, y};
}

int main() {
    double x0 = 0.0, y0 = -2.0;
    double x_end = 100.0, h = 0.1;
    
    auto [x_arr, y_arr] = adams_bashforth_2(y0, x0, x_end, h);
    
    //файл
    ofstream file("adams_results.txt");
    for (size_t i = 0; i < x_arr.size(); i++) {
        file << x_arr[i] << " " << y_arr[i] << endl;
    }
    file.close();
    cout << "Всего точек: " << x_arr.size() << endl;
    return 0;
}