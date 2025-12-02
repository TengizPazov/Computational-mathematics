#include <iostream>
#include <vector>
#include <cmath>

std::vector<double> shooting_system(double x, const std::vector<double>& z) {
    double y = z[0];
    double v = z[1];
    double p = z[2];
    double q = z[3];
    
    double sqrt_y, dsqrt_dy;
    if (y < 0) {
        sqrt_y = 0;
        dsqrt_dy = 0;
    } else {
        sqrt_y = std::sqrt(y);
        if (y > 1e-12) {
            dsqrt_dy = 1.0 / (2.0 * sqrt_y);
        } else {
            dsqrt_dy = 0;
        }
    }
    
    double y_prime = v;
    double v_prime = x * sqrt_y;
    double df_dv = 0;
    double df_dy = -x * dsqrt_dy;
    
    double p_prime = q;
    double q_prime = -df_dv * q - df_dy * p;
    
    return {y_prime, v_prime, p_prime, q_prime};
}

std::vector<double> runge_kutta_step(double x, const std::vector<double>& z, double h) {
    auto k1 = shooting_system(x, z);
    
    std::vector<double> z2(4);
    for (int i = 0; i < 4; i++) z2[i] = z[i] + h * k1[i] / 2.0;
    auto k2 = shooting_system(x + h/2.0, z2);
    
    for (int i = 0; i < 4; i++) z2[i] = z[i] + h * k2[i] / 2.0;
    auto k3 = shooting_system(x + h/2.0, z2);
    
    for (int i = 0; i < 4; i++) z2[i] = z[i] + h * k3[i];
    auto k4 = shooting_system(x + h, z2);
    
    std::vector<double> result(4);
    for (int i = 0; i < 4; i++) {
        result[i] = z[i] + h * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
    }
    return result;
}

std::pair<double, double> solve_shooting(double alpha, int n_steps = 100) {
    double x = 0;
    double h = 1.0 / n_steps;
    std::vector<double> z = {0.0, alpha, 0.0, 1.0};
    
    for (int i = 0; i < n_steps; i++) {
        z = runge_kutta_step(x, z, h);
        x += h;
    }
    
    return {z[0], z[2]};
}

double shooting_method(double tol = 1e-9, int max_iter = 100) {
    double alpha = 1.0;
    
    for (int i = 0; i < max_iter; i++) {
        auto [y_end, p_end] = solve_shooting(alpha);
        
        double r = y_end - 2.0;
        
        if (std::abs(r) < tol) {
            std::cout << "alpha = " << alpha << std::endl;
            return alpha;
        }
        
        alpha = alpha - r / p_end;
    }
    return alpha;
}

std::pair<std::vector<double>, std::vector<double>> get_solution(double alpha, int n_steps = 1000) {
    double x = 0;
    double h = 1.0 / n_steps;
    std::vector<double> xs(n_steps + 1);
    std::vector<double> ys(n_steps + 1);
    
    std::vector<double> z = {0.0, alpha, 0.0, 1.0};
    
    xs[0] = 0;
    ys[0] = z[0];
    
    for (int i = 0; i < n_steps; i++) {
        z = runge_kutta_step(x, z, h);
        x += h;
        xs[i+1] = x;
        ys[i+1] = z[0];
    }
    
    return {xs, ys};
}

int main() {
    double optimal_alpha = shooting_method();
    auto [xs, ys] = get_solution(optimal_alpha, 1000);
    
    std::cout << "y values array:" << std::endl;
    for (size_t i = 0; i < ys.size(); i++) {
        std::cout << ys[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nx values array:" << std::endl;
    for (size_t i = 0; i < xs.size(); i++) {
        std::cout << xs[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}