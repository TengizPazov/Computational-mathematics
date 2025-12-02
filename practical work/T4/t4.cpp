#include <iostream>
#include <vector>
#include <cmath>

const double E = std::exp(1.0);

std::pair<std::vector<double>, std::vector<double>> t4_problem_function(
    double x, const std::vector<double>& y, const std::vector<double>& p) {
    
    double sqrt_val = 0;
    double df_dv = 0;
    double df_dy = 0;
    
    if (x <= 1 || std::log(x) <= 0) {
        sqrt_val = 0;
        df_dv = 0;
        df_dy = 0;
    } else {
        double arg = (E / std::log(x)) * y[0] * y[0] + 1.0 / (x * x) - std::exp(y[1]) * y[0];
        if (arg < 0) {
            sqrt_val = 0;
            df_dv = 0;
            df_dy = 0;
        } else {
            sqrt_val = std::sqrt(arg);
            df_dv = (std::exp(y[1]) * y[0] * 0.5) / sqrt_val;
            df_dy = -((2 * E * y[0] / std::log(x) - std::exp(y[1])) * 0.5) / sqrt_val;
        }
    }
    
    std::vector<double> y_system = {y[1], sqrt_val};
    std::vector<double> p_system = {p[1], -df_dv * p[1] - df_dy * p[0]};
    
    return {y_system, p_system};
}

std::pair<std::vector<double>, std::vector<std::vector<double>>> t4_y_solution(
    const std::vector<double>& y0, const std::vector<double>& p0,
    double x_start, double x_end, double h) {
    
    int n_steps = static_cast<int>((x_end - x_start) / h) + 1;
    std::vector<double> x_points(n_steps);
    std::vector<std::vector<double>> y_values(n_steps, std::vector<double>(2));
    
    for (int i = 0; i < n_steps; i++) {
        x_points[i] = x_start + i * h;
    }
    
    y_values[0] = y0;
    
    for (int i = 1; i < n_steps; i++) {
        double x_current = x_points[i-1];
        std::vector<double> y = y_values[i-1];
        
        auto k1 = t4_problem_function(x_current, y, p0).first;
        
        std::vector<double> y_temp1(2);
        for (int j = 0; j < 2; j++) y_temp1[j] = y[j] + (h/2) * k1[j];
        auto k2 = t4_problem_function(x_current + h/2, y_temp1, p0).first;
        
        std::vector<double> y_temp2(2);
        for (int j = 0; j < 2; j++) y_temp2[j] = y[j] + (h/2) * k2[j];
        auto k3 = t4_problem_function(x_current + h/2, y_temp2, p0).first;
        
        std::vector<double> y_temp3(2);
        for (int j = 0; j < 2; j++) y_temp3[j] = y[j] + h * k3[j];
        auto k4 = t4_problem_function(x_current + h, y_temp3, p0).first;
        
        for (int j = 0; j < 2; j++) {
            y_values[i][j] = y[j] + (h/6) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
        }
    }
    
    return {x_points, y_values};
}

double t4_p_solution(const std::vector<double>& y0, const std::vector<double>& p0,
                     double x_start, double x_end, double h) {
    
    int n_steps = static_cast<int>((x_end - x_start) / h) + 1;
    std::vector<double> x_points(n_steps);
    std::vector<std::vector<double>> p_values(n_steps, std::vector<double>(2));
    
    for (int i = 0; i < n_steps; i++) {
        x_points[i] = x_start + i * h;
    }
    
    p_values[0] = p0;
    std::vector<double> y_temp = y0;
    
    for (int i = 1; i < n_steps; i++) {
        double x_current = x_points[i-1];
        std::vector<double> p = p_values[i-1];
        
        auto func_res = t4_problem_function(x_current, y_temp, p);
        auto y_rhs = func_res.first;
        auto p_rhs = func_res.second;
        
        auto k1 = p_rhs;
        
        std::vector<double> p_temp1(2);
        for (int j = 0; j < 2; j++) p_temp1[j] = p[j] + (h/2) * k1[j];
        auto k2 = t4_problem_function(x_current + h/2, y_temp, p_temp1).second;
        
        std::vector<double> p_temp2(2);
        for (int j = 0; j < 2; j++) p_temp2[j] = p[j] + (h/2) * k2[j];
        auto k3 = t4_problem_function(x_current + h/2, y_temp, p_temp2).second;
        
        std::vector<double> p_temp3(2);
        for (int j = 0; j < 2; j++) p_temp3[j] = p[j] + h * k3[j];
        auto k4 = t4_problem_function(x_current + h, y_temp, p_temp3).second;
        
        for (int j = 0; j < 2; j++) {
            p_values[i][j] = p[j] + (h/6) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
        }
        
        for (int j = 0; j < 2; j++) {
            y_temp[j] += h * y_rhs[j];
        }
    }
    
    return p_values.back()[0];
}

int main() {
    double x_start = 2.718;
    double x_end = 7.389;
    double h = 0.001;
    
    double r0 = 1e-6;
    double a0 = 1.0;
    std::vector<double> y0 = {E, a0};
    std::vector<double> p0 = {0, 1};
    
    int max_iter = 1000;
    int i = 0;
    double final_alpha = a0;
    std::vector<std::vector<double>> final_y_solution;
    std::vector<double> final_x_points;
    
    while (i < max_iter) {
        auto [x_points, y_full_solution] = t4_y_solution(y0, p0, x_start, x_end, h);
        double y_end = y_full_solution.back()[0];
        double residual = y_end - 2 * E * E;
        
        if (std::abs(residual) < r0) {
            std::cout << "Converged at iteration: " << i << std::endl;
            std::cout << "alpha = " << std::round(a0 * 100000) / 100000.0 << std::endl;
            final_alpha = a0;
            final_y_solution = y_full_solution;
            final_x_points = x_points;
            break;
        }
        
        double p_end = t4_p_solution(y0, p0, x_start, x_end, h);
        
        if (std::abs(p_end) > 1e-8) {
            a0 = a0 - residual / p_end;
        } else {
            a0 = a0 - 0.5 * residual;
        }
        
        y0 = {E, a0};
        i++;
    }
    
    return 0;
}