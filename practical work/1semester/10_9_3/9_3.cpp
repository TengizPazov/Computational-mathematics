#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

std::pair<std::vector<double>, std::vector<double>> 
rayleigh_equation(double x0, double v0, double mu, double t_k, double h, const std::string& filename = "results.csv") {
    
    int n_steps = static_cast<int>(t_k / h) + 1;
    
    std::vector<double> time(n_steps);
    std::vector<double> x_results(n_steps);
    std::vector<double> v_results(n_steps);
    
    std::ofstream file(filename);
    file << "time,x,v\n";
    
    // Начальные условия
    double x = x0;
    double v = v0;
    x_results[0] = x;
    v_results[0] = v;
    time[0] = 0.0;
    
    file << time[0] << "," << x_results[0] << "," << v_results[0] << "\n";
    
    for (int i = 1; i < n_steps; ++i) {
        // Коэффициенты k1
        double k1x = v;
        double k1v = mu * (1 - v*v) * v - x;
        
        // Коэффициенты k2
        double temp_x2 = x + (h/2) * k1x;
        double temp_v2 = v + (h/2) * k1v;
        double k2x = temp_v2;
        double k2v = mu * (1 - temp_v2*temp_v2) * temp_v2 - temp_x2;
        
        // Коэффициенты k3
        double temp_x3 = x + (h/2) * k2x;
        double temp_v3 = v + (h/2) * k2v;
        double k3x = temp_v3;
        double k3v = mu * (1 - temp_v3*temp_v3) * temp_v3 - temp_x3;
        
        // Коэффициенты k4
        double temp_x4 = x + h * k3x;
        double temp_v4 = v + h * k3v;
        double k4x = temp_v4;
        double k4v = mu * (1 - temp_v4*temp_v4) * temp_v4 - temp_x4;
        
        x += (h / 6) * (k1x + 2*k2x + 2*k3x + k4x);
        v += (h / 6) * (k1v + 2*k2v + 2*k3v + k4v);
        
        x_results[i] = x;
        v_results[i] = v;
        time[i] = i * h;
    
        file << time[i] << "," << x_results[i] << "," << v_results[i] << "\n";
    }
    
    file.close();
    std::cout << "Данные сохранены в файл: " << filename << std::endl;
    
    return {time, x_results};
}

int main() {
    double mu = 1000.0;
    double t_k = 1000.0;
    double h = 0.001;
    
    auto [time, x] = rayleigh_equation(0.0, 0.001, mu, t_k, h, "rayleigh_data.csv");
    
    std::cout << "Размер массива: " << time.size() << std::endl;
    std::cout << "x(" << t_k << ") = " << x.back() << std::endl;
    
    return 0;
}