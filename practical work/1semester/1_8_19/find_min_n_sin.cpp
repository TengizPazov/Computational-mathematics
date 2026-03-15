#include <iostream>
#include <cmath>
#include <cstdint>

// Для sin(t): M = 1 всегда
uint32_t find_min_n_sin(double t_max, double delta = 1e-3, double safety_factor = 0.1) {
    uint32_t n = 0;
    double target = delta * safety_factor;
    
    while (true) {
        double term = std::pow(t_max, n + 1) / std::tgamma(n + 2); // (n+1)! = gamma(n+2)
        if (term <= target) {
            return n;
        }
        n++;
    }
}
// Для e^t: M = e^t (наихудший случай)
uint32_t find_min_n_exp(double t_max, double delta = 1e-3, double safety_factor = 0.1) {
    uint32_t n = 0;
    double M = std::exp(t_max); // Максимальное значение производной
    double target = delta * safety_factor;
    while (true) {
        double term = M * std::pow(t_max, n + 1) / std::tgamma(n + 2);
        if (term <= target) {
            return n;
        }
        n++;
    }
}

int main() {
    // Для отрезка [0, 1]
    uint32_t n1_sin = find_min_n_sin(1.0, 1e-3);
    uint32_t n1_exp = find_min_n_exp(1.0, 1e-3);
    
    std::cout << "Для отрезка [0, 1]:" << std::endl;
    std::cout << "sin(t): можно отбрасывать члены, начиная с n = " << n1_sin << std::endl;
    std::cout << "exp(t): можно отбрасывать члены, начиная с n = " << n1_exp << std::endl;
    
    // Для отрезка [10, 11]
    uint32_t n2_sin = find_min_n_sin(11.0, 1e-3);
    uint32_t n2_exp = find_min_n_exp(11.0, 1e-3);
    
    std::cout << "\nДля отрезка [10, 11]:" << std::endl;
    std::cout << "sin(t): можно отбрасывать члены, начиная с n = " << n2_sin << std::endl;
    std::cout << "exp(t): можно отбрасывать члены, начиная с n = " << n2_exp << std::endl;
    
    return 0;
}