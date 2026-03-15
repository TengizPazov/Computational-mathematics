#include <iostream>
#include <cmath>

int main() {
    // первый корень
    // приближенное значение x0 = 0.5
    double x0 = 0.5;
    double epsilon1 = 1e-3 / 2;
    double x = 0.214 * exp(x0 * x0);
    std::cout << "Метод 1 (итерации):" << std::endl;
    int iter1 = 0;
    while (std::abs(x - x0) >= epsilon1) {
        x0 = x;
        x = 0.214 * exp(x0 * x0);
        iter1++;
        std::cout << "Итерация " << iter1 << ": x = " << x << std::endl;
    }
    std::cout << "Первый корень: " << x << std::endl;
    // второй корень
    // приближенное значение x0 = 0.5
    x0 = 1;
    epsilon1 = 1e-3 / 2;
    x = std::sqrt(std::log(x0 / 0.214));
    
    std::cout << "\nМетод 2 (итерации):" << std::endl;
    int iter2 = 0;
    while (std::abs(x - x0) >= epsilon1) {
        x0 = x;
        x = std::sqrt(std::log(x0 / 0.214));
        iter2++;
        std::cout << "Итерация " << iter2 << ": x = " << x << std::endl;
    }
    std::cout << "Второй корень: " << x << std::endl;
    std::cout << "\nШирина на полувысоте = " << x - 0.22513135400649073 << std::endl;
    return 0;
}