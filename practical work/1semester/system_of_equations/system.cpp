#include <iostream>
#include <vector>
#include <cmath>

// Функция нахождения вектора F в методе Ньютона
std::vector<double> find_F(const std::vector<double>& w0) {
    double F1 = w0[0] * w0[0] + w0[1] * w0[1] - 1;
    double F2 = w0[1] - std::tan(w0[0]);
    return {F1, F2};
}
// Функция нахождения обратной матрицы Якоби
std::vector<std::vector<double>> inverse_Jacobi_matrix(const std::vector<double>& w0) {
    double cos_x = std::cos(w0[0]);
    double det_J = 2 * (w0[0] + w0[1] / (cos_x * cos_x));
    
    // Матрица A
    std::vector<std::vector<double>> A = {
        {1, -2 * w0[1]},
        {1/(cos_x * cos_x), 2 * w0[0]}
    };
    
    // Умножаем матрицу A на 1/det_J
    std::vector<std::vector<double>> inverse_J(2, std::vector<double>(2));
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            inverse_J[i][j] = (1.0 / det_J) * A[i][j];
        }
    }
    
    return inverse_J;
}

// Функция для вычисления нормы вектора
double vector_norm(const std::vector<double>& v) {
    double sum = 0;
    for (double x : v) {
        sum += x * x;
    }
    return std::sqrt(sum);
}

// Функция умножения матрицы на вектор
std::vector<double> matrix_vector_mult(const std::vector<std::vector<double>>& matrix, 
                                      const std::vector<double>& vector) {
    std::vector<double> result(2, 0);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

// Функция вычитания векторов
std::vector<double> vector_subtract(const std::vector<double>& v1, 
                                   const std::vector<double>& v2) {
    std::vector<double> result(2);
    result[0] = v1[0] - v2[0];
    result[1] = v1[1] - v2[1];
    return result;
}

int main() {
    // Для (x1, y1) начальное приближение (x0, y0) = (0.7, 0.7)
    std::vector<double> w0 = {0.7, 0.7};
    std::vector<double> F_now = find_F(w0);
    std::vector<double> w = vector_subtract(w0, matrix_vector_mult(inverse_Jacobi_matrix(w0), find_F(w0)));
    while (vector_norm(F_now) >= 1e-6) {
        w0 = w;
        F_now = find_F(w0);
        w = vector_subtract(w0, matrix_vector_mult(inverse_Jacobi_matrix(w0), find_F(w0)));
    }
    std::cout << "Решение 1: (" << w[0] << ", " << w[1] << ")" << std::endl;
    // Для (x2, y2) начальное приближение (x0, y0) = (-0.7, -0.7)
    w0 = {-0.7, -0.7};
    F_now = find_F(w0);
    w = vector_subtract(w0, matrix_vector_mult(inverse_Jacobi_matrix(w0), find_F(w0)));
    while (vector_norm(F_now) >= 1e-6) {
        w0 = w;
        F_now = find_F(w0);
        w = vector_subtract(w0, matrix_vector_mult(inverse_Jacobi_matrix(w0), find_F(w0)));
    }
    std::cout << "Решение 2: (" << w[0] << ", " << w[1] << ")" << std::endl;
    return 0;
}