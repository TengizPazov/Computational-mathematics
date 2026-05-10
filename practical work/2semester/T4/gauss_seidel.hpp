#pragma once
#include "SLAE.hpp"
#include <cmath>
#include <stdexcept>

struct GaussSeidelResult {
    Vector<double> x;
    std::size_t    iterations;
    double         residual;
};

GaussSeidelResult gauss_seidel(const CSR_matrix& A,
                               const Vector<double>& b,
                               Vector<double> x0,
                               double tol        = 1e-10,
                               std::size_t maxiter = 100000)
{
    const std::size_t n = A.rows;

    Vector<double> x = x0;

    std::vector<double> diag(n, 0.0);
    std::vector<std::size_t> diag_pos(n, SIZE_MAX);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t p = A.row_ptr[i]; p < A.row_ptr[i+1]; ++p)
            if (A.col_idx[p] == i) { diag[i] = A.val[p]; diag_pos[i] = p; break; }

    std::size_t iter = 0;
    double residual  = 0.0;

    for (; iter < maxiter; ++iter) {
        double max_delta = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            double sigma = 0.0;
            for (std::size_t p = A.row_ptr[i]; p < A.row_ptr[i+1]; ++p) {
                std::size_t j = A.col_idx[p];
                if (j != i) sigma += A.val[p] * x(j);
            }

            double x_new = (b(i) - sigma) / diag[i];
            double delta = std::fabs(x_new - x(i));
            if (delta > max_delta) max_delta = delta;
            x(i) = x_new;
        }

        if (max_delta < tol) {
            residual = max_delta;
            ++iter;
            break;
        }
        residual = max_delta;
    }

    return { x, iter, residual };
}
