#pragma once
#include <vector>
#include <map>
#include <array>
#include <cstddef>
#include "SLAE.hpp"

template<typename LeftF, typename RightF, typename BottomF, typename TopF>
SLAE<double> create_matrix(const std::size_t M,
                            const std::size_t K,
                            const double tau,
                            const double Dx,
                            const double Dy,
                            const double hx,
                            const double hy,
                            const LeftF   &left,
                            const RightF  &right,
                            const BottomF &bottom,
                            const TopF    &top,
                            const Vector<double> &u_i)
{
    const std::size_t N  = M * K;
    const double sx      = tau * Dx / (hx * hx);
    const double sy      = tau * Dy / (hy * hy);
    const double diag_val = 1.0 + 2.0*sx + 2.0*sy;

    std::map<std::array<std::size_t,2>, double> A;
    Vector<double> b(N);

    for (std::size_t k = 0; k < K; ++k) {
        const double y = static_cast<double>(k + 1) * hy;
        for (std::size_t m = 0; m < M; ++m) {
            const std::size_t row = k * M + m;
            const double x = static_cast<double>(m + 1) * hx;

            A[{row, row}] = diag_val;
            b(row) = u_i(row);

            // left neighbour
            if (m > 0)        A[{row, row - 1}] = -sx;
            else              b(row) += sx * left(y);

            // right neighbour
            if (m + 1 < M)    A[{row, row + 1}] = -sx;
            else              b(row) += sx * right(y);

            // bottom neighbour
            if (k > 0)        A[{row, row - M}] = -sy;
            else              b(row) += sy * bottom(x);

            // top neighbour
            if (k + 1 < K)    A[{row, row + M}] = -sy;
            else              b(row) += sy * top(x);
        }
    }

    return SLAE<double>{ CSR_matrix{A, N, N}, b };
}
