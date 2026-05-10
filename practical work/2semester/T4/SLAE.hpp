#pragma once
#include <vector>
#include <map>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <cmath>

// ─────────────────────────────────────────────
//  Dense vector
// ─────────────────────────────────────────────
template<typename T>
class Vector {
public:
    std::vector<T> data;
    explicit Vector(std::size_t n, T val = T{}) : data(n, val) {}
    T&       operator()(std::size_t i)       { return data[i]; }
    const T& operator()(std::size_t i) const { return data[i]; }
    std::size_t size() const { return data.size(); }
};

// ─────────────────────────────────────────────
//  CSR sparse matrix (5-diagonal structure)
// ─────────────────────────────────────────────
struct CSR_matrix {
    std::size_t rows, cols;
    std::vector<double>      val;
    std::vector<std::size_t> col_idx;
    std::vector<std::size_t> row_ptr;   // size = rows+1

    CSR_matrix() : rows(0), cols(0) {}

    // Build from COO map
    CSR_matrix(const std::map<std::array<std::size_t,2>, double>& A,
               std::size_t nrows, std::size_t ncols)
        : rows(nrows), cols(ncols)
    {
        row_ptr.resize(rows + 1, 0);
        // count entries per row
        for (auto& [key, v] : A) row_ptr[key[0] + 1]++;
        for (std::size_t i = 1; i <= rows; ++i) row_ptr[i] += row_ptr[i-1];

        val.resize(A.size());
        col_idx.resize(A.size());

        std::vector<std::size_t> pos(row_ptr.begin(), row_ptr.end());
        for (auto& [key, v] : A) {
            std::size_t p = pos[key[0]]++;
            col_idx[p] = key[1];
            val[p]     = v;
        }
    }

    // Matrix–vector product  y = A*x
    Vector<double> matvec(const Vector<double>& x) const {
        Vector<double> y(rows, 0.0);
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t p = row_ptr[i]; p < row_ptr[i+1]; ++p)
                y(i) += val[p] * x(col_idx[p]);
        return y;
    }
};

// ─────────────────────────────────────────────
//  SLAE container
// ─────────────────────────────────────────────
template<typename T>
struct SLAE {
    CSR_matrix     A;
    Vector<double> b;
};
