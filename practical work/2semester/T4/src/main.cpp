#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "SLAE.hpp"
#include "create_matrix.hpp"
#include "gauss_seidel.hpp"

namespace fs = std::filesystem;

static constexpr double PI      = M_PI;
static constexpr double LAMBDA  = 1e-4;
static constexpr double Dx      = 25.0 * LAMBDA;
static constexpr double Dy      = LAMBDA;
static constexpr double T_OUT   = 0.01;

//аналитическое решение
inline double phi_analytical(double x, double y, double t) {
    return std::cos(PI*x) * std::sin(5.0*PI*y)
           * std::exp(-50.0*PI*PI*LAMBDA*t);
}

//граничные условия
inline double bc_left  (double y, double t) {  return  std::sin(5.0*PI*y)*std::exp(-50.0*PI*PI*LAMBDA*t); }
inline double bc_right (double y, double t) {  return -std::sin(5.0*PI*y)*std::exp(-50.0*PI*PI*LAMBDA*t); }
inline double bc_bottom(double /*x*/, double /*t*/) { return 0.0; }
inline double bc_top   (double /*x*/, double /*t*/) { return 0.0; }

double solve_grid(std::size_t N,
                  std::ofstream* snapshot_out = nullptr,
                  double snapshot_t           = -1.0)
{
    const std::size_t M  = N;
    const std::size_t K  = N;
    const std::size_t Ntot = M * K;

    const double hx  = 1.0 / static_cast<double>(N + 1);
    const double hy  = hx;

    const double tau_cfl = 0.5 * hx * hx / Dx;
    const double tau_max = T_OUT / 2.0;
    double tau = std::min(tau_cfl, tau_max);

    std::size_t nsteps = static_cast<std::size_t>(std::ceil(T_OUT / tau));
    tau = T_OUT / static_cast<double>(nsteps);

    //начальное состояние
    Vector<double> phi(Ntot);
    for (std::size_t k = 0; k < K; ++k) {
        double y = (k + 1) * hy;
        for (std::size_t m = 0; m < M; ++m) {
            double x = (m + 1) * hx;
            phi(k*M + m) = phi_analytical(x, y, 0.0);
        }
    }

    double t = 0.0;
    for (std::size_t step = 0; step < nsteps; ++step) {
        double t_new = t + tau;

        //неявные
        auto left   = [&](double y){ return bc_left  (y, t_new); };
        auto right  = [&](double y){ return bc_right (y, t_new); };
        auto bottom = [&](double x){ return bc_bottom(x, t_new); };
        auto top    = [&](double x){ return bc_top   (x, t_new); };

        auto slae = create_matrix(M, K, tau, Dx, Dy, hx, hy,
                                  left, right, bottom, top, phi);

        auto res  = gauss_seidel(slae.A, slae.b, phi, 1e-10, 200000);
        phi = res.x;
        t   = t_new;
    }

    double err_max = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        double y = (k + 1) * hy;
        for (std::size_t m = 0; m < M; ++m) {
            double x   = (m + 1) * hx;
            double exact = phi_analytical(x, y, t);
            double diff  = std::fabs(phi(k*M+m) - exact);
            if (diff > err_max) err_max = diff;
        }
    }

    if (snapshot_out && std::fabs(t - snapshot_t) < 1e-12) {
        auto& out = *snapshot_out;
        for (std::size_t k = 0; k < K; ++k) {
            double y = (k + 1) * hy;
            for (std::size_t m = 0; m < M; ++m) {
                double x = (m + 1) * hx;
                out << x << "," << y << ","
                    << phi(k*M+m) << ","
                    << phi_analytical(x, y, t) << "\n";
            }
        }
    }

    return err_max;
}

int main()
{
    const std::vector<std::size_t> grid_sizes = {10, 20, 50, 100, 200, 500, 1000};

    // convergence table
    std::ofstream conv("convergence.csv");
    conv << "N,h,error_max\n";

    for (std::size_t N : grid_sizes) {
        std::cout << "Solving N = " << N << " ... " << std::flush;


        std::ofstream* snap_ptr = nullptr;
        std::ofstream  snap_file;
        if (N == 50) {
            snap_file.open("snapshot_N50.csv");
            snap_file << "x,y,numerical,analytical\n";
            snap_ptr = &snap_file;
        }

        double err = solve_grid(N, snap_ptr, T_OUT);
        double h   = 1.0 / static_cast<double>(N + 1);

        conv << N << "," << h << "," << err << "\n";
        std::cout << "  h = " << h << "  max-error = " << err << "\n";
    }
    return 0;
}
