// ─────────────────────────────────────────────────────────────────────────────
//  2-D heat equation:   25λ φ_xx + λ φ_yy = φ_t ,   λ = 1e-4
//
//  Analytical solution: φ(x,y,t) = cos(πx) sin(5πy) exp(-50π²λ t)
//
//  Boundary / initial conditions (from the problem statement):
//    φ(x,y,0) = cos(πx) sin(5πy)
//    φ(x,0,t) = φ(x,1,t) = 0
//    φ(0,y,t) =  sin(5πy) exp(-50π²λ t)
//    φ(1,y,t) = -sin(5πy) exp(-50π²λ t)
//
//  Domain: x,y,t ∈ [0,1].  Interior (M×M) uniform grid, NX = NY = N.
//
//  We run the simulation for each grid size N ∈ {10,20,50,100,200,500,1000},
//  advance to a fixed output time T_out ≈ 0.01 with time step τ chosen so
//  that the CFL-like number stays modest, then record the max-norm error.
//
//  Results are written to  results/convergence.csv  for Python post-processing.
// ─────────────────────────────────────────────────────────────────────────────

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
static constexpr double Dx      = 25.0 * LAMBDA;   // coefficient in front of φ_xx
static constexpr double Dy      = LAMBDA;           // coefficient in front of φ_yy
static constexpr double T_OUT   = 0.01;             // fixed time at which error is measured

// ── analytical solution ──────────────────────────────────────────────────────
inline double phi_analytical(double x, double y, double t) {
    return std::cos(PI*x) * std::sin(5.0*PI*y)
           * std::exp(-50.0*PI*PI*LAMBDA*t);
}

// ── boundary conditions at time t ───────────────────────────────────────────
inline double bc_left  (double y, double t) {  return  std::sin(5.0*PI*y)*std::exp(-50.0*PI*PI*LAMBDA*t); }
inline double bc_right (double y, double t) {  return -std::sin(5.0*PI*y)*std::exp(-50.0*PI*PI*LAMBDA*t); }
inline double bc_bottom(double /*x*/, double /*t*/) { return 0.0; }
inline double bc_top   (double /*x*/, double /*t*/) { return 0.0; }

// ─────────────────────────────────────────────────────────────────────────────
//  Solve for a single grid size N (interior nodes NxN)
//  Returns max-norm error at T_OUT
// ─────────────────────────────────────────────────────────────────────────────
double solve_grid(std::size_t N,
                  std::ofstream* snapshot_out = nullptr,  // optional field dump
                  double snapshot_t           = -1.0)
{
    const std::size_t M  = N;             // interior nodes in x
    const std::size_t K  = N;             // interior nodes in y
    const std::size_t Ntot = M * K;

    const double hx  = 1.0 / static_cast<double>(N + 1);
    const double hy  = hx;

    // τ: choose so that sx = τ·Dx/hx² ≤ 0.5  (generous for implicit, but keeps
    // the iteration count of Gauss–Seidel reasonable)
    // We also need to land near T_OUT, so we compute the number of steps.
    const double tau_cfl = 0.5 * hx * hx / Dx;   // stability-motivated upper bound
    const double tau_max = T_OUT / 2.0;            // at least 2 steps to T_OUT
    double tau = std::min(tau_cfl, tau_max);

    std::size_t nsteps = static_cast<std::size_t>(std::ceil(T_OUT / tau));
    tau = T_OUT / static_cast<double>(nsteps);     // adjust to hit T_OUT exactly

    // ── initial condition ──
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

        // boundary functions at t_new (fully implicit)
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

    // ── compute max-norm error at T_OUT ──
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

    // ── optional snapshot dump ──
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

// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    fs::create_directories("results");

    const std::vector<std::size_t> grid_sizes = {10, 20, 50, 100, 200, 500, 1000};

    // convergence table
    std::ofstream conv("results/convergence.csv");
    conv << "N,h,error_max\n";

    for (std::size_t N : grid_sizes) {
        std::cout << "Solving N = " << N << " ... " << std::flush;

        // For N=50 also save a 2-D field snapshot (for visualisation)
        std::ofstream* snap_ptr = nullptr;
        std::ofstream  snap_file;
        if (N == 50) {
            snap_file.open("results/snapshot_N50.csv");
            snap_file << "x,y,numerical,analytical\n";
            snap_ptr = &snap_file;
        }

        double err = solve_grid(N, snap_ptr, T_OUT);
        double h   = 1.0 / static_cast<double>(N + 1);

        conv << N << "," << h << "," << err << "\n";
        std::cout << "  h = " << h << "  max-error = " << err << "\n";
    }

    std::cout << "\nResults written to results/convergence.csv\n";
    return 0;
}
