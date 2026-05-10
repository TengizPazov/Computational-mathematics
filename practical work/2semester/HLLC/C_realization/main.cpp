#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <array>
//params
const double gamma_ = 5.0 / 3.0;
const int    N      = 100;
const double L      = 10.0;
const double x_start = -L;
const double x_end   =  L;
const double t_end   = 0.02;
const double CFL     = 0.9;
const int SAVE_EVERY = 1;
// начальные условия
const double ro_L_init = 13.0;
const double ro_R_init = 1.3;
const double u_L_init  = 0.0;
const double u_R_init  = 0.0;
const double p_L_init  = 10.0 * 101325.0;
const double p_R_init  =  1.0 * 101325.0;

using Vec5 = std::array<double, 5>;

//hllc
Vec5 HLLC(const Vec5& U_L, const Vec5& U_R) {
    double ro_L = U_L[0], ro_R = U_R[0];
    double u_L  = U_L[1] / ro_L;
    double u_R  = U_R[1] / ro_R;
    double p_L  = (gamma_ - 1.0) * (U_L[4] - 0.5 * ro_L * u_L * u_L);
    double p_R  = (gamma_ - 1.0) * (U_R[4] - 0.5 * ro_R * u_R * u_R);
    double a_L  = std::sqrt(gamma_ * p_L / ro_L);
    double a_R  = std::sqrt(gamma_ * p_R / ro_R);

    double S_L = std::min(u_L - a_L, u_R - a_R);
    double S_R = std::max(u_L + a_L, u_R + a_R);

    double S_star = (p_R - p_L + ro_L * u_L * (S_L - u_L) - ro_R * u_R * (S_R - u_R))
                  / (ro_L * (S_L - u_L) - ro_R * (S_R - u_R));

    Vec5 F_L = {ro_L * u_L,
                ro_L * u_L * u_L + p_L,
                U_L[2] * u_L,
                U_L[3] * u_L,
                u_L * (U_L[4] + p_L)};

    Vec5 F_R = {ro_R * u_R,
                ro_R * u_R * u_R + p_R,
                U_R[2] * u_R,
                U_R[3] * u_R,
                u_R * (U_R[4] + p_R)};
    Vec5 D_star = {0.0, 1.0, 0.0, 0.0, S_star};

    auto compute_F_star = [&](double S_K, const Vec5& U_K, const Vec5& F_K,
                               double p_K, double u_K) -> Vec5 {
        double coeff = p_K + ro_L * (S_K - u_K) * (S_star - u_K);
        Vec5 res;
        for (int k = 0; k < 5; k++)
            res[k] = (S_star * (S_K * U_K[k] - F_K[k]) + S_K * coeff * D_star[k])
                   / (S_K - S_star);
        return res;
    };

    if (S_L >= 0.0)
        return F_L;
    else if (S_L <= 0.0 && 0.0 <= S_star)
        return compute_F_star(S_L, U_L, F_L, p_L, u_L);
    else if (S_star <= 0.0 && 0.0 <= S_R)
        return compute_F_star(S_R, U_R, F_R, p_R, u_R);
    else
        return F_R;
}

int main() {
    double dx = (x_end - x_start) / N;
    std::vector<double> x(N);
    for (int i = 0; i < N; i++)
        x[i] = x_start + dx / 2.0 + i * dx;

    std::vector<Vec5> U(N);
    for (int i = 0; i < N; i++) {
        double ro = (x[i] < 0.0) ? ro_L_init : ro_R_init;
        double u  = (x[i] < 0.0) ? u_L_init  : u_R_init;
        double p  = (x[i] < 0.0) ? p_L_init  : p_R_init;
        double e  = p / ((gamma_ - 1.0) * ro);
        double E  = ro * e + 0.5 * ro * u * u;
        U[i] = {ro, ro * u, 0.0, 0.0, E};
    }

    std::ofstream fout("output.csv");
    fout << "t,x,ro,u,p,e\n";

    double t    = 0.0;
    int    step = 0;

    std::vector<Vec5> F(N + 1);

    while (t < t_end) {
        double max_speed = 0.0;
        for (int i = 0; i < N; i++) {
            double ro = U[i][0];
            double u  = U[i][1] / ro;
            double p  = (gamma_ - 1.0) * (U[i][4] - 0.5 * ro * u * u);
            double a  = std::sqrt(gamma_ * p / ro);
            max_speed = std::max(max_speed, std::abs(u) + a);
        }
        double dt = CFL * dx / max_speed;
        dt = std::min(dt, t_end - t);

        // потоки на гранях
        for (int i = 1; i < N; i++)
            F[i] = HLLC(U[i-1], U[i]);
        F[0] = HLLC(U[0],   U[0]);
        F[N] = HLLC(U[N-1], U[N-1]);

        //обновление U
        for (int i = 0; i < N; i++)
            for (int k = 0; k < 5; k++)
                U[i][k] -= dt / dx * (F[i+1][k] - F[i][k]);

        t += dt;
        step++;

        if (step % SAVE_EVERY == 0) {
            for (int i = 0; i < N; i++) {
                double ro = U[i][0];
                double u  = U[i][1] / ro;
                double p  = (gamma_ - 1.0) * (U[i][4] - 0.5 * ro * u * u);
                double e  = p / ((gamma_ - 1.0) * ro);
                fout << t << "," << x[i] << "," << ro << "," << u << ","
                     << p / 101325.0 << "," << e / 1000.0 << "\n";
            }
        }
    }

    fout.close();
    std::cout << "Кол-во шагов: " << step << ", t = " << t << std::endl;
    return 0;
}