#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <iomanip>
#include <sys/stat.h>

const double GAMMA = 5.0 / 3.0;  // показатель адиабаты
const double L = 10.0;            // половина длины области
const int NX = 100;               // число ячеек
const double T_END = 0.02;        // конечное время
const double CFL_MAX = 0.01;      // максимальное число Куранта
const double TAU_INIT = 1e-7;     // начальный шаг по времени

// Начальные условия
const double rho_L = 13.0;
const double u_L   = 0.0;
const double P_L   = 10.0 * 101325.0;

const double rho_R = 1.3;
const double u_R   = 0.0;
const double P_R   = 1.0 * 101325.0;

// Скорость звука
inline double sound_speed(double rho, double P) {
    return std::sqrt(GAMMA * P / rho);
}

// Удельная внутренняя энергия из давления
inline double energy_from_P(double rho, double P) {
    return P / ((GAMMA - 1.0) * rho);
}

// Давление из удельной внутренней энергии
inline double pressure(double rho, double e) {
    return (GAMMA - 1.0) * rho * e;
}

struct Matrix3x3 {
    double a[3][3];

    Matrix3x3() { for(int i=0;i<3;i++) for(int j=0;j<3;j++) a[i][j]=0.0; }

    // Умножение матрицы на вектор
    std::vector<double> matvec(const std::vector<double>& v) const {
        std::vector<double> res(3, 0.0);
        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                res[i] += a[i][j] * v[j];
        return res;
    }

    // Умножение матриц
    Matrix3x3 matmat(const Matrix3x3& B) const {
        Matrix3x3 C;
        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                for(int k=0;k<3;k++)
                    C.a[i][j] += a[i][k] * B.a[k][j];
        return C;
    }
};

// Построение Omega^T
Matrix3x3 build_OmegaT(double u, double c) {
    Matrix3x3 M;
    double gm1 = GAMMA - 1.0;
    M.a[0][0] = -u*c;  M.a[0][1] =  c;  M.a[0][2] = gm1;
    M.a[1][0] = -c*c;  M.a[1][1] =  0;  M.a[1][2] = gm1;
    M.a[2][0] =  u*c;  M.a[2][1] = -c;  M.a[2][2] = gm1;
    return M;
}

Matrix3x3 invert3x3(const Matrix3x3& M) {
    double det =
        M.a[0][0]*(M.a[1][1]*M.a[2][2] - M.a[1][2]*M.a[2][1])
       -M.a[0][1]*(M.a[1][0]*M.a[2][2] - M.a[1][2]*M.a[2][0])
       +M.a[0][2]*(M.a[1][0]*M.a[2][1] - M.a[1][1]*M.a[2][0]);

    Matrix3x3 inv;
    inv.a[0][0] =  (M.a[1][1]*M.a[2][2] - M.a[1][2]*M.a[2][1]) / det;
    inv.a[0][1] = -(M.a[0][1]*M.a[2][2] - M.a[0][2]*M.a[2][1]) / det;
    inv.a[0][2] =  (M.a[0][1]*M.a[1][2] - M.a[0][2]*M.a[1][1]) / det;

    inv.a[1][0] = -(M.a[1][0]*M.a[2][2] - M.a[1][2]*M.a[2][0]) / det;
    inv.a[1][1] =  (M.a[0][0]*M.a[2][2] - M.a[0][2]*M.a[2][0]) / det;
    inv.a[1][2] = -(M.a[0][0]*M.a[1][2] - M.a[0][2]*M.a[1][0]) / det;

    inv.a[2][0] =  (M.a[1][0]*M.a[2][1] - M.a[1][1]*M.a[2][0]) / det;
    inv.a[2][1] = -(M.a[0][0]*M.a[2][1] - M.a[0][1]*M.a[2][0]) / det;
    inv.a[2][2] =  (M.a[0][0]*M.a[1][1] - M.a[0][1]*M.a[1][0]) / det;
    return inv;
}


struct EigenData {
    double lam[3];      // собственные значения
    double abs_lam[3];
};

EigenData compute_eigen(double u, double c) {
    EigenData ed;
    ed.lam[0] = u + c;
    ed.lam[1] = u;
    ed.lam[2] = u - c;
    for(int k=0;k<3;k++) ed.abs_lam[k] = std::abs(ed.lam[k]);
    return ed;
}


std::vector<double> apply_LambdaFactor_OmegaT(
    const Matrix3x3& OmegaT,
    const EigenData& ed,
    const std::vector<double>& v,
    bool positive_part)
{
    std::vector<double> z = OmegaT.matvec(v);

    for(int k=0;k<3;k++) {
        double factor;
        if(positive_part)
            factor = 0.5*(ed.lam[k] + ed.abs_lam[k]);
        else
            factor = 0.5*(ed.lam[k] - ed.abs_lam[k]);
        z[k] *= factor;
    }
    return z;
}

int main() {
    const double h = 2.0 * L / (NX - 1);  // шаг по пространству

    // Сетка
    std::vector<double> x(NX);
    for(int i = 0; i < NX; i++)
        x[i] = -L + i * h;

    std::vector<double> rho(NX), u(NX), e(NX), P(NX);

    // Начальные условия
    for(int i = 0; i < NX; i++) {
        if(x[i] < 0.0) {
            rho[i] = rho_L;   
            u[i]   = u_L;
            P[i]   = P_L;
        } else {
            rho[i] = rho_R;
            u[i]   = u_R;
            P[i]   = P_R;
        }
        e[i] = energy_from_P(rho[i], P[i]);
    }

    // Создаём папку для вывода
    mkdir("output", 0777);

    // Файл для записи временных срезов
    std::ofstream fout("output/solution.csv");
    fout << std::scientific << std::setprecision(6);
    fout << "t,x,rho,u,e,P\n";

    // Сохраняем начальное состояние
    double t = 0.0;
    auto save_snapshot = [&]() {
        for(int i = 0; i < NX; i++)
            fout << t << "," << x[i] << "," << rho[i] << ","
                 << u[i] << "," << e[i] << "," << P[i] << "\n";
    };
    save_snapshot();

    // Временной цикл
    double tau = TAU_INIT;
    int step = 0;
    int save_every = 50;

    while(t < T_END) {
        double max_lam = 0.0;
        for(int i = 0; i < NX; i++) {
            double c = sound_speed(rho[i], P[i]);
            double lam = std::abs(u[i]) + c;
            max_lam = std::max(max_lam, lam);
        }
        double tau_cfl = CFL_MAX * h / max_lam;
        tau = std::min(tau_cfl, TAU_INIT);
        if(t + tau > T_END) tau = T_END - t;

        std::vector<std::vector<double>> w(NX, std::vector<double>(3));
        for(int i = 0; i < NX; i++) {
            w[i][0] = rho[i];
            w[i][1] = rho[i] * u[i];
            w[i][2] = rho[i] * e[i];
        }

        std::vector<std::vector<double>> w_new(NX, std::vector<double>(3));

        for(int i = 0; i < NX; i++) {
            // Граничные условия: нулевой градиент
            int im = (i == 0)    ? 1   : i - 1;
            int ip = (i == NX-1) ? NX-2: i + 1;

            double ci = sound_speed(rho[i], P[i]);
            Matrix3x3 OmT = build_OmegaT(u[i], ci);
            Matrix3x3 OmT_inv = invert3x3(OmT);
            EigenData ed = compute_eigen(u[i], ci);

            std::vector<double> dw_left(3), dw_right(3);
            for(int k=0;k<3;k++) {
                dw_left[k]  = w[i][k] - w[im][k];
                dw_right[k] = w[ip][k] - w[i][k];
            }

            std::vector<double> term_plus  = apply_LambdaFactor_OmegaT(OmT, ed, dw_left,  true);
            std::vector<double> term_minus = apply_LambdaFactor_OmegaT(OmT, ed, dw_right, false);

            std::vector<double> Ow = OmT.matvec(w[i]);

            // Правая часть в пространстве характеристик
            std::vector<double> rhs(3);
            for(int k=0;k<3;k++)
                rhs[k] = Ow[k]
                         - tau / (2.0*h) * term_plus[k]
                         - tau / (2.0*h) * term_minus[k];

            w_new[i] = OmT_inv.matvec(rhs);
        }

        for(int i = 0; i < NX; i++) {
            rho[i] = w_new[i][0];
            if(rho[i] < 1e-10) rho[i] = 1e-10;
            u[i]   = w_new[i][1] / rho[i];
            e[i]   = w_new[i][2] / rho[i];
            if(e[i] < 1e-10) e[i] = 1e-10;
            P[i]   = pressure(rho[i], e[i]);
        }

        //Граничные условия: нулевой градиент
        rho[0] = rho[1];   u[0] = u[1];   e[0] = e[1];   P[0] = P[1];
        rho[NX-1] = rho[NX-2]; u[NX-1] = u[NX-2]; e[NX-1] = e[NX-2]; P[NX-1] = P[NX-2];

        t += tau;
        step++;

        // Сохраняем
        if(step % save_every == 0 || t >= T_END) {
            save_snapshot();
            std::cout << "t = " << t << " s,  step = " << step
                      << ",  tau = " << tau << " s\n";
        }
    }

    fout.close();
    return 0;
}