#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
using namespace std;
//параметры
const double L       = 500.0;
const double B       = 50.0;
const double dz      = 10.0;
const double k       = 1e-14;
const double mu      = 1e-3;
const double phi     = 0.2;
const double cf      = 1e-4 / 101325.;
const double ro0     = 1000.0;
const double p_init  = 100.0 * 101325.0;
const double p0_ref  = 100.0 * 101325.0;
const double p_inj   = 150.0 * 101325.0;
const double p_prod  =  50.0 * 101325.0;
const double T_max   = 10.0 * 86400.0;
const int    NX      = 100;
const double tau     = 3600.0;  

// Функция плотности
double rho(double p_val) {
    return ro0 * (1.0 + cf * (p_val - p0_ref));
}

// Метод прогонки для трёхдиагональной системы
vector<double> thomas_algorithm(const vector<double>& a_in,
                                const vector<double>& b,
                                const vector<double>& c,
                                const vector<double>& d_in)
{
    int N = (int)a_in.size();
    vector<double> a = a_in;
    vector<double> d = d_in;
    vector<double> x(N, 0.0);

    // Прямой ход
    for (int i = 1; i < N; ++i) {
        double m = c[i] / a[i - 1];
        a[i] -= m * b[i - 1];
        d[i] -= m * d[i - 1];
    }

    // Обратный ход
    x[N - 1] = d[N - 1] / a[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        x[i] = (d[i] - b[i] * x[i + 1]) / a[i];
    }

    return x;
}

int main() {
    double h = L / (NX - 1);

    vector<double> p(NX, p_init);
    double t = 0.0;

    vector<double> save_times = {
        0.1 * 86400.0,
        0.25 * 86400.0,
        0.5 * 86400.0,
        1.0 * 86400.0,
        1.5 * 86400.0,
        2.0 * 86400.0,
        3.0 * 86400.0,
        5.0 * 86400.0,
        7.0 * 86400.0,
        10.0 * 86400.0
    };

    int frame_id = 0;

    while (t < T_max && frame_id < save_times.size()) {
        vector<double> a(NX, 0.0);
        vector<double> b(NX, 0.0);
        vector<double> c(NX, 0.0);
        vector<double> d(NX, 0.0);

        // Внутренние узлы
        for (int i = 1; i < NX - 1; ++i) {
            double rho_right = (p[i] >= p[i+1]) ? rho(p[i]) : rho(p[i+1]);
            double rho_left  = (p[i-1] >= p[i]) ? rho(p[i-1]) : rho(p[i]);

            c[i] = k * rho_left  / (mu * h * h);
            b[i] = k * rho_right / (mu * h * h);
            a[i] = -c[i] - b[i] - phi * cf * ro0 / tau;
            d[i] = -(phi * cf * ro0 / tau) * p[i];
        }

        // Границы
        a[0] = 1.0;  d[0] = p_inj;
        a[NX-1] = 1.0; d[NX-1] = p_prod;

        p = thomas_algorithm(a, b, c, d);

        t += tau;

        //пересечение нужного моменты времени
        if (t >= save_times[frame_id]) {
            ostringstream fname;
            fname << "pressure_" << setw(2) << setfill('0') << frame_id << ".dat";
            ofstream fout(fname.str());

            fout << "# t = " << t << " s, " << t/86400.0 << " days\n";
            fout << "# x[m]\tP[Pa]\tP[atm]\n";

            for (int i = 0; i < NX; ++i) {
                double x = i * h;
                double P_atm = p[i] / 101325.0;
                fout << x << "\t" << p[i] << "\t" << P_atm << "\n";
            }

            fout.close();
            cout << "Saved frame " << frame_id << " at t=" << t/86400.0 << " days\n";

            frame_id++;
        }
    }

    cout << "Total frames saved: " << frame_id << endl;
    return 0;
}
