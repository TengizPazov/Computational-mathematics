#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

double exact(double x, double t, double L, double a) {
    return sin(4.0 * M_PI * (x - a * t) / L);
}

void init(vector<double>& y, double L, double h) {
    for (int i = 0; i < y.size(); ++i) {
        double x = i * h;
        y[i] = sin(4.0 * M_PI * x / L);
    }
}

void step_upwind(vector<double>& y, double lambda) {
    int N = y.size();
    vector<double> yn = y;
    for (int i = 1; i < N; ++i)
        y[i] = yn[i] - lambda * (yn[i] - yn[i - 1]);
    y[0] = y[N - 1];
}

void step_lw(vector<double>& y, double lambda) {
    int N = y.size();
    vector<double> yn = y;

    for (int i = 1; i < N - 1; ++i) {
        y[i] = yn[i]
             - 0.5 * lambda * (yn[i + 1] - yn[i - 1])
             + 0.5 * lambda * lambda * (yn[i + 1] - 2*y[i] + yn[i - 1]);
    }

    y[0] = yn[0]
         - 0.5 * lambda * (yn[1] - yn[N - 1])
         + 0.5 * lambda * lambda * (yn[1] - 2*yn[0] + yn[N - 1]);

    y[N - 1] = yn[N - 1]
             - 0.5 * lambda * (yn[0] - yn[N - 2])
             + 0.5 * lambda * lambda * (yn[0] - 2*yn[N - 1] + yn[N - 2]);
}

void output_frame(const vector<double>& y, double h, double t) {
    cout << "FRAME t=" << fixed << setprecision(4) << t << "\n";
    for (int i = 0; i < y.size(); ++i)
        cout << i*h << " " << y[i] << "\n";
    cout << "END\n";
}

void run_scheme(const string& name,
                void (*step)(vector<double>&, double),
                double L, double T, double a,
                double h, double CFL)
{
    int NX = int(L / h) + 1;
    vector<double> y(NX);

    init(y, L, h);

    double tau = CFL * h / a;
    double lambda = a * tau / h;

    double t = 0.0;
    double dt_out = 0.05;

    double next_out = 0.0;

    cout << "SCHEME " << name << " CFL " << CFL << "\n";

    output_frame(y, h, t);

    int Nt = int(T / tau) + 5;

    for (int n = 0; n < Nt && t < T; ++n) {
        step(y, lambda);
        t += tau;

        if (t >= next_out - 1e-12) {
            output_frame(y, h, t);
            next_out += dt_out;
        }
    }

    cout << "DONE\n";
}

int main() {
    double L = 20, T = 18, a = 1, h = 0.5;

    vector<double> CFLs = {1.0, 0.6, 0.3};

    for (double CFL : CFLs) {
        run_scheme("upwind", step_upwind, L, T, a, h, CFL);
        run_scheme("laxwendroff", step_lw, L, T, a, h, CFL);
    }
}
