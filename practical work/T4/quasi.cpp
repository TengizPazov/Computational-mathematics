#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

const double E = exp(1.0);
const double EPS = 1e-12; 

vector<double> thomas_algorithm(const vector<double>& a, const vector<double>& b, 
                                 const vector<double>& c, const vector<double>& d) {
    int n = d.size();
    vector<double> alpha(n), beta(n), x(n);
    
    alpha[0] = -c[0] / b[0];
    beta[0] = d[0] / b[0];
    
    for (int i = 1; i < n - 1; i++) {
        double denom = b[i] + a[i] * alpha[i-1];
        alpha[i] = -c[i] / denom;
        beta[i] = (d[i] - a[i] * beta[i-1]) / denom;
    }
    
    x[n-1] = (d[n-1] - a[n-1] * beta[n-2]) / (b[n-1] + a[n-1] * alpha[n-2]);
    for (int i = n - 2; i >= 0; i--) {
        x[i] = alpha[i] * x[i+1] + beta[i];
    }
    return x;
}

inline double safe_exp(double x, double max_val = 50.0) {
    return exp(max(min(x, max_val), -max_val));
}

inline double clip(double x, double lo, double hi) {
    return max(lo, min(x, hi));
}

double f_original(double x, double y, double yp) {
    double log_x = abs(log(x)) < EPS ? EPS : log(x);
    double exp_term = safe_exp(yp);
    double under_sqrt = 1.0/(x*x) + E*y*y/log_x - y*exp_term;
    return sqrt(max(under_sqrt, EPS));
}

double f_y(double x, double y, double yp) {
    double log_x = abs(log(x)) < EPS ? EPS : log(x);
    double exp_term = safe_exp(yp);
    double under_sqrt = 1.0/(x*x) + E*y*y/log_x - y*exp_term;
    double denom = 2.0 * sqrt(max(under_sqrt, EPS));
    double numer = 2.0*E*y/log_x - exp_term;
    return clip(numer/denom, -1e10, 1e10);
}

double f_yp(double x, double y, double yp) {
    double log_x = abs(log(x)) < EPS ? EPS : log(x);
    double exp_term = safe_exp(yp);
    double under_sqrt = 1.0/(x*x) + E*y*y/log_x - y*exp_term;
    double denom = 2.0 * sqrt(max(under_sqrt, EPS));
    double numer = -y * exp_term;
    return clip(numer/denom, -1e10, 1e10);
}

double g_n(double x, double y, double yp) {
    return f_original(x,y,yp) - f_y(x,y,yp)*y - f_yp(x,y,yp)*yp;
}

vector<double> gradient(const vector<double>& y, double h) {
    int n = y.size();
    vector<double> g(n);
    g[0] = (y[1] - y[0]) / h;
    for (int i = 1; i < n-1; i++)
        g[i] = (y[i+1] - y[i-1]) / (2*h);
    g[n-1] = (y[n-1] - y[n-2]) / h;
    return g;
}

pair<vector<double>, vector<double>> quasilinear_solve(int N = 200, double tol = 1e-6, int max_iter = 10000) {
    double a_val = E, b_val = E*E;
    double y_a = E, y_b = 2*E*E;
    
    vector<double> x(N), y(N);
    double h = (b_val - a_val) / (N - 1);
    
    for (int i = 0; i < N; i++) {
        x[i] = a_val + i * h;
        y[i] = x[i] * log(x[i]);
    }
    
    for (int it = 0; it < max_iter; it++) {
        vector<double> yp = gradient(y, h);
        
        vector<double> A(N), B(N), C(N), D(N);
        A[0] = 0; B[0] = 1; C[0] = 0; D[0] = y_a;
        A[N-1] = 0; B[N-1] = 1; C[N-1] = 0; D[N-1] = y_b;
        
        for (int i = 1; i < N-1; i++) {
            double p = -f_yp(x[i], y[i], yp[i]);
            double q = -f_y(x[i], y[i], yp[i]);
            double r = g_n(x[i], y[i], yp[i]);
            
            A[i] = 1.0/(h*h) - p/(2*h);
            B[i] = -2.0/(h*h) + q;
            C[i] = 1.0/(h*h) + p/(2*h);
            D[i] = r;
        }
        
        vector<double> y_new = thomas_algorithm(A, B, C, D);
        
        double diff = 0;
        for (int i = 0; i < N; i++)
            diff = max(diff, abs(y_new[i] - y[i]));
        
        cout << "Iter " << it+1 << ": " << diff << endl;
        
        y = y_new;
        if (diff < tol) {
            cout << "Converged" << endl;
            break;
        }
    }
    return {x, y};
}

int main() {
    auto [x, y] = quasilinear_solve(200, 1e-10);
    return 0;
}