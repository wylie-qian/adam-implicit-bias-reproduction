#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Full direct Adam recurrence in native long double; no moment or loss scaling.
// Inputs are parsed as double first, then cast, preserving the frozen binary64 data.
int main(int argc, char** argv) {
    if (argc != 6) throw std::runtime_error("usage: replay INPUT OUTPUT A EPS STEPS");
    const long double a = std::stold(argv[3]);
    const long double eps = std::stold(argv[4]);
    const int steps = std::stoi(argv[5]);
    std::ifstream in(argv[1]);
    if (!in) throw std::runtime_error("missing input");
    int n, d, ck_count; in >> n >> d >> ck_count;
    if(n <= 0 || d <= 0 || ck_count <= 0) throw std::runtime_error("invalid input shape");
    std::vector<int> checkpoints(ck_count);
    for(auto& x : checkpoints) in >> x;
    std::vector<long double> z(n*d), w(d), m(d,0), v(d,0), h(n), g(d);
    for(auto& x : z) { double q; in >> q; x = static_cast<long double>(q); }
    for(auto& x : w) { double q; in >> q; x = static_cast<long double>(q); }
    if(!in) throw std::runtime_error("invalid input contents");
    std::ofstream out(argv[2]);
    if(!out) throw std::runtime_error("output unavailable");
    out << std::setprecision(LDBL_DECIMAL_DIG) << std::scientific;
    out << "step";
    for (auto label : {"w","m","v"}) for(int j=0;j<d;j++) out << ',' << label << j;
    out << '\n';
    auto record = [&](int t) {
        out << t;
        for(auto const* arr : {&w,&m,&v}) for(auto x : *arr) out << ',' << x;
        out << '\n'; out.flush();
    };
    size_t next = 0;
    if(checkpoints[next]==0) {record(0); ++next;}
    unsigned long long zero_h_count=0, zero_gradient_square_count=0;
    int first_zero_h_step=0, first_zero_gradient_square_step=0;
    long double max_margin=-LDBL_MAX, min_nonzero_h=LDBL_MAX, min_positive_v=LDBL_MAX;
    const auto begin = std::chrono::steady_clock::now();
    for(int t=1; t<=steps; ++t) {
        for(int i=0; i<n; ++i) {
            long double margin = 0;
            for(int j=0;j<d;++j) margin += z[i*d+j]*w[j];
            max_margin=std::max(max_margin,margin);
            if(margin>=0) { auto e=std::exp(-margin); h[i]=e/(1+e); }
            else h[i]=1/(1+std::exp(margin));
            if(h[i]==0) { ++zero_h_count; if(!first_zero_h_step) first_zero_h_step=t; }
            else min_nonzero_h=std::min(min_nonzero_h,h[i]);
        }
        std::fill(g.begin(),g.end(),0.L);
        for(int i=0;i<n;++i) for(int j=0;j<d;++j) g[j] -= z[i*d+j]*h[i]/n;
        const long double eta = std::pow(static_cast<long double>(t),-a);
        for(int j=0;j<d;++j) {
            m[j]=0.9L*m[j]+0.1L*g[j];
            const long double squared_gradient=g[j]*g[j];
            if(g[j]!=0 && squared_gradient==0) {
                ++zero_gradient_square_count;
                if(!first_zero_gradient_square_step) first_zero_gradient_square_step=t;
            }
            v[j]=0.99L*v[j]+0.01L*squared_gradient;
            min_positive_v=std::min(min_positive_v,v[j]);
            if(!(v[j]>0) || !std::isfinite(v[j]) || !std::isfinite(m[j]))
                throw std::runtime_error("invalid moment at step "+std::to_string(t));
            w[j] -= eta*m[j]/(std::sqrt(v[j])+eps);
            if(!std::isfinite(w[j])) throw std::runtime_error("invalid weight");
        }
        if(next<checkpoints.size() && t==checkpoints[next]) {record(t); ++next;}
    }
    if(next!=checkpoints.size()) throw std::runtime_error("checkpoint mismatch");
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout << std::setprecision(LDBL_DECIMAL_DIG) << "{\"completed_steps\":" << steps << ",\"a\":" << a << ",\"epsilon\":" << eps
              << ",\"LDBL_MANT_DIG\":" << LDBL_MANT_DIG << ",\"LDBL_MAX_EXP\":" << LDBL_MAX_EXP
              << ",\"zero_logistic_weight_count\":" << zero_h_count
              << ",\"first_zero_logistic_weight_step\":" << first_zero_h_step
              << ",\"zero_nonzero_gradient_square_count\":" << zero_gradient_square_count
              << ",\"first_zero_nonzero_gradient_square_step\":" << first_zero_gradient_square_step
              << ",\"max_margin_evaluated\":" << max_margin
              << ",\"min_nonzero_logistic_weight_log10\":" << std::log10(min_nonzero_h)
              << ",\"min_positive_second_moment_log10\":" << std::log10(min_positive_v)
              << ",\"long_double_min_subnormal_log10\":" << std::log10(std::numeric_limits<long double>::denorm_min())
              << ",\"checkpoints\":" << next << ",\"wall_seconds\":" << elapsed << "}\n";
}
