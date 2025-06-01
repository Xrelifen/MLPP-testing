//
//  UniLinReg.cpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "UniLinReg.hpp"
#include "LinAlg/LinAlg.hpp"
#include "Stat/Stat.hpp"
#include <iostream>

// revise
#include <numeric>  

// General Multivariate Linear Regression Model
// ŷ = b0 + b1x1 + b2x2 + ... + bkxk


// Univariate Linear Regression Model
// ŷ = b0 + b1x1

namespace MLPP{
    UniLinReg::UniLinReg(std::vector<double> x, std::vector<double> y)
    : inputSet(x), outputSet(y)
    {
        /*Stat estimator;
        b1 = estimator.b1Estimation(inputSet, outputSet);
        b0 = estimator.b0Estimation(inputSet, outputSet);
        */
        // revise
        if (x.size() != y.size() || x.empty()) {
            throw std::invalid_argument("UniLinReg: x and y must be the same non‑zero length");
        }
    
        // -------- robust single‑pass mean calculation --------
        const std::size_t n = x.size();
        const long double mean_x = std::accumulate(x.begin(), x.end(), 0.0L) / static_cast<long double>(n);
        const long double mean_y = std::accumulate(y.begin(), y.end(), 0.0L) / static_cast<long double>(n);
    
        long double num = 0.0L;   // Σ (x‑μx)(y‑μy)
        long double den = 0.0L;   // Σ (x‑μx)^2
        for (std::size_t i = 0; i < n; ++i) {
            const long double dx = static_cast<long double>(x[i]) - mean_x;
            const long double dy = static_cast<long double>(y[i]) - mean_y;
            num += dx * dy;
            den += dx * dx;
        }
    
        if (std::abs(den) < 1e-20L) {
            // Degenerate: all x identical (or single sample). (as sklearn)
            // or assert an error (as statsmodels OLS)
            b1 = 0.0;
            b0 = static_cast<double>(mean_y);
        } else {
            b1 = static_cast<double>(num / den);
            b0 = static_cast<double>(mean_y - static_cast<long double>(b1) * mean_x);
        }
    }

    std::vector<double> UniLinReg::modelSetTest(std::vector<double> x){
        LinAlg alg;
        return alg.scalarAdd(b0, alg.scalarMultiply(b1, x));
    }

    double UniLinReg::modelTest(double input){
        return b0 + b1 * input;
    }
}
