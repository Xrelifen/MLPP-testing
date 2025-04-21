// test_unilinreg.cpp
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <numeric>
#include <cmath>
#include "UniLinReg/UniLinReg.hpp"

using namespace MLPP;

static constexpr double EPS       = 1e-9;  // exact fits
static constexpr double NOISY_EPS = 1e-2;  // noisy / ill‑conditioned

//-----------------------------------------------------------
// BASIC TESTS
//-----------------------------------------------------------

// 1) Perfect line  y = 1 + 2x
TEST(UniLinRegBasic, ExactLineFit)
{
    std::vector<double> x{0, 1, 2, 3, 4};
    std::vector<double> y{1, 3, 5, 7, 9};
    UniLinReg mdl(x, y);

    EXPECT_NEAR(mdl.modelTest(0), 1.0, EPS);
    EXPECT_NEAR(mdl.modelTest(4), 9.0, EPS);
}

// 2) Constant output  y = 3 (slope should be 0)
TEST(UniLinRegBasic, ConstantOutput)
{
    std::vector<double> x{-5, -2, 0, 1, 4};
    std::vector<double> y(x.size(), 3.0);
    UniLinReg mdl(x, y);

    EXPECT_NEAR(mdl.modelTest(10), 3.0, EPS);
}

// 3) Single sample → model degenerates to y0
TEST(UniLinRegBasic, SinglePoint)
{
    std::vector<double> x{7};
    std::vector<double> y{-2};
    UniLinReg mdl(x, y);

    EXPECT_NEAR(mdl.modelTest(123.45), -2.0, EPS);
}

//-----------------------------------------------------------
// ADVANCED / EDGE‑CASE TESTS
//-----------------------------------------------------------

// 4) All x identical → slope must collapse to 0
TEST(UniLinRegAdvanced, IdenticalInputs)
{
    std::vector<double> x(10, 2.0);               // all inputs equal
    std::vector<double> y{1,2,3,4,5,6,7,8,9,10};  // arbitrary outputs
    UniLinReg mdl(x, y);

    double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    EXPECT_NEAR(mdl.modelTest(2.0),  mean_y, EPS);
    EXPECT_NEAR(mdl.modelTest(99.0), mean_y, EPS);
}

// 5) Random line + Gaussian noise → coefficients should be close
TEST(UniLinRegAdvanced, NoisyRandomLine)
{
    constexpr int N = 200;
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> coef_dist(-10.0, 10.0);
    std::normal_distribution<double> noise(0.0, 0.05);

    double b0_true = coef_dist(rng);
    double b1_true = coef_dist(rng);

    std::vector<double> x, y;
    x.reserve(N); y.reserve(N);
    for (int i = 0; i < N; ++i) {
        double xi = coef_dist(rng);
        x.push_back(xi);
        y.push_back(b0_true + b1_true * xi + noise(rng));
    }

    UniLinReg mdl(x, y);

    double b0_hat = mdl.modelTest(0.0);
    double b1_hat = mdl.modelTest(1.0) - b0_hat;

    EXPECT_NEAR(b0_hat, b0_true, NOISY_EPS);
    EXPECT_NEAR(b1_hat, b1_true, NOISY_EPS);
}

// 6) Very large magnitudes → check overflow / precision
TEST(UniLinRegAdvanced, LargeScaleValues)
{
    std::vector<double> x{1e8, 2e8, 3e8};
    std::vector<double> y{5e8, 7e8, 9e8};   // y = 1e8 + 2x
    UniLinReg mdl(x, y);

    EXPECT_NEAR(mdl.modelTest(0),    3e8, 1e2);
    EXPECT_NEAR(mdl.modelTest(4e8),  11e8, 1e2);
}
