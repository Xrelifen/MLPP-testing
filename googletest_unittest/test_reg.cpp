// test_reg_combined.cpp

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include "Regularization/Reg.hpp"

using namespace MLPP;
static constexpr double EPS = 1e-9;

// Helper to compare two vectors element‑wise
namespace {
    void expectVectorNear(const std::vector<double>& actual,
                          const std::vector<double>& expected,
                          double tol = EPS) {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], tol)
                << "Mismatch at index " << i;
        }
    }

    // Manual computations for advanced tests
    double manualRidge(const std::vector<double>& w, double lambda) {
        double sum_sq = 0;
        for (auto v : w) sum_sq += v * v;
        return lambda * sum_sq / 2;
    }
    double manualLasso(const std::vector<double>& w, double lambda) {
        double sum_abs = 0;
        for (auto v : w) sum_abs += std::abs(v);
        return lambda * sum_abs;
    }
    double manualElastic(const std::vector<double>& w, double lambda, double alpha) {
        double total = 0;
        for (auto v : w) {
            total += alpha * std::abs(v) + (1 - alpha) * v * v / 2;
        }
        return lambda * total;
    }
    std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> r(a.size());
        for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
        return r;
    }
}

class RegTest : public ::testing::Test {
protected:
    Reg reg;
};

// --- Original basic tests ---
TEST_F(RegTest, RidgeTermVector) {
    std::vector<double> w{1.0, 2.0, -3.0};
    double lambda = 0.5;
    EXPECT_DOUBLE_EQ(reg.regTerm(w, lambda, 0.0, "Ridge"), 3.5);
}

TEST_F(RegTest, LassoTermVector) {
    std::vector<double> w{1.0, -2.0, 3.0};
    double lambda = 2.0;
    EXPECT_DOUBLE_EQ(reg.regTerm(w, lambda, 0.0, "Lasso"), 12.0);
}

TEST_F(RegTest, ElasticNetTermVector) {
    std::vector<double> w{1.0, -2.0};
    double lambda = 1.0, alpha = 0.5;
    // per-element: [0.75, 2.0], sum = 2.75
    EXPECT_DOUBLE_EQ(reg.regTerm(w, lambda, alpha, "ElasticNet"), 2.75);
}

TEST_F(RegTest, ZeroLambdaGivesZeroTerm) {
    std::vector<double> w{1.0, -2.0, 3.0};
    EXPECT_DOUBLE_EQ(reg.regTerm(w, 0.0, 0.7, "Ridge"), 0.0);
    EXPECT_DOUBLE_EQ(reg.regTerm(w, 0.0, 0.7, "Lasso"), 0.0);
    EXPECT_DOUBLE_EQ(reg.regTerm(w, 0.0, 0.7, "ElasticNet"), 0.0);
}

TEST_F(RegTest, RidgeDerivativeAndUpdate) {
    std::vector<double> w{1.0, -2.0, 0.5};
    double lambda = 0.1, alpha = 0.0;
    std::vector<double> deriv{0.1, -0.2, 0.05};
    expectVectorNear(reg.regDerivTerm(w, lambda, alpha, "Ridge"), deriv);
    std::vector<double> updated{0.9, -1.8, 0.45};
    expectVectorNear(reg.regWeights(w, lambda, alpha, "Ridge"), updated);
}

TEST_F(RegTest, EmptyWeights) {
    std::vector<double> w;
    EXPECT_DOUBLE_EQ(reg.regTerm(w, 1.0, 0.5, "Ridge"), 0.0);
    EXPECT_TRUE(reg.regDerivTerm(w, 1.0, 0.5, "Lasso").empty());
    EXPECT_TRUE(reg.regWeights(w, 1.0, 0.5, "ElasticNet").empty());
}

TEST_F(RegTest, AlphaZeroAndOneForElasticNet) {
    std::vector<double> w{3.0, -4.0};
    double lambda = 2.0;
    EXPECT_DOUBLE_EQ(reg.regTerm(w, lambda, 0.0, "ElasticNet"),
                     reg.regTerm(w, lambda, 0.0, "Ridge"));
    EXPECT_DOUBLE_EQ(reg.regTerm(w, lambda, 1.0, "ElasticNet"),
                     reg.regTerm(w, lambda, 1.0, "Lasso"));
}

TEST_F(RegTest, WeightClippingBehavior) {
    std::vector<double> w{-5.0, 0.0, 10.0};
    double lower = -1.0, upper = 2.0;
    std::vector<double> clipped{-1.0, 0.0, 2.0};
    expectVectorNear(reg.regWeights(w, lower, upper, "WeightClipping"), clipped);
}

TEST_F(RegTest, RidgeTermMatrix) {
    std::vector<std::vector<double>> W{{1,2},{-3,4}};
    double lambda = 0.5;
    EXPECT_DOUBLE_EQ(reg.regTerm(W, lambda, 0.0, "Ridge"), 7.5);
}

// --- Advanced randomized and edge‑case tests ---
class RegAdvancedTest : public ::testing::Test {
protected:
    Reg reg;
};

TEST_F(RegAdvancedTest, RandomizedTermsMatchManual) {
    std::mt19937_64 rnd(42);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int iter = 0; iter < 100; ++iter) {
        int len = 1 + (rnd() % 10);
        std::vector<double> w(len);
        for (double &v : w) v = dist(rnd);
        double lambda = std::abs(dist(rnd));
        double alpha = std::fmod(std::abs(dist(rnd)), 1.0001);
        EXPECT_NEAR(reg.regTerm(w, lambda, alpha, "Ridge"), manualRidge(w, lambda), EPS);
        EXPECT_NEAR(reg.regTerm(w, lambda, alpha, "Lasso"), manualLasso(w, lambda), EPS);
        EXPECT_NEAR(reg.regTerm(w, lambda, alpha, "ElasticNet"), manualElastic(w, lambda, alpha), EPS);
    }
}

TEST_F(RegAdvancedTest, WeightsUpdateConsistency) {
    std::vector<double> w{2.3, -1.7, 0.0, 5.5};
    double lambda = 0.8, alpha = 0.3;
    auto deriv = reg.regDerivTerm(w, lambda, alpha, "ElasticNet");
    auto updated = reg.regWeights(w, lambda, alpha, "ElasticNet");
    auto manualUpd = subtract(w, deriv);
    ASSERT_EQ(updated.size(), manualUpd.size());
    for (size_t i = 0; i < updated.size(); ++i) {
        EXPECT_NEAR(updated[i], manualUpd[i], EPS) << "Mismatch at index " << i;
    }
}

TEST_F(RegAdvancedTest, LassoDerivativeAtZero) {
    std::vector<double> w{0.0, 1.0, -1.0};
    double lambda = 1.5;
    auto deriv = reg.regDerivTerm(w, lambda, 0.0, "Lasso");
    EXPECT_DOUBLE_EQ(deriv[0], 0.0);
    EXPECT_DOUBLE_EQ(deriv[1], lambda);
    EXPECT_DOUBLE_EQ(deriv[2], -lambda);
}

TEST_F(RegAdvancedTest, UnknownRegType) {
    std::vector<double> w{1.0, 2.0};
    double lambda = 2.0, alpha = 0.5;
    EXPECT_DOUBLE_EQ(reg.regTerm(w, lambda, alpha, "Unknown"), 0.0);
    auto deriv = reg.regDerivTerm(w, lambda, alpha, "Unknown");
    EXPECT_TRUE(deriv.empty() || std::all_of(deriv.begin(), deriv.end(), [](double v){ return v==0.0;}));
    auto upd = reg.regWeights(w, lambda, alpha, "Unknown");
    EXPECT_EQ(upd, w);
}

TEST_F(RegAdvancedTest, MatrixElasticNetMatchesFlatten) {
    std::vector<std::vector<double>> W{{1.0, -2.0}, {3.0, -4.0}};
    double lambda = 0.7, alpha = 0.4;
    double gotM = reg.regTerm(W, lambda, alpha, "ElasticNet");
    std::vector<double> flat;
    for (auto &row : W)
        for (auto v : row)
            flat.push_back(v);
    double gotV = reg.regTerm(flat, lambda, alpha, "ElasticNet");
    EXPECT_NEAR(gotM, gotV, EPS);
}

TEST_F(RegAdvancedTest, WeightClippingBounds) {
    std::vector<double> w{-10.0, 0.0, 5.0};
    double low = -2.0, high = 3.0;
    auto deriv = reg.regDerivTerm(w, low, high, "WeightClipping");
    EXPECT_DOUBLE_EQ(deriv[0], low);
    EXPECT_DOUBLE_EQ(deriv[1], 0.0);
    EXPECT_DOUBLE_EQ(deriv[2], high);
    auto clipped = reg.regWeights(w, low, high, "WeightClipping");
    EXPECT_EQ(clipped, deriv);
}