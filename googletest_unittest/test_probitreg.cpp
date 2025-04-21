// test_probit_reg.cpp
// Revised GoogleTest suite – compiles and tests ProbitReg using only its public API.
// Build example (adjust include/src paths as needed):
// g++ -std=c++17 -I/path/to/gtest/include -pthread \
//     test_probit_reg.cpp -lgtest -lgtest_main -o test_probit_reg

#include <gtest/gtest.h>
#include <random>
#include <cmath>

#include "ProbitReg/ProbitReg.hpp" 
#include "Regularization/Reg.hpp"

using namespace MLPP;
static constexpr double EPS = 1e-8;

/* ------------------------------------------------------------
   Helper utilities for testing
   ------------------------------------------------------------*/
// Tiny toy data: 2D, linearly separable: y = 1 if x0 + x1 > 0
static void makeTinyDataset(std::vector<std::vector<double>>& X,
                            std::vector<double>& y) {
    X = {{ 2,  1},
         {-1, -3},
         { 4,  2},
         {-3, -2}};
    y = {1, 0, 1, 0};
}

// Simple MSE helper (avoiding private Cost)
static double mse(const std::vector<double>& pred,
                  const std::vector<double>& truth) {
    double s = 0.0;
    for (size_t i = 0; i < pred.size(); ++i)
        s += (pred[i] - truth[i]) * (pred[i] - truth[i]);
    return s / pred.size();
}

/* ============================================================
   BASIC TESTS
   ============================================================*/

// 1D and batch inference consistency
TEST(ProbitRegBasic, SingleAndBatchConsistency) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);
    ProbitReg model(X, y, "None", 0.0, 0.0);

    auto batch_preds = model.modelSetTest(X);
    for (size_t i = 0; i < X.size(); ++i) {
        double single = model.modelTest(X[i]);
        EXPECT_NEAR(batch_preds[i], single, EPS);
    }
}

// Predictions shape and range
TEST(ProbitRegBasic, PredictionsShapeAndRange) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);

    ProbitReg model(X, y, "None", 0.0, 0.0);
    auto preds = model.modelSetTest(X);

    ASSERT_EQ(preds.size(), X.size());
    for (double p : preds) {
        EXPECT_GT(p, 0.0);
        EXPECT_LT(p, 1.0);
    }
}

/* ============================================================
   ADVANCED / EDGE-CASE TESTS
   ============================================================*/

// Training (GD) should improve or maintain the score
TEST(ProbitRegAdvanced, TrainingImprovesScore) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);

    ProbitReg model(X, y, "None", 0.0, 0.0);
    double b = model.score();
    model.gradientDescent(0.2, 20, false);
    double a = model.score();

    EXPECT_GE(a + 1e-4, b);
    EXPECT_GT(a, 0.5);
}

// SGD on zero inputs must not produce NaN
TEST(ProbitRegAdvanced, SGDZeroInputsStaysFinite) {
    std::vector<std::vector<double>> X(10, std::vector<double>(3, 0.0));
    std::vector<double> y(10, 0.0);
    ProbitReg m(X, y, "Ridge", 0.1, 0.0);

    EXPECT_NO_THROW(m.SGD(0.05, 30, false));
    auto p = m.modelSetTest(X);
    for (double v : p) EXPECT_TRUE(std::isfinite(v));
    EXPECT_TRUE(std::isfinite(m.score()));
}

// Mini-batch with remainder should not hang or crash
TEST(ProbitRegAdvanced, MBGDHandlesRemainderBatch) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);
    ProbitReg m(X, y, "None", 0.0, 0.0);
    EXPECT_NO_THROW(m.MBGD(0.1, 3, 3, false));
}

// MLE update step reduces MSE on toy data
TEST(ProbitRegAdvanced, MLEStepLowersMSE) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);
    ProbitReg m(X, y, "None", 0.0, 0.0);

    auto before_preds = m.modelSetTest(X);
    double before = mse(before_preds, y);
    m.MLE(0.2, 1, false);
    auto after_preds = m.modelSetTest(X);
    double after = mse(after_preds, y);

    EXPECT_LT(after, before + 1e-6);
}

// Regularization helper sanity for Ridge and Lasso on vector
TEST(RegHelper, RidgeAndLassoBasic) {
    Reg r;
    std::vector<double> w = {1.0, -2.0, 3.0};
    EXPECT_NEAR(r.regTerm(w, 0.5, 0.0, "Ridge"), 0.5*(1+4+9)/2, EPS);
    EXPECT_NEAR(r.regTerm(w, 0.5, 0.0, "Lasso"), 0.5*(1+2+3), EPS);
}
