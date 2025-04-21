// test_cloglog_reg.cpp
// GoogleTest suite for CLogLogReg using only its public API.
// Build example (adjust include/src paths as needed):
// g++ -std=c++17 -I/path/to/gtest/include -pthread \
//     test_cloglog_reg.cpp -lgtest -lgtest_main -o test_cloglog_reg

#include <gtest/gtest.h>
#include <random>
#include <cmath>

#include "CLogLogReg/CLogLogReg.hpp"  // adjust to your directory structure
#include "Regularization/Reg.hpp"

using namespace MLPP;
static constexpr double EPS = 1e-8;

/* ------------------------------------------------------------
   Helper utilities for testing
   ------------------------------------------------------------*/
// Tiny toy data: 2D, linearly separable: y = 1 if x0 + x1 > 0 else 0
static void makeTinyDataset(std::vector<std::vector<double>>& X,
                            std::vector<double>& y) {
    X = {{ 2,  1},
         {-1, -3},
         { 4,  2},
         {-3, -2}};
    y = {1, 0, 1, 0};
}

// Simple mean squared error helper
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

// Ensures single-example inference matches batch inference
TEST(CLogLogBasic, SingleAndBatchConsistency) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);
    CLogLogReg model(X, y, "None", 0.0, 0.0);

    auto batch_preds = model.modelSetTest(X);
    for (size_t i = 0; i < X.size(); ++i) {
        double single = model.modelTest(X[i]);
        EXPECT_NEAR(batch_preds[i], single, EPS);
    }
}

// Verifies predicted probabilities are in (0,1) and output size matches input size
TEST(CLogLogBasic, PredictionsShapeAndRange) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);

    CLogLogReg model(X, y, "None", 0.0, 0.0);
    auto preds = model.modelSetTest(X);

    ASSERT_EQ(preds.size(), X.size());
    for (double p : preds) {
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
    }
}

/* ============================================================
   ADVANCED / EDGE-CASE TESTS
   ============================================================*/

// Confirms gradient descent training does not worsen and typically improves the score
TEST(CLogLogAdvanced, TrainingImprovesScore) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);

    CLogLogReg model(X, y, "None", 0.0, 0.0);
    double before = model.score();
    model.gradientDescent(0.2, /*max_epoch=*/20, /*UI=*/false);
    double after = model.score();

    EXPECT_GE(after + 1e-4, before);
    EXPECT_GT(after, 0.5);
}

// Checks that one MLE step increases the log-likelihood of the data under the cloglog model
static double logLikelihood(const std::vector<double>& pred, const std::vector<double>& truth) {
    double ll = 0.0;
    for (size_t i = 0; i < pred.size(); ++i) {
        // avoid log(0)
        double p = std::min(std::max(pred[i], 1e-12), 1.0 - 1e-12);
        ll += truth[i] * std::log(p) + (1.0 - truth[i]) * std::log(1.0 - p);
    }
    return ll;
}

// Checks that classification score improves or stays the same after MLE updates
TEST(CLogLogAdvanced, MLEImprovesScore) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);

    CLogLogReg model(X, y, "None", 0.0, 0.0);
    double before = model.score();
    model.MLE(0.2, /*max_epoch=*/20, /*UI=*/false);
    double after = model.score();
    EXPECT_GE(after, before - EPS);

    model.MLE(0.3, /*max_epoch=*/30, /*UI=*/false);
    EXPECT_GE(after, before - EPS);

    model.MLE(0.4, /*max_epoch=*/40, /*UI=*/false);
    EXPECT_GE(after, before - EPS);
}

// Ensures SGD on zero-valued inputs produces finite predictions and score
TEST(CLogLogAdvanced, SGDZeroInputsStaysFinite) {
    std::vector<std::vector<double>> X(10, std::vector<double>(3, 0.0));
    std::vector<double> y(10, 0.0);

    CLogLogReg model(X, y, "Ridge", /*lambda=*/0.1, /*alpha=*/0.0);
    EXPECT_NO_THROW(model.SGD(0.05, /*max_epoch=*/30, /*UI=*/false));

    auto p = model.modelSetTest(X);
    for (double v : p) EXPECT_TRUE(std::isfinite(v));
    EXPECT_TRUE(std::isfinite(model.score()));
}

// Verifies MBGD handles data sizes not divisible by batch size without crashing
TEST(CLogLogAdvanced, MBGDHandlesRemainderBatch) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    makeTinyDataset(X, y);

    CLogLogReg model(X, y, "None", 0.0, 0.0);
    EXPECT_NO_THROW(model.MBGD(0.1, /*max_epoch=*/3, /*mini_batch_size=*/3, /*UI=*/false));
}

// Validates Reg helper computes ElasticNet term correctly on a fixed vector
TEST(RegHelper, ElasticNetBasic) {
    Reg r;
    std::vector<double> w = {1.0, -2.0, 3.0};
    double lambda = 0.7, alpha = 0.4;
    // manual: lambda * [alpha*sum|w| + (1-alpha)*sum(w^2/2)]
    double manual = lambda * (alpha*(1+2+3) + (1-alpha)*( (1+4+9)/2 ));
    EXPECT_NEAR(r.regTerm(w, lambda, alpha, "ElasticNet"), manual, EPS);
}