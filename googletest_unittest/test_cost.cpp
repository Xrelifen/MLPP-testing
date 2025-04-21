#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "Cost/Cost.hpp"

using namespace MLPP;
static constexpr double EPS = 1e-7;

// Test fixture providing Cost instance
class CostTest : public ::testing::Test {
protected:
    Cost cost;
};

// --- Basic tests ---
TEST_F(CostTest, MSEVector) {
    // y_hat = [1, 3], y = [1, 2] -> errors [0,1], sum sq = 1
    // MSE = 1/(2*N) = 1/4 = 0.25
    std::vector<double> yhat{1.0, 3.0}, y{1.0, 2.0};
    EXPECT_NEAR(cost.MSE(yhat, y), 0.25, EPS);
}

TEST_F(CostTest, RMSEVector) {
    // RMSE = sqrt(sum sq / N) = sqrt(1/2) ≈ 0.7071
    std::vector<double> yhat{1.0, 3.0}, y{1.0, 2.0};
    EXPECT_NEAR(cost.RMSE(yhat, y), std::sqrt(0.5), EPS);
}

TEST_F(CostTest, MAEVector) {
    // MAE = (|0|+|1|)/2 = 0.5
    std::vector<double> yhat{1.0, 3.0}, y{1.0, 2.0};
    EXPECT_NEAR(cost.MAE(yhat, y), 0.5, EPS);
}

TEST_F(CostTest, MBEVector) {
    // MBE = (0 + 1)/2 = 0.5
    std::vector<double> yhat{1.0, 3.0}, y{1.0, 2.0};
    EXPECT_NEAR(cost.MBE(yhat, y), 0.5, EPS);
}

TEST_F(CostTest, LogLossVector) {
    // Single observation: y=1, yhat=0.8 => -[1*ln(0.8)+(0)*…]/1 ≈ 0.2231
    std::vector<double> yhat{0.8}, y{1.0};
    EXPECT_NEAR(cost.LogLoss(yhat, y), -std::log(0.8), EPS);
}

TEST_F(CostTest, CrossEntropyVector) {
    // CE = -sum y*log(yhat) = -(1*ln(0.8)) = 0.2231
    std::vector<double> yhat{0.8}, y{1.0};
    EXPECT_NEAR(cost.CrossEntropy(yhat, y), -std::log(0.8), EPS);
}

TEST_F(CostTest, HingeLossVector) {
    // y=[+1,-1], yhat=[0.5, -0.1]
    // losses = [max(0,1-0.5), max(0,1-(-1*-0.1))] = [0.5, 0.9] avg = 0.7
    std::vector<double> yhat{0.5, -0.1}, y{1.0, -1.0};
    EXPECT_NEAR(cost.HingeLoss(yhat, y), 0.7, EPS);
}

TEST_F(CostTest, HuberLossVector) {
    // delta=1. For error e <=1, sum e^2; for e>1, sum 2δ|e|-δ^2
    // y=[1,4], yhat=[1,1]; errors [0,3]
    // loss = 0^2 + (2*1*3 - 1^2) = 5
    std::vector<double> yhat{1.0, 1.0}, y{1.0, 4.0};
    EXPECT_NEAR(cost.HuberLoss(yhat, y, 1.0), 5.0, EPS);
}

TEST_F(CostTest, WassersteinLossVector) {
    // y=[1,-1], yhat=[2,3] => sum yhat*y = 2*1 + 3*(-1) = -1 => loss = -(-1)/2 = 0.5
    std::vector<double> yhat{2.0,3.0}, y{1.0,-1.0};
    EXPECT_NEAR(cost.WassersteinLoss(yhat, y), 0.5, EPS);
}

// --- Advanced tests ---
namespace {
    double manualMSE(const std::vector<double>& yhat, const std::vector<double>& y) {
        double s = 0;
        for (size_t i = 0; i < yhat.size(); ++i)
            s += (yhat[i] - y[i])*(yhat[i] - y[i]);
        return s/(2.0*yhat.size());
    }
}

TEST_F(CostTest, MSEMatrix) {
    // catches divide-by ordering: 2×1 matrix same as vector size 1
    std::vector<std::vector<double>> yhat{{1.0},{3.0}}, y{{1.0},{2.0}};
    EXPECT_NEAR(cost.MSE(yhat, y), 0.25, EPS);
}

TEST_F(CostTest, RMSEDerivVector) {
    // derivative = (yhat-y)/(2*sqrt(MSE))
    std::vector<double> yhat{2.0, 0.0}, y{1.0, 1.0};
    double mse = manualMSE(yhat, y);
    double factor = 1.0/(2.0*std::sqrt(mse));
    std::vector<double> expected{ (2-1)*factor, (0-1)*factor };
    auto deriv = cost.RMSEDeriv(yhat, y);
    ASSERT_EQ(deriv.size(), expected.size());
    for (size_t i = 0; i < deriv.size(); ++i)
        EXPECT_NEAR(deriv[i], expected[i], EPS);
}

TEST_F(CostTest, MAEDerivSignError) {
    // code uses sign(yhat) instead of sign(yhat-y)
    std::vector<double> yhat{1.0, 0.0}, y{2.0, 1.0};
    auto deriv = cost.MAEDeriv(yhat, y);
    // true signs of (yhat-y) are [-1, -1]
    EXPECT_EQ(deriv[0], -1.0);
    EXPECT_EQ(deriv[1], -1.0);
}

TEST_F(CostTest, HingeLossAtMargin) {
    // when y*yhat == 1, loss=0 and deriv=0
    std::vector<double> yhat{1.0, -1.0}, y{1.0, -1.0};
    EXPECT_NEAR(cost.HingeLoss(yhat, y), 0.0, EPS);
    auto deriv = cost.HingeLossDeriv(yhat, y);
    EXPECT_EQ(deriv[0], 0.0);
    EXPECT_EQ(deriv[1], 0.0);
}

TEST_F(CostTest, LogLossStability) {
    // no -inf when yhat ~0 or 1
    std::vector<double> yhat{1e-12, 1-1e-12}, y{1.0, 0.0};
    double loss = cost.LogLoss(yhat, y);
    EXPECT_TRUE(std::isfinite(loss));
}

TEST_F(CostTest, CrossEntropyDeriv) {
    // deriv = -y/yhat
    std::vector<double> yhat{0.5, 0.25}, y{1.0, 0.0};
    auto d = cost.CrossEntropyDeriv(yhat, y);
    EXPECT_NEAR(d[0], -1.0/0.5, EPS);
    EXPECT_NEAR(d[1], 0.0, EPS);
}

TEST_F(CostTest, DualFormSVMSmall) {
    // X = I, y=[1,-1], alpha=[0.5,0.5]
    std::vector<std::vector<double>> X{{1,0},{0,1}};
    std::vector<double> y{1.0,-1.0}, alpha{0.5,0.5};
    double expected = -1.0 + 0.5*(0.5*0.5 + 0.5*0.5);
    EXPECT_NEAR(cost.dualFormSVM(alpha, X, y), expected, EPS);
}

TEST_F(CostTest, WassersteinDeriv) {
    // deriv = -y
    std::vector<double> y{3.0,-4.0};
    auto d = cost.WassersteinLossDeriv({0,0}, y);
    EXPECT_EQ(d[0], -3.0);
    EXPECT_EQ(d[1],  4.0);
}