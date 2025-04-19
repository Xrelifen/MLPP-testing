// test_linreg.cpp

#include <gtest/gtest.h>
#include "LinReg/LinReg.hpp"

// If you need to inspect private members in tests, consider adding these getters in LinReg:
// std::vector<double> getWeights() const { return weights; }
// double getBias() const { return bias; }

// --- Basic Tests ---

// Test univariate linear model y = 2x + 1 using normal equation
TEST(LinRegNormalEquation, Univariate) {
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0};

    MLPP::LinReg lr(X, y, "None", 0.0, 0.0);
    lr.normalEquation();

    auto preds = lr.modelSetTest(X);
    ASSERT_EQ(preds.size(), y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(preds[i], y[i], 1e-6)
            << "Index " << i << ": got " << preds[i] << ", expected " << y[i];
    }
}

// Test multivariate linear model y = x1 + x2 using normal equation
TEST(LinRegNormalEquation, Multivariate) {
    std::vector<std::vector<double>> X = {
        {1.0, 2.0},
        {2.0, 1.0},
        {3.0, 4.0},
        {4.0, 3.0}
    };
    std::vector<double> y = {3.0, 3.0, 7.0, 7.0};

    MLPP::LinReg lr(X, y, "None", 0.0, 0.0);
    lr.normalEquation();

    auto preds = lr.modelSetTest(X);
    ASSERT_EQ(preds.size(), y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(preds[i], y[i], 1e-6)
            << "Index " << i << ": got " << preds[i] << ", expected " << y[i];
    }
}

// --- Advanced Edge Cases ---

// 1. Singular matrix (perfect multicollinearity)
TEST(LinRegNormalEquation, SingularMatrix) {
    std::vector<std::vector<double>> X = {
        {1.0, 2.0},
        {2.0, 4.0},
        {3.0, 6.0}
    };
    std::vector<double> y = {3.0, 6.0, 9.0};

    MLPP::LinReg lr(X, y, "None", 0.0, 0.0);
    testing::internal::CaptureStdout();
    lr.normalEquation();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("ERR 99"), std::string::npos);
}

// 2. Ridge regression closed‑form shrinkage test
TEST(LinRegNormalEquation, RidgeRegressionShrinkage) {
    std::vector<std::vector<double>> X = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };
    std::vector<double> y = {1.0, 1.0, 2.0};

    double lambda = 10.0;
    MLPP::LinReg lr(X, y, "Ridge", lambda, 0.0);
    lr.normalEquation();

    auto preds = lr.modelSetTest(X);
    ASSERT_EQ(preds.size(), y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(preds[i], y[i], 1e-3)
            << "Index " << i;
    }
}

// 3. Underdetermined system (n < k)
TEST(LinRegNormalEquation, UnderdeterminedSystem) {
    std::vector<std::vector<double>> X = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    std::vector<double> y = {1.0, 2.0};

    MLPP::LinReg lr(X, y, "None", 0.0, 0.0);
    testing::internal::CaptureStdout();
    lr.normalEquation();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("ERR 99"), std::string::npos);
}

// 4. Single sample (n = 1)
TEST(LinRegNormalEquation, SingleSample) {
    std::vector<std::vector<double>> X = {{42.0, -7.0, 3.14}};
    std::vector<double> y = {100.0};

    MLPP::LinReg lr(X, y, "None", 0.0, 0.0);
    testing::internal::CaptureStdout();
    lr.normalEquation();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("ERR 99"), std::string::npos);
}

// 5. gradientDescent convergence on simple univariate model
TEST(LinRegGradientDescent, ConvergenceUnivariate) {
    std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}, {3.0}};
    std::vector<double> y = {1.0, 3.0, 5.0, 7.0};

    MLPP::LinReg lr(X, y, "None", 0.0, 0.0);
    lr.gradientDescent(0.1, 1000, false);

    auto preds = lr.modelSetTest(X);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(preds[i], y[i], 1e-2)
            << "Index " << i;
    }
}

// 6. Equivalence of SGD and MBGD (batch sizes 1 and n)
TEST(LinRegMBGD, EquivalenceToSGDAndGD) {
    std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}};
    std::vector<double> y = {1.0, 3.0, 5.0};

    MLPP::LinReg lr_sgd(X, y, "None", 0.0, 0.0);
    lr_sgd.SGD(0.1, 500, false);
    auto p_sgd = lr_sgd.modelSetTest(X);

    MLPP::LinReg lr_mb1(X, y, "None", 0.0, 0.0);
    lr_mb1.MBGD(0.1, 500, 1, false);
    auto p_mb1 = lr_mb1.modelSetTest(X);

    MLPP::LinReg lr_gd(X, y, "None", 0.0, 0.0);
    lr_gd.gradientDescent(0.1, 500, false);
    auto p_gd = lr_gd.modelSetTest(X);

    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(p_sgd[i], p_mb1[i], 1e-2);
        EXPECT_NEAR(p_gd[i], p_mb1[i], 1e-2);
    }
}

// 7. score() should return 1.0 for perfect prediction
TEST(LinRegScore, PerfectPrediction) {
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
    std::vector<double> y = {4.0, 5.0, 6.0};

    MLPP::LinReg lr(X, y, "None", 0.0, 0.0);
    lr.normalEquation();
    double r2 = lr.score();
    EXPECT_NEAR(r2, 1.0, 1e-4);
}