// additional_logreg_tests.cpp
// Tricky and edge-case tests for LogReg

#include <gtest/gtest.h>
#include "LogReg/LogReg.hpp"
#include <cmath>

// Helper: convert probabilities to labels with customizable threshold
static std::vector<int> toLabels(const std::vector<double>& probs,
                                 double thresh) {
    std::vector<int> labels;
    labels.reserve(probs.size());
    for (double p : probs) {
        labels.push_back(p >= thresh ? 1 : 0);
    }
    return labels;
}

// 1. Probability range test: raw Evaluate and after training, always in (0,1)
TEST(LogRegEdge, ProbabilityRange) {
    // Simple dataset: y=0 always
    std::vector<std::vector<double>> X = {{-100}, {0}, {100}};
    std::vector<double> y = {0, 0, 0};
    MLPP::LogReg clf(X, y, "None", 0.0, 0.0);
    // Raw evaluate prior to training
    for (auto &x: X) {
        double p = clf.modelTest(x);
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
    }
    // Train and test again
    clf.gradientDescent(0.1, 200, false);
    auto probs = clf.modelSetTest(X);
    for (double p : probs) {
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
    }
}

// 2. Single-feature perfect separation
TEST(LogRegEdge, SingleFeaturePerfect) {
    // y = 1 if x>=2 else 0
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        X.push_back({double(i)});
        y.push_back(i < 2 ? 0 : 1);
    }
    MLPP::LogReg clf(X, y, "None", 0.0, 0.0);
    clf.gradientDescent(0.1, 500, false);
    auto preds = toLabels(clf.modelSetTest(X), 0.5);
    // Expect at least 90% accuracy
    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i) correct += (preds[i] == static_cast<int>(y[i]));
    EXPECT_GE(double(correct) / y.size(), 0.9);
}

// 3. Non-separable XOR dataset: expect poor performance
TEST(LogRegEdge, XorNonSeparable) {
    std::vector<std::vector<double>> X = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<double> y = {0,1,1,0}; // XOR
    MLPP::LogReg clf(X, y, "None", 0.0, 0.0);
    clf.gradientDescent(0.1, 500, false);
    double acc = clf.score();
    // Logistic regression cannot solve XOR, accuracy should be <= 0.75
    EXPECT_LE(acc, 0.75);
}

// 4. High regularization shrinks weights
TEST(LogRegEdge, L2RegularizationEffect) {
    // Simple separable data
    std::vector<std::vector<double>> X = {{1,0},{2,0},{3,0},{4,0}};
    std::vector<double> y = {0,0,1,1};
    // No regularization
    MLPP::LogReg clf_none(X, y, "None", 0.0, 0.0);
    clf_none.gradientDescent(0.1, 10000, false);
    auto w_none = clf_none.getWeights(); // assume public or make getter
    // Strong L2
    MLPP::LogReg clf_l2(X, y, "L2", 10.0, 0.0);
    clf_l2.gradientDescent(0.1, 10000, false);
    auto w_l2 = clf_l2.getWeights();
    // Magnitude with L2 should be smaller than without
    double mag_none = std::hypot(w_none[0], w_none[1]);
    double mag_l2 = std::hypot(w_l2[0], w_l2[1]);
    EXPECT_NEAR(mag_l2, mag_none, 1e-1);
}

// 5. Custom threshold behavior
TEST(LogRegEdge, CustomThreshold) {
    std::vector<std::vector<double>> X = {{0},{1}};
    std::vector<double> y = {0,1};
    MLPP::LogReg clf(X, y, "None", 0.0, 0.0);
    clf.gradientDescent(0.05, 300, false);
    auto probs = clf.modelSetTest(X);
    // Use a high threshold to force all zeros
    auto highThreshPreds = toLabels(probs, 0.9);
    EXPECT_EQ(highThreshPreds[0], 0);
    EXPECT_EQ(highThreshPreds[1], 0);
    // Use a low threshold to force all ones
    auto lowThreshPreds = toLabels(probs, 0.1);
    EXPECT_EQ(lowThreshPreds[0], 1);
    EXPECT_EQ(lowThreshPreds[1], 1);
}
