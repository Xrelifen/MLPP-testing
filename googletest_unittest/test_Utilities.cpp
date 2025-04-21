// test_utilities.cpp

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "Utilities/Utilities.hpp"

using namespace MLPP;

// --- Basic tests ---

TEST(UtilitiesBasic, WeightInitSizeAndType) {
    auto w1 = Utilities::weightInitialization(10, "XavierUniform");
    EXPECT_EQ(w1.size(), 10);
    auto w2 = Utilities::weightInitialization(5, "UnknownType");
    EXPECT_EQ(w2.size(), 5);
    // UnknownType should default to uniform [0,1)
    for (auto v : w2) {
        EXPECT_GE(v, 0.0);
        EXPECT_LT(v, 1.0);
    }
}

TEST(UtilitiesBasic, WeightInitUniformRange) {
    int n = 1000;
    auto w = Utilities::weightInitialization(n, "Uniform");
    double bound = 1.0 / std::sqrt(n);
    for (auto v : w) {
        EXPECT_GE(v, -bound);
        EXPECT_LE(v, +bound);
    }
}

TEST(UtilitiesBasic, BiasInitSingleAndVector) {
    double b = Utilities::biasInitialization();
    EXPECT_GE(b, 0.0);
    EXPECT_LT(b, 1.0);

    int n = 7;
    auto bv = Utilities::biasInitialization(n);
    EXPECT_EQ(bv.size(), n);
    for (auto v : bv) {
        EXPECT_GE(v, 0.0);
        EXPECT_LT(v, 1.0);
    }
}

TEST(UtilitiesBasic, PerformanceBinaryPerfectAndPartial) {
    Utilities util;                       // ← 物件
    std::vector<double> yhat1 {0.2, 1.8, 2.4};
    std::vector<double> y1    {0.0, 2.0, 2.0};
    EXPECT_DOUBLE_EQ(util.performance(yhat1, y1), 1.0);   // ← util.performance

    std::vector<double> yhat2 {0.2, 1.4, 0.7, 1.2};
    std::vector<double> y2    {0.0, 2.0, 1.0, 2.0};
    EXPECT_NEAR(util.performance(yhat2, y2), 0.5, 1e-9);  // ← util.performance
}

TEST(UtilitiesBasic, PerformanceMatrix) {
    Utilities util;
    std::vector<std::vector<double>> yhat {{0.1,0.9},{1.2,1.8},{0.4,1.4}};
    std::vector<std::vector<double>> y    {{0,1},{1,2},{0,2}};
    EXPECT_NEAR(util.performance(yhat, y), 2.0/3.0, 1e-9);      // ← util.performance
}

TEST(UtilitiesBasic, TFPNAndMetrics) {
    Utilities util;
    std::vector<double> yhat {1,0,1,0,1};
    std::vector<double> y    {1,0,0,1,1};

    auto [TP, FP, TN, FN] = util.TF_PN(yhat, y);           // ← util.TF_PN
    EXPECT_EQ(TP, 2); EXPECT_EQ(TN, 1);
    EXPECT_EQ(FP, 1); EXPECT_EQ(FN, 1);

    EXPECT_NEAR(util.recall(yhat, y),    2.0/3.0, 1e-9);
    EXPECT_NEAR(util.precision(yhat, y), 2.0/3.0, 1e-9);
    EXPECT_NEAR(util.accuracy(yhat, y),  3.0/5.0, 1e-9);
    EXPECT_NEAR(util.f1_score(yhat, y),  2.0/3.0, 1e-9);
}

// --- Advanced tests ---

TEST(UtilitiesAdvanced, CreateMiniBatchesEvenSplit) {
    // 6 samples, 3 mini‑batches → each of size 2
    std::vector<std::vector<double>> input(6);
    for (int i = 0; i < 6; ++i) input[i] = {double(i)};
    auto batches = Utilities::createMiniBatches(input, 3);
    ASSERT_EQ(batches.size(), 3);
    EXPECT_EQ(batches[0].size(), 2);
    EXPECT_EQ(batches[1].size(), 2);
    EXPECT_EQ(batches[2].size(), 2);
    // check contents
    EXPECT_EQ(batches[0][0][0], 0.0);
    EXPECT_EQ(batches[2][1][0], 5.0);
}

TEST(UtilitiesAdvanced, CreateMiniBatchesUnevenSplit) {
    // 5 samples, 3 mini‑batches → base size 1, remainder 2 → last batch size = 3
    std::vector<std::vector<double>> input(5);
    for (int i = 0; i < 5; ++i) input[i] = {double(i)};
    auto batches = Utilities::createMiniBatches(input, 3);
    ASSERT_EQ(batches.size(), 3);
    EXPECT_EQ(batches[0].size(), 1);
    EXPECT_EQ(batches[1].size(), 1);
    EXPECT_EQ(batches[2].size(), 3);
    EXPECT_EQ(batches[2][2][0], 4.0);
}

TEST(UtilitiesAdvanced, CreateMiniBatchesWithOutputVector) {
    // pair input with scalar output
    std::vector<std::vector<double>> in(5);
    std::vector<double> out(5);
    for (int i = 0; i < 5; ++i) {
        in[i]  = {double(i)};
        out[i] = double(i + 1);
    }
    auto [inB, outB] = Utilities::createMiniBatches(in, out, 2);
    ASSERT_EQ(inB.size(), 2);
    ASSERT_EQ(outB.size(), 2);
    // 5/2=2, rem=1 ⇒ sizes [2,3]
    EXPECT_EQ(inB[0].size(), 2);
    EXPECT_EQ(outB[0].size(), 2);
    EXPECT_EQ(inB[1].size(), 3);
    EXPECT_EQ(outB[1].size(), 3);
    EXPECT_EQ(outB[1][2], 5.0);
}

TEST(UtilitiesAdvanced, CreateMiniBatchesWithOutputMatrix) {
    // pair input with vector output
    std::vector<std::vector<double>> in(4), out(4);
    for (int i = 0; i < 4; ++i) {
        in[i]  = {double(i)};
        out[i] = {double(i+1), double(i+2)};
    }
    auto [inB, outB] = Utilities::createMiniBatches(in, out, 3);
    ASSERT_EQ(inB.size(), 3);
    ASSERT_EQ(outB.size(), 3);
    // 4/3=1, rem=1 ⇒ sizes [1,1,2]
    EXPECT_EQ(inB[2].size(), 2);
    EXPECT_EQ(outB[2].size(), 2);
    EXPECT_EQ(outB[2][1][0], 4.0);
    EXPECT_EQ(outB[2][1][1], 5.0);
}
