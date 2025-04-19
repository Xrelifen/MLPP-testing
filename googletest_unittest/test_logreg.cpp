#include <gtest/gtest.h>
#include "LogReg/LogReg.hpp"  // 根據你專案裡的實際路徑調整

// 測試資料：AND Gate
TEST(LogRegBasic, AndGate) {
    std::vector<std::vector<double>> X = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };
    std::vector<double> y = {0, 0, 0, 1};

    MLPP::LogReg clf;
    clf.setLearningRate(0.1);
    clf.fit(X, y, /*max_iters=*/1000);

    auto probs = clf.predictProb(X);
    ASSERT_EQ(probs.size(), y.size());

    auto preds = clf.predict(X);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_EQ(preds[i], static_cast<int>(y[i]))
            << "at sample " << i;
        // 同時檢查機率分佈
        EXPECT_GE(probs[i], 0.0);
        EXPECT_LE(probs[i], 1.0);
    }
}

// Fixture：測試不同正則化強度
class LogRegRegTest : public ::testing::TestWithParam<double> {
protected:
    void SetUp() override {
        X = {
            {1,2}, {2,1}, {2,3}, {3,2},
            {4,5}, {5,4}, {5,6}, {6,5}
        };
        y = {0,0,0,0,1,1,1,1};
    }
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

TEST_P(LogRegRegTest, Regularization) {
    double lambda = GetParam();
    MLPP::LogReg clf;
    clf.setLearningRate(0.05);
    clf.setRegularization(lambda);
    clf.fit(X, y, 500);

    // 檢查非負權重範圍及分類準確度
    auto preds = clf.predict(X);
    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (preds[i] == static_cast<int>(y[i])) correct++;
    }
    EXPECT_GE(correct, 6) << "至少 75% 正確率";
}

INSTANTIATE_TEST_SUITE_P(
    RegStrength,
    LogRegRegTest,
    ::testing::Values(0.0, 0.1, 1.0)
);
