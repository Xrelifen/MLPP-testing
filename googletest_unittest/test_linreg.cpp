#include <gtest/gtest.h>
#include "LinReg/LinReg.hpp"  // 根據你專案裡的實際路徑調整

// 測試資料：y = 2x + 1
TEST(LinRegBasic, FitAndPredict) {
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0};

    MLPP::LinReg lr;             // 假設 LinReg 在 MLPP 命名空間下
    lr.fit(X, y);                // 假設有 fit(X, y) 介面

    auto preds = lr.predict(X);  // 假設 predict 回傳 vector<double>
    ASSERT_EQ(preds.size(), y.size());

    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(preds[i], y[i], 1e-6)
            << "at index " << i;
    }
}

// Fixture 範例：測試不同學習率對收斂速度的影響
class LinRegLRTest : public ::testing::TestWithParam<double> {
protected:
    void SetUp() override {
        X = {{1}, {2}, {3}, {4}};
        y = {3, 5, 7, 9};
    }
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

TEST_P(LinRegLRTest, Converges) {
    double lr_rate = GetParam();
    MLPP::LinReg lr;
    lr.setLearningRate(lr_rate);
    lr.fit(X, y, /*max_iters=*/1000, /*tol=*/1e-8);

    auto preds = lr.predict(X);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(preds[i], y[i], 1e-3);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DifferentLR,
    LinRegLRTest,
    ::testing::Values(0.01, 0.05, 0.1)
);
