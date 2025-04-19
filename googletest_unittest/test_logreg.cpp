#include <gtest/gtest.h>
#include "LogReg/LogReg.hpp"

// Helper：从概率转离散预测
static std::vector<int> toLabels(const std::vector<double>& probs, double thresh = 0.5) {
    std::vector<int> labels;
    labels.reserve(probs.size());
    for (double p : probs) labels.push_back(p >= thresh ? 1 : 0);
    return labels;
}

// 基础测试：AND 门
TEST(LogRegBasic, AndGate) {
    std::vector<std::vector<double>> X = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };
    std::vector<double> y = {0, 0, 0, 1};

    // 1) 用带数据的构造器创建模型
    MLPP::LogReg clf(X, y, "None", /*lambda=*/0.0, /*alpha=*/0.0);

    // 2) 训练模型
    clf.gradientDescent(/*learning_rate=*/0.1, /*max_epoch=*/200, /*UI=*/false);

    // 3) 获取概率并转离散预测
    auto probs = clf.modelSetTest(X);
    ASSERT_EQ(probs.size(), y.size());
    for (double p : probs) {
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
    }
    auto preds = toLabels(probs);

    // 4) 验证预测结果
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_EQ(preds[i], static_cast<int>(y[i]))
            << "Sample " << i << " failed";
    }
}

// 参数化测试：不同正则化强度下的准确率
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

TEST_P(LogRegRegTest, RegularizationImpact) {
    double lambda = GetParam();
    MLPP::LogReg clf(X, y, "L2", /*lambda=*/lambda, /*alpha=*/0.0);
    clf.gradientDescent(/*learning_rate=*/0.05, /*max_epoch=*/500, /*UI=*/false);

    // 评估整体准确率
    double acc = clf.score();
    EXPECT_GE(acc, 0.75) << "Lambda=" << lambda << " yielded low accuracy";
}

INSTANTIATE_TEST_SUITE_P(
    RegStrengths,
    LogRegRegTest,
    ::testing::Values(0.0, 0.1, 1.0)
);
