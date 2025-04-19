#include <gtest/gtest.h>
#include "LinReg/LinReg.hpp"  // 确保路径正确

// 测试：简单线性回归 y = 2x + 1
TEST(LinRegBasic, NormalEquationFit) {
    // 准备数据 y = 2x + 1
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0};

    // 用 X, y 构造 LinReg，使用默认 reg="None"
    MLPP::LinReg lr(X, y);

    // 调用解析解方法
    lr.normalEquation();

    // 获取对同一 X 的预测
    auto preds = lr.modelSetTest(X);
    ASSERT_EQ(preds.size(), y.size());

    // 逐个比较预测值与真实值
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(preds[i], y[i], 1e-6)
            << "预测值与真实值在索引 " << i << " 处差异过大";
    }
}

// 参照不同正则化选项也可以做参数化测试
class LinRegRegTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        // 简单数据 y = x1 + x2
        X = {{1,2}, {2,1}, {3,4}, {4,3}};
        y = {3, 3, 7, 7};
    }
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

TEST_P(LinRegRegTest, RegularizationEffects) {
    std::string reg_type = GetParam();
    // 构造时使用不同正则化类型
    MLPP::LinReg lr(X, y, reg_type);

    // 用梯度下降做拟合，并允许足够迭代
    lr.gradientDescent(0.1, 1000, false);

    auto preds = lr.modelSetTest(X);
    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (std::abs(preds[i] - y[i]) < 1e-2) correct++;
    }
    // 验证至少 75% 样本拟合良好
    EXPECT_GE(correct, 3) << "在正则化类型 " << reg_type << " 下拟合效果不佳";
}

// 对 "None" 与 "Ridge" 两种情况做参数化
INSTANTIATE_TEST_SUITE_P(
    RegTypes,
    LinRegRegTest,
    ::testing::Values("None", "Ridge")
);
