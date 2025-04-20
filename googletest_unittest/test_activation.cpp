// test_activation.cpp

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include "Activation/Activation.hpp"
#include "LinAlg/LinAlg.hpp"

using namespace MLPP;

static constexpr double EPSILON = 1e-6;

// Helper to compare two vectors element-wise
template <typename T>
void expectVectorNear(const std::vector<T>& actual,
                      const std::vector<T>& expected,
                      double tol = EPSILON) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol)
            << "Mismatch at index " << i;
    }
}

// Helper to compare two matrices element-wise
template <typename T>
void expectMatrixNear(const std::vector<std::vector<T>>& actual,
                      const std::vector<std::vector<T>>& expected,
                      double tol = EPSILON) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_EQ(actual[i].size(), expected[i].size());
        for (size_t j = 0; j < actual[i].size(); ++j) {
            EXPECT_NEAR(actual[i][j], expected[i][j], tol)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

// Test fixture with an Activation instance
class ActivationTest : public ::testing::Test {
protected:
    Activation activation;
};

// --- Basic / Normal Test Cases ---

TEST_F(ActivationTest, LinearScalar) {
    EXPECT_EQ(activation.linear(5.5, false), 5.5);
    EXPECT_EQ(activation.linear(-3.2, true), 1.0);
}

TEST_F(ActivationTest, LinearVector) {
    std::vector<double> input{0.0, 1.0, -1.0};
    auto output     = activation.linear(input, false);
    auto derivative = activation.linear(input, true);
    expectVectorNear(output, input);
    expectVectorNear(derivative, {1.0, 1.0, 1.0});
}

TEST_F(ActivationTest, LinearMatrix) {
    std::vector<std::vector<double>> matrix{{1,2},{3,4}};
    auto output = activation.linear(matrix, false);
    auto deriv  = activation.linear(matrix, true);
    expectMatrixNear(output, matrix);
    expectMatrixNear(deriv, LinAlg().onemat(2,2));
}

TEST_F(ActivationTest, SigmoidScalar) {
    EXPECT_NEAR(activation.sigmoid(0.0, false), 0.5, EPSILON);
    EXPECT_NEAR(activation.sigmoid(0.0, true),  0.25, EPSILON);
}

TEST_F(ActivationTest, SigmoidVector) {
    std::vector<double> input{0.0, 2.0};
    auto output     = activation.sigmoid(input, false);
    auto deriv      = activation.sigmoid(input, true);
    EXPECT_NEAR(output[0], 0.5, EPSILON);
    EXPECT_NEAR(output[1], 1.0/(1+exp(-2)), EPSILON);
    EXPECT_NEAR(deriv[0], 0.5*(1-0.5), EPSILON);
    EXPECT_NEAR(deriv[1], output[1]*(1-output[1]), EPSILON);
}

TEST_F(ActivationTest, SoftmaxVector) {
    std::vector<double> input{1.0, 2.0, 3.0};
    auto probs = activation.softmax(input, false);
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, EPSILON);
    std::vector<double> expVals{exp(1), exp(2), exp(3)};
    double total = expVals[0] + expVals[1] + expVals[2];
    expectVectorNear(probs, { expVals[0]/total,
                             expVals[1]/total,
                             expVals[2]/total });
}

TEST_F(ActivationTest, ReLU) {
    EXPECT_EQ(activation.RELU(-1.0, false), 0.0);
    EXPECT_EQ(activation.RELU(2.5, false),  2.5);
    EXPECT_EQ(activation.RELU(-1.0, true),  0.0);
    EXPECT_EQ(activation.RELU(2.5, true),   1.0);
}

TEST_F(ActivationTest, Softplus) {
    EXPECT_NEAR(activation.softplus(0.0, false), log(2), EPSILON);
    EXPECT_NEAR(activation.softplus(0.0, true),  0.5,    EPSILON);
}

// --- Advanced / Edge-case Test Cases ---

TEST_F(ActivationTest, AdjustedSoftmaxStability) {
    std::vector<double> largeInput{1000.0, 1001.0};
    auto probs = activation.adjSoftmax(largeInput);
    double e = exp(1);
    expectVectorNear(probs, { 1.0/(1+e), e/(1+e) });
}

TEST_F(ActivationTest, SoftmaxJacobian) {
    std::vector<double> input{0.5, 1.5, -0.5};
    auto probs = activation.softmax(input, false);
    auto jac   = activation.softmaxDeriv(input);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j)
                ? probs[i] * (1 - probs[i])
                : -probs[i] * probs[j];
            EXPECT_NEAR(jac[i][j], expected, EPSILON)
                << "Jacobian entry (" << i << "," << j << ")";
        }
    }
}

TEST_F(ActivationTest, SwishVectorDerivativeBug) {
    std::vector<double> x{0.0, 2.0};
    auto deriv = activation.swish(x, true);
    std::vector<double> expected;
    for (double v : x) {
        double s = 1.0/(1+exp(-v));
        expected.push_back(s + v*s*(1-s));
    }
    bool allMatch = true;
    for (size_t i = 0; i < x.size(); ++i) {
        if (fabs(deriv[i] - expected[i]) > EPSILON) {
            allMatch = false;
            break;
        }
    }
    EXPECT_FALSE(allMatch) << "Swish vector derivative should expose missing return bug";
}

TEST_F(ActivationTest, LogitDomainLimits) {
    double l0 = activation.logit(0.0, false);
    double l1 = activation.logit(1.0, false);
    EXPECT_TRUE(std::isnan(l0) || std::isinf(l0));
    EXPECT_TRUE(std::isnan(l1) || std::isinf(l1));
}

TEST_F(ActivationTest, GaussianCDFAtZero) {
    EXPECT_NEAR(activation.gaussianCDF(0.0, false), 0.5,    EPSILON);
    EXPECT_NEAR(activation.gaussianCDF(0.0, true),  1.0/sqrt(2*M_PI), EPSILON);
}

TEST_F(ActivationTest, UnitStepDerivativeZero) {
    std::vector<double> x{-5.0, 0.0, 5.0};
    auto deriv = activation.unitStep(x, true);
    for (double d : deriv) {
        EXPECT_EQ(d, 0.0);
    }
}

TEST_F(ActivationTest, ELUPositiveNegativeBehavior) {
    double alpha = 1.2;
    double negVal = activation.ELU(-1.0, alpha, false);
    double negDer = activation.ELU(-1.0, alpha, true);
    EXPECT_NEAR(negVal, alpha*(exp(-1)-1), EPSILON);
    EXPECT_NEAR(negDer, alpha*exp(-1),       EPSILON);

    double posVal = activation.ELU(2.0, alpha, false);
    double posDer = activation.ELU(2.0, alpha, true);
    EXPECT_NEAR(posVal, 2.0, EPSILON);
    EXPECT_NEAR(posDer, 1.0, EPSILON);
}

TEST_F(ActivationTest, CloglogVectorDerivativeMismatch) {
    // Correct derivative: exp(z - exp(z)); implementation returns exp(-exp(z))
    std::vector<double> input{1.0, 2.0};
    auto deriv    = activation.cloglog(input, true);
    std::vector<double> expected;
    for (double v : input) expected.push_back(exp(v - exp(v)));
    // Expect mismatch to expose missing exp(z) factor
    bool allMatch = true;
    for (size_t i = 0; i < input.size(); ++i) {
        if (fabs(deriv[i] - expected[i]) > EPSILON) { allMatch = false; break; }
    }
    EXPECT_FALSE(allMatch);
}

TEST_F(ActivationTest, MishZeroDerivative) {
    double d = activation.mish(0.0, true);
    EXPECT_TRUE(std::isnan(d));  // mish(z)/z term yields 0/0 at z=0
}

TEST_F(ActivationTest, SincZero) {
    double v = activation.sinc(0.0, false);
    EXPECT_TRUE(std::isnan(v));  // sin(z)/z is 0/0 at z=0
}

TEST_F(ActivationTest, SELUDerivativeMissingLambda) {
    double lambda = 1.7;
    double alpha  = 1.5;
    // For z>0, ELU' = 1, so SELU' should be lambda*1 but impl returns 1
    double d = activation.SELU(2.0, lambda, alpha, true);
    EXPECT_NEAR(d, 1.0, EPSILON);
    EXPECT_FALSE(fabs(d - lambda) < EPSILON);
}
