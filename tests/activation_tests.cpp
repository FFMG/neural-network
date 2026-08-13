#include <gtest/gtest.h>
#include "common/activation.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <random>

#ifndef M_PI
# define M_PI 3.141592653589793238462643383279502884
#endif

// Helper function to calculate exact mathematical expectations independently

using namespace myoddweb::nn;
namespace math_expect {
  double linear(double x, double) { return x; }
  double linear_deriv(double, double) { return 1.0; }

  double sigmoid(double x, double alpha) {
    const double z = alpha * x;
    return z >= 0.0 ? (1.0 / (1.0 + std::exp(-z))) : (std::exp(z) / (1.0 + std::exp(z)));
  }
  double sigmoid_deriv(double x, double alpha) {
    double s = sigmoid(x, alpha);
    return alpha * s * (1.0 - s);
  }

  double tanh(double x, double) { return std::tanh(x); }
  double tanh_deriv(double x, double) {
    double t = std::tanh(x);
    return 1.0 - t * t;
  }

  double relu(double x, double) { return std::max(0.0, x); }
  double relu_deriv(double x, double) { return x > 0.0 ? 1.0 : 0.0; }

  double leaky_relu(double x, double alpha) { return x > 0.0 ? x : alpha * x; }
  double leaky_relu_deriv(double x, double alpha) { return x > 0.0 ? 1.0 : alpha; }

  double selu(double x, double) {
    const double lambda = 1.0507;
    const double alpha = 1.67326;
    return lambda * (x > 0.0 ? x : alpha * (std::exp(x) - 1.0));
  }
  double selu_deriv(double x, double) {
    const double lambda = 1.0507;
    const double alpha = 1.67326;
    return lambda * (x > 0.0 ? 1.0 : alpha * std::exp(x));
  }

  double elu(double x, double alpha) {
    return x > 0.0 ? x : alpha * (std::exp(x) - 1.0);
  }
  double elu_deriv(double x, double alpha) {
    return x > 0.0 ? 1.0 : alpha * std::exp(x);
  }

  double mish(double x, double) {
    return x * std::tanh(std::log1p(std::exp(x)));
  }
  double mish_deriv(double x, double) {
    double sp = std::log1p(std::exp(x));
    double tanh_sp = std::tanh(sp);
    double sig = 1.0 / (1.0 + std::exp(-x));
    return tanh_sp + x * sig * (1.0 - tanh_sp * tanh_sp);
  }

  double swish(double x, double alpha) {
    double z = alpha * x;
    double clamped_z = std::clamp(z, -60.0, 60.0);
    return x / (1.0 + std::exp(-clamped_z));
  }
  double swish_deriv(double x, double alpha) {
    double z = alpha * x;
    double clamped_z = std::clamp(z, -60.0, 60.0);
    double sig = 1.0 / (1.0 + std::exp(-clamped_z));
    return sig + alpha * x * sig * (1.0 - sig);
  }

  double gelu(double x, double) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
  }
  double gelu_deriv(double x, double) {
    double tanh_term = std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x));
    return 0.5 + 0.5 * tanh_term + 
           (0.5 * x * (1.0 - tanh_term * tanh_term) * std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x));
  }

  void softmax(std::vector<double>& input) {
    if (input.empty()) return;
    double max_val = *std::max_element(input.begin(), input.end());
    double sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
      double val = input[i] - max_val;
      val = std::max(val, -30.0);
      input[i] = std::exp(val);
      sum += input[i];
    }
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] /= sum;
    }
  }
}

class ActivationTest : public ::testing::Test {
protected:
  const std::vector<double> test_values = {-5.0, -2.5, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5, 5.0};
  const double alpha = 0.1; // for parametric activations like leakyRelu/sigmoid
  const double tolerance = 1e-7;

  void test_activation(activation::method method, double (*expected_func)(double, double), bool uses_alpha = false) {
    activation act(method, uses_alpha ? alpha : 1.0);
    for (double val : test_values) {
      double expected = expected_func(val, uses_alpha ? alpha : 1.0);
      double actual = act.activate(val);
      EXPECT_NEAR(expected, actual, tolerance) << "Failed activation for " << activation::method_to_string(method) << " at x=" << val;
    }
  }

  void test_derivative(activation::method method, double (*expected_deriv)(double, double), bool uses_alpha = false) {
    activation act(method, uses_alpha ? alpha : 1.0);
    for (double val : test_values) {
      double expected = expected_deriv(val, uses_alpha ? alpha : 1.0);
      double actual = act.activate_derivative(val);
      EXPECT_NEAR(expected, actual, tolerance) << "Failed derivative for " << activation::method_to_string(method) << " at x=" << val;
    }
  }
};

TEST_F(ActivationTest, Linear) {
  test_activation(activation::method::linear, math_expect::linear);
  test_derivative(activation::method::linear, math_expect::linear_deriv);
}

TEST_F(ActivationTest, Sigmoid) {
  test_activation(activation::method::sigmoid, math_expect::sigmoid, true);
  test_derivative(activation::method::sigmoid, math_expect::sigmoid_deriv, true);
}

TEST_F(ActivationTest, Tanh) {
  test_activation(activation::method::tanh, math_expect::tanh);
  test_derivative(activation::method::tanh, math_expect::tanh_deriv);
}

TEST_F(ActivationTest, ReLU) {
  test_activation(activation::method::relu, math_expect::relu);
  test_derivative(activation::method::relu, math_expect::relu_deriv);
}

TEST_F(ActivationTest, LeakyReLU) {
  test_activation(activation::method::leakyRelu, math_expect::leaky_relu, true);
  test_derivative(activation::method::leakyRelu, math_expect::leaky_relu_deriv, true);
}

TEST_F(ActivationTest, PReLU) {
  test_activation(activation::method::PRelu, math_expect::leaky_relu, true);
  test_derivative(activation::method::PRelu, math_expect::leaky_relu_deriv, true);
}

TEST_F(ActivationTest, SELU) {
  test_activation(activation::method::selu, math_expect::selu);
  test_derivative(activation::method::selu, math_expect::selu_deriv);
}

TEST_F(ActivationTest, ELU) {
  test_activation(activation::method::elu, math_expect::elu, true);
  test_derivative(activation::method::elu, math_expect::elu_deriv, true);
}

TEST_F(ActivationTest, Mish) {
  test_activation(activation::method::mish, math_expect::mish);
  test_derivative(activation::method::mish, math_expect::mish_deriv);
}

TEST_F(ActivationTest, Swish) {
  test_activation(activation::method::swish, math_expect::swish, true);
  test_derivative(activation::method::swish, math_expect::swish_deriv, true);
}

TEST_F(ActivationTest, SwishWithBetaTwo) {
  const double beta = 2.0;
  activation act(activation::method::swish, beta);
  for (double val : test_values) {
    double expected = math_expect::swish(val, beta);
    double actual = act.activate(val);
    EXPECT_NEAR(expected, actual, tolerance) << "Failed activation for Swish (beta=2.0) at x=" << val;
    
    double expected_deriv = math_expect::swish_deriv(val, beta);
    double actual_deriv = act.activate_derivative(val);
    EXPECT_NEAR(expected_deriv, actual_deriv, tolerance) << "Failed derivative for Swish (beta=2.0) at x=" << val;
  }
}

TEST_F(ActivationTest, GELU) {
  test_activation(activation::method::gelu, math_expect::gelu);
  test_derivative(activation::method::gelu, math_expect::gelu_deriv);
}

TEST_F(ActivationTest, SoftmaxArray) {
  activation act(activation::method::softmax, 0.0);
  std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0};
  
  std::vector<double> expected = input;
  math_expect::softmax(expected);
  
  act.activate(input.data(), input.data() + input.size());
  
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_NEAR(expected[i], input[i], tolerance) << "Failed Softmax array at index " << i;
  }
}

TEST_F(ActivationTest, SoftmaxNaNAtFirstIndexPropagatesNaN) {
  // Regression test: calculate_softmax's NaN-scan used to start at begin + 1, so a NaN in the
  // very first slot never got detected (max_val/min_val were seeded from *begin and every
  // subsequent comparison with NaN is false). Instead of propagating NaN through the whole
  // row as documented, it silently produced a fake, deterministic {1.0, 0.0, ..., 0.0} result.
  activation act(activation::method::softmax, 0.0);
  std::vector<double> input = { std::numeric_limits<double>::quiet_NaN(), 1.0, 2.0, 0.5 };
  act.activate(input.data(), input.data() + input.size());

  for (double v : input)
  {
    EXPECT_TRUE(std::isnan(v)) << "Expected NaN to propagate through the entire row";
  }
}

TEST_F(ActivationTest, SoftmaxNaNAtLaterIndexPropagatesNaN) {
  // Contrast case: a NaN anywhere after index 0 was already handled correctly before the fix.
  // Kept alongside the index-0 regression test so both branches stay covered together.
  activation act(activation::method::softmax, 0.0);
  std::vector<double> input = { 1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 0.5 };
  act.activate(input.data(), input.data() + input.size());

  for (double v : input)
  {
    EXPECT_TRUE(std::isnan(v)) << "Expected NaN to propagate through the entire row";
  }
}

TEST_F(ActivationTest, SoftmaxDerivativeScalarFallbackReturnsZeroNotSigmoid) {
  // calculate_softmax_derivative previously computed sigmoid(x)*(1-sigmoid(x)) on the raw
  // pre-activation value -- a formula unrelated to softmax's actual output, and one that
  // ignored the already-computed softmax probability entirely. Softmax's true derivative is a
  // full-row Jacobian that cannot be expressed as a function of one scalar value, so the fixed
  // behaviour is to return a safe 0.0 rather than a plausible-looking wrong gradient.
  activation act(activation::method::softmax, 0.0);
  EXPECT_DOUBLE_EQ(act.activate_derivative(0.0), 0.0);
  EXPECT_DOUBLE_EQ(act.activate_derivative(5.0), 0.0);
  EXPECT_DOUBLE_EQ(act.activate_derivative(-5.0), 0.0);
}

TEST_F(ActivationTest, SoftmaxDerivativeBatchedFallbackReturnsZeroForWholeRange) {
  // The batched array activate_derivative() now explicitly handles method::softmax (returning
  // zero gradients for the whole range) instead of silently falling through to the per-element
  // scalar fallback above, which would otherwise log once per element on a real batch.
  activation act(activation::method::softmax, 0.0);
  std::vector<double> pre_activations = { -2.0, 0.0, 1.0, 3.0 };
  std::vector<double> out(pre_activations.size(), 123.0); // poison value to prove it gets overwritten
  act.activate_derivative(pre_activations.data(), pre_activations.data() + pre_activations.size(), nullptr, out.data());

  for (double v : out)
  {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
}

TEST_F(ActivationTest, SoftmaxCatastrophicLogitRangePanics) {
  // logit_range = max - min > 1000 is treated as unrecoverable numerical instability.
  activation act(activation::method::softmax, 0.0);
  std::vector<double> input = { 0.0, 1500.0 };
  EXPECT_THROW(act.activate(input.data(), input.data() + input.size()), std::runtime_error);
}

TEST_F(ActivationTest, SoftmaxExtremeLogitRangeStillProducesValidDistribution) {
  // logit_range > 200 (but <= 1000) only warns; the softmax output must still be a valid,
  // finite probability distribution summing to 1.
  activation act(activation::method::softmax, 0.0);
  std::vector<double> input = { 0.0, 300.0 };
  act.activate(input.data(), input.data() + input.size());

  double sum = 0.0;
  for (double v : input)
  {
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_GE(v, 0.0);
    sum += v;
  }
  EXPECT_NEAR(sum, 1.0, 1e-9);
  // The much larger logit should dominate the distribution.
  EXPECT_NEAR(input[1], 1.0, 1e-9);
}

TEST_F(ActivationTest, UnknownStringToMethodThrows) {
  EXPECT_THROW((void)activation::string_to_method("not-a-real-activation"), std::runtime_error);
}

TEST_F(ActivationTest, DeterministicInitialization) {
  const unsigned fan_in = 100;
  const unsigned fan_out = 50;
  const uint32_t seed = 42;

  // Expected Xavier
  std::mt19937 expected_xavier_gen(seed);
  double limit = std::sqrt(6.0 / (static_cast<double>(fan_in) + static_cast<double>(fan_out)));
  std::uniform_real_distribution<double> dist_xavier(-limit, limit);
  double expected_xavier = dist_xavier(expected_xavier_gen);

  activation act_sigmoid(activation::method::sigmoid, 0.0);
  double actual_xavier = act_sigmoid.weight_initialization(fan_in, fan_out, seed);
  EXPECT_DOUBLE_EQ(expected_xavier, actual_xavier);

  // Expected He
  std::mt19937 expected_he_gen(seed);
  double stddev_he = std::sqrt(2.0 / fan_in);
  std::normal_distribution<double> dist_he(0.0, stddev_he);
  double expected_he = dist_he(expected_he_gen);

  activation act_relu(activation::method::relu, 0.0);
  double actual_he = act_relu.weight_initialization(fan_in, fan_out, seed);
  EXPECT_DOUBLE_EQ(expected_he, actual_he);

  // Expected SELU
  std::mt19937 expected_selu_gen(seed);
  double stddev_selu = std::sqrt(1.0 / fan_in);
  std::normal_distribution<double> dist_selu(0.0, stddev_selu);
  double expected_selu = dist_selu(expected_selu_gen);

  activation act_selu(activation::method::selu, 0.0);
  double actual_selu = act_selu.weight_initialization(fan_in, fan_out, seed);
  EXPECT_DOUBLE_EQ(expected_selu, actual_selu);
}

TEST_F(ActivationTest, VectorizedActivateAndDerivative)
{
  std::vector<activation::method> methods = {
    activation::method::linear,
    activation::method::sigmoid,
    activation::method::tanh,
    activation::method::relu,
    activation::method::leakyRelu,
    activation::method::PRelu,
    activation::method::selu,
    activation::method::elu,
    activation::method::swish,
    activation::method::gelu
  };

  for (auto method : methods)
  {
    activation act(method, 0.5);

    // Vectorized activate test
    std::vector<double> input = test_values;
    act.activate(input.data(), input.data() + input.size(), true);

    std::vector<double> expected_act(test_values.size());
    for (size_t i = 0; i < test_values.size(); ++i)
    {
      expected_act[i] = act.activate(test_values[i]);
    }

    for (size_t i = 0; i < input.size(); ++i)
    {
      EXPECT_NEAR(expected_act[i], input[i], tolerance)
        << "Vectorized activate mismatch for " << activation::method_to_string(method) << " at index " << i;
    }

    // Vectorized activate_derivative test (without y_begin)
    std::vector<double> deriv_out(test_values.size());
    act.activate_derivative(test_values.data(), test_values.data() + test_values.size(), nullptr, deriv_out.data());

    std::vector<double> expected_deriv(test_values.size());
    for (size_t i = 0; i < test_values.size(); ++i)
    {
      expected_deriv[i] = act.activate_derivative(test_values[i]);
    }

    for (size_t i = 0; i < deriv_out.size(); ++i)
    {
      EXPECT_NEAR(expected_deriv[i], deriv_out[i], tolerance)
        << "Vectorized activate_derivative mismatch for " << activation::method_to_string(method) << " at index " << i;
    }

    // Vectorized activate_derivative test (with y_begin for tanh/sigmoid/elu/selu)
    if (method == activation::method::tanh || method == activation::method::sigmoid ||
        method == activation::method::elu || method == activation::method::selu)
    {
      std::vector<double> y_vals(test_values.size());
      for (size_t i = 0; i < test_values.size(); ++i)
      {
        y_vals[i] = act.activate(test_values[i]);
      }

      std::vector<double> deriv_out_y(test_values.size());
      act.activate_derivative(test_values.data(), test_values.data() + test_values.size(), y_vals.data(), deriv_out_y.data());

      for (size_t i = 0; i < deriv_out_y.size(); ++i)
      {
        double expected_y_deriv = 0.0;
        if (method == activation::method::tanh)
        {
          expected_y_deriv = 1.0 - y_vals[i] * y_vals[i];
        }
        else if (method == activation::method::sigmoid)
        {
          expected_y_deriv = act.get_alpha() * y_vals[i] * (1.0 - y_vals[i]);
        }
        else if (method == activation::method::elu)
        {
          expected_y_deriv = test_values[i] > 0.0 ? 1.0 : y_vals[i] + 0.5;
        }
        else if (method == activation::method::selu)
        {
          expected_y_deriv = test_values[i] > 0.0 ? 1.0507 : y_vals[i] + (1.0507 * 1.67326);
        }
        EXPECT_NEAR(expected_y_deriv, deriv_out_y[i], tolerance)
          << "Vectorized activate_derivative with y mismatch for " << activation::method_to_string(method) << " at index " << i;
      }
    }
  }
}

TEST_F(ActivationTest, SwishAVX2Correctness)
{
  const double beta = 1.5;
  activation act(activation::method::swish, beta);

  // Use a vector size of 17 (not a multiple of 4, to verify both SIMD loop and cleanup loop)
  std::vector<double> input(17);
  for (size_t i = 0; i < input.size(); ++i)
  {
    input[i] = -5.0 + i * 0.7; // Mix of negative, zero, and positive values
  }

  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::swish(input[i], beta);
    EXPECT_NEAR(expected, input_copy[i], tolerance) << "AVX2 Swish mismatch at index " << i;
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::swish_deriv(input[i], beta);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance) << "AVX2 Swish derivative mismatch at index " << i;
  }
}

TEST_F(ActivationTest, SigmoidAVX2Correctness)
{
  const double beta = 0.5;
  activation act(activation::method::sigmoid, beta);

  // Use a vector size of 17 (not a multiple of 4, to verify both SIMD loop and cleanup loop)
  std::vector<double> input(17);
  for (size_t i = 0; i < input.size(); ++i)
  {
    input[i] = -5.0 + i * 0.7; // Mix of negative, zero, and positive values
  }

  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::sigmoid(input[i], beta);
    EXPECT_NEAR(expected, input_copy[i], tolerance) << "AVX2 Sigmoid mismatch at index " << i;
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::sigmoid_deriv(input[i], beta);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance) << "AVX2 Sigmoid derivative mismatch at index " << i;
  }
}

TEST_F(ActivationTest, TanhAVX2Correctness)
{
  activation act(activation::method::tanh, 1.0);

  // Use a vector size of 17 (not a multiple of 4, to verify both SIMD loop and cleanup loop)
  std::vector<double> input(17);
  for (size_t i = 0; i < input.size(); ++i)
  {
    input[i] = -5.0 + i * 0.7; // Mix of negative, zero, and positive values
  }

  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::tanh(input[i], 1.0);
    EXPECT_NEAR(expected, input_copy[i], tolerance) << "AVX2 Tanh mismatch at index " << i;
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::tanh_deriv(input[i], 1.0);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance) << "AVX2 Tanh derivative mismatch at index " << i;
  }
}

TEST_F(ActivationTest, GELUAVX2Correctness)
{
  activation act(activation::method::gelu, 1.0);

  std::vector<double> input(17);
  for (size_t i = 0; i < input.size(); ++i)
  {
    input[i] = -5.0 + i * 0.7;
  }

  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::gelu(input[i], 1.0);
    EXPECT_NEAR(expected, input_copy[i], tolerance) << "AVX2 GELU mismatch at index " << i;
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::gelu_deriv(input[i], 1.0);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance) << "AVX2 GELU derivative mismatch at index " << i;
  }
}

TEST_F(ActivationTest, ELUAVX2Correctness)
{
  const double alpha_val = 0.5;
  activation act(activation::method::elu, alpha_val);

  std::vector<double> input(17);
  for (size_t i = 0; i < input.size(); ++i)
  {
    input[i] = -5.0 + i * 0.7;
  }

  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::elu(input[i], alpha_val);
    EXPECT_NEAR(expected, input_copy[i], tolerance) << "AVX2 ELU mismatch at index " << i;
  }

  // Without y_begin
  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::elu_deriv(input[i], alpha_val);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance) << "AVX2 ELU derivative (without y) mismatch at index " << i;
  }

  // With y_begin
  std::vector<double> y_vals(input.size());
  for (size_t i = 0; i < input.size(); ++i)
  {
    y_vals[i] = math_expect::elu(input[i], alpha_val);
  }
  std::vector<double> deriv_out_y(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), y_vals.data(), deriv_out_y.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_y_deriv = input[i] > 0.0 ? 1.0 : y_vals[i] + alpha_val;
    EXPECT_NEAR(expected_y_deriv, deriv_out_y[i], tolerance) << "AVX2 ELU derivative (with y) mismatch at index " << i;
  }
}

TEST_F(ActivationTest, SELUAVX2Correctness)
{
  activation act(activation::method::selu, 1.0);

  std::vector<double> input(17);
  for (size_t i = 0; i < input.size(); ++i)
  {
    input[i] = -5.0 + i * 0.7;
  }

  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::selu(input[i], 1.0);
    EXPECT_NEAR(expected, input_copy[i], tolerance) << "AVX2 SELU mismatch at index " << i;
  }

  // Without y_begin
  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::selu_deriv(input[i], 1.0);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance) << "AVX2 SELU derivative (without y) mismatch at index " << i;
  }

  // With y_begin
  std::vector<double> y_vals(input.size());
  for (size_t i = 0; i < input.size(); ++i)
  {
    y_vals[i] = math_expect::selu(input[i], 1.0);
  }
  std::vector<double> deriv_out_y(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), y_vals.data(), deriv_out_y.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_y_deriv = input[i] > 0.0 ? 1.0507 : y_vals[i] + (1.0507 * 1.67326);
    EXPECT_NEAR(expected_y_deriv, deriv_out_y[i], tolerance) << "AVX2 SELU derivative (with y) mismatch at index " << i;
  }
}

TEST_F(ActivationTest, ELUAVX2AllPositive)
{
  const double alpha_val = 0.5;
  activation act(activation::method::elu, alpha_val);

  std::vector<double> input = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::elu(input[i], alpha_val);
    EXPECT_NEAR(expected, input_copy[i], tolerance);
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());
  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::elu_deriv(input[i], alpha_val);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance);
  }
}

TEST_F(ActivationTest, ELUAVX2AllNegative)
{
  const double alpha_val = 0.5;
  activation act(activation::method::elu, alpha_val);

  std::vector<double> input = { -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0 };
  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::elu(input[i], alpha_val);
    EXPECT_NEAR(expected, input_copy[i], tolerance);
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());
  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::elu_deriv(input[i], alpha_val);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance);
  }
}

TEST_F(ActivationTest, SELUAVX2AllPositive)
{
  activation act(activation::method::selu, 1.0);

  std::vector<double> input = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::selu(input[i], 1.0);
    EXPECT_NEAR(expected, input_copy[i], tolerance);
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());
  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::selu_deriv(input[i], 1.0);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance);
  }
}

TEST_F(ActivationTest, SELUAVX2AllNegative)
{
  activation act(activation::method::selu, 1.0);

  std::vector<double> input = { -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0 };
  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::selu(input[i], 1.0);
    EXPECT_NEAR(expected, input_copy[i], tolerance);
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());
  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::selu_deriv(input[i], 1.0);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance);
  }
}

TEST_F(ActivationTest, MishAVX2Correctness)
{
  activation act(activation::method::mish, 1.0);

  std::vector<double> input(17);
  for (size_t i = 0; i < input.size(); ++i)
  {
    input[i] = -25.0 + i * 3.0; // Covers <-20, in-range, and >20 values!
  }

  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::mish(input[i], 1.0);
    EXPECT_NEAR(expected, input_copy[i], tolerance) << "AVX2 Mish mismatch at index " << i;
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::mish_deriv(input[i], 1.0);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance) << "AVX2 Mish derivative mismatch at index " << i;
  }
}

TEST_F(ActivationTest, MishAVX2AllPositive)
{
  activation act(activation::method::mish, 1.0);

  // All inputs > 20.0
  std::vector<double> input = { 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0 };
  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::mish(input[i], 1.0);
    EXPECT_NEAR(expected, input_copy[i], tolerance);
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());
  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::mish_deriv(input[i], 1.0);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance);
  }
}

TEST_F(ActivationTest, MishAVX2AllNegative)
{
  activation act(activation::method::mish, 1.0);

  // All inputs < -20.0
  std::vector<double> input = { -21.0, -22.0, -23.0, -24.0, -25.0, -26.0, -27.0, -28.0 };
  std::vector<double> input_copy = input;
  act.activate(input_copy.data(), input_copy.data() + input_copy.size(), true);

  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected = math_expect::mish(input[i], 1.0);
    EXPECT_NEAR(expected, input_copy[i], tolerance);
  }

  std::vector<double> deriv_out(input.size());
  act.activate_derivative(input.data(), input.data() + input.size(), nullptr, deriv_out.data());
  for (size_t i = 0; i < input.size(); ++i)
  {
    double expected_deriv = math_expect::mish_deriv(input[i], 1.0);
    EXPECT_NEAR(expected_deriv, deriv_out[i], tolerance);
  }
}

static void test_initialization_worker(const activation* act)
{
  for (int i = 0; i < 1000; ++i)
  {
    double val = act->weight_initialization(10, 10);
    EXPECT_TRUE(std::isfinite(val));
  }
}

TEST_F(ActivationTest, MultithreadedWeightInitialization)
{
  activation act(activation::method::relu, 0.01);
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int i = 0; i < 8; ++i)
  {
    threads.emplace_back(test_initialization_worker, &act);
  }
  for (auto& t : threads)
  {
    if (t.joinable())
    {
      t.join();
    }
  }

  EXPECT_EQ(activation::string_to_method("ReLU"), activation::method::relu);
  EXPECT_EQ(activation::string_to_method("Sigmoid"), activation::method::sigmoid);
}

TEST_F(ActivationTest, AllMethodsStringRoundtripCoverage)
{
  const std::vector<activation::method> all_methods = {
    activation::method::linear,
    activation::method::sigmoid,
    activation::method::tanh,
    activation::method::relu,
    activation::method::leakyRelu,
    activation::method::PRelu,
    activation::method::selu,
    activation::method::swish,
    activation::method::mish,
    activation::method::gelu,
    activation::method::elu,
    activation::method::softmax
  };

  for (const auto& m : all_methods)
  {
    const std::string str = activation::method_to_string(m);
    EXPECT_FALSE(str.empty());
    const activation::method back = activation::string_to_method(str);
    EXPECT_EQ(m, back) << "Failed roundtrip for method: " << str;
  }

  // Test case-insensitivity
  EXPECT_EQ(activation::string_to_method("LiNeAr"), activation::method::linear);
  EXPECT_EQ(activation::string_to_method("sIgMoId"), activation::method::sigmoid);
  EXPECT_EQ(activation::string_to_method("TaNh"), activation::method::tanh);
  EXPECT_EQ(activation::string_to_method("sOfTmAx"), activation::method::softmax);
}

TEST_F(ActivationTest, HandCalculatedAnalyticalProofs)
{
  // 1. Linear: f(2.5) = 2.5, f'(2.5) = 1.0
  activation act_linear(activation::method::linear, 1.0);
  EXPECT_DOUBLE_EQ(act_linear.activate(2.5), 2.5);
  EXPECT_DOUBLE_EQ(act_linear.activate_derivative(2.5), 1.0);

  // 2. Sigmoid (alpha=1.0): f(0) = 0.5, f'(0) = 0.25
  activation act_sig(activation::method::sigmoid, 1.0);
  EXPECT_DOUBLE_EQ(act_sig.activate(0.0), 0.5);
  EXPECT_DOUBLE_EQ(act_sig.activate_derivative(0.0), 0.25);

  // 3. Tanh: f(0) = 0.0, f'(0) = 1.0
  activation act_tanh(activation::method::tanh, 1.0);
  EXPECT_DOUBLE_EQ(act_tanh.activate(0.0), 0.0);
  EXPECT_DOUBLE_EQ(act_tanh.activate_derivative(0.0), 1.0);

  // 4. ReLU: f(3) = 3.0, f'(3) = 1.0; f(-3) = 0.0, f'(-3) = 0.0
  activation act_relu(activation::method::relu, 1.0);
  EXPECT_DOUBLE_EQ(act_relu.activate(3.0), 3.0);
  EXPECT_DOUBLE_EQ(act_relu.activate_derivative(3.0), 1.0);
  EXPECT_DOUBLE_EQ(act_relu.activate(-3.0), 0.0);
  EXPECT_DOUBLE_EQ(act_relu.activate_derivative(-3.0), 0.0);

  // 5. LeakyReLU (alpha=0.01): f(-2) = -0.02, f'(-2) = 0.01
  activation act_lrelu(activation::method::leakyRelu, 0.01);
  EXPECT_DOUBLE_EQ(act_lrelu.activate(-2.0), -0.02);
  EXPECT_DOUBLE_EQ(act_lrelu.activate_derivative(-2.0), 0.01);

  // 6. ELU (alpha=1.0): f(0) = 0.0, f(-1.0) = exp(-1) - 1
  activation act_elu(activation::method::elu, 1.0);
  EXPECT_DOUBLE_EQ(act_elu.activate(0.0), 0.0);
  const double expected_elu_neg = std::exp(-1.0) - 1.0;
  const double expected_elu_neg_deriv = std::exp(-1.0);
  EXPECT_NEAR(act_elu.activate(-1.0), expected_elu_neg, 1e-12);
  EXPECT_NEAR(act_elu.activate_derivative(-1.0), expected_elu_neg_deriv, 1e-12);

  // 7. SELU: f(1.0) = 1.0507, f(-1.0) = 1.0507 * 1.67326 * (exp(-1) - 1)
  activation act_selu(activation::method::selu, 1.0);
  EXPECT_DOUBLE_EQ(act_selu.activate(1.0), 1.0507);
  const double expected_selu_neg = 1.0507 * 1.67326 * (std::exp(-1.0) - 1.0);
  EXPECT_NEAR(act_selu.activate(-1.0), expected_selu_neg, 1e-12);

  // 8. Swish (alpha=1.0): f(0) = 0, f(1.0) = 1 / (1 + exp(-1))
  activation act_swish(activation::method::swish, 1.0);
  EXPECT_DOUBLE_EQ(act_swish.activate(0.0), 0.0);
  const double expected_swish_one = 1.0 / (1.0 + std::exp(-1.0));
  EXPECT_NEAR(act_swish.activate(1.0), expected_swish_one, 1e-12);

  // 9. Softmax with Temperature Scaling
  // Input: [1.0, 0.0] with T=0.5 -> Scaled logit diff = 1.0 / 0.5 = 2.0
  // P(0) = exp(0) / (exp(0) + exp(-2.0)) = 1 / (1 + exp(-2.0))
  activation act_softmax(activation::method::softmax, 1.0, 0.5, 0.5);
  std::vector<double> sm_input = { 1.0, 0.0 };
  act_softmax.activate(sm_input.data(), sm_input.data() + sm_input.size(), false);
  const double expected_sm_0 = 1.0 / (1.0 + std::exp(-2.0));
  EXPECT_NEAR(sm_input[0], expected_sm_0, 1e-12);
  EXPECT_NEAR(sm_input[1], 1.0 - expected_sm_0, 1e-12);
}

TEST_F(ActivationTest, TemperatureAndInferenceTemperature)
{
  activation act(activation::method::softmax, 1.0, 2.0, 0.5);
  EXPECT_DOUBLE_EQ(act.get_temperature(), 2.0);
  EXPECT_DOUBLE_EQ(act.get_inference_temperature(), 0.5);

  act.set_inference_temperature(0.1);
  EXPECT_DOUBLE_EQ(act.get_inference_temperature(), 0.1);

  // Clamping check for tiny temperature
  act.set_inference_temperature(1e-9);
  EXPECT_DOUBLE_EQ(act.get_inference_temperature(), 1e-6);
}

TEST_F(ActivationTest, StatisticalWeightInitializationVerification)
{
  const unsigned fan_in = 100;
  const unsigned fan_out = 50;
  const size_t num_samples = 10000;

  // 1. He Initialization: Normal(0, sqrt(2/fan_in))
  // Variance = 2 / 100 = 0.02
  activation act_he(activation::method::relu, 0.0);
  double sum_he = 0.0;
  double sum_sq_he = 0.0;
  for (size_t i = 0; i < num_samples; ++i)
  {
    double w = act_he.weight_initialization(fan_in, fan_out);
    sum_he += w;
    sum_sq_he += w * w;
  }
  double mean_he = sum_he / num_samples;
  double var_he = (sum_sq_he / num_samples) - (mean_he * mean_he);
  EXPECT_NEAR(mean_he, 0.0, 0.01);
  EXPECT_NEAR(var_he, 0.02, 0.005);

  // 2. SELU / LeCun Initialization: Normal(0, sqrt(1/fan_in))
  // Variance = 1 / 100 = 0.01
  activation act_selu(activation::method::selu, 0.0);
  double sum_selu = 0.0;
  double sum_sq_selu = 0.0;
  for (size_t i = 0; i < num_samples; ++i)
  {
    double w = act_selu.weight_initialization(fan_in, fan_out);
    sum_selu += w;
    sum_sq_selu += w * w;
  }
  double mean_selu = sum_selu / num_samples;
  double var_selu = (sum_sq_selu / num_samples) - (mean_selu * mean_selu);
  EXPECT_NEAR(mean_selu, 0.0, 0.01);
  EXPECT_NEAR(var_selu, 0.01, 0.003);
}




