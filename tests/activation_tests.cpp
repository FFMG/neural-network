#include <gtest/gtest.h>
#include "common/activation.h"
#include <cmath>
#include <algorithm>
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
