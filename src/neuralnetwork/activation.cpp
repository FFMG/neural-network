#include "activation.h"
#include <cmath>
#include <utility>

// Sigmoid function
double activation::sigmoid(double x)
{
  return 1 / (1 + std::exp(-x));
}

// Sigmoid derivative
double activation::sigmoid_derivative(double x)
{
  return x * (1 - x);
}

double activation::relu(double x)
{
  return std::max(0.0, x);
}

double activation::relu_derivative(double x)
{
  return (x > 0.0) ? 1.0 : 0.0;
}

double activation::leakyRelu(double x, double alpha)
{
  return (x > 0) ? x : alpha * x;
}

double activation::leakyRelu_derivative(double x, double alpha)
{
  return (x > 0) ? 1.0 : alpha;
}

double activation::tanh(double x)
{
  return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
}

double activation::tanh_derivative(double x)
{
  return 1 - std::pow(tanh(x), 2);
}
