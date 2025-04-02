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

// PReLU Activation Function
double activation::PReLU(double x, double alpha)
{
  return (x > 0) ? x : alpha * x;
}

// PReLU Derivative Function
double activation::PReLU_derivative(double x, double alpha)
{
  return (x > 0) ? 1.0 : alpha;
}

double activation::activate(method method, double x)
{
  switch (method)
  {
  case activation::relu_activation:
    return activation::relu(x);

  case activation::leakyRelu_activation:
    return activation::leakyRelu(x);

  case activation::tanh_activation:
    return activation::tanh(x);

  case activation::PRelu_activation:
    return activation::PReLU(x);

  case activation::sigmoid_activation:
  default:
    return activation::sigmoid(x);
  }
}

double activation::activate_derivative(method method, double x)
{
  switch (method)
  {
  case activation::relu_activation:
    return activation::relu_derivative(x);

  case activation::leakyRelu_activation:
    return activation::leakyRelu_derivative(x);

  case activation::tanh_activation:
    return activation::tanh_derivative(x);

  case activation::PRelu_activation:
    return activation::PReLU_derivative(x);

  case activation::sigmoid_activation:
  default:
    return activation::sigmoid_derivative(x);
  }
}