#include "activation.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#define SELU_LAMBDA 1.0507
#define SELU_ALPHA 1.67326

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

double activation::selu(double x) 
{
  return SELU_LAMBDA * (x > 0 ? x : SELU_ALPHA * (std::exp(x) - 1));
}

double activation::selu_derivative(double x) 
{
  return SELU_LAMBDA * (x > 0 ? 1.0 : SELU_ALPHA * std::exp(x));
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

  case activation::Selu_activation:
    return activation::selu(x);

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

  case activation::Selu_activation:
    return activation::selu_derivative(x);

  case activation::sigmoid_activation:
  default:
    return activation::sigmoid_derivative(x);
  }
}

std::vector<double> activation::weight_initialization(
  int num_neurons_prev_layer, 
  int num_neurons_current_layer,
  const activation::method& activation)
{
  switch (activation)
  {
  case activation::method::sigmoid_activation:
  case activation::method::tanh_activation:
    // return lecun_initialization(num_neurons_current_layer);
    return xavier_initialization(num_neurons_prev_layer, num_neurons_current_layer);

  case activation::Selu_activation:
    return selu_initialization(num_neurons_prev_layer);

  case activation::method::relu_activation:
  case activation::method::leakyRelu_activation:
  case activation::method::PRelu_activation:
    return he_initialization(num_neurons_prev_layer);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

std::vector<double> activation::xavier_initialization(int num_neurons_prev_layer, int num_neurons_current_layer)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  // Glorot/Xavier initialization uses a uniform distribution:
  double limit = std::sqrt(6.0 / (num_neurons_prev_layer + num_neurons_current_layer));
  std::uniform_real_distribution<double> dist(-limit, limit);

  std::vector<double> weights(num_neurons_prev_layer);
  for (double& w : weights) {
    w = dist(gen);
  }

  return weights;
}

std::vector<double> activation::he_initialization(int num_neurons_prev_layer)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / num_neurons_prev_layer));

  std::vector<double> weights(num_neurons_prev_layer);
  for (double& w : weights) {
    w = dist(gen);  // Initialize weights
  }
  return weights;
}

std::vector<double> activation::selu_initialization(int num_neurons_prev_layer)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  // Same as LeCun normal
  std::normal_distribution<double> dist(0.0, std::sqrt(1.0 / num_neurons_prev_layer));

  std::vector<double> weights(num_neurons_prev_layer);
  for (double& w : weights) {
    w = dist(gen);
  }

  return weights;
}

std::vector<double> activation::lecun_initialization(int num_neurons_prev_layer)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  std::normal_distribution<double> dist(0.0, std::sqrt(1.0 / num_neurons_prev_layer));

  std::vector<double> weights(num_neurons_prev_layer);
  for (double& w : weights) {
    w = dist(gen);
  }

  return weights;
}
