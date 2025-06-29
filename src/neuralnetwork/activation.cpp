#include "activation.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#ifndef M_PI
# define M_PI   3.141592653589793238462643383279502884
#endif

#define SELU_LAMBDA 1.0507
#define SELU_ALPHA 1.67326

activation::activation(const method method, double alpha)  noexcept :
  _method(method),
  _alpha(alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
}

activation::activation(const activation& src) noexcept :
  _method(src._method),
  _alpha(src._alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
}

activation::activation(activation&& src) noexcept :
  _method(src._method),
  _alpha(src._alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
}

activation& activation::operator=(const activation& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  if(this != &src)
  {
    _method = src._method;
    _alpha = src._alpha;
  }
  return *this;
}

activation& activation::operator=(activation&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  if (this != &src)
  {
    _method = src._method;
    _alpha = src._alpha;
  }
  return *this;
}

void activation::set_alpha(double alpha)
{
  _alpha = alpha;
}

double activation::get_alpha() const
{
  return _alpha;
}

double activation::linear(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return x;
}

double activation::linear_derivative(double /*x*/)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return 1.0;
}

// Sigmoid function
double activation::sigmoid(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return 1 / (1 + std::exp(-x));
}

// Sigmoid derivative
double activation::sigmoid_derivative(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return x * (1 - x);
}

double activation::selu(double x) 
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return SELU_LAMBDA * (x > 0 ? x : SELU_ALPHA * (std::exp(x) - 1));
}

double activation::selu_derivative(double x) 
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return SELU_LAMBDA * (x > 0 ? 1.0 : SELU_ALPHA * std::exp(x));
}

double activation::relu(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return std::max(0.0, x);
}

double activation::relu_derivative(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0.0) ? 1.0 : 0.0;
}

double activation::leakyRelu(double x, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? x : alpha * x;
}

double activation::leakyRelu_derivative(double x, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? 1.0 : alpha;
}

double activation::tanh(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
}

double activation::tanh_derivative(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return 1 - std::pow(tanh(x), 2);
}

// PReLU Activation Function
double activation::PReLU(double x, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? x : alpha * x;
}

// PReLU Derivative Function
double activation::PReLU_derivative(double x, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? 1.0 : alpha;
}

double activation::swish(double x) 
{
  return x / (1.0 + std::exp(-x));
}

double activation::swish_derivative(double x) 
{
  double sigmoid = 1.0 / (1.0 + std::exp(-x));
  return sigmoid + x * sigmoid * (1 - sigmoid);
}

// Approximate GELU (fast, used in transformers)
double activation::gelu(double x) 
{
  return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

double activation::gelu_derivative(double x)
{
  // Optional: derivative is complex; can use numerical approximation or skip exact
  const double tanh_term = std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3)));
  return 0.5 * tanh_term +
    (0.5 * x * (1 - tanh_term * tanh_term) *
      std::sqrt(2.0 / M_PI) * (1 + 3 * 0.044715 * x * x));
}
double activation::activate(double x) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::linear_activation:
    return activation::linear(x);

  case activation::relu_activation:
    return activation::relu(x);

  case activation::leakyRelu_activation:
    return activation::leakyRelu(x, _alpha);

  case activation::tanh_activation:
    return activation::tanh(x);

  case activation::PRelu_activation:
    return activation::PReLU(x, _alpha);

  case activation::selu_activation:
    return activation::selu(x);

  case activation::gelu_activation:
    return activation::gelu(x);

  case activation::swish_activation:
    return activation::swish(x);

  case activation::sigmoid_activation:
    return activation::sigmoid(x);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

double activation::activate_derivative(double x) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::linear_activation:
    return activation::linear_derivative(x);

  case activation::relu_activation:
    return activation::relu_derivative(x);

  case activation::leakyRelu_activation:
    return activation::leakyRelu_derivative(x, _alpha);

  case activation::tanh_activation:
    return activation::tanh_derivative(x);

  case activation::PRelu_activation:
    return activation::PReLU_derivative(x, _alpha);

  case activation::selu_activation:
    return activation::selu_derivative(x);

  case activation::gelu_activation:
    return activation::gelu_derivative(x);

  case activation::swish_activation:
    return activation::swish_derivative(x);

  case activation::sigmoid_activation:
    return activation::sigmoid_derivative(x);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

std::vector<double> activation::weight_initialization(int num_neurons_prev_layer, int num_neurons_current_layer) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::method::sigmoid_activation:
  case activation::method::tanh_activation:
    // return lecun_initialization(num_neurons_current_layer);
    return xavier_initialization(num_neurons_prev_layer, num_neurons_current_layer);

  case activation::selu_activation:
    return selu_initialization(num_neurons_prev_layer);

  case activation::method::linear_activation:
  case activation::method::relu_activation:
  case activation::method::leakyRelu_activation:
  case activation::method::PRelu_activation:
  case activation::method::gelu_activation:
  case activation::method::swish_activation:
    return he_initialization(num_neurons_prev_layer);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

std::vector<double> activation::xavier_initialization(int num_neurons_prev_layer, int num_neurons_current_layer)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());

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
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());

  std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / num_neurons_prev_layer));

  std::vector<double> weights(num_neurons_prev_layer);
  for (double& w : weights) {
    w = dist(gen);  // Initialize weights
  }
  return weights;
}

std::vector<double> activation::selu_initialization(int num_neurons_prev_layer)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());

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
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());

  std::normal_distribution<double> dist(0.0, std::sqrt(1.0 / num_neurons_prev_layer));

  std::vector<double> weights(num_neurons_prev_layer);
  for (double& w : weights) {
    w = dist(gen);
  }
  return weights;
}
