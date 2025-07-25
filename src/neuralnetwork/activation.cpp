#include "activation.h"
#include <algorithm>
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

double activation::calculate_linear(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return x;
}

double activation::calculate_linear_derivative(double /*x*/)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return 1.0;
}

// Sigmoid function
double activation::calculate_sigmoid(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return 1 / (1 + std::exp(-x));
}

// Sigmoid derivative
double activation::calculate_sigmoid_derivative(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return x * (1 - x);
}

double activation::calculate_selu(double x) 
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return SELU_LAMBDA * (x > 0 ? x : SELU_ALPHA * (std::exp(x) - 1));
}

double activation::calculate_selu_derivative(double x) 
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return SELU_LAMBDA * (x > 0 ? 1.0 : SELU_ALPHA * std::exp(x));
}

double activation::calculate_relu(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return std::max(0.0, x);
}

double activation::calculate_relu_derivative(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0.0) ? 1.0 : 0.0;
}

double activation::calculate_leakyRelu(double x, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? x : alpha * x;
}

double activation::calculate_leakyRelu_derivative(double x, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? 1.0 : alpha;
}

double activation::calculate_tanh(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return std::tanh(x);
}

double activation::calculate_tanh_derivative(double x)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  const auto t = std::tanh(x);
  return 1.0 - t * t;
}

// PReLU Activation Function
double activation::calculate_PReLU(double x, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? x : alpha * x;
}

// PReLU Derivative Function
double activation::calculate_PReLU_derivative(double x, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? 1.0 : alpha;
}

double activation::calculate_mish(double x)
{
  return x * std::tanh(std::log1p(std::exp(x)));
}

double activation::calculate_mish_derivative(double x)
{
  double sp = std::log1p(std::exp(x)); // softplus
  double tanh_sp = std::tanh(sp);
  double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
  return tanh_sp + x * sigmoid_x * (1 - tanh_sp * tanh_sp);
}

double activation::calculate_swish(double x) 
{
  constexpr double MAX_EXP_INPUT = 60.0;
  const double exp_term = std::exp(std::clamp(-x, -MAX_EXP_INPUT, MAX_EXP_INPUT));
  return x / (1.0 + exp_term);
}

double activation::calculate_swish_derivative(double x) 
{
  constexpr double MAX_EXP_INPUT = 60.0;
  double clamped_x = std::clamp(x, -MAX_EXP_INPUT, MAX_EXP_INPUT);
  double sigmoid = 1.0 / (1.0 + std::exp(-clamped_x));
  return sigmoid + x * sigmoid * (1.0 - sigmoid);
}

// Approximate GELU (fast, used in transformers)
double activation::calculate_gelu(double x) 
{
  return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

double activation::calculate_gelu_derivative(double x)
{
  // Optional: derivative is complex; can use numerical approximation or skip exact
  const double tanh_term = std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3)));
  return 0.5 * tanh_term +
    (0.5 * x * (1 - tanh_term * tanh_term) *
      std::sqrt(2.0 / M_PI) * (1 + 3 * 0.044715 * x * x));
}

double activation::momentum() const
{
  switch (_method)
  {
  case activation::method::linear:
  case activation::method::relu:
  case activation::method::leakyRelu:
  case activation::method::tanh:
  case activation::method::PRelu:
  case activation::method::selu:
    return 0.9;

  case activation::method::gelu:
  case activation::method::swish:
  case activation::method::mish:
    return 0.95;

  case activation::method::sigmoid:
    return 0.99;

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

double activation::activate(double x) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::method::linear:
    return calculate_linear(x);

  case activation::method::relu:
    return calculate_relu(x);

  case activation::method::leakyRelu:
    return calculate_leakyRelu(x, _alpha);

  case activation::method::tanh:
    return calculate_tanh(x);

  case activation::method::PRelu:
    return calculate_PReLU(x, _alpha);

  case activation::method::selu:
    return calculate_selu(x);

  case activation::method::gelu:
    return calculate_gelu(x);

  case activation::method::swish:
    return calculate_swish(x);

  case activation::method::mish:
    return calculate_mish(x);

  case activation::method::sigmoid:
    return calculate_sigmoid(x);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

double activation::activate_derivative(double x) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::method::linear:
    return calculate_linear_derivative(x);

  case activation::method::relu:
    return calculate_relu_derivative(x);

  case activation::method::leakyRelu:
    return calculate_leakyRelu_derivative(x, _alpha);

  case activation::method::tanh:
    return calculate_tanh_derivative(x);

  case activation::method::PRelu:
    return calculate_PReLU_derivative(x, _alpha);

  case activation::method::selu:
    return calculate_selu_derivative(x);

  case activation::method::gelu:
    return calculate_gelu_derivative(x);

  case activation::method::swish:
    return calculate_swish_derivative(x);

  case activation::method::mish:
    return calculate_mish_derivative(x);

  case activation::method::sigmoid:
    return calculate_sigmoid_derivative(x);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

std::vector<double> activation::weight_initialization(int num_neurons_next_layer, int num_neurons_current_layer) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::method::sigmoid:
  case activation::method::tanh:
    // return lecun_initialization(num_neurons_current_layer);
    return xavier_initialization(num_neurons_next_layer, num_neurons_current_layer);

  case activation::method::selu:
    return selu_initialization(num_neurons_next_layer);

  case activation::method::linear:
  case activation::method::relu:
  case activation::method::leakyRelu:
  case activation::method::PRelu:
  case activation::method::gelu:
  case activation::method::swish:
  case activation::method::mish:
    return he_initialization(num_neurons_next_layer);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

std::vector<double> activation::xavier_initialization(int num_neurons_next_layer, int num_neurons_current_layer)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());

  // Glorot/Xavier initialization uses a uniform distribution:
  double limit = std::sqrt(6.0 / (num_neurons_next_layer + num_neurons_current_layer));
  std::uniform_real_distribution<double> dist(-limit, limit);

  std::vector<double> weights(num_neurons_next_layer);
  for (double& w : weights) 
  {
    w = dist(gen);
  }

  return weights;
}

std::vector<double> activation::he_initialization(int num_neurons_next_layer)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());

  std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / num_neurons_next_layer));

  std::vector<double> weights(num_neurons_next_layer);
  for (double& w : weights) {
    w = dist(gen);  // Initialize weights
  }
  return weights;
}

std::vector<double> activation::selu_initialization(int num_neurons_next_layer)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());

  // Same as LeCun normal
  std::normal_distribution<double> dist(0.0, std::sqrt(1.0 / num_neurons_next_layer));

  std::vector<double> weights(num_neurons_next_layer);
  for (double& w : weights) {
    w = dist(gen);
  }

  return weights;
}

std::vector<double> activation::lecun_initialization(int num_neurons_next_layer)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());

  std::normal_distribution<double> dist(0.0, std::sqrt(1.0 / num_neurons_next_layer));

  std::vector<double> weights(num_neurons_next_layer);
  for (double& w : weights) {
    w = dist(gen);
  }
  return weights;
}

std::string activation::method_to_string() const
{
  return method_to_string(_method);
}

activation::method activation::string_to_method(const std::string& str)
{
  std::string lower_str = str;
  // Convert the string to lowercase for case-insensitive comparison
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
    [](unsigned char c) { return std::tolower(c); });

  if (lower_str == "linear")
  {
    return method::linear;
  }
  if (lower_str == "sigmoid")
  {
    return method::sigmoid;
  }
  if (lower_str == "tanh")
  {
    return method::tanh;
  }
  if (lower_str == "relu")
  {
    return method::relu;
  }
  if (lower_str == "leakyrelu")
  {
    return method::leakyRelu;
  }
  if (lower_str == "prelu")
  {
    return method::PRelu;
  }
  if (lower_str == "selu")
  {
    return method::selu;
  }
  if (lower_str == "swish")
  {
    return method::swish;
  }
  if (lower_str == "mish")
  {
    return method::mish;
  }
  if (lower_str == "gelu")
  {
    return method::gelu;
  }

  // If no match is found, throw an exception
  throw std::invalid_argument("Unknown method: " + str);
}

std::string activation::method_to_string(method m)
{
  switch (m)
  {
  case method::linear:    return "linear";
  case method::sigmoid:   return "sigmoid";
  case method::tanh:      return "tanh";
  case method::relu:      return "relu";
  case method::leakyRelu: return "leakyRelu";
  case method::PRelu:     return "PRelu";
  case method::selu:      return "selu";
  case method::swish:     return "swish";
  case method::mish:      return "mish";
  case method::gelu:      return "gelu";
  default:
    // Handle unknown enum values by throwing an exception
    throw std::invalid_argument("Unknown or unsupported 'method' enum value.");
  }
}

