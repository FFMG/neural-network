#include "activation.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

#include "logger.h"

#ifndef M_PI
# define M_PI   3.141592653589793238462643383279502884
#endif

#define SELU_LAMBDA 1.0507
#define SELU_ALPHA 1.67326

activation::activation(const method method, double alpha) :
  _method(method),
  _alpha(alpha)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::method::linear:
    _activate_ptr = &calculate_linear;
    _derivative_ptr = &calculate_linear_derivative;
    break;
  case activation::method::relu:
    _activate_ptr = &calculate_relu;
    _derivative_ptr = &calculate_relu_derivative;
    break;
  case activation::method::leakyRelu:
    _activate_ptr = &calculate_leakyRelu;
    _derivative_ptr = &calculate_leakyRelu_derivative;
    break;
  case activation::method::tanh:
    _activate_ptr = &calculate_tanh;
    _derivative_ptr = &calculate_tanh_derivative;
    break;
  case activation::method::PRelu:
    _activate_ptr = &calculate_PReLU;
    _derivative_ptr = &calculate_PReLU_derivative;
    break;
  case activation::method::selu:
    _activate_ptr = &calculate_selu;
    _derivative_ptr = &calculate_selu_derivative;
    break;
  case activation::method::elu:
    _activate_ptr = &calculate_elu;
    _derivative_ptr = &calculate_elu_derivative;
    break;
  case activation::method::gelu:
    _activate_ptr = &calculate_gelu;
    _derivative_ptr = &calculate_gelu_derivative;
    break;
  case activation::method::swish:
    _activate_ptr = &calculate_swish;
    _derivative_ptr = &calculate_swish_derivative;
    break;
  case activation::method::mish:
    _activate_ptr = &calculate_mish;
    _derivative_ptr = &calculate_mish_derivative;
    break;
  case activation::method::sigmoid:
    _activate_ptr = &calculate_sigmoid;
    _derivative_ptr = &calculate_sigmoid_derivative;
    break;
  case activation::method::softmax:
    _activate_ptr = &calculate_softmax;
    _derivative_ptr = &calculate_softmax_derivative;
    break;
  default:
    Logger::panic("Unknown activation type!");
  }
}

activation::activation(const activation& src) noexcept :
  _method(src._method),
  _alpha(src._alpha),
  _activate_ptr(src._activate_ptr),
  _derivative_ptr(src._derivative_ptr)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
}

activation::activation(activation&& src) noexcept :
  _method(src._method),
  _alpha(src._alpha),
  _activate_ptr(src._activate_ptr),
  _derivative_ptr(src._derivative_ptr)
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
    _activate_ptr = src._activate_ptr;
    _derivative_ptr = src._derivative_ptr;
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
    _activate_ptr = src._activate_ptr;
    _derivative_ptr = src._derivative_ptr;
  }
  return *this;
}

double activation::calculate_linear(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return x;
}

double activation::calculate_linear_derivative(double, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return 1.0;
}

void activation::activate(double* begin, double* end) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  if (_method == method::softmax)
  {
    calculate_softmax(begin, end);
  }
  else
  {
    for (double* it = begin; it != end; ++it)
    {
      *it = activate(*it);
    }
  }
}

double activation::calculate_sigmoid(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return 1 / (1 + std::exp(-x));
}

double activation::calculate_sigmoid_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  const auto sigmoid_output = calculate_sigmoid(x, 0.0);
  return sigmoid_output * (1.0 - sigmoid_output);
}

double activation::calculate_selu(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return SELU_LAMBDA * (x > 0 ? x : SELU_ALPHA * (std::exp(x) - 1));
}

double activation::calculate_selu_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return SELU_LAMBDA * (x > 0 ? 1.0 : SELU_ALPHA * std::exp(x));
}

double activation::calculate_elu(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return x > 0.0 ? x : alpha * (std::exp(x) - 1.0);
}

double activation::calculate_elu_derivative(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return x > 0.0 ? 1.0 : alpha * std::exp(x);
}

void activation::calculate_softmax(double* begin, double* end) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  if (begin == end)
  {
    return;
  }

  // Find max for numerical stability
  double max_val = *begin;
  for (const double* it = begin + 1; it != end; ++it)
  {
    if (std::isnan(*it))
    {
      // If any input is NaN produce NaN outputs (propagate NaN)
      for (double* jt = begin; jt != end; ++jt)
      {
        *jt = std::numeric_limits<double>::quiet_NaN();
      }
      return;
    }
    if (*it > max_val)
    {
      max_val = *it;
    }
  }

  // Exponentiate and accumulate in higher precision
  long double sum = 0.0L;
  for (double* it = begin; it != end; ++it)
  {
    long double v = std::exp(static_cast<long double>(*it) - static_cast<long double>(max_val));
    *it = static_cast<double>(v);
    sum += v;
  }

  // If sum is zero or not finite (extremely rare), fall back:
  if (sum == 0.0L || !std::isfinite(static_cast<double>(sum)))
  {
    // After exp(* - max) the max entries are 1.0 (or very close). Distribute mass
    // evenly among entries that equal 1.0, set others to 0.
    int count_max = 0;
    for (const double* it = begin; it != end; ++it)
    {
      if (*it == 1.0) ++count_max;
    }

    if (count_max == 0)
    {
      // Defensive: set first element to 1
      for (double* it = begin; it != end; ++it) *it = 0.0;
      *begin = 1.0;
    }
    else
    {
      double inv = 1.0 / static_cast<double>(count_max);
      for (double* it = begin; it != end; ++it)
      {
        *it = (*it == 1.0) ? inv : 0.0;
      }
    }
    return;
  }

  const long double inv_sum = 1.0L / sum;
  for (double* it = begin; it != end; ++it)
  {
    *it = static_cast<double>(static_cast<long double>(*it) * inv_sum);
  }
}

double activation::calculate_softmax(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  Logger::warning("Calling the softmax activation indicate that the wrong error type/activation pair was used!");
  // This is not really correct for a single value, 
  // but we need it for the function pointer.
  // Softmax of a single value is always 1.0.
  return 1.0;
}

double activation::calculate_softmax_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  Logger::warning("Calling the softmax activation derivative indicate that the wrong error type/activation pair was used!");
  // This is also not strictly correct as it depends on all other outputs.
  // However, often we use S(1-S) as a placeholder if we don't have the full Jacobian.
  const double s = calculate_softmax(x, 0.0);
  return s * (1.0 - s);
}

double activation::calculate_relu(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return std::max(0.0, x);
}

double activation::calculate_relu_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0.0) ? 1.0 : 0.0;
}

double activation::calculate_leakyRelu(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? x : alpha * x;
}

double activation::calculate_leakyRelu_derivative(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? 1.0 : alpha;
}

double activation::calculate_tanh(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return std::tanh(x);
}

double activation::calculate_tanh_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  const auto t = std::tanh(x);
  return 1.0 - t * t;
}

double activation::calculate_PReLU(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? x : alpha * x;
}

double activation::calculate_PReLU_derivative(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return (x > 0) ? 1.0 : alpha;
}

double activation::calculate_mish(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return x * std::tanh(std::log1p(std::exp(x)));
}

double activation::calculate_mish_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  double sp = std::log1p(std::exp(x)); // softplus
  double tanh_sp = std::tanh(sp);
  double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
  return tanh_sp + x * sigmoid_x * (1 - tanh_sp * tanh_sp);
}

double activation::calculate_swish(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  constexpr double MAX_EXP_INPUT = 60.0;
  const double exp_term = std::exp(std::clamp(-x, -MAX_EXP_INPUT, MAX_EXP_INPUT));
  return x / (1.0 + exp_term);
}

double activation::calculate_swish_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  constexpr double MAX_EXP_INPUT = 60.0;
  double clamped_x = std::clamp(x, -MAX_EXP_INPUT, MAX_EXP_INPUT);
  double sigmoid = 1.0 / (1.0 + std::exp(-clamped_x));
  return sigmoid + x * sigmoid * (1.0 - sigmoid);
}

double activation::calculate_gelu(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

double activation::calculate_gelu_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  const double tanh_term = std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3)));
  return 0.5 + 0.5 * tanh_term +
    (0.5 * x * (1 - tanh_term * tanh_term) *
      std::sqrt(2.0 / M_PI) * (1 + 3 * 0.044715 * x * x));
}

[[nodiscard]] double activation::momentum() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::method::linear:
  case activation::method::relu:
  case activation::method::leakyRelu:
  case activation::method::tanh:
  case activation::method::PRelu:
  case activation::method::selu:
  case activation::method::elu:
    return 0.9;

  case activation::method::gelu:
  case activation::method::swish:
  case activation::method::mish:
    return 0.95;

  case activation::method::sigmoid:
  case activation::method::softmax:
    return 0.99;

  default:
    Logger::panic("Unknown activation type!");
  }
}

double activation::weight_initialization() const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return weight_initialization(1, 1).front();
}

std::vector<double> activation::weight_initialization(int num_neurons_next_layer, int num_neurons_current_layer) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::method::sigmoid:
  case activation::method::tanh:
  case activation::method::softmax:
    return xavier_initialization(num_neurons_current_layer, num_neurons_next_layer);

  case activation::method::selu:
    return selu_initialization(num_neurons_current_layer, num_neurons_next_layer);

  case activation::method::linear:
  case activation::method::relu:
  case activation::method::leakyRelu:
  case activation::method::PRelu:
  case activation::method::gelu:
  case activation::method::elu:
  case activation::method::swish:
  case activation::method::mish:
    return he_initialization(num_neurons_current_layer, num_neurons_next_layer);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

std::vector<double> activation::xavier_initialization(int fan_in, int fan_out) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());
  double limit = std::sqrt(6.0 / (fan_in + fan_out));
  std::uniform_real_distribution<double> dist(-limit, limit);

  std::vector<double> weights(fan_out);
  for (double& w : weights) 
  {
    w = dist(gen);
  }

  return weights;
}

std::vector<double> activation::he_initialization(int fan_in, int fan_out) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / fan_in));

  std::vector<double> weights(fan_out);
  for (double& w : weights) {
    w = dist(gen);
  }
  return weights;
}

std::vector<double> activation::selu_initialization(int fan_in, int fan_out) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, std::sqrt(1.0 / fan_in));

  std::vector<double> weights(fan_out);
  for (double& w : weights) {
    w = dist(gen);
  }
  return weights;
}

std::vector<double> activation::lecun_initialization(int fan_in, int fan_out) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, std::sqrt(1.0 / fan_in));

  std::vector<double> weights(fan_out);
  for (double& w : weights) {
    w = dist(gen);
  }
  return weights;
}

std::string activation::method_to_string() const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  return method_to_string(_method);
}

activation::method activation::string_to_method(const std::string& str)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  std::string lower_str = str;
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
  if (lower_str == "elu")
  {
    return method::elu;
  }
  if (lower_str == "softmax")
  {
    return method::softmax;
  }
  
  Logger::panic("Unknown method: ", str);
}

std::string activation::method_to_string(method m)
{
  switch (m)
  {
  case method::linear:
    return "linear";
  case method::sigmoid:
    return "sigmoid";
  case method::tanh:
    return "tanh";
  case method::relu:
    return "relu";
  case method::leakyRelu:
    return "leakyRelu";
  case method::PRelu:
    return "PRelu";
  case method::selu:
    return "selu";
  case method::swish:
    return "swish";
  case method::mish:
    return "mish";
  case method::gelu:
    return "gelu";
  case method::elu:
    return "elu";
  case method::softmax:
    return "softmax";
  default:
    Logger::panic("Unknown or unsupported 'method' enum value.");
  }
}
