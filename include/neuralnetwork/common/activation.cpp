#include "activation.h"
#include "simd_utils.h"
#include <algorithm>
#include <cmath>
#include <random>
// Check if AVX2 is available on x86/x64 architectures
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__AVX2__)
#include <immintrin.h>
#define SIMD_AVX2_ENABLED
#endif

#include "logger.h"

#ifndef M_PI
# define M_PI   3.141592653589793238462643383279502884
#endif

#define SELU_LAMBDA 1.0507
#define SELU_ALPHA 1.67326


namespace myoddweb::nn
{
activation::activation(const method method, double alpha, double temperature, double inference_temperature) :
  _method(method),
  _alpha(alpha),
  _temperature(temperature),
  _inference_temperature(inference_temperature)
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

activation::activation(const method method, double alpha, double temperature) :
  activation(method, alpha, temperature, temperature)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
}

activation::activation(const activation& src) noexcept :
  _method(src._method),
  _alpha(src._alpha),
  _temperature(src._temperature),
  _inference_temperature(src._inference_temperature),
  _activate_ptr(src._activate_ptr),
  _derivative_ptr(src._derivative_ptr)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
}

activation::activation(activation&& src) noexcept :
  _method(src._method),
  _alpha(src._alpha),
  _temperature(src._temperature),
  _inference_temperature(src._inference_temperature),
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
    _temperature = src._temperature;
    _inference_temperature = src._inference_temperature;
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
    _temperature = src._temperature;
    _inference_temperature = src._inference_temperature;
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

void activation::activate(double* begin, double* end, bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  if (_method == method::softmax)
  {
    calculate_softmax(begin, end, is_training ? _temperature : _inference_temperature);
    return;
  }

  const size_t size = end - begin;
  switch (_method)
  {
  case method::linear:
    break; // Nothing to do
  case method::relu:
    simd::relu_activate(begin, size);
    break;
  case method::leakyRelu:
  case method::PRelu:
    simd::leaky_relu_activate(begin, size, _alpha);
    break;
  case method::tanh:
    simd::tanh_activate(begin, size);
    break;
  case method::sigmoid:
    simd::sigmoid_activate(begin, size, _alpha);
    break;
  case method::selu:
    simd::selu_activate(begin, size);
    break;
  case method::elu:
    simd::elu_activate(begin, size, _alpha);
    break;
  case method::swish:
    simd::swish_activate(begin, size, _alpha);
    break;
  case method::gelu:
    simd::gelu_activate(begin, size);
    break;
  case method::mish:
    simd::mish_activate(begin, size);
    break;
  default:
    for (double* it = begin; it != end; ++it)
    {
      *it = _activate_ptr(*it, _alpha);
    }
    break;
  }
}

void activation::activate_derivative(const double* begin, const double* end, const double* y_begin, double* out) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  const size_t size = end - begin;
  switch (_method)
  {
  case method::linear:
    simd::linear_derivative(out, size);
    break;
  case method::relu:
    simd::relu_derivative(begin, size, out);
    break;
  case method::leakyRelu:
  case method::PRelu:
    simd::leaky_relu_derivative(begin, size, out, _alpha);
    break;
  case method::tanh:
    simd::tanh_derivative(begin, size, y_begin, out);
    break;
  case method::sigmoid:
    simd::sigmoid_derivative(begin, size, y_begin, out, _alpha);
    break;
  case method::selu:
    simd::selu_derivative(begin, size, y_begin, out);
    break;
  case method::elu:
    simd::elu_derivative(begin, size, y_begin, out, _alpha);
    break;
  case method::swish:
    simd::swish_derivative(begin, size, out, _alpha);
    break;
  case method::gelu:
    simd::gelu_derivative(begin, size, out);
    break;
  case method::mish:
    simd::mish_derivative(begin, size, out);
    break;
  default:
    for (size_t i = 0; i < size; ++i)
    {
      out[i] = _derivative_ptr(begin[i], _alpha);
    }
    break;
  }
}

double activation::calculate_sigmoid(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  // Compute z = alpha * x in a numerically stable way for the sigmoid.
  const double z = alpha * x;
  if (z >= 0.0)
  {
    const double exp_neg = std::exp(-z);
    return 1.0 / (1.0 + exp_neg);
  }
  else
  {
    const double exp_pos = std::exp(z);
    return exp_pos / (1.0 + exp_pos);
  }
}

double activation::calculate_sigmoid_derivative(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  // d/dx sigmoid(alpha*x) = alpha * sigmoid(alpha*x) * (1 - sigmoid(alpha*x))
  const double s = calculate_sigmoid(x, alpha);
  return alpha * s * (1.0 - s);
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

void activation::calculate_softmax(double* begin, double* end, double temperature)
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  if (begin == end)
  {
    return;
  }

  // Find max for numerical stability
  double max_val = *begin;
  // Also find min_val to detect extreme range for warning
  double min_val = *begin; 
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
    if (*it < min_val) 
    {
      min_val = *it;
    }
  }

  // --- Add warning for extreme logit range ---
  const double logit_range = max_val - min_val;
  const double EXTREME_LOGIT_THRESHOLD = 200.0; // Increased to reduce noise for highly confident models
  const double CATASTROPHIC_LOGIT_THRESHOLD = 1000.0;

  if (logit_range > CATASTROPHIC_LOGIT_THRESHOLD)
  {
    Logger::panic("CRITICAL: Catastrophic logit range detected (", logit_range, "). Initialization or weight update is unstable!");
  }
  if (logit_range > EXTREME_LOGIT_THRESHOLD)
  {
    Logger::warning("Softmax logits exhibit extreme range (max-min diff: ", logit_range, "). Consider increasing weight decay or reducing learning rate.");
  }

  // --- End warning addition ---

  // Exponentiate and accumulate in higher precision
  double sum = 0.0;
  constexpr double LOGIT_CLAMP = 30.0;
  const double inv_temperature = 1.0 / temperature;

  for (double* it = begin; it != end; ++it)
  {
    // Apply temperature scaling and clamp the exponent to prevent explosion
    double val = (static_cast<double>(*it) - static_cast<double>(max_val)) * inv_temperature;
    if (val < -LOGIT_CLAMP)
    {
      val = -LOGIT_CLAMP;
    }
    
    double v = std::exp(val);
    *it = v;
    sum += v;
  }

  // If sum is zero or not finite (extremely rare), fall back:
  if (sum == 0.0 || !std::isfinite(sum))
  {
    // After exp(* - max) the max entries are 1.0 (or very close). Distribute mass
    // evenly among entries that equal 1.0, set others to 0.
    int count_max = 0;
    for (const double* it = begin; it != end; ++it)
    {
      if (*it == 1.0)
      {
        ++count_max;
      }
    }

    if (count_max == 0)
    {
      // Defensive: set first element to 1
      for (double* it = begin; it != end; ++it)
      {
        *it = 0.0;
      }
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

  const double inv_sum = 1.0 / sum;
  for (double* it = begin; it != end; ++it)
  {
    *it = *it * inv_sum;
  }
}

double activation::calculate_softmax(double, double) noexcept
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
  // This is a simplified scalar derivative (S(1-S)). 
  // Note: Softmax derivative is actually a Jacobian matrix.
  // Standard practice is to skip this derivative when combined with Cross-Entropy.
  double s = 1.0 / (1.0 + std::exp(-x)); 
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
  if (x > 20.0)
  {
    return x;
  }
  else if (x < -20.0)
  {
    return 0.0;
  }
  else
  {
    return x * std::tanh(std::log1p(std::exp(x)));
  }
}

double activation::calculate_mish_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  if (x > 20.0)
  {
    return 1.0;
  }
  else if (x < -20.0)
  {
    return 0.0;
  }
  else
  {
    double sp = std::log1p(std::exp(x)); // softplus
    double tanh_sp = std::tanh(sp);
    double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
    return tanh_sp + x * sigmoid_x * (1.0 - tanh_sp * tanh_sp);
  }
}

double activation::calculate_swish(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  constexpr double MAX_EXP_INPUT = 60.0;
  const double z = alpha * x;
  const double exp_term = std::exp(std::clamp(-z, -MAX_EXP_INPUT, MAX_EXP_INPUT));
  return x / (1.0 + exp_term);
}

double activation::calculate_swish_derivative(double x, double alpha) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  constexpr double MAX_EXP_INPUT = 60.0;
  const double z = alpha * x;
  const double clamped_z = std::clamp(z, -MAX_EXP_INPUT, MAX_EXP_INPUT);
  const double sigmoid = 1.0 / (1.0 + std::exp(-clamped_z));
  return sigmoid + alpha * x * sigmoid * (1.0 - sigmoid);
}

double activation::calculate_gelu(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  const double sqrt_2_over_pi = 0.7978845608028654;
  return 0.5 * x * (1.0 + std::tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
}

double activation::calculate_gelu_derivative(double x, double) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  const double sqrt_2_over_pi = 0.7978845608028654;
  const double tanh_term = std::tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x));
  return 0.5 + 0.5 * tanh_term +
    (0.5 * x * (1.0 - tanh_term * tanh_term) *
      sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x));
}

double activation::weight_initialization(unsigned fan_in, unsigned fan_out, std::optional<uint32_t> seed) const
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  switch (_method)
  {
  case activation::method::sigmoid:
  case activation::method::tanh:
  case activation::method::softmax:
    return xavier_initialization(fan_in, fan_out, seed);

  case activation::method::selu:
    return selu_initialization(fan_in, seed);

  case activation::method::linear:
  case activation::method::relu:
  case activation::method::leakyRelu:
  case activation::method::PRelu:
  case activation::method::gelu:
  case activation::method::elu:
  case activation::method::swish:
  case activation::method::mish:
    return he_initialization(fan_in, seed);

  default:
    throw std::invalid_argument("Unknown activation type!");
  }
}

double activation::xavier_initialization(unsigned fan_in, unsigned fan_out, std::optional<uint32_t> seed) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  
  // Standard Xavier initialization: Uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
  double limit = std::sqrt(6.0 / (static_cast<double>(fan_in) + static_cast<double>(fan_out))); 
  std::uniform_real_distribution<double> dist(-limit, limit);

  if (seed.has_value())
  {
    std::mt19937 local_gen(seed.value());
    return dist(local_gen);
  }

  static std::random_device rd;
  static std::mt19937 gen(rd());
  return dist(gen);
}

double activation::he_initialization(unsigned fan_in, std::optional<uint32_t> seed) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  
  // Standard He initialization: Normal(0, sqrt(2/fan_in))
  double stddev = std::sqrt(2.0 / std::max(1u, fan_in));
  std::normal_distribution<double> dist(0.0, stddev);

  if (seed.has_value())
  {
    std::mt19937 local_gen(seed.value());
    return dist(local_gen);
  }

  static std::random_device rd;
  static std::mt19937 gen(rd());
  return dist(gen);
}

double activation::selu_initialization(unsigned fan_in, std::optional<uint32_t> seed) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  
  // SELU initialization (LeCun): Normal(0, sqrt(1/fan_in))
  double stddev = std::sqrt(1.0 / std::max(1u, fan_in));
  std::normal_distribution<double> dist(0.0, stddev);

  if (seed.has_value())
  {
    std::mt19937 local_gen(seed.value());
    return dist(local_gen);
  }

  static std::random_device rd;
  static std::mt19937 gen(rd());
  return dist(gen);
}

double activation::lecun_initialization(unsigned fan_in, std::optional<uint32_t> seed) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("activation");
  
  double stddev = std::sqrt(1.0 / std::max(1u, fan_in));
  std::normal_distribution<double> dist(0.0, stddev);

  if (seed.has_value())
  {
    std::mt19937 local_gen(seed.value());
    return dist(local_gen);
  }

  static std::random_device rd;
  static std::mt19937 gen(rd());
  return dist(gen);
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
    [](unsigned char c) { return static_cast<char>(std::tolower(static_cast<int>(c))); });

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
  MYODDWEB_PROFILE_FUNCTION("activation");
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

} // namespace myoddweb::nn
