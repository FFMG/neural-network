#pragma once
#include "../libraries/instrumentor.h"

#include <optional>
#include <string>


namespace myoddweb::nn
{
class activation
{
private:
  using activation_function = double (*)(double, double);

public:
  enum class method
  {
    linear,
    sigmoid,
    tanh,
    relu,
    leakyRelu,
    PRelu,
    selu,
    swish,
    mish,
    gelu,
    elu,
    softmax
  };

  activation(const method method, double alpha, double temperature, double inference_temperature);
  activation(const method method, double alpha, double temperature = 1.0);
  activation(const activation& src) noexcept;
  activation(activation&& src) noexcept;
  activation& operator=(const activation& src) noexcept;
  activation& operator=(activation&& src) noexcept;
  ~activation() = default;

  [[nodiscard]] inline double activate(double x) const noexcept 
  {
    MYODDWEB_PROFILE_FUNCTION("activation");
    switch (_method)
    {
    case method::linear:
      return x;
    case method::relu:
      return (x > 0.0) ? x : 0.0;
    case method::leakyRelu:
    case method::PRelu:
      return (x > 0.0) ? x : _alpha * x;
    case method::tanh:
      return std::tanh(x);
    case method::sigmoid:
      return calculate_sigmoid(x, _alpha);
    case method::selu:
      return calculate_selu(x, _alpha);
    case method::elu:
      return calculate_elu(x, _alpha);
    case method::swish:
      return calculate_swish(x, _alpha);
    case method::mish:
      return calculate_mish(x, _alpha);
    case method::gelu:
      return calculate_gelu(x, _alpha);
    case method::softmax:
      return calculate_softmax(x, _alpha);
    }
    return _activate_ptr(x, _alpha); 
  }
  void activate(double* begin, double* end, bool is_training = false) const;
  void activate_derivative(const double* begin, const double* end, const double* y_begin, double* out) const;

  [[nodiscard]] inline double activate_derivative(double x) const noexcept 
  {
    MYODDWEB_PROFILE_FUNCTION("activation");
    switch (_method)
    {
    case method::linear:
      return 1.0;
    case method::relu:
      return (x > 0.0) ? 1.0 : 0.0;
    case method::leakyRelu:
    case method::PRelu:
      return (x > 0.0) ? 1.0 : _alpha;
    case method::tanh:
    {
      const double t = std::tanh(x);
      return 1.0 - t * t;
    }
    case method::sigmoid:
      return calculate_sigmoid_derivative(x, _alpha);
    case method::selu:
      return calculate_selu_derivative(x, _alpha);
    case method::elu:
      return calculate_elu_derivative(x, _alpha);
    case method::swish:
      return calculate_swish_derivative(x, _alpha);
    case method::mish:
      return calculate_mish_derivative(x, _alpha);
    case method::gelu:
      return calculate_gelu_derivative(x, _alpha);
    case method::softmax:
      return calculate_softmax_derivative(x, _alpha);
    }
    return _derivative_ptr(x, _alpha); 
  }

  [[nodiscard]] double weight_initialization(unsigned fan_in, unsigned fan_out, std::optional<uint32_t> seed = std::nullopt) const;

  [[nodiscard]] std::string method_to_string() const;
  [[nodiscard]] static std::string method_to_string(method m);
  [[nodiscard]] static method string_to_method(const std::string& str);
  
  [[nodiscard]] inline method get_method() const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("activation");
    return _method; 
  }

  [[nodiscard]] inline double get_alpha() const noexcept
  {
	  MYODDWEB_PROFILE_FUNCTION("activation");
	  return _alpha;
  }

  [[nodiscard]] inline double get_temperature() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("activation");
    return _temperature;
  }

  [[nodiscard]] inline double get_inference_temperature() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("activation");
    return _inference_temperature;
  }

  inline void set_inference_temperature(double t) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("activation");
    _inference_temperature = t;
    if (_inference_temperature < 1e-6)
    {
      _inference_temperature = 1e-6;
    }
  }

private:
  [[nodiscard]] static bool iequals(const std::string& str, const char* lit) noexcept;
  [[nodiscard]] double he_initialization(unsigned fan_in, std::optional<uint32_t> seed) const noexcept;
  [[nodiscard]] double xavier_initialization(unsigned fan_in, unsigned fan_out, std::optional<uint32_t> seed) const noexcept;
  [[nodiscard]] double lecun_initialization(unsigned fan_in, std::optional<uint32_t> seed) const noexcept;
  [[nodiscard]] double selu_initialization(unsigned fan_in, std::optional<uint32_t> seed) const noexcept;

  [[nodiscard]] static double calculate_selu(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_selu_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_sigmoid(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_sigmoid_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_tanh(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_tanh_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_relu(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_relu_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_leakyRelu(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_leakyRelu_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_PReLU(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_PReLU_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_linear(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_linear_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_swish(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_swish_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_mish(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_mish_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_gelu(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_gelu_derivative(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_elu(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_elu_derivative(double x, double alpha) noexcept;
  static void calculate_softmax(double* begin, double* end, double temperature);
  [[nodiscard]] static double calculate_softmax(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_softmax_derivative(double x, double alpha) noexcept;

  method _method;
  double _alpha;
  double _temperature;
  double _inference_temperature;
  activation_function _activate_ptr;
  activation_function _derivative_ptr;
};
} // namespace myoddweb::nn
