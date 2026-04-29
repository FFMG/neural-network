#pragma once
#include "./libraries/instrumentor.h"

#include <string>

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

  activation() noexcept;
  activation(const method method, double alpha);
  activation(const activation& src) noexcept;
  activation(activation&& src) noexcept;
  activation& operator=(const activation& src) noexcept;
  activation& operator=(activation&& src) noexcept;
  ~activation() = default;

  [[nodiscard]] inline double activate(double x) const noexcept 
  {
    MYODDWEB_PROFILE_FUNCTION("activation");
    return _activate_ptr(x, _alpha); 
  }
  void activate(double* begin, double* end) const;

  [[nodiscard]] inline double activate_derivative(double x) const noexcept 
  {
    MYODDWEB_PROFILE_FUNCTION("activation");
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

private:
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
  [[nodiscard]] static void calculate_softmax(double* begin, double* end) noexcept;
  [[nodiscard]] static double calculate_softmax(double x, double alpha) noexcept;
  [[nodiscard]] static double calculate_softmax_derivative(double x, double alpha) noexcept;

  method _method;
  double _alpha;
  activation_function _activate_ptr;
  activation_function _derivative_ptr;
};