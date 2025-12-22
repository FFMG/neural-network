#pragma once
#include "./libraries/instrumentor.h"

#include <string>
#include <vector>

class activation
{
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
    elu
  };

  using activation_function = double (*)(double, double);

  activation(const method method, double alpha = 0.01);
  activation(const activation& src) noexcept;
  activation(activation&& src) noexcept;
  activation& operator=(const activation& src) noexcept;
  activation& operator=(activation&& src) noexcept;
  ~activation() = default;

  inline double activate(double x) const { return _activate_ptr(x, _alpha); }
  inline double activate_derivative(double x) const { return _derivative_ptr(x, _alpha); }

  double momentum() const;

  std::vector<double> weight_initialization(int num_neurons_prev_layer, int num_neurons_current_layer) const;
  double weight_initialization() const;

  std::string method_to_string() const;
  static std::string method_to_string(method m);
  static method string_to_method(const std::string& str);
  
  inline method get_method() const noexcept 
  { 
    MYODDWEB_PROFILE_FUNCTION("activation");
    return _method; 
  }

private:
  static std::vector<double> he_initialization(int fan_in, int fan_out) noexcept;
  static std::vector<double> xavier_initialization(int fan_in, int fan_out) noexcept;
  static std::vector<double> lecun_initialization(int fan_in, int fan_out) noexcept;
  static std::vector<double> selu_initialization(int fan_in, int fan_out) noexcept;

  static double calculate_selu(double x, double alpha) noexcept;
  static double calculate_selu_derivative(double x, double alpha) noexcept;
  static double calculate_sigmoid(double x, double alpha) noexcept;
  static double calculate_sigmoid_derivative(double x, double alpha) noexcept;
  static double calculate_tanh(double x, double alpha) noexcept;
  static double calculate_tanh_derivative(double x, double alpha) noexcept;
  static double calculate_relu(double x, double alpha) noexcept;
  static double calculate_relu_derivative(double x, double alpha) noexcept;
  static double calculate_leakyRelu(double x, double alpha) noexcept;
  static double calculate_leakyRelu_derivative(double x, double alpha) noexcept;
  static double calculate_PReLU(double x, double alpha) noexcept;
  static double calculate_PReLU_derivative(double x, double alpha) noexcept;
  static double calculate_linear(double x, double alpha) noexcept;
  static double calculate_linear_derivative(double x, double alpha) noexcept;
  static double calculate_swish(double x, double alpha) noexcept;
  static double calculate_swish_derivative(double x, double alpha) noexcept;
  static double calculate_mish(double x, double alpha) noexcept;
  static double calculate_mish_derivative(double x, double alpha) noexcept;
  static double calculate_gelu(double x, double alpha) noexcept;
  static double calculate_gelu_derivative(double x, double alpha) noexcept;
  static double calculate_elu(double x, double alpha) noexcept;
  static double calculate_elu_derivative(double x, double alpha) noexcept;

  method _method;
  double _alpha;
  activation_function _activate_ptr;
  activation_function _derivative_ptr;
};