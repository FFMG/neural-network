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

  activation(const method method, double alpha = 0.01) noexcept;
  activation(const activation& src) noexcept;
  activation(activation&& src) noexcept;
  activation& operator=(const activation& src) noexcept;
  activation& operator=(activation&& src) noexcept;
  ~activation() = default;

  double activate(double x) const;
  double activate_derivative(double x) const ;
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

  static double calculate_selu(double x) noexcept;
  static double calculate_selu_derivative(double x) noexcept;
  static double calculate_sigmoid(double x) noexcept;
  static double calculate_sigmoid_derivative(double x) noexcept;
  static double calculate_tanh(double x) noexcept;
  static double calculate_tanh_derivative(double x) noexcept;
  static double calculate_relu(double x) noexcept;
  static double calculate_relu_derivative(double x) noexcept;
  static double calculate_leakyRelu(double x, double alpha) noexcept;
  static double calculate_leakyRelu_derivative(double x, double alpha) noexcept;
  static double calculate_PReLU(double x, double alpha) noexcept;
  static double calculate_PReLU_derivative(double x, double alpha) noexcept;
  static double calculate_linear(double x) noexcept;
  static double calculate_linear_derivative(double x) noexcept;
  static double calculate_swish(double x) noexcept;
  static double calculate_swish_derivative(double x) noexcept;
  static double calculate_mish(double x) noexcept;
  static double calculate_mish_derivative(double x) noexcept;
  static double calculate_gelu(double x) noexcept;
  static double calculate_gelu_derivative(double x) noexcept;
  static double calculate_elu(double x, double alpha) noexcept;
  static double calculate_elu_derivative(double x, double alpha) noexcept;

  method _method;
  double _alpha;
};