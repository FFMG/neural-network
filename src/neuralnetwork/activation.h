#pragma once
#include "./libraries/instrumentor.h"

#include <vector>

class activation
{
public:
  enum method
  {
    linear,
    sigmoid,
    tanh,
    relu,
    leakyRelu,
    PRelu,
    selu,
    swish,
    gelu
  };

  activation(const method method, double alpha = 0.01) noexcept;
  activation(const activation& src) noexcept;
  activation(activation&& src) noexcept;
  activation& operator=(const activation& src) noexcept;
  activation& operator=(activation&& src) noexcept;
  ~activation() = default;

  double activate(double x) const;
  double activate_derivative(double x) const;

  std::vector<double> weight_initialization(int num_neurons_prev_layer, int num_neurons_current_layer) const;

  void set_alpha(double alpha);
  double get_alpha() const;

  std::string method_to_string() const;
  static std::string method_to_string(method m);

private:
  static std::vector<double> he_initialization(int num_neurons_prev_layer);
  static std::vector<double> xavier_initialization(int num_neurons_prev_layer, int num_neurons_current_layer);
  static std::vector<double> lecun_initialization(int num_neurons_prev_layer);
  static std::vector<double> selu_initialization(int num_neurons_prev_layer);

  static double calculate_selu(double x);
  static double calculate_selu_derivative(double x);
  static double calculate_sigmoid(double x);
  static double calculate_sigmoid_derivative(double x);
  static double calculate_tanh(double x);
  static double calculate_tanh_derivative(double x);
  static double calculate_relu(double x);
  static double calculate_relu_derivative(double x);
  static double calculate_leakyRelu(double x, double alpha);
  static double calculate_leakyRelu_derivative(double x, double alpha);
  static double calculate_PReLU(double x, double alpha);
  static double calculate_PReLU_derivative(double x, double alpha);
  static double calculate_linear(double x);
  static double calculate_linear_derivative(double x);
  static double calculate_swish(double x);
  static double calculate_swish_derivative(double x);
  static double calculate_gelu(double x);
  static double calculate_gelu_derivative(double x);

  method _method;
  double _alpha;
};

