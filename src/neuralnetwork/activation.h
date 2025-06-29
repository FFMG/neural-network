#pragma once
#include "./libraries/instrumentor.h"

#include <vector>

class activation
{
public:
  enum method
  {
    linear_activation,
    sigmoid_activation,
    tanh_activation,
    relu_activation,
    leakyRelu_activation,
    PRelu_activation,
    selu_activation,
    swish_activation,
    gelu_activation
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

private:
  static std::vector<double> he_initialization(int num_neurons_prev_layer);
  static std::vector<double> xavier_initialization(int num_neurons_prev_layer, int num_neurons_current_layer);
  static std::vector<double> lecun_initialization(int num_neurons_prev_layer);
  static std::vector<double> selu_initialization(int num_neurons_prev_layer);

  static double selu(double x);
  static double selu_derivative(double x);
  static double sigmoid(double x);
  static double sigmoid_derivative(double x);
  static double tanh(double x);
  static double tanh_derivative(double x);
  static double relu(double x);
  static double relu_derivative(double x);
  static double leakyRelu(double x, double alpha);
  static double leakyRelu_derivative(double x, double alpha);
  static double PReLU(double x, double alpha);
  static double PReLU_derivative(double x, double alpha);
  static double linear(double x);
  static double linear_derivative(double x);
  static double swish(double x);
  static double swish_derivative(double x);
  static double gelu(double x);
  static double gelu_derivative(double x);

  method _method;
  double _alpha;
};

