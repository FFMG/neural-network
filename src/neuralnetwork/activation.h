#pragma once

#include <vector>

class activation
{
public:
  enum method
  {
    sigmoid_activation,
    tanh_activation,
    relu_activation,
    leakyRelu_activation,
    PRelu_activation,
    Selu_activation,
  };

  activation(const method method, double alpha = 0.001);
  activation(const activation& src);
  activation& operator=(const activation& src);

  double activate(double x) const;
  double activate_derivative(double x) const;

  std::vector<double> weight_initialization(int num_neurons_prev_layer, int num_neurons_current_layer) const;

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

  method _method;
  double _alpha;
};

