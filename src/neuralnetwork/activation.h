#pragma once

#include <vector>

class activation
{
public:
  activation() = delete;

  enum method
  {
    sigmoid_activation,
    tanh_activation,
    relu_activation,
    leakyRelu_activation,
    PRelu_activation,
    Selu_activation,
  };

  static double activate(method method, double x);
  static double activate_derivative(method method, double x);

  static std::vector<double> weight_initialization(int num_neurons_prev_layer, int num_neurons_current_layer, const method& activation);

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
  static double leakyRelu(double x, double alpha = 0.01);
  static double leakyRelu_derivative(double x, double alpha = 0.01);
  static double PReLU(double x, double alpha = 0.01);
  static double PReLU_derivative(double x, double alpha = 0.01);
};

