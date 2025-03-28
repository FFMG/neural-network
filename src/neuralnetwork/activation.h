#pragma once

class activation
{
public:
  activation() = delete;

  enum method
  {
    sigmoid_activation,
    tanh_activation,
    relu_activation,
    leakyRelu_activation
  };

  static double activate(method method, double x);
  static double activate_derivative(method method, double x);

private:
  static double sigmoid(double x);
  static double sigmoid_derivative(double x);
  static double tanh(double x);
  static double tanh_derivative(double x);
  static double relu(double x);
  static double relu_derivative(double x);
  static double leakyRelu(double x, double alpha = 0.01);
  static double leakyRelu_derivative(double x, double alpha = 0.01);
};

