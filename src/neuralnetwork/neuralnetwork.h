#pragma once
#include <random>
#include <iomanip>
#include <vector>

class NeuralNetwork
{
public:
  enum activation_method
  {
    sigmoid_activation,
    tanh_activation,
    relu_activation,
    leakyRelu_activation
  };

  NeuralNetwork(int number_of_inputs, int number_of_outputs, activation_method activation);
  virtual ~NeuralNetwork();

  void train(
    const std::vector<std::vector<double>>& training_inputs,  
    const std::vector<std::vector<double>>& training_outputs,
    int number_of_epoch
  );

  std::vector<std::vector<double>> think(
    const std::vector<std::vector<double>>& inputs
  ) const;
  double think(
    const std::vector<double>& inputs
  ) const;

private:
  double sigmoid(double x) const;
  double sigmoid_derivative(double x) const;
  double tanh(double x) const;
  double tanh_derivative(double x) const;
  double relu(double x) const;
  double relu_derivative(double x) const;
  double leakyRelu(double x, double alpha = 0.01) const;
  double leakyRelu_derivative(double x, double alpha = 0.01) const;

  double activation(double x) const;
  double activation_derivative(double x) const;

  void prepare_synaptic_weights(int number_of_inputs, int number_of_outputs);

  std::uniform_real_distribution<>* _dis;
  std::mt19937 *_gen;
  std::vector<std::vector<double>>* _synaptic_weights;
  const activation_method _activation_method;
};