#pragma once
#include "activation.h"
#include <random>
#include <iomanip>
#include <vector>

class NeuralNetwork
{
public:
  NeuralNetwork(int number_of_inputs, const activation::method& activation);
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
  double activation(double x) const;
  double activation_derivative(double x) const;

  void prepare_synaptic_weights(int number_of_inputs);

  std::uniform_real_distribution<>* _dis;
  std::mt19937 *_gen;
  std::vector<double>* _synaptic_weights;
  const activation::method _activation_method;
};