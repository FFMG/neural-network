#pragma once
#include "activation.h"
#include "neuron.h"
#include <vector>

class NeuralNetwork
{
public:
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& activation);
  NeuralNetwork(const NeuralNetwork&) = delete;
  virtual ~NeuralNetwork();

  void train(
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs,
    int number_of_epoch
  );

  std::vector<std::vector<double>> think(
    const std::vector<std::vector<double>>& inputs
  ) const;
  std::vector<double> think(
    const std::vector<double>& inputs
  ) const;

private:
  void forward_feed(const std::vector<double>& inputVals, std::vector<Neuron::Layer>& layers_src) const;
  void back_propagation(const std::vector<double>& targetVals, std::vector<Neuron::Layer>& layers_src) const;

  std::vector<Neuron::Layer>* _layers;
  const activation::method _activation_method;
};