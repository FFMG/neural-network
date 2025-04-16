#pragma once
#include <vector>
#include <functional>

#include "activation.h"
#include "neuron.h"

class NeuralNetwork
{
public:
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& activation);
  NeuralNetwork(const std::vector<Neuron::Layer>& layers, const activation::method& activation);
  NeuralNetwork(const NeuralNetwork&) = delete;
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  virtual ~NeuralNetwork();

  void train(
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs,
    int number_of_epoch,
    const std::function<void(int, double)>& progress_callback = nullptr
  );

  std::vector<std::vector<double>> think(
    const std::vector<std::vector<double>>& inputs
  ) const;
  std::vector<double> think(
    const std::vector<double>& inputs
  ) const;

  std::vector<unsigned> get_topology() const;
  const std::vector<Neuron::Layer>& get_layers() const;
  activation::method get_activation_method() const;
  long double get_error() const;

private:
  void get_outputs( std::vector<double>& outputs, const std::vector<Neuron::Layer>& layers) const;
  void forward_feed(const std::vector<double>& inputVals, std::vector<Neuron::Layer>& layers_src) const;
  double back_propagation(
    const std::vector<double>& targetVals, 
    std::vector<Neuron::Layer>& layers_src,
    const double current_recent_average_error
  ) const;

  long double _error;
  std::vector<Neuron::Layer>* _layers;
  const activation::method _activation_method;
};