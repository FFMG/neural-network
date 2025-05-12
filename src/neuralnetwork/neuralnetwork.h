#pragma once
#include <vector>
#include <functional>

#include "activation.h"
#include "neuron.h"

class NeuralNetwork
{
public:
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& activation, double learning_rate);
  NeuralNetwork(const std::vector<Neuron::Layer>& layers, const activation::method& activation, double learning_rate, double error);
  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&) = delete;

  virtual ~NeuralNetwork();

  void train(
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs,
    int number_of_epoch,
    const std::function<bool(int, NeuralNetwork&)>& progress_callback = nullptr
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
  double get_learning_rate() const;

private:
  std::vector<double> get_outputs(const Neuron::Layer& output_layers) const;
  static void forward_feed(const std::vector<double>& inputs, std::vector<Neuron::Layer>& layers_src);
  static void back_propagation(
    const std::vector<double>& current_output, 
    std::vector<Neuron::Layer>& layers_src
  );
  static void calculate_output_gradients(const std::vector<double>& current_output, Neuron::Layer& output_layer);
  static double norm_output_gradients(Neuron::Layer& output_layer);

  double calculate_batch_error(const std::vector<double>& targets, const Neuron::Layer& output_layer) const;
  double calculate_batch_mse_error( const std::vector<double>& targets, const Neuron::Layer& output_layer) const;
  double calculate_batch_rmse_error(const std::vector<double>& targets, const Neuron::Layer& output_layer) const;
  
  long double _error;
  std::vector<Neuron::Layer>* _layers;
  const activation::method _activation_method;
  double _learning_rate;
};