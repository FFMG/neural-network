#pragma once
#include "logger.h"
#include "neuron.h"
#include "optimiser.h"

#include <vector>

class Neuron;
class Layer
{
private:  
  enum class LayerType 
  {
    Input,
    Hidden,
    Output
  };
  Layer(unsigned num_neurons_in_previous_layer, unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, LayerType layer_type, const activation::method& activation, const OptimiserType& optimiser_type, const Logger& logger);
  Layer(LayerType layer_type, const Logger& logger);

public:  
  Layer(const Layer& src) noexcept;
  Layer(Layer&& src) noexcept;
  Layer& operator=(const Layer& src) noexcept;
  Layer& operator=(Layer&& src) noexcept;
  virtual ~Layer() = default;

  unsigned size() const;
  const std::vector<Neuron>& get_neurons() const;
  std::vector<Neuron>& get_neurons();

  const Neuron& get_neuron(unsigned index) const;
  Neuron& get_neuron(unsigned index);

  static Layer create_input_layer(const std::vector<Neuron>& neurons, const Logger& logger);
  static Layer create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Logger& logger);

  static Layer create_hidden_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer, const Logger& logger);
  static Layer create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, const Logger& logger);

  static Layer create_output_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer, const Logger& logger);
  static Layer create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, const Logger& logger);

  std::vector<double> get_outputs() const;
  
private:
  void add_neuron(const Neuron& neuron);

  std::vector<Neuron> _neurons;
  unsigned _number_input_neurons;  //  number of neurons in previous layer
  unsigned _number_output_neurons; //  number of neurons in this layer
  LayerType _layer_type;
  Logger _logger;
};
