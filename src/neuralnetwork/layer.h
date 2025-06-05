#pragma once
#include "neuron.h"

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
  Layer(unsigned num_neurons_in_previous_layer, unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, LayerType layer_type, const activation::method& activation, double learning_rate);
  Layer(LayerType layer_type);
public:  
  
  Layer(const Layer& src);
  Layer(Layer&& src);
  Layer& operator=(const Layer& src);
  Layer& operator=(Layer&& src);
  virtual ~Layer() = default;

  unsigned size() const;
  const std::vector<Neuron>& get_neurons() const;
  std::vector<Neuron>& get_neurons();

  const Neuron& get_neuron(unsigned index) const;
  Neuron& get_neuron(unsigned index);

  static Layer create_input_layer(const std::vector<Neuron>& neurons);
  static Layer create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const activation::method& activation, double learning_rate);

  static Layer create_hidden_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer);
  static Layer create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Layer& previous_layer, const activation::method& activation, double learning_rate);

  static Layer create_output_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer);
  static Layer create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const activation::method& activation, double learning_rate);

  std::vector<double> get_outputs() const;
  
private:
  void add_neuron(const Neuron& neuron);
  const LayerType&  get_layer_type()const;

  std::vector<Neuron> _neurons;
  unsigned _number_input_neurons;  //  number of neurons in previous layer
  unsigned _number_output_neurons; //  number of neurons in this layer
  LayerType _layer_type;
};
