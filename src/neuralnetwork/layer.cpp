#include "layer.h"

#include <iostream>
#include <numeric>

Layer::Layer(unsigned num_neurons_in_previous_layer, unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, LayerType layer_type, const activation::method& activation, double learning_rate) :
  _number_input_neurons(num_neurons_in_previous_layer),
  _number_output_neurons(num_neurons_in_this_layer),
  _layer_type(layer_type),
  _activation(activation),
  _learning_rate(learning_rate)
{
  if (num_neurons_in_this_layer == 0) 
  {
    std::cerr << "Warning: Creating a layer with 0 neurons." << std::endl;
    throw std::invalid_argument("Warning: Creating a layer with 0 neurons.");
  }
  if (layer_type != LayerType::Input && num_neurons_in_previous_layer == 0) 
  {
    std::cerr << "Warning: Non-input layer created with 0 inputs." << std::endl;
  }

  // We have a new layer, now fill it with neurons, and add a bias neuron in each layer.
  for (unsigned neuron_number = 0; neuron_number <= _number_output_neurons; ++neuron_number) // +1 for bias
  {
    // force the bias node's output to 1.0
    auto neuron = Neuron(
      num_neurons_in_next_layer,
      _number_output_neurons,
      neuron_number, 
      activation, 
      learning_rate);
    neuron.set_output_value(1.0);
    add_neuron(neuron);
  }
}

Layer::Layer(const Layer& src) :
  _neurons(src._neurons),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _layer_type(src._layer_type),
  _activation(src._activation),
  _learning_rate(src._learning_rate)
{
}

Layer& Layer::operator=(const Layer& src)
{
  if(this != &src)
  {
    _neurons = src._neurons;
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _activation = src._activation;
    _learning_rate = src._learning_rate;
  }
  return *this;
}


unsigned Layer::size() const
{
  return _number_output_neurons + 1;  //  add one for the bias
}

void Layer::add_neuron(const Neuron& neuron)
{
  _neurons.push_back(neuron);
}

Layer Layer::create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const activation::method& activation, double learning_rate)
{
  return Layer(0, num_neurons_in_this_layer, num_neurons_in_next_layer, LayerType::Input, activation, learning_rate);
}

Layer Layer::create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const activation::method& activation, double learning_rate)
{
  return Layer(previous_layer._number_output_neurons, num_neurons_in_this_layer, 0, LayerType::Output, activation, learning_rate);
}

Layer Layer::create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Layer& previous_layer, const activation::method& activation, double learning_rate)
{
  return Layer(previous_layer._number_output_neurons, num_neurons_in_this_layer, num_neurons_in_next_layer, LayerType::Hidden, activation, learning_rate);
}

std::vector<double> Layer::get_outputs() const
{
  std::vector<double> outputs;
  outputs.reserve(size() - 1); //  exclude the bias Neuron
  for (auto it = _neurons.begin(); it != _neurons.end() - 1; ++it) 
  {
    outputs.push_back(it->get_output_value());
  }
  return outputs;
}

void Layer::normalise_gradients()
{
  const double max_norm = 10.0;
  auto norm = calulcate_normalised_gradients();
  if (norm < max_norm)
  {
    return;
  }

  // update all the gradients.
  double scale = max_norm / (norm == 0 ? 1e-8 : norm);
  for (size_t n = 0; n < _neurons.size() - 1; ++n)
  {
    auto gradient = _neurons[n].get_gradient();
    gradient *= scale;
    _neurons[n].set_gradient_value(gradient);
  }
}

double Layer::calulcate_normalised_gradients()
{
  auto acc = std::accumulate(
    _neurons.begin(),
    _neurons.end(),
    0.0,
    [](double sum, Neuron& n) {
      auto grad = n.get_gradient();
      return sum + grad * grad;
    });
  return std::sqrt(acc);
}
