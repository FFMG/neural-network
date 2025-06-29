#include "./libraries/instrumentor.h"
#include "layer.h"

#include <iostream>
#include <numeric>

Layer::Layer(LayerType layer_type, Logger& logger) :
  _number_input_neurons(0),
  _number_output_neurons(0),
  _layer_type(layer_type),
  _logger(logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
}

Layer::Layer(unsigned num_neurons_in_previous_layer, unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, LayerType layer_type, const activation::method& activation, Logger& logger) :
  _number_input_neurons(num_neurons_in_previous_layer),
  _number_output_neurons(num_neurons_in_this_layer),
  _layer_type(layer_type),
  _logger(logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (num_neurons_in_this_layer == 0) 
  {
    _logger.log_warning("Warning: Creating a layer with 0 neurons.");
    throw std::invalid_argument("Warning: Creating a layer with 0 neurons.");
  }
  if (layer_type != LayerType::Input && num_neurons_in_previous_layer == 0) 
  {
    _logger.log_warning("Warning: Non-input layer created with 0 inputs.");
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
      logger);
    neuron.set_output_value(1.0);
    add_neuron(neuron);
  }
}

Layer::Layer(const Layer& src) noexcept :
  _neurons(src._neurons),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _layer_type(src._layer_type),
  _logger(src._logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
}

Layer::Layer(Layer&& src) noexcept :
  _neurons(std::move(src._neurons)),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _layer_type(src._layer_type),
  _logger(src._logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  src._number_output_neurons = 0;
  src._number_input_neurons = 0;
}

Layer& Layer::operator=(const Layer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(this != &src)
  {
    _neurons = src._neurons;
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _logger = src._logger;
  }
  return *this;
}

Layer& Layer::operator=(Layer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(this != &src)
  {
    _neurons = std::move(src._neurons);
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _logger = src._logger;
    src._number_output_neurons = 0;
    src._number_input_neurons = 0;
  }
  return *this;
}

unsigned Layer::size() const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _number_output_neurons + 1;  //  add one for the bias
}

void Layer::add_neuron(const Neuron& neuron)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  _neurons.push_back(neuron);
}

Layer Layer::create_input_layer(const std::vector<Neuron>& neurons, Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (neurons.size() <= 1) 
  {
    logger.log_error("Creating a layer with 1 neurons, (bias is needed).");
    throw std::invalid_argument("Warning: Creating a layer with 1 neurons, (bias is needed).");
  }
  auto layer = Layer(LayerType::Input, logger);
  layer._number_input_neurons = 0;
  layer._number_output_neurons = static_cast<unsigned>(neurons.size()) -1; // remove bias
  layer._neurons = neurons;
  return layer;
}

Layer Layer::create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(0, num_neurons_in_this_layer, num_neurons_in_next_layer, LayerType::Input, activation::linear, logger);
}

Layer Layer::create_hidden_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer, Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (neurons.size() <= 1) 
  {
    logger.log_error("Creating a layer with 1 neurons, (bias is needed).");
    throw std::invalid_argument("Warning: Creating a layer with 1 neurons, (bias is needed).");
  }
  auto layer = Layer(LayerType::Hidden, logger);
  layer._number_input_neurons = num_neurons_in_previous_layer;
  layer._number_output_neurons = static_cast<unsigned>(neurons.size()) -1; // remove bias
  layer._neurons = neurons;
  return layer;
}

Layer Layer::create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Layer& previous_layer, const activation::method& activation, Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(previous_layer._number_output_neurons, num_neurons_in_this_layer, num_neurons_in_next_layer, LayerType::Hidden, activation, logger);
}

Layer Layer::create_output_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer, Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (neurons.size() <= 1) 
  {
    logger.log_error("Creating a layer with 1 neurons, (bias is needed).");
    throw std::invalid_argument("Warning: Creating a layer with 1 neurons, (bias is needed).");
  }
  auto layer = Layer(LayerType::Output, logger);
  layer._number_input_neurons = num_neurons_in_previous_layer;
  layer._number_output_neurons = static_cast<unsigned>(neurons.size()) -1; // remove bias
  layer._neurons = neurons;
  return layer;
}

Layer Layer::create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const activation::method& activation, Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(previous_layer._number_output_neurons, num_neurons_in_this_layer, 0, LayerType::Output, activation, logger);
}

std::vector<double> Layer::get_outputs() const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  std::vector<double> outputs;
  outputs.reserve(size() - 1); //  exclude the bias Neuron
  for (auto it = _neurons.begin(); it != _neurons.end() - 1; ++it) 
  {
    outputs.emplace_back(it->get_output_value());
  }
  return outputs;
}

const std::vector<Neuron>& Layer::get_neurons() const 
{ 
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _neurons;
}

std::vector<Neuron>& Layer::get_neurons() 
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _neurons;
}

const Neuron& Layer::get_neuron(unsigned index) const 
{ 
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _neurons[index];
}

Neuron& Layer::get_neuron(unsigned index) 
{ 
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _neurons[index];
}
