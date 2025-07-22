#include "./libraries/instrumentor.h"
#include "layer.h"

#include <iostream>
#include <numeric>

constexpr bool _has_bias_neuron = true;

Layer::Layer(LayerType layer_type, const Logger& logger) :
  _number_input_neurons(0),
  _number_output_neurons(0),
  _residual_layer_number(-1),
  _residual_projector(nullptr),
  _layer_type(layer_type),
  _logger(logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
}

Layer::Layer(unsigned num_neurons_in_previous_layer, unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, int residual_layer_number, LayerType layer_type, const activation::method& activation, const OptimiserType& optimiser_type, const Logger& logger) :
  _number_input_neurons(num_neurons_in_previous_layer),
  _number_output_neurons(num_neurons_in_this_layer),
  _residual_layer_number(residual_layer_number),
  _residual_projector(nullptr),
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
  _neurons.reserve(_number_output_neurons+1); // for bias
  for (unsigned neuron_number = 0; neuron_number < _number_output_neurons; ++neuron_number)
  {
    // force the bias node's output to 1.0
    auto neuron = Neuron(
      layer_type == LayerType::Input ? 0 : num_neurons_in_previous_layer,  //  previous
      _number_output_neurons,         //  current 
      layer_type == LayerType::Output ? 0 : num_neurons_in_next_layer+1,      //  next
      neuron_number, 
      activation,
      optimiser_type,
      Neuron::Type::Normal,
      logger);
    _neurons.emplace_back(neuron);
  }

  if(_has_bias_neuron)
  {
    // +1 for bias neuron has no weights.
    auto neuron = Neuron(
      layer_type == LayerType::Input ? 0 : num_neurons_in_previous_layer,  //  previous
      _number_output_neurons,       //  current
      layer_type == LayerType::Output ? 0 : num_neurons_in_next_layer,    //  next
      _number_output_neurons,
      activation,
      optimiser_type,
      Neuron::Type::Bias,
      logger);
    _neurons.emplace_back(neuron);
  }
}

Layer::Layer(const Layer& src) noexcept :
  _neurons(src._neurons),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _residual_layer_number(src._residual_layer_number),
  _residual_projector(nullptr),
  _layer_type(src._layer_type),
  _logger(src._logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(src._residual_projector != nullptr)
  {
    _residual_projector = new ResidualProjector(*src._residual_projector);
  }
}

Layer::Layer(Layer&& src) noexcept :
  _neurons(std::move(src._neurons)),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _residual_layer_number(src._residual_layer_number),
  _residual_projector(src._residual_projector),
  _layer_type(src._layer_type),
  _logger(src._logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  src._number_output_neurons = 0;
  src._number_input_neurons = 0;
  src._residual_layer_number = -1;
  src._residual_projector = nullptr;
}

Layer& Layer::operator=(const Layer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(this != &src)
  {
    clean();
    _neurons = src._neurons;
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _residual_layer_number = src._residual_layer_number;
    _residual_projector = new ResidualProjector(*_residual_projector);
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
    clean();
    _neurons = std::move(src._neurons);
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _residual_layer_number = src._residual_layer_number;
    _residual_projector = src._residual_projector;
    _logger = src._logger;
    
    src._number_output_neurons = 0;
    src._number_input_neurons = 0;
    src._residual_layer_number = -1;
    src._residual_projector = nullptr;
  }
  return *this;
}

Layer::~Layer()
{
  clean();
}

void Layer::clean()
{
  delete _residual_projector;
  _residual_projector = nullptr;
}

void Layer::move_residual_projector(ResidualProjector* residual_projector)
{
  if(residual_projector != _residual_projector)
  {
    delete _residual_projector;
    _residual_projector = residual_projector;
  }
}

unsigned Layer::number_neurons() const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _number_output_neurons + 1;  //  add one for the bias
}

Layer Layer::create_input_layer(const std::vector<Neuron>& neurons, const Logger& logger)
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
  layer._residual_layer_number = -1;
  return layer;
}

Layer Layer::create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(0, num_neurons_in_this_layer, num_neurons_in_next_layer, -1, LayerType::Input, activation::method::linear, OptimiserType::None, logger);
}

Layer Layer::create_hidden_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer, int residual_layer_number, const Logger& logger)
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
  layer._residual_layer_number = residual_layer_number;
  return layer;
}

Layer Layer::create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number, const Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(previous_layer._number_output_neurons, num_neurons_in_this_layer, num_neurons_in_next_layer, residual_layer_number, LayerType::Hidden, activation, optimiser_type, logger);
}

Layer Layer::create_output_layer(const std::vector<Neuron>& neurons, unsigned num_neurons_in_previous_layer, int residual_layer_number, const Logger& logger)
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
  layer._residual_layer_number = residual_layer_number;
  return layer;
}

Layer Layer::create_output_layer(unsigned num_neurons_in_this_layer, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number, const Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(previous_layer._number_output_neurons, num_neurons_in_this_layer, 0, residual_layer_number, LayerType::Output, activation, optimiser_type, logger);
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

std::vector<double> Layer::project_residual_layer_output_values(const std::vector<double>& residual_layer_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(nullptr == _residual_projector)
  {
    return {};
  }
  return _residual_projector->project(residual_layer_outputs);
}