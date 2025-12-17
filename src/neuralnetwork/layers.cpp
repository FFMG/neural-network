#include "layers.h"
#include "logger.h"
#include "fflayer.h"
#include "elmanrnnlayer.h"
#include <cassert>

Layers::Layers(
  const std::vector<unsigned>& topology,
  double weight_decay_param, // Renamed for clarity in initializer list
  const std::vector<unsigned>& recurrent_layers_param, // Renamed for clarity in initializer list
  const std::vector<double>& dropout_layers,
  const activation::method& hidden_activation,
  const activation::method& output_activation,
  const OptimiserType& optimiser_type,
  int residual_layer_jump) noexcept :
  _weight_decay(weight_decay_param), // Initialize _weight_decay here
  _recurrent_layers(recurrent_layers_param) // Initialize _recurrent_layers here
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  assert(dropout_layers.size() == topology.size() -2 && "Dropout layers size must match the number of hidden layers");
  assert(recurrent_layers_param.size() == topology.size() && "The recurrence layer size must match the topology");
  const auto& number_of_layers = topology.size();
  _layers.reserve(number_of_layers);
  _residual_layer_numbers.reserve(number_of_layers);

  // add the input layer
  auto layer = create_input_layer(topology[0], _weight_decay, -1);
  _layers.emplace_back(std::move(layer));
  _residual_layer_numbers.push_back(-1); // No residual for input layer

  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    auto dropout_rate = dropout_layers[layer_number-1];
    const auto& previous_layer = *_layers.back();
    const auto residual_layer_number = compute_residual_layer(static_cast<int>(layer_number), residual_layer_jump);
    _residual_layer_numbers.push_back(residual_layer_number);

    layer = create_hidden_layer(num_neurons_current_layer, _weight_decay, previous_layer, hidden_activation, optimiser_type, recurrent_layers_param, residual_layer_number, dropout_rate);

    _layers.emplace_back(std::move(layer));
  }

  // finally, the output layer
  const auto residual_layer_number = compute_residual_layer(static_cast<int>(number_of_layers)-1, residual_layer_jump);
  _residual_layer_numbers.push_back(residual_layer_number);
  layer = create_output_layer(topology.back(), _weight_decay, *_layers.back(), output_activation, optimiser_type, recurrent_layers_param, residual_layer_number);

  _layers.emplace_back(std::move(layer));
}

Layers::~Layers() = default;

Layers::Layers(const Layers& src) noexcept :
  _recurrent_layers(src._recurrent_layers) // Copy _recurrent_layers
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _layers.reserve(src._layers.size());
  for(const auto& layer : src._layers)
  {
    _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
  }
  _residual_layer_numbers = src._residual_layer_numbers;
}

Layers::Layers(Layers&& src) noexcept = default;

Layers& Layers::operator=(const Layers& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if(this != &src)
  {
    _layers.clear();
    _layers.reserve(src._layers.size());
    for(const auto& layer : src._layers)
    {
      _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
    }
    _residual_layer_numbers = src._residual_layer_numbers;
    _recurrent_layers = src._recurrent_layers; // Copy _recurrent_layers
  }
  return *this;
}

Layers& Layers::operator=(Layers&& src) noexcept = default;

const Layer& Layers::operator[](unsigned index ) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  #ifndef NDEBUG
  assert(index < _layers.size());
  #endif
  return *_layers[index];
}

Layer& Layers::operator[](unsigned index )
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  #ifndef NDEBUG
  assert(index < _layers.size());
  #endif
  return *_layers[index];
}

int Layers::residual_layer_number(unsigned index) const
{
  #ifndef NDEBUG
  assert(index < _residual_layer_numbers.size());
  #endif
  return _residual_layer_numbers[index];
}

const std::vector<std::unique_ptr<Layer>>& Layers::get_layers() const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _layers;
}

std::vector<std::unique_ptr<Layer>>& Layers::get_layers()
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _layers;
}

int Layers::compute_residual_layer(int current_layer_index, int residual_layer_jump) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if(residual_layer_jump <= 0)
  {
    return -1;  // disabled
  }

  int residual_layer_index = current_layer_index - residual_layer_jump;
  if (residual_layer_index < 0) 
  {
      return -1;
  }
  return residual_layer_index;
}

std::unique_ptr<Layer> Layers::create_input_layer(unsigned num_neurons_in_this_layer, double weight_decay, int residual_layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return std::make_unique<FFLayer>(
    0, 
    0, 
    num_neurons_in_this_layer, 
    weight_decay, 
    Layer::LayerType::Input, 
    activation::method::linear, 
    OptimiserType::None, 
    residual_layer_number, 
    0.0);
}

std::unique_ptr<Layer> Layers::create_hidden_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, const std::vector<unsigned>& recurrent_layers, int residual_layer_number, double dropout_rate)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  unsigned layer_index = previous_layer.get_layer_index() + 1;
  if (recurrent_layers[layer_index] > 0) 
  {
    return std::make_unique<ElmanRNNLayer>(
      layer_index, 
      previous_layer.get_number_neurons(), 
      num_neurons_in_this_layer, 
      weight_decay, 
      Layer::LayerType::Hidden, 
      activation, 
      optimiser_type, 
      residual_layer_number, 
      dropout_rate);
  } 
  else 
  {
    return std::make_unique<FFLayer>(
      layer_index, 
      previous_layer.get_number_neurons(),
      num_neurons_in_this_layer, 
      weight_decay, 
      Layer::LayerType::Hidden, 
      activation, 
      optimiser_type, 
      residual_layer_number,
      dropout_rate);
  }
}

std::unique_ptr<Layer> Layers::create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, const std::vector<unsigned>& recurrent_layers, int residual_layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  unsigned layer_index = previous_layer.get_layer_index() + 1;
  if (recurrent_layers[layer_index] > 0) 
  { 
    // Check if this layer should be recurrent
    return std::make_unique<ElmanRNNLayer>(
      layer_index, 
      previous_layer.get_number_neurons(),
      num_neurons_in_this_layer, 
      weight_decay, 
      Layer::LayerType::Output, 
      activation, 
      optimiser_type, 
      residual_layer_number, 
      0.0);
  } 
  else 
  {
    return std::make_unique<FFLayer>(
      layer_index, 
      previous_layer.get_number_neurons(),
      num_neurons_in_this_layer, 
      weight_decay, 
      Layer::LayerType::Output, 
      activation, 
      optimiser_type, 
      residual_layer_number,
      0.0);
  }
}

const std::vector<unsigned>& Layers::recurrent_layers() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _recurrent_layers;
}
