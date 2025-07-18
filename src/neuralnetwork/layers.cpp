#include "layers.h"

Layers::Layers(
  const std::vector<unsigned>& topology, 
  const activation::method& hidden_activation,
  const activation::method& output_activation,
  const OptimiserType& optimiser_type,
  const Logger& logger)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  const auto& number_of_layers = topology.size();
  _layers.reserve(number_of_layers);

  // add the input layer
  auto layer = Layer::create_input_layer(topology[0], topology[1], logger);
  _layers.emplace_back(std::move(layer));
  
  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    const auto& previous_layer = _layers.back();
    layer = Layer::create_hidden_layer(num_neurons_current_layer, num_neurons_next_layer, previous_layer, hidden_activation, optimiser_type, logger);
    _layers.emplace_back(std::move(layer));
  }

  // finally, the output layer
  layer = Layer::create_output_layer(topology.back(), _layers.back(), output_activation, optimiser_type, logger);
  _layers.emplace_back(std::move(layer));
}

Layers::Layers(const std::vector<Layer>& layers)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _layers.reserve(layers.size());
  for (const auto& layer : layers)
  {
    auto copy_layer = Layer(layer);
    _layers.emplace_back(std::move(copy_layer));
  }
}

Layers::Layers(const Layers& layers)
 : _layers(layers._layers)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
}

Layers::Layers(Layers&& layers) 
  : _layers(std::move(layers._layers))
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
}

const Layer& Layers::operator[](unsigned index ) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _layers[index];
}

const std::vector<Layer>& Layers::get_layers() const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _layers;
}

std::vector<Layer>& Layers::get_layers()
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _layers;
}
