#include "layers.h"
#include "logger.h"
#include <cassert>

Layers::Layers(
  const std::vector<unsigned>& topology,
  const std::vector<double>& dropout_layers,
  const activation::method& hidden_activation,
  const activation::method& output_activation,
  const OptimiserType& optimiser_type,
  int residual_layer_jump) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  assert(dropout_layers.size() == topology.size() -2 && "Dropout layers size must match the number of hidden layers");
  const auto& number_of_layers = topology.size();
  _layers.reserve(number_of_layers);

  // add the input layer
  auto layer = Layer::create_input_layer(topology[0], topology[1]);
  _layers.emplace_back(std::move(layer));

  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    auto dropout_rate = dropout_layers[layer_number-1]; // remove input
    const auto& previous_layer = _layers.back();
    const auto residual_layer_number = compute_residual_layer(layer_number, residual_layer_jump);
    layer = Layer::create_hidden_layer(num_neurons_current_layer, num_neurons_next_layer, previous_layer, hidden_activation, optimiser_type, residual_layer_number, dropout_rate);

    add_residual_layer(layer, hidden_activation);
    Logger::log_tracef([&]
      {
        std::string trace = Logger::log_factory("Layer: ", layer_number, ", residual layer number: ", residual_layer_number);
        if (residual_layer_number != -1)
        {
          auto number_of_neuron_in_that_layer_x = _layers[residual_layer_number].number_neurons();
          auto num_neurons_current_layer_x = layer.number_neurons();

          trace += Logger::log_factory("  Number of neurons in residual: ", number_of_neuron_in_that_layer_x);
          trace += Logger::log_factory("  Number of neurons in layer   : ", num_neurons_current_layer_x);
        }
        return trace;
      });
    _layers.emplace_back(std::move(layer));
  }

  // finally, the output layer
  const auto residual_layer_number = compute_residual_layer(number_of_layers-1, residual_layer_jump);
  layer = Layer::create_output_layer(topology.back(), _layers.back(), output_activation, optimiser_type, residual_layer_number);

  add_residual_layer(layer, output_activation);
  if (Logger::can_log_trace())
  {
    Logger::log_trace("Layer: ", number_of_layers - 1, ", residual layer number: ", residual_layer_number);
    if (residual_layer_number != -1)
    {
      auto number_of_neuron_in_that_layer = _layers[residual_layer_number].number_neurons();
      auto num_neurons_current_layer = layer.number_neurons();
      Logger::log_trace("  Number of neurons in residual: ", number_of_neuron_in_that_layer);
      Logger::log_trace("  Number of neurons in layer   : ", num_neurons_current_layer);
    }
  }
  _layers.emplace_back(std::move(layer));
}

Layers::Layers(const std::vector<Layer>& layers) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _layers.reserve(layers.size());
  for (const auto& layer : layers)
  {
    auto copy_layer = Layer(layer);
    _layers.emplace_back(std::move(copy_layer));
  }
}

Layers::Layers(const Layers& layers) noexcept
 : _layers(layers._layers)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
}

Layers::Layers(Layers&& layers) noexcept
  : _layers(std::move(layers._layers))
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
}

Layers& Layers::operator=(const Layers& layers) noexcept
{
  if(this != &layers)
  {
    _layers = layers._layers;
  }
  return *this;
}

Layers& Layers::operator=(Layers&& layers) noexcept
{
  if(this != &layers)
  {
    _layers = std::move(layers._layers);
  }
  return *this;
}

const Layer& Layers::operator[](unsigned index ) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  #ifndef NDEBUG
  assert(index < _layers.size());
  #endif
  return _layers[index];
}

Layer& Layers::operator[](unsigned index )
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  #ifndef NDEBUG
  assert(index < _layers.size());
  #endif
  return _layers[index];
}

int Layers::residual_layer_number(unsigned index) const
{
  #ifndef NDEBUG
  assert(index < _layers.size());
  #endif
  return _layers[index].residual_layer_number();
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

void Layers::add_residual_layer(Layer& layer, const activation::method& activation_method) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  auto residual_layer_number = layer.residual_layer_number();
  if( -1 == residual_layer_number)
  {
    return;
  }
  
  auto number_of_neuron_in_that_layer = _layers[residual_layer_number].number_neurons();
  auto num_neurons_current_layer = layer.number_neurons();
  auto residual_projector = new Layer::ResidualProjector(number_of_neuron_in_that_layer, num_neurons_current_layer, activation_method);

  // pass ownership
  layer.move_residual_projector(residual_projector);
}