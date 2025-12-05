#include "layers.h"
#include "logger.h"
#include "fflayer.h"
#include "elmanrnnlayer.h"
#include <cassert>

Layers::Layers(
  const std::vector<unsigned>& topology,
  double weight_decay,
  const std::vector<unsigned>& recurrent_layers,
  const std::vector<double>& dropout_layers,
  const activation::method& hidden_activation,
  const activation::method& output_activation,
  const OptimiserType& optimiser_type,
  int residual_layer_jump) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  assert(dropout_layers.size() == topology.size() -2 && "Dropout layers size must match the number of hidden layers");
  assert(recurrent_layers.size() == topology.size() && "The recurrence layer size must match the topology");
  const auto& number_of_layers = topology.size();
  _layers.reserve(number_of_layers);
  _residual_layer_numbers.reserve(number_of_layers);

  // add the input layer
  auto layer = create_input_layer(topology[0], topology[1], weight_decay);
  _layers.emplace_back(std::move(layer));
  _residual_layer_numbers.push_back(-1); // No residual for input layer

  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    auto dropout_rate = dropout_layers[layer_number-1]; // remove input
    const auto& previous_layer = *_layers.back();
    const auto residual_layer_number = compute_residual_layer(static_cast<int>(layer_number), residual_layer_jump);
    _residual_layer_numbers.push_back(residual_layer_number);

    // TODO: Use recurrent_layers to decide between FFLayer and ElmanRNNLayer
    layer = create_hidden_layer(num_neurons_current_layer, num_neurons_next_layer, weight_decay, previous_layer, hidden_activation, optimiser_type, residual_layer_number, dropout_rate);

    // add_residual_layer(*layer, hidden_activation); // TODO: Re-implement for BaseLayer
    _layers.emplace_back(std::move(layer));
  }

  // finally, the output layer
  const auto residual_layer_number = compute_residual_layer(static_cast<int>(number_of_layers)-1, residual_layer_jump);
  _residual_layer_numbers.push_back(residual_layer_number);
  layer = create_output_layer(topology.back(), weight_decay, *_layers.back(), output_activation, optimiser_type, residual_layer_number);

  // add_residual_layer(*layer, output_activation); // TODO: Re-implement for BaseLayer
  _layers.emplace_back(std::move(layer));
}

Layers::~Layers() = default;

Layers::Layers(const Layers& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _layers.reserve(src._layers.size());
  for(const auto& layer : src._layers)
  {
    _layers.emplace_back(std::unique_ptr<BaseLayer>(layer->clone()));
  }
  _residual_layer_numbers = src._residual_layer_numbers;
}

Layers::Layers(Layers&& layers) noexcept = default;

Layers& Layers::operator=(const Layers& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if(this != &src)
  {
    _layers.clear();
    _layers.reserve(src._layers.size());
    for(const auto& layer : src._layers)
    {
      _layers.emplace_back(std::unique_ptr<BaseLayer>(layer->clone()));
    }
    _residual_layer_numbers = src._residual_layer_numbers;
  }
  return *this;
}

Layers& Layers::operator=(Layers&& layers) noexcept = default;

const BaseLayer& Layers::operator[](unsigned index ) const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  #ifndef NDEBUG
  assert(index < _layers.size());
  #endif
  return *_layers[index];
}

BaseLayer& Layers::operator[](unsigned index )
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

const std::vector<std::unique_ptr<BaseLayer>>& Layers::get_layers() const
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return _layers;
}

std::vector<std::unique_ptr<BaseLayer>>& Layers::get_layers()
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

void Layers::add_residual_layer(BaseLayer& layer, const activation::method& activation_method) const
{
  // TODO: Re-implement this for the new BaseLayer architecture.
  // The old implementation used Layer::ResidualProjector which is not part of BaseLayer.
  MYODDWEB_PROFILE_FUNCTION("Layers");
}

std::unique_ptr<BaseLayer> Layers::create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  // The private constructor for FFLayer is being used here.
  // This is allowed because 'Layers' is a friend class of 'FFLayer'.
  return std::make_unique<FFLayer>(0, 0, num_neurons_in_this_layer, num_neurons_in_next_layer, weight_decay, BaseLayer::LayerType::Input, activation::method::linear, OptimiserType::None, 0.0);
}

std::unique_ptr<BaseLayer> Layers::create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay, const BaseLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number, double dropout_rate)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return std::make_unique<FFLayer>(previous_layer.get_layer_index() + 1, previous_layer.number_neurons(), num_neurons_in_this_layer, num_neurons_in_next_layer, weight_decay, BaseLayer::LayerType::Hidden, activation, optimiser_type, dropout_rate);
}

std::unique_ptr<BaseLayer> Layers::create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const BaseLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return std::make_unique<FFLayer>(previous_layer.get_layer_index() + 1, previous_layer.number_neurons(), num_neurons_in_this_layer, 0, weight_decay, BaseLayer::LayerType::Output, activation, optimiser_type, 0.0);
}
