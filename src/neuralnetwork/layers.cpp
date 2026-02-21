#include "elmanrnnlayer.h"
#include "fflayer.h"
#include "grurnnlayer.h"
#include "layers.h"
#include "logger.h"
#include <cassert>

Layers::Layers(
  const std::vector<unsigned>& topology,
  const std::vector<LayerDetails>& hidden_layers,
  double weight_decay,
  const activation& output_activation,
  const OptimiserType& optimiser_type,
  int residual_layer_jump,
  int number_of_threads) noexcept :
  _weight_decay(weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA == 1
  if(hidden_layers.size() != topology.size() - 2)
  {
    Logger::panic("The topology size does not match the layer details size!");
  }
#endif

  const auto& number_of_layers = topology.size();
  _layers.reserve(number_of_layers);

  // add the input layer
  auto layer = create_input_layer(topology[0], _weight_decay, -1, number_of_threads);
  _layers.emplace_back(std::move(layer));

  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    const auto hidden_layer_number = layer_number - 1;
    const auto& layer_details = hidden_layers[hidden_layer_number];
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    auto dropout_rate = layer_details.get_dropout();
    const auto& previous_layer = *_layers.back();
    const auto residual_layer_number = compute_residual_layer(static_cast<int>(layer_number), residual_layer_jump);
    
    layer = create_hidden_layer(_weight_decay, previous_layer, optimiser_type, residual_layer_number, dropout_rate, layer_details, number_of_threads);

    _layers.emplace_back(std::move(layer));
  }

  // finally, the output layer
  const auto residual_layer_number = compute_residual_layer(static_cast<int>(number_of_layers)-1, residual_layer_jump);
  layer = create_output_layer(topology.back(), _weight_decay, *_layers.back(), output_activation, optimiser_type, residual_layer_number, number_of_threads);

  _layers.emplace_back(std::move(layer));
}

Layers::~Layers() = default;

Layers::Layers(const Layers& src) noexcept :
  _weight_decay(src._weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _layers.reserve(src._layers.size());
  for(const auto& layer : src._layers)
  {
    _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
  }
}

Layers::Layers(const std::vector<std::unique_ptr<Layer>>& layers, double weight_decay) noexcept :
  _weight_decay(weight_decay)
{
  _layers.reserve(layers.size());
  for (const auto& layer : layers)
  {
    _layers.emplace_back(std::unique_ptr<Layer>(layer->clone()));
  }
}

Layers::Layers(Layers&& src) noexcept :
  _layers(std::move(src._layers))
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  _weight_decay = src._weight_decay;
}

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
    _weight_decay = src._weight_decay;
  }
  return *this;
}

Layers& Layers::operator=(Layers&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if (this != &src)
  {
    _layers = std::move(src._layers);
    _weight_decay = src._weight_decay;
    src._weight_decay = 0.0;
  }
  return *this;
}

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
#if VALIDATE_DATA ==1
  assert(index < _layers.size());
#endif
  return *_layers[index];
}

const ResidualProjector* Layers::get_residual_layer_projector(unsigned index) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA ==1
  assert(index < _layers.size());
#endif
  return _layers[index]->get_residual_projector();
}

int Layers::get_residual_layer_number(unsigned index) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
#if VALIDATE_DATA ==1
  assert(index < _layers.size());
#endif
  return _layers[index]->get_residual_layer_number();
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

std::unique_ptr<Layer> Layers::create_input_layer(unsigned num_neurons_in_this_layer, double weight_decay, int residual_layer_number, int number_of_threads)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  return std::make_unique<FFLayer>(
    0, 
    0, 
    num_neurons_in_this_layer, 
    weight_decay, 
    Layer::LayerType::Input, 
    activation(activation::method::linear, 0.00),   //  Linear has no activation apha
    OptimiserType::None, 
    residual_layer_number, 
    0.0,    // no dropout for input layer
    nullptr, // no residual projector for input
    number_of_threads
  );
}

std::unique_ptr<Layer> Layers::create_hidden_layer(
  double weight_decay, 
  const Layer& previous_layer, 
  const OptimiserType& optimiser_type, 
  int residual_layer_number, 
  double dropout_rate, 
  const LayerDetails& layer_details,
  int number_of_threads)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  unsigned layer_index = previous_layer.get_layer_index() + 1;

  unsigned num_neurons_in_this_layer = layer_details.get_size();

  switch (layer_details.get_type())
  {
  case LayerDetails::LayerType::Elman:
    return std::make_unique<ElmanRNNLayer>(
      layer_index, 
      previous_layer.get_number_neurons(), 
      num_neurons_in_this_layer, 
      weight_decay, 
      Layer::LayerType::Hidden, 
      layer_details.get_activation(),
      optimiser_type, 
      residual_layer_number, 
      dropout_rate,
      create_residual_projector(layer_details.get_activation(), residual_layer_number, num_neurons_in_this_layer, _weight_decay),
      number_of_threads);

  case LayerDetails::LayerType::Gru:
    return std::make_unique<GRURNNLayer>(
      layer_index,
      previous_layer.get_number_neurons(),
      num_neurons_in_this_layer,
      weight_decay,
      Layer::LayerType::Hidden,
      layer_details.get_activation(),
      optimiser_type,
      residual_layer_number,
      dropout_rate,
      create_residual_projector(layer_details.get_activation(), residual_layer_number, num_neurons_in_this_layer, _weight_decay),
      number_of_threads);

  case LayerDetails::LayerType::FF:
    return std::make_unique<FFLayer>(
      layer_index, 
      previous_layer.get_number_neurons(),
      num_neurons_in_this_layer, 
      weight_decay, 
      Layer::LayerType::Hidden, 
      layer_details.get_activation(),
      optimiser_type, 
      residual_layer_number,
      dropout_rate,
      create_residual_projector(layer_details.get_activation(), residual_layer_number, num_neurons_in_this_layer, _weight_decay),
      number_of_threads);

  default:
    Logger::panic("Unknown or unsupported layer type!");
  }
}

std::unique_ptr<Layer> Layers::create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation& activation, const OptimiserType& optimiser_type, int residual_layer_number, int number_of_threads)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  unsigned layer_index = previous_layer.get_layer_index() + 1;
  return std::make_unique<FFLayer>(
    layer_index, 
    previous_layer.get_number_neurons(),
    num_neurons_in_this_layer, 
    weight_decay, 
    Layer::LayerType::Output, 
    activation, 
    optimiser_type, 
    residual_layer_number,
    0.0, // no dropout for output layer
    create_residual_projector(activation, residual_layer_number, num_neurons_in_this_layer, _weight_decay), 
    number_of_threads);
}

ResidualProjector* Layers::create_residual_projector(
  const activation& activation_method,
  int residual_layer_number,
  int number_of_neurons_in_current_layer,
  double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("Layers");
  if (residual_layer_number < 0)
  {
    return nullptr;
  }

  auto number_of_neurons_in_that_layer = _layers[residual_layer_number]->get_number_neurons();

  return ResidualProjector::create(
    residual_layer_number,
    activation_method, 
    number_of_neurons_in_that_layer, 
    number_of_neurons_in_current_layer, 
    weight_decay);
}
