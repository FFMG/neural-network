#include "neuron.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include "logger.h"
#include <random>

// TODO remove all the no longer used values moved to layer.
Neuron::Neuron(
  unsigned /*num_neurons_prev_layer*/,
  unsigned num_neurons_current_layer,
  unsigned num_neurons_next_layer,
  unsigned index,
  const activation& activation,
  const OptimiserType& optimiser_type,
  const Type& type,
  const double dropout_rate
) :
  _index(index),
  _activation_method(activation),
  _optimiser_type(optimiser_type),
  _alpha(LEARNING_ALPHA),
  _type(type),
  _dropout_rate(dropout_rate),
  _recurrent_weight(0, 0.0, 0.0, 0.0, 0.0)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  auto weight = _activation_method.weight_initialization();

  // TODO set weigh_decay in the options and add it to the info output on start of training.
  //      does the recurent weight even need a decay?
  _recurrent_weight = WeightParam(weight, 0.0, 0.0, 0.0, 0.0);
}

Neuron::Neuron(
  unsigned index,
  const activation& activation,
  const OptimiserType& optimiser_type,
  const Type& type,
  const double dropout_rate,
  const double recurrent_weight
) :
  _index(index),
  _activation_method(activation),
  _optimiser_type(optimiser_type),
  _alpha(LEARNING_ALPHA),
  _type(type),
  _dropout_rate(dropout_rate),
  // TODO set weigh_decay in the options and add it to the info output on start of training.
  //      does the recurent weight even need a decay?
  _recurrent_weight(WeightParam(recurrent_weight, 0.0, 0.0, 0.0, 0.0))
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
}

Neuron::Neuron(const Neuron& src)  noexcept : 
  _index(src._index),
  _activation_method(src._activation_method),
  _optimiser_type(src._optimiser_type),
  _alpha(LEARNING_ALPHA),
  _type(src._type),
  _dropout_rate(src._dropout_rate),
  _recurrent_weight(src._recurrent_weight)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
}

Neuron& Neuron::operator=(const Neuron& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _activation_method = src._activation_method;
    _optimiser_type = src._optimiser_type;
    _type = src._type;
    _dropout_rate = src._dropout_rate;
    _recurrent_weight = src._recurrent_weight;
  }
  return *this;
}

Neuron::Neuron(Neuron&& src) noexcept :
  _index(src._index),
  _activation_method(src._activation_method),
  _optimiser_type(src._optimiser_type),
  _alpha(LEARNING_ALPHA),
  _type(src._type),
  _dropout_rate(src._dropout_rate),
  _recurrent_weight(src._recurrent_weight)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");

  src._optimiser_type = OptimiserType::None;
  src._index = 0;
  src._type = Neuron::Type::Normal;
}

Neuron& Neuron::operator=(Neuron&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _activation_method = src._activation_method;
    _optimiser_type = src._optimiser_type;
    _dropout_rate = src._dropout_rate;
    _type = src._type;
    _recurrent_weight = std::move(src._recurrent_weight);

    src._optimiser_type = OptimiserType::None;
    src._index = 0;
    src._dropout_rate = 0.0;
    src._type = Neuron::Type::Normal;
    src._recurrent_weight = WeightParam(0, 0.0, 0.0, 0.0, 0.0);
  }
  return *this;
}

Neuron::~Neuron()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  Clean();
}

void Neuron::Clean()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
}

unsigned Neuron::get_index() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _index;
}

void Neuron::apply_residual_weight_gradients(
  Layer& layer,
  Layer& residual_layer,
  const std::vector<double>& residual_outputs,
  const std::vector<double>& gradients,  // same as deltas for this layer
  double learning_rate,
  double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");

  const size_t residual_output_count = residual_outputs.size(); // includes bias
  assert(gradients.size() == residual_output_count);  

  for (size_t residual_source_index = 0; residual_source_index < residual_output_count; ++residual_source_index)
  {
    auto& current_layer_neuron = residual_layer.get_neuron(static_cast<unsigned>(residual_source_index));
    // auto& weight_param = current_layer_neuron._residual_weight_params[get_index()];
    auto& weight_param = layer.residual_weight_param(get_index(), static_cast<unsigned>(residual_source_index));

    const auto& gradient = gradients[residual_source_index];
    apply_weight_gradient(gradient, learning_rate, false, weight_param, clipping_scale);
  }
}

// TODO REMOVE THIS
void Neuron::apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (!std::isfinite(gradient))
  {
    Logger::error("Error while calculating input weigh gradient it invalid.");
    throw std::invalid_argument("Error while calculating input weight.");
  }
  auto old_velocity = weight_param.get_velocity();
  if (!std::isfinite(old_velocity))
  {
    Logger::error("Error while calculating input weigh old velocity is invalid.");
    throw std::invalid_argument("Error while calculating input weigh old velocity is invalid.");
  }

  if (clipping_scale < 0.0)
  {
    // If clipping scale is negative, we clip the gradient to a fixed range
    // This is useful for debugging or when we want to ensure gradients are not too large.
    Logger::warning("Clipping gradient to a fixed range.");
  }

  auto clipped_gradient = clipping_scale <= 0.0 ? Layer::clip_gradient(gradient) : gradient * clipping_scale;
  switch (_optimiser_type)
  {
  case OptimiserType::None:
    Layer::apply_none_update(weight_param, clipped_gradient, learning_rate);
    break;

  case OptimiserType::SGD:
    Layer::apply_sgd_update(weight_param, clipped_gradient, learning_rate, _activation_method.momentum(), is_bias);
    break;

  case OptimiserType::Adam:
    Layer::apply_adam_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias);
    break;

  case OptimiserType::AdamW:
    Layer::apply_adamw_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8);
    break;

  case OptimiserType::Nadam:
    Layer::apply_nadam_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8);
    break;

  case OptimiserType::NadamW:
    Layer::apply_nadamw_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias);
    break;

  default:
    throw std::runtime_error("Unknown optimizer type.");
  }
}

void Neuron::apply_weight_gradients(Layer& previous_layer, const std::vector<double>& gradients, const double learning_rate, unsigned /*epoch*/, double clipping_scale)
{
  // TODO this shou;d be removed.
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  throw new std::exception("Never called!");

  /*
  // we should not be doing anything with out weights.
  assert(!is_bias());
  assert(gradients.size() == previous_layer.number_neurons());
  for (unsigned neuron_number = 0; neuron_number < static_cast<unsigned>(gradients.size()); ++neuron_number)
  {
    auto& previous_layer_neuron = previous_layer.get_neuron(neuron_number);
    auto& weight_param = previous_layer.get_weight_params(neuron_number, get_index());

    const auto& gradient = gradients[neuron_number];         // from prev layer, averaged over batch
    apply_weight_gradient(gradient, learning_rate, previous_layer_neuron.is_bias(), weight_param, clipping_scale);

    if (neuron_number == get_index())
    {
      apply_weight_gradient(gradient, learning_rate, false, previous_layer_neuron._recurrent_weight, clipping_scale);
    }
  }
  */
}

double Neuron::get_dropout_rate() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  assert(_type == Neuron::Type::Dropout);
  return _dropout_rate;
}

bool Neuron::must_randomly_drop() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  assert(_type == Neuron::Type::Dropout);
  static thread_local std::mt19937 rng(std::random_device{}());
  std::bernoulli_distribution drop(1.0 - get_dropout_rate());
  return !drop(rng);  // true means keep, false means drop
}

const OptimiserType& Neuron::get_optimiser_type() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _optimiser_type; 
}

bool Neuron::is_dropout() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _type == Neuron::Type::Dropout;
}

const Neuron::Type& Neuron::get_type() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _type;
}

const WeightParam& Neuron::get_recurrent_weight() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _recurrent_weight;
}
