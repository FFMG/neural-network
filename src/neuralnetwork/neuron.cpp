#include "neuron.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

Neuron::Neuron(
  unsigned num_neurons_next_layer,
  unsigned num_neurons_current_layer,
  unsigned index,
  const activation& activation,
  const Logger& logger
) :
  _index(index),
  _output_value(0),
  _activation_method(activation),
  _weight_params({}),
  _alpha(LEARNING_ALPHA),
  _logger(logger)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  auto weights = _activation_method.weight_initialization(num_neurons_next_layer, num_neurons_current_layer);
  for (auto weight : weights)
  {
    _weight_params.push_back(WeightParam(weight, 0.0, logger));
  }
}

Neuron::Neuron(
  unsigned index,
  double output_value,
  const activation& activation,
  const std::vector<std::array<double,2>>& output_weights,
  const Logger& logger
) :
  _index(index),
  _output_value(output_value),
  _activation_method(activation),
  _weight_params({}),
  _alpha(LEARNING_ALPHA),
  _logger(logger)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  _weight_params.reserve(output_weights.size());
  for (auto& weights : output_weights)
  {
    auto weightParam = WeightParam(weights[0], weights[1], logger);
    _weight_params.emplace_back(std::move(weightParam));
  }
}

Neuron::Neuron(const Neuron& src)  noexcept : 
  _index(src._index),
  _output_value(src._output_value),
  _activation_method(src._activation_method),
  _weight_params({}),
  _alpha(LEARNING_ALPHA),
  _logger(src._logger)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  _weight_params = src._weight_params;
}

Neuron& Neuron::operator=(const Neuron& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _output_value = src._output_value;
    _activation_method = src._activation_method;
    _weight_params = src._weight_params;
    _logger = src._logger;
  }
  return *this;
}

Neuron::Neuron(Neuron&& src) noexcept :
  _index(src._index),
  _output_value(src._output_value),
  _activation_method(src._activation_method),
  _alpha(LEARNING_ALPHA),
  _logger(src._logger)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  _weight_params = std::move(src._weight_params);
}

Neuron& Neuron::operator=(Neuron&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _output_value = src._output_value;
    _activation_method = src._activation_method;
    _weight_params = std::move(src._weight_params);
    _logger = src._logger;

    src._output_value = 0;
    src._index = 0;
  }
  return *this;
}

Neuron::~Neuron()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  Clean();
}

std::vector<std::array<double, 2>> Neuron::get_weight_params() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  std::vector<std::array<double, 2>> weights;
  for(const auto& output_weight : _weight_params)
  {
    weights.push_back({output_weight.value(), output_weight.gradient()});
  }
  return weights;
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

void Neuron::update_input_weights(Layer& previous_layer, const std::vector<double>& weights_gradients, double learning_rate)
{
  const double gradient_clip_threshold = 1.0;
  const double weight_decay_factor = 0.0001;

  assert(weights_gradients.size() == previous_layer.size());
  for (size_t i = 0; i < weights_gradients.size(); ++i)
  {
    auto& neuron = previous_layer.get_neuron(static_cast<unsigned>(i));
    auto& WeightParam = neuron._weight_params[_index];

    const auto& weights_gradient = weights_gradients[i];         // from prev layer, averaged over batch
    if (!std::isfinite(weights_gradient))
    {
      _logger.log_error("Error while calculating input weigh gradient it invalid.");
      throw std::invalid_argument("Error while calculating input weight.");
    }
    auto old_delta_weight = WeightParam.gradient();
    if (!std::isfinite(old_delta_weight))
    {
      old_delta_weight = 0.0;
      _logger.log_error("Error while calculating input weigh old weight is invalid.");
      throw std::invalid_argument("Error while calculating input weigh old weight is invalid.");
    }

    double weight_decay = 0.0;
    double clipped_gradient = weights_gradient;
    if (clipped_gradient > gradient_clip_threshold)
    {
      clipped_gradient = gradient_clip_threshold;
      weight_decay = weight_decay_factor;
    }
    else if (clipped_gradient < -gradient_clip_threshold)
    {
      clipped_gradient = -gradient_clip_threshold;
      weight_decay = weight_decay_factor;
    }

    double new_delta_weight =
      learning_rate * clipped_gradient +   // batch-based weight update
      _alpha * old_delta_weight;            // momentum term

    WeightParam.set_gradient(new_delta_weight);

    double current_weight = WeightParam.value();
    double new_weight = current_weight * (1.0 - weight_decay) + new_delta_weight;
    WeightParam.set_value(new_weight);
  }
}

double Neuron::get_output_weight(int index) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _weight_params[index].value();
}

double Neuron::sum_of_derivatives_of_weights(const Layer& next_layer, const std::vector<double>& activation_gradients) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  double sum = 0.0;
  assert(activation_gradients.size() == next_layer.size());
  for (unsigned n = 0; n < next_layer.size() - 1; ++n) 
  {
    auto weights_and_gradients = get_output_weight(n) * activation_gradients[n];
    sum += std::isinf(weights_and_gradients) ? std::numeric_limits<double>::infinity() : weights_and_gradients;
  }
  if (!std::isfinite(sum))
  {
    _logger.log_error("Error while calculating sum of the derivatives of the weights.");
    throw std::invalid_argument("Error while calculating sum of the derivatives of the weights.");
    return std::numeric_limits<double>::quiet_NaN();
  }
  return sum;
}

double Neuron::clip_gradient(double gradient) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  constexpr double GRADIENT_CLIP_THRESHOLD = 1.0;
  return std::clamp(gradient, -GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD);
}

double Neuron::calculate_output_gradients(double target_value, double output_value) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  double delta = target_value - output_value;
  auto gradient = delta * _activation_method.activate_derivative(output_value);
  gradient = clip_gradient(gradient);
  if (!std::isfinite(gradient))
  {
    _logger.log_error("Error while calculating output gradients.");
    throw std::invalid_argument("Error while calculating output gradients.");
    return std::numeric_limits<double>::quiet_NaN();
  }
  return gradient;
}

double Neuron::calculate_hidden_gradients(const Layer& next_layer, const std::vector<double>& activation_gradients, double output_value) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  auto derivatives_of_weights = sum_of_derivatives_of_weights(next_layer, activation_gradients);
  auto gradient = derivatives_of_weights * _activation_method.activate_derivative(output_value);
  gradient = clip_gradient(gradient);
  if (!std::isfinite(gradient))
  {
    _logger.log_error("Error while calculating hidden gradients.");
    throw std::invalid_argument("Error while calculating hidden gradients.");
    return std::numeric_limits<double>::quiet_NaN();
  }  
  return gradient;
}

void Neuron::set_output_value(double val) 
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (!std::isfinite(val))
  {
    _logger.log_error("Error while calculating output values.");
    throw std::invalid_argument("Error while calculating output values.");
    return;
  }
  _output_value = val;
}

double Neuron::get_output_value() const
{ 
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _output_value; 
}

double Neuron::calculate_forward_feed(const Layer& previous_layer, const std::vector<double>& previous_layer_output_values) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  double sum = 0.0;

  // Sum the previous layer's outputs (which are our inputs)
  // Include the bias node from the previous layer.
  assert(previous_layer_output_values.size() == previous_layer.size());
  for (unsigned neuron_index = 0; neuron_index < previous_layer.size(); ++neuron_index) 
  {
    const auto& previous_layer_neuron = previous_layer.get_neuron(neuron_index);
    const auto output_weight = previous_layer_neuron.get_output_weight(_index);
    if (std::abs(output_weight) > 1e5)
    {
      _logger.log_error("Exploding weight detected");
      throw std::runtime_error("Exploding weight detected");
    }
    const auto output_value  = previous_layer_output_values[neuron_index];
    sum +=  output_value * output_weight;
    if (!std::isfinite(sum))
    {
      _logger.log_error("Error while calculating forward feed.");
      throw std::invalid_argument("Error while calculating forward feed.");
    }
  }
  return _activation_method.activate(sum);
}

void Neuron::forward_feed(const Layer& previous_layer)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  // build the output values
  std::vector<double> previous_layer_output_values;
  previous_layer_output_values.reserve(previous_layer.size());
  for(auto neuron : previous_layer.get_neurons())
  {
    previous_layer_output_values.push_back(neuron.get_output_value());
  }
  set_output_value(calculate_forward_feed(previous_layer, previous_layer_output_values));
}