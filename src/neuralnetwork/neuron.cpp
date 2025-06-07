#include "neuron.h"

#include <cassert>
#include <cmath>
#include <iostream>

#define GRADIENT_CLIP double(1.0)

Neuron::Neuron(
  unsigned num_neurons_next_layer,
  unsigned num_neurons_current_layer,
  unsigned index,
  const activation& activation
) :
  _index(index),
  _output_value(0),
  _activation_method(activation),
  _output_weights({}),
  _alpha(LEARNING_ALPHA)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  auto weights = _activation_method.weight_initialization(num_neurons_next_layer, num_neurons_current_layer);
  for (auto weight : weights)
  {
    _output_weights.push_back(Connection(weight, 0.0));
  }
}

Neuron::Neuron(
  unsigned index,
  double output_value,
  const activation& activation,
  const std::vector<std::array<double,2>>& output_weights
) :
  _index(index),
  _output_value(output_value),
  _activation_method(activation),
  _output_weights({}),
  _alpha(LEARNING_ALPHA)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  for (auto& weights : output_weights)
  {
    auto connection = Connection(weights[0], weights[1]);
    _output_weights.push_back(connection);
  }
}

Neuron::Neuron(const Neuron& src) : 
  _index(src._index),
  _output_value(src._output_value),
  _activation_method(src._activation_method),
  _output_weights({}),
  _alpha(LEARNING_ALPHA)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  _output_weights = src._output_weights;
}

const Neuron& Neuron::operator=(const Neuron& src)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _output_value = src._output_value;
    _activation_method = src._activation_method;
    _output_weights = src._output_weights;
  }
  return *this;
}

Neuron::~Neuron()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  Clean();
}

std::vector<std::array<double, 2>> Neuron::get_weights() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  std::vector<std::array<double, 2>> weights;
  for(const auto& output_weight : _output_weights)
  {
    weights.push_back({output_weight.weight(), output_weight.delta_weight()});
  }
  return weights;
}

void Neuron::Clean()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
}

void Neuron::update_input_weights(Layer& previous_layer, const std::vector<double>& weights_gradients, double learning_rate)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  assert(weights_gradients.size() == previous_layer.size());
  for (size_t i = 0; i < weights_gradients.size(); ++i) 
  {
    auto& neuron = previous_layer.get_neuron(static_cast<unsigned>(i));
    auto& connection = neuron._output_weights[_index];

    const auto& weights_gradient = weights_gradients[i];         // from prev layer, averaged over batch
    if (!std::isfinite(weights_gradient))
    {
      std::cout << "Error while calculating input weigh gradient it invalid." << std::endl;
      throw std::invalid_argument("Error while calculating input weight.");
    }
    auto old_delta_weight = connection.delta_weight();
    if (!std::isfinite(old_delta_weight))
    {
      old_delta_weight = 0.0;
      std::cout << "Error while calculating input weigh old weight is invalid." << std::endl;
      throw std::invalid_argument("Error while calculating input weigh old weight is invalid.");
    }

    double new_delta_weight =
      learning_rate * weights_gradient +   // batch-based weight update
      _alpha * old_delta_weight;            // momentum term

    connection.set_delta_weight(new_delta_weight);
    connection.set_weight(connection.weight() + new_delta_weight);
  }
}

double Neuron::get_output_weight(int index) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _output_weights[index].weight();
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
    std::cout << "Error while calculating sum of the derivatives of the weights." << std::endl;
    throw std::invalid_argument("Error while calculating sum of the derivatives of the weights.");
    return std::numeric_limits<double>::quiet_NaN();
  }
  return sum;
}

double Neuron::clip_gradient(double val, double clip_val) 
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return std::max(-clip_val, std::min(clip_val, val));
}

double Neuron::calculate_output_gradients(double target_value, double output_value) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  double delta = target_value - output_value;
  auto gradient = delta * _activation_method.activate_derivative(output_value);
  if (!std::isfinite(gradient))
  {
    std::cout << "Error while calculating output gradients." << std::endl;
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
  gradient = clip_gradient(gradient, GRADIENT_CLIP);
  if (!std::isfinite(gradient))
  {
    std::cout << "Error while calculating hidden gradients." << std::endl;
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
    std::cout << "Error while calculating output values." << std::endl;
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
    const auto output_value  = previous_layer_output_values[neuron_index];
    sum +=  output_value * output_weight;
    if (!std::isfinite(sum))
    {
      std::cout << "Error while calculating forward feed." << std::endl;
      throw std::invalid_argument("Error while calculating forward feed.");
      return std::numeric_limits<double>::quiet_NaN();
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