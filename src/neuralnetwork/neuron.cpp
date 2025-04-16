#include "neuron.h"
#include <cmath>
#include <random>

Neuron::Neuron(
  unsigned numOutputs, 
  unsigned index,
  const activation::method& activation,
  double learning_rate
) :
  _index(index),
  _output_value(0),
  _gradient(0),
  _activation_method(activation),
  _output_weights(nullptr),
  _learning_rate(learning_rate),
  _alpha(LEARNING_ALPHA)
{
  _output_weights = new std::vector<Connection>();
  auto weights = he_initialization(numOutputs);
  for (auto weight : weights)
  {
    _output_weights->push_back(Connection( weight));
  }
}

Neuron::Neuron(
  unsigned index,
  double output_value,
  double gradient,
  const activation::method& activation,
  const std::vector<std::array<double,2>>& output_weights,
  double learning_rate
) :
  _index(index),
  _output_value(output_value),
  _gradient(gradient),
  _activation_method(activation),
  _output_weights(nullptr),
  _learning_rate(learning_rate),
  _alpha(LEARNING_ALPHA)
{
  _output_weights = new std::vector<Connection>();
  for (auto& weights : output_weights)
  {
    auto connection = Connection(weights[0]);
    connection.set_delta_weight(weights[1]);
    _output_weights->push_back(connection);
  }
}

Neuron::Neuron(const Neuron& src) : 
  _index(src._index),
  _output_value(src._output_value),
  _gradient(src._gradient),
  _activation_method(src._activation_method),
  _output_weights(nullptr),
  _learning_rate(src._learning_rate),
  _alpha(LEARNING_ALPHA)
{
  _output_weights = new std::vector<Connection>();
  for (auto& connection : *src._output_weights)
  {
    _output_weights->push_back(connection);
  }
}

const Neuron& Neuron::operator=(const Neuron& src)
{
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _output_value = src._output_value;
    _gradient = src._gradient;
    _activation_method = src._activation_method;

    _output_weights = new std::vector<Connection>();
    if (src._output_weights != nullptr)
    {
      for (auto& connection : *src._output_weights)
      {
        _output_weights->push_back(connection);
      }
    }
  }
  return *this;
}

Neuron::~Neuron()
{
  Clean();
}

std::vector<std::array<double, 2>> Neuron::get_weights() const
{
  std::vector<std::array<double, 2>> weights;
  if(nullptr == _output_weights)
  {
    return weights;
  }
  for(const auto& output_weight : *_output_weights)
  {
    weights.push_back({output_weight.weight(), output_weight.delta_weight()});
  }
  return weights;
}

std::vector<double> Neuron::he_initialization(int num_neurons_prev_layer) 
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / num_neurons_prev_layer));

  std::vector<double> weights(num_neurons_prev_layer);
  for (double& w : weights) {
    w = dist(gen);  // Initialize weights
  }
  return weights;
}

void Neuron::Clean()
{
  delete _output_weights;
  _output_weights = nullptr;
}

void Neuron::update_input_weights(Layer& previous_layer)
{
  for (auto& neuron : previous_layer)
  {
    auto& connection = neuron._output_weights->at(_index);
    const auto& old_delta_weight = connection.delta_weight();

    auto new_delta_weight =
      _learning_rate  // Individual input, magnified by the gradient and train rate:
      * neuron.get_output_value()
      * _gradient
      + _alpha  // momentum = a fraction of the previous delta weight;
      * old_delta_weight;

    connection.set_delta_weight( new_delta_weight );
    connection.set_weight(connection.weight() + new_delta_weight);
  }
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
  double sum = 0.0;

  // Sum our contributions of the errors at the nodes we feed.

  for (unsigned n = 0; n < nextLayer.size() - 1; ++n) 
  {
    auto weights_and_gradients = _output_weights->at(n).weight() * nextLayer[n]._gradient;
    sum += std::isinf(weights_and_gradients) ? 0 : weights_and_gradients;
  }
  if (!std::isfinite(sum))
  {
    return 0.0;
  }
  return sum;
}

void Neuron::calculate_hidden_gradients(const Layer& nextLayer)
{
  double dow = sumDOW(nextLayer);
  auto gradient = dow * activation::activate_derivative(_activation_method, get_output_value());
  if (!std::isfinite(gradient))
  {
    return;
  }
  _gradient = gradient;
}

void Neuron::set_output_value(double val) 
{
  if (!std::isfinite(val))
  {
    return;
  }
  _output_value = val;
}

double Neuron::get_output_value() const
{ 
  return _output_value; 
}

void Neuron::calculate_output_gradients(double targetVal)
{
  double delta = targetVal - get_output_value();
  auto gradient = delta * activation::activate_derivative(_activation_method, get_output_value());
  if (!std::isfinite(gradient))
  {
    return;
  }
  _gradient = gradient;
}

void Neuron::forward_feed(const Layer& prevLayer)
{
  double sum = 0.0;

  // Sum the previous layer's outputs (which are our inputs)
  // Include the bias node from the previous layer.

  for (const auto& layer : prevLayer) 
  {
    const auto weight = layer._output_weights->at(_index).weight();
    sum += layer.get_output_value() * weight;

    if (!std::isfinite(sum))
    {
      return;
    }
  }

  set_output_value( activation::activate(_activation_method, sum) );
}

