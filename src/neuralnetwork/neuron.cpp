#include "neuron.h"
#include <cmath>
#include <random>

#define LEARNING_RATE 0.05    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

Neuron::Neuron(
  unsigned numOutputs, 
  unsigned index,
  const activation::method& activation
) :
  _index(index),
  _output_value(0),
  _gradient(0),
  _activation_method(activation),
  _output_weights(nullptr),
  _learning_rate(LEARNING_RATE)
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
  double outputVal,
  double gradient,
  const activation::method& activation,
  const std::vector<Connection>& output_weights
) :
  _index(index),
  _output_value(outputVal),
  _gradient(gradient),
  _activation_method(activation),
  _output_weights(nullptr),
  _learning_rate(LEARNING_RATE)
{
  _output_weights = new std::vector<Connection>();
  for (auto& connection : output_weights)
  {
    _output_weights->push_back(connection);
  }
}

Neuron::Neuron(const Neuron& src) :
  Neuron(src._index, src._output_value, src._gradient, src._activation_method, *src._output_weights)
{
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
      // Individual input, magnified by the gradient and train rate:
      _learning_rate
      * neuron.get_output_value()
      * _gradient
      // Also add momentum = a fraction of the previous delta weight;
      + alpha
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
  }

  set_output_value( activation::activate(_activation_method, sum) );
}

