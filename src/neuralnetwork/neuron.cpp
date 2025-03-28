#include "neuron.h"

double Neuron::eta = 0.15;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

Neuron::Neuron(
  unsigned numOutputs, 
  unsigned index,
  const activation::method& activation
) :
  _index(index),
  _outputVal(0),
  _gradient(0),
  _activation_method(activation),
  _output_weights(nullptr)
{
  _output_weights = new std::vector<Connection>();
  for (unsigned c = 0; c < numOutputs; ++c) 
  {
    _output_weights->push_back(Connection());
    _output_weights->back().weight = randomWeight();
  }
}

Neuron::Neuron(const Neuron& src) :
  _index(src._index),
  _outputVal(src._outputVal),
  _gradient(src._gradient),
  _activation_method(src._activation_method),
  _output_weights(nullptr)
{
  *this = src;
}

const Neuron& Neuron::operator=(const Neuron& src)
{
  if (this != &src)
  {
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
  delete _output_weights;
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
  // The weights to be updated are in the Connection container
  // in the neurons in the preceding layer

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    Neuron& neuron = prevLayer[n];
    double oldDeltaWeight = neuron._output_weights->at(_index).delta_weight;

    double newDeltaWeight =
      // Individual input, magnified by the gradient and train rate:
      eta
      * neuron.getOutputVal()
      * _gradient
      // Also add momentum = a fraction of the previous delta weight;
      + alpha
      * oldDeltaWeight;

    neuron._output_weights->at(_index).delta_weight = newDeltaWeight;
    neuron._output_weights->at(_index).weight += newDeltaWeight;
  }
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
  double sum = 0.0;

  // Sum our contributions of the errors at the nodes we feed.

  for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
    sum += _output_weights->at(n).weight * nextLayer[n]._gradient;
  }

  return sum;
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
  double dow = sumDOW(nextLayer);
  _gradient = dow * activation::activate_derivative(_activation_method, _outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
  double delta = targetVal - _outputVal;
  _gradient = delta * activation::activate_derivative(_activation_method, _outputVal);
}

void Neuron::forward_feed(const Layer& prevLayer)
{
  double sum = 0.0;

  // Sum the previous layer's outputs (which are our inputs)
  // Include the bias node from the previous layer.

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    sum += prevLayer[n].getOutputVal() *
      prevLayer[n]._output_weights->at(_index).weight;
  }

  _outputVal = activation::activate(_activation_method, sum);
}

