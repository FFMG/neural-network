#include "neuralnetwork.h"
#include <cassert>

NeuralNetwork::NeuralNetwork(
  const std::vector<unsigned>& topology, 
  const activation::method& activation) :
  _layers(nullptr),
  _activation_method(activation)
{
  const auto& numLayers = topology.size();
  _layers = new std::vector<Neuron::Layer>();
  for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) 
  {
    _layers->push_back(Neuron::Layer());
    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

    // We have a new layer, now fill it with neurons, and
    // add a bias neuron in each layer.
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) 
    {
      _layers->back().push_back(Neuron(numOutputs, neuronNum, activation));
    }

    // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
    (*_layers).back().back().set_output_value(1.0);
  }
}

NeuralNetwork::~NeuralNetwork()
{
  delete _layers;
}

std::vector<double> NeuralNetwork::think(
  const std::vector<double>& inputs
) const
{
  auto layers = *_layers;
  forward_feed(inputs, layers);

  std::vector<double> resultVals;
  const auto& back_layer = layers.back();
  for (unsigned n = 0; n < back_layer.size() - 1; ++n)
  {
    resultVals.push_back(back_layer[n].get_output_value());
  }
  return resultVals;
}

std::vector<std::vector<double>> NeuralNetwork::think(
  const std::vector<std::vector<double>>& inputs
) const
{
  std::vector<std::vector<double>> outputs(inputs.size(), std::vector<double>(1));
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    outputs[i] = think(inputs[i]);
  }
  return outputs;
}

void NeuralNetwork::train(
  const std::vector<std::vector<double>>& training_inputs,
  const std::vector<std::vector<double>>& training_outputs,
  int number_of_epoch)
{
  for (auto i = 0; i < number_of_epoch; ++i)
  {
    for (auto j = 0; j < training_inputs.size(); ++j)
    {
      const auto& inputs = training_inputs[j];
      const auto& outputs = training_outputs[j];

      forward_feed(inputs, *_layers);
      back_propagation(outputs, *_layers);
    }
  }
}

void NeuralNetwork::back_propagation(const std::vector<double>& targetVals, std::vector<Neuron::Layer>& layers_src) const
{
  // Calculate overall net error (RMS of output neuron errors)

  auto& outputLayer = layers_src.back();
  auto error = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) 
  {
    double delta = targetVals[n] - outputLayer[n].get_output_value();
    error += delta * delta;
  }
  error /= outputLayer.size() - 1; // get average error squared
  error = sqrt(error); // RMS

  // Calculate output layer gradients

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) 
  {
    outputLayer[n].calculate_output_gradients(targetVals[n]);
  }

  // Calculate hidden layer gradients

  for (auto layerNum = layers_src.size() - 2; layerNum > 0; --layerNum) {
    auto& hiddenLayer = layers_src[layerNum];
    auto& nextLayer = layers_src[layerNum + 1];

    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calculate_hidden_gradients(nextLayer);
    }
  }

  // update the hidden layers
  for (auto layerNum = layers_src.size() - 1; layerNum > 0; --layerNum) {
    auto& layer = layers_src[layerNum];
    auto& prevLayer = layers_src[layerNum - 1];

    for (unsigned n = 0; n < layer.size() - 1; ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void NeuralNetwork::forward_feed(const std::vector<double>& inputVals, std::vector<Neuron::Layer>& layers) const
{
  // Assign (latch) the input values into the input neurons
  for (auto i = 0; i < inputVals.size(); ++i) 
  {
    layers[0][i].set_output_value(inputVals[i]);
  }

  // forward propagate
  for (auto layerNum = 1; layerNum < layers.size(); ++layerNum) 
  {
    const auto& prevLayer = layers[layerNum - 1];
    for (auto n = 0; n < layers[layerNum].size() - 1; ++n) 
    {
      layers[layerNum][n].forward_feed(prevLayer);
    }
  }

  std::vector<double> resultVals;
  const auto& back_layer = layers.back();
  for (unsigned n = 0; n < back_layer.size() - 1; ++n)
  {
    resultVals.push_back(back_layer[n].get_output_value());
  }
}