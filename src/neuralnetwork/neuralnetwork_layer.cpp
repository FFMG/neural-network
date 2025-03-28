#include "neuralnetwork_layer.h"
#include <cassert>

NeuralNetworkLayer::NeuralNetworkLayer(
  const std::vector<unsigned>& topology, 
  const activation::method& activation) :
  _layers(nullptr),
  _activation_method(activation)
{
  unsigned numLayers = topology.size();
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
    (*_layers).back().back().setOutputVal(1.0);
  }
}

NeuralNetworkLayer::~NeuralNetworkLayer()
{
  delete _layers;
}

std::vector<double> NeuralNetworkLayer::think(
  const std::vector<double>& inputs
) const
{
  auto layers = *_layers;
  forward_feed(inputs, layers);

  std::vector<double> resultVals;
  for (unsigned n = 0; n < layers.back().size() - 1; ++n) {
    resultVals.push_back(layers.back()[n].getOutputVal());
  }
  return resultVals;
}

void NeuralNetworkLayer::train(
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

void NeuralNetworkLayer::back_propagation(const std::vector<double>& targetVals, std::vector<Neuron::Layer>& layers_src) const
{
  // Calculate overall net error (RMS of output neuron errors)

  auto& outputLayer = layers_src.back();
  auto error = 0.0;

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    error += delta * delta;
  }
  error /= outputLayer.size() - 1; // get average error squared
  error = sqrt(error); // RMS

  // Calculate output layer gradients

  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) 
  {
    outputLayer[n].calcOutputGradients(targetVals[n]);
  }

  // Calculate hidden layer gradients

  for (unsigned layerNum = layers_src.size() - 2; layerNum > 0; --layerNum) {
    auto& hiddenLayer = layers_src[layerNum];
    auto& nextLayer = layers_src[layerNum + 1];

    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calcHiddenGradients(nextLayer);
    }
  }

  // update the hidden layers
  for (unsigned layerNum = layers_src.size() - 1; layerNum > 0; --layerNum) {
    auto& layer = layers_src[layerNum];
    auto& prevLayer = layers_src[layerNum - 1];

    for (unsigned n = 0; n < layer.size() - 1; ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void NeuralNetworkLayer::forward_feed(const std::vector<double>& inputVals, std::vector<Neuron::Layer>& layers) const
{
  // Assign (latch) the input values into the input neurons
  for (unsigned i = 0; i < inputVals.size(); ++i) {
    layers[0][i].setOutputVal(inputVals[i]);
  }

  // forward propagate
  for (unsigned layerNum = 1; layerNum < layers.size(); ++layerNum) {
    auto& prevLayer = layers[layerNum - 1];
    for (unsigned n = 0; n < layers[layerNum].size() - 1; ++n) {
      layers[layerNum][n].forward_feed(prevLayer);
    }
  }

  std::vector<double> resultVals;
  for (unsigned n = 0; n < layers.back().size() - 1; ++n) {
    resultVals.push_back(layers.back()[n].getOutputVal());
  }
}