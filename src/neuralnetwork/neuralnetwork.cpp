#include "neuralnetwork.h"
#include <cassert>

static const double RecentAverageSmoothingFactor = 100.0;

NeuralNetwork::NeuralNetwork(
  const std::vector<unsigned>& topology, 
  const activation::method& activation) :
  _layers(nullptr),
  _activation_method(activation)
{
  const auto& number_of_layers = topology.size();
  _layers = new std::vector<Neuron::Layer>();
  for (auto layer_number = 0; layer_number < number_of_layers; ++layer_number)
  {
    auto number_of_outputs = layer_number == topology.size() - 1 ? 0 : topology[layer_number + 1];
    auto layer = Neuron::Layer();

    // We have a new layer, now fill it with neurons, and add a bias neuron in each layer.
    for (unsigned neuronNum = 0; neuronNum <= topology[layer_number]; ++neuronNum)
    {
      // force the bias node's output to 1.0
      auto neuron = Neuron(number_of_outputs, neuronNum, activation);
      neuron.set_output_value(1.0);
      layer.push_back(neuron);
    }
    _layers->push_back(layer);
  }
}

NeuralNetwork::~NeuralNetwork()
{
  delete _layers;
}

const std::vector<Neuron::Layer>& NeuralNetwork::get_layers() const
{
  return *_layers;
}

std::vector<unsigned> NeuralNetwork::get_topology() const
{
  std::vector<unsigned> topology = {};
  for(const auto& layer : *_layers)
  {
    // remove the bias Neuron.
    topology.push_back(layer.size() -1);
  }
  return topology;
}

std::vector<double> NeuralNetwork::think(
  const std::vector<double>& inputs
) const
{
  std::vector<double> outputs;
  auto layers = *_layers;
  forward_feed(inputs, layers);
  get_outputs(outputs, layers);
  return outputs;
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
  int number_of_epoch,
  const std::function<void(int, double)>& progress_callback
)
{
  auto percent = 0;
  // initial callback
  if (progress_callback != nullptr)
  {
    progress_callback(0, 0.0);
  }
  auto error = 0.0;
  for (auto i = 0; i < number_of_epoch; ++i)
  {
    for (auto j = 0; j < training_inputs.size(); ++j)
    {
      const auto& inputs = training_inputs[j];
      const auto& outputs = training_outputs[j];

      forward_feed(inputs, *_layers);
      error = back_propagation(outputs, *_layers, error);
    }

    if (progress_callback != nullptr)
    {
      auto this_percent = (int)(((float)i / number_of_epoch)*100);
      if (this_percent != percent && percent != 100)
      {
        percent = this_percent;
        progress_callback(percent, error);
      }
    }
  }

  // final callback if needed
  if (progress_callback != nullptr && 100 != percent)
  {
    progress_callback(100, error);
  }
}

double NeuralNetwork::back_propagation(
  const std::vector<double>& targetVals, 
  std::vector<Neuron::Layer>& layers_src,
  const double current_recent_average_error
) const
{
  auto& output_layer = layers_src.back();
  auto error = 0.0;

  for (unsigned n = 0; n < output_layer.size() - 1; ++n)
  {
    auto delta = targetVals[n] - output_layer[n].get_output_value();
    error += delta * delta;
  }
  error /= output_layer.size() - 1; // get average error squared
  error = sqrt(error); // RMS

  // Implement a recent average measurement
  auto recent_average_error =
    (current_recent_average_error * RecentAverageSmoothingFactor + error)
    / (RecentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradients

  for (auto n = 0; n < output_layer.size() - 1; ++n)
  {
    output_layer[n].calculate_output_gradients(targetVals[n]);
  }

  // Calculate hidden layer gradients

  for (auto layer_number = layers_src.size() - 2; layer_number > 0; --layer_number)
  {
    auto& hidden_layer = layers_src[layer_number];
    auto& next_layer = layers_src[layer_number + 1];

    for (auto n = 0; n < hidden_layer.size(); ++n)
    {
      hidden_layer[n].calculate_hidden_gradients(next_layer);
    }
  }

  // update the hidden layers
  for (auto layerNum = layers_src.size() - 1; layerNum > 0; --layerNum) 
  {
    auto& layer = layers_src[layerNum];
    auto& prevLayer = layers_src[layerNum - 1];

    for (unsigned n = 0; n < layer.size() - 1; ++n) 
    {
      layer[n].update_input_weights(prevLayer);
    }
  }
  return recent_average_error;
}

void NeuralNetwork::forward_feed(
  const std::vector<double>& inputVals,
  std::vector<Neuron::Layer>& layers
) const
{
  // Assign (latch) the input values into the input neurons
  for (auto i = 0; i < inputVals.size(); ++i)
  {
    layers[0][i].set_output_value(inputVals[i]);
  }

  // forward propagate
  for (auto layer_number = 1; layer_number < layers.size(); ++layer_number)
  {
    const auto& previous_layer = layers[layer_number - 1];
    for (auto n = 0; n < layers[layer_number].size() - 1; ++n)
    {
      layers[layer_number][n].forward_feed(previous_layer);
    }
  }
}

void NeuralNetwork::get_outputs(std::vector<double>& outputs, const std::vector<Neuron::Layer>& layers) const
{
  outputs.erase(outputs.begin(), outputs.end());
  const auto& back_layer = layers.back();
  for (unsigned n = 0; n < back_layer.size() - 1; ++n)
  {
    outputs.push_back(back_layer[n].get_output_value());
  }
}