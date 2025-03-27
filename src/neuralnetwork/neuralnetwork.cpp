#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(
  int number_of_inputs, 
  const activation::method& activation
) :
  _dis(nullptr),
  _gen(nullptr),
  _synaptic_weights(nullptr),
  _activation_method(activation)
{
  // seed random numbers to make calculation
  std::random_device rd;
  _gen = new std::mt19937(rd());
  _dis = new std::uniform_real_distribution<>(-1.0, 1.0);
  prepare_synaptic_weights(number_of_inputs);
}

NeuralNetwork::~NeuralNetwork()
{
  delete _gen;
  delete _dis;
  delete _synaptic_weights;
}

double NeuralNetwork::activation(double x) const
{
  switch (_activation_method)
  {
  case activation::relu_activation:
    return activation::relu(x);

  case activation::leakyRelu_activation:
    return activation::leakyRelu(x);

  case activation::tanh_activation:
    return activation::tanh(x);

  case activation::sigmoid_activation:
  default:
    return activation::sigmoid(x);
  }
}

double NeuralNetwork::activation_derivative(double x) const
{
  switch (_activation_method)
  {
  case activation::relu_activation:
    return activation::relu_derivative(x);

  case activation::leakyRelu_activation:
    return activation::leakyRelu_derivative(x);

  case activation::tanh_activation:
    return activation::tanh_derivative(x);

  case activation::sigmoid_activation:
  default:
    return activation::sigmoid_derivative(x);
  }
}

void NeuralNetwork::prepare_synaptic_weights(int number_of_inputs)
{
  delete _synaptic_weights;
  _synaptic_weights = nullptr;
  _synaptic_weights = new std::vector<double>(number_of_inputs, 0);

  for (int i = 0; i < number_of_inputs; ++i) 
  {
    (*_synaptic_weights)[i] = (*_dis)(*_gen);
  }
}

double NeuralNetwork::think(
  const std::vector<double>& inputs
) const
{
  double outputs = 0;
  double sum = 0.0;
  for (auto i = 0; i < inputs.size(); ++i) 
  {
    sum += inputs[i] * (*_synaptic_weights)[i];
  }
  outputs = activation(sum);
  return outputs;
}

std::vector<std::vector<double>> NeuralNetwork::think(
    const std::vector<std::vector<double>>& inputs
  ) const
{
  std::vector<std::vector<double>> outputs(inputs.size(), std::vector<double>(1));
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    outputs[i][0] = think(inputs[i]);
  }
  return outputs;
}

void NeuralNetwork::train(
  const std::vector<std::vector<double>>& training_inputs,
  const std::vector<std::vector<double>>& training_outputs,
  int number_of_epoch)
{
  // Iterate 10,000 times
  for (int iteration = 0; iteration < number_of_epoch; ++iteration)
  {
    // Define input layer
    std::vector<std::vector<double>> input_layer = training_inputs;

    std::vector<std::vector<double>> outputs(input_layer.size(), std::vector<double>(1));

    // Normalize the product of the input layer with the synaptic weights
    for (size_t i = 0; i < input_layer.size(); ++i) 
    {
      double sum = 0.0;
      for (size_t j = 0; j < input_layer[i].size(); ++j) {
        sum += input_layer[i][j] * (*_synaptic_weights)[j];
      }
      outputs[i][0] = activation(sum);
    }

    // how much did we miss?
    std::vector<std::vector<double>> error(training_outputs.size(), std::vector<double>(1));
    for (size_t i = 0; i < training_outputs.size(); ++i) {
      error[i][0] = training_outputs[i][0] - outputs[i][0];
    }

    // multiply how much we missed by the slope of the activation at the values in outputs
    std::vector<std::vector<double>> adjustments(outputs.size(), std::vector<double>(1));
    for (size_t i = 0; i < outputs.size(); ++i) {
      adjustments[i][0] = error[i][0] * activation_derivative(outputs[i][0]);
    }

    // update weights
    std::vector<std::vector<double>> input_layer_transpose(input_layer[0].size(), std::vector<double>(input_layer.size()));
    for (size_t i = 0; i < input_layer.size(); ++i) {
      for (size_t j = 0; j < input_layer[i].size(); ++j) {
        input_layer_transpose[j][i] = input_layer[i][j];
      }
    }

    std::vector<std::vector<double>> weight_adjustments((*_synaptic_weights).size(), std::vector<double>(1, 0.0));

    for (size_t i = 0; i < (*_synaptic_weights).size(); ++i) {
      for (size_t j = 0; j < adjustments.size(); ++j) {
        weight_adjustments[i][0] += input_layer_transpose[i][j] * adjustments[j][0];
      }
    }

    for (size_t i = 0; i < (*_synaptic_weights).size(); ++i) {
      (*_synaptic_weights)[i] += weight_adjustments[i][0];
    }
  }
}