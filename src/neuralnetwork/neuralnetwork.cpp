#include "neuralnetwork.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>

static const double RecentAverageSmoothingFactor = 100.0;

NeuralNetwork::NeuralNetwork(
  const std::vector<unsigned>& topology, 
  const activation::method& activation,
  double learning_rate
  ) :
  _error(0.0),
  _layers(nullptr),
  _activation_method(activation),
  _learning_rate(learning_rate)
{
  const auto& number_of_layers = topology.size();
  _layers = new std::vector<Neuron::Layer>();
  for (size_t layer_number = 0; layer_number < number_of_layers; ++layer_number)
  {
    auto num_neurons_prev_layer = layer_number == topology.size() - 1 ? 0 : topology[layer_number + 1];
    auto num_neurons_current_layer = layer_number == topology.size() ? 0 : topology[layer_number];
    auto layer = Neuron::Layer();

    // We have a new layer, now fill it with neurons, and add a bias neuron in each layer.
    for (unsigned neuronNum = 0; neuronNum <= topology[layer_number]; ++neuronNum)
    {
      // force the bias node's output to 1.0
      auto neuron = Neuron(
        num_neurons_prev_layer,
        num_neurons_current_layer,
        neuronNum, activation, learning_rate);
      neuron.set_output_value(1.0);
      layer.push_back(neuron);
    }
    _layers->push_back(layer);
  }
}

NeuralNetwork::NeuralNetwork(
  const std::vector<Neuron::Layer>& layers, 
  const activation::method& activation,
  double learning_rate,
  double error
  ) :
  _error(error),
  _layers(nullptr),
  _activation_method(activation),
  _learning_rate(learning_rate)
{
  _layers = new std::vector<Neuron::Layer>();
  for (auto layer : layers)
  {
    auto copy_layer = Neuron::Layer(layer);
    _layers->push_back(copy_layer);
  }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& src) :
  _error(src._error),
  _layers(nullptr),
  _activation_method(src._activation_method),
  _learning_rate(src._learning_rate)
{
  _layers = new std::vector<Neuron::Layer>();
  for (const auto& layer : *src._layers)
  {
    auto copy_layer = Neuron::Layer(layer);
    _layers->push_back(copy_layer);
  }
}

NeuralNetwork::~NeuralNetwork()
{
  delete _layers;
}

double NeuralNetwork::get_learning_rate() const
{
  return _learning_rate;
}

activation::method NeuralNetwork::get_activation_method() const
{
  return _activation_method;
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
  auto layers = *_layers;
  forward_feed(inputs, layers);
  return get_outputs(layers.back());
}

long double NeuralNetwork::get_error() const
{
  return _error;
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
  const std::function<bool(int, NeuralNetwork&)>& progress_callback
)
{
  const auto interval = std::chrono::seconds(5);
  auto last_callback_time = std::chrono::high_resolution_clock::now();
  // initial callback
  _error = 0.0;
  if (progress_callback != nullptr)
  {
    if( !progress_callback(0, *this))
    {
      return;
    }
  }
  const auto& output_layer = _layers->back();

  for (auto i = 0; i < number_of_epoch; ++i)
  {
    std::vector<std::vector<double>> predictions = {};
    for (size_t j = 0; j < training_inputs.size(); ++j)
    {
      const auto& inputs = training_inputs[j];
      const auto& outputs = training_outputs[j];

      forward_feed(inputs, *_layers);
      predictions.push_back(get_outputs(output_layer));
      back_propagation(outputs, *_layers);
    }

    _error = calculate_batch_error(training_outputs, predictions);

    if (progress_callback != nullptr)
    {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto elapsed_time = current_time - last_callback_time;
      auto percent = (int)(((float)i / number_of_epoch)*100);
      if (elapsed_time >= interval)
      {
        if( !progress_callback(percent, *this))
        {
          return;
        }
        last_callback_time = current_time;
      }
    }
  }

  // final callback if needed
  if (progress_callback != nullptr)
  {
    progress_callback(100, *this);
  }
}

double NeuralNetwork::calculate_batch_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
{
  return calculate_batch_rmse_error(ground_truth, predictions);
}

double NeuralNetwork::calculate_batch_rmse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
{
  auto mean_squared_error = calculate_batch_mse_error(ground_truth, predictions);
  return std::sqrt(mean_squared_error); // RMSE
}

double NeuralNetwork::calculate_batch_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
{
  if (ground_truth.size() != predictions.size()) 
  {
    std::cerr << "Mismatch in batch sizes.\n";
    return std::numeric_limits<double>::quiet_NaN();
  }

  double mean_squared_error = 0.0;
  size_t valid_count = 0;

  for (size_t i = 0; i < ground_truth.size(); ++i) 
  {
    const auto& true_output = ground_truth[i];
    const auto& predicted_output = predictions[i];

    if (true_output.size() != predicted_output.size()) 
    {
      std::cerr << "Mismatch in output vector sizes at index " << i << "\n";
      continue;
    }

    for (size_t j = 0; j < true_output.size(); ++j) 
    {
      double error = predicted_output[j] - true_output[j];

      if (!std::isfinite(error))
      {
        continue;
      }

      double squared_error = error * error;
      if (!std::isfinite(squared_error))
      {
        continue;
      }
      ++valid_count;
      mean_squared_error += (squared_error - mean_squared_error) / valid_count;
    }
  }

  if (valid_count == 0)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return mean_squared_error;
}

void NeuralNetwork::back_propagation(
  const std::vector<double>& current_output,
  std::vector<Neuron::Layer>& layers_src
)
{
  auto& output_layer = layers_src.back();
  
  // Calculate output layer gradients
  calculate_output_gradients(current_output, output_layer);

  // Calculate hidden layer gradients

  for (auto layer_number = layers_src.size() - 2; layer_number > 0; --layer_number)
  {
    auto& hidden_layer = layers_src[layer_number];
    auto& next_layer = layers_src[layer_number + 1];

    for (size_t n = 0; n < hidden_layer.size(); ++n)
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
}

void NeuralNetwork::forward_feed( const std::vector<double>& inputs, std::vector<Neuron::Layer>& layers )
{
  // Assign (latch) the input values into the input neurons
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    layers[0][i].set_output_value(inputs[i]);
  }

  // forward propagate
  for (size_t layer_number = 1; layer_number < layers.size(); ++layer_number)
  {
    const auto& previous_layer = layers[layer_number - 1];
    for (size_t n = 0; n < layers[layer_number].size() - 1; ++n)
    {
      layers[layer_number][n].forward_feed(previous_layer);
    }
  }
}

std::vector<double> NeuralNetwork::get_outputs(const Neuron::Layer& output_layer) const
{
  std::vector<double> outputs;
  outputs.reserve(output_layer.size() - 1); //  exclude the bias Neuron
  for (auto it = output_layer.begin(); it != output_layer.end() - 1; ++it) 
  {
    outputs.push_back(it->get_output_value());
  }
  return outputs;
}

void NeuralNetwork::calculate_output_gradients( const std::vector<double>& targetVals, Neuron::Layer& output_layer)
{
  for (size_t n = 0; n < output_layer.size() - 1; ++n)
  {
    output_layer[n].calculate_output_gradients(targetVals[n]);
  }

  const double max_norm = 10.0;
  auto norm = norm_output_gradients(output_layer);
  if (norm < max_norm)
  {
    return;
  }

  // update all the gradients.
  double scale = max_norm / (norm == 0 ? 1e-8 : norm);
  for (size_t n = 0; n < output_layer.size() - 1; ++n)
  {
    auto gradient = output_layer[n].get_gradient();
    gradient *= scale;
    output_layer[n].set_gradient_value(gradient);
  }
}

double NeuralNetwork::norm_output_gradients(Neuron::Layer& output_layer)
{
  auto acc = std::accumulate(
    output_layer.begin(),
    output_layer.end(),
    0.0,
    [](double sum, Neuron& n) {
      auto grad = n.get_gradient();
      return sum + grad * grad;
    });
  return std::sqrt(acc);
}
