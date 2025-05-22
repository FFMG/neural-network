#include "neuralnetwork.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <string>

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
  if(topology.size() < 2)
  {
    std::cerr << "The topology is not value, you must have at lease 2 layers." << std::endl;
    throw std::invalid_argument("The topology is not value, you must have at lease 2 layers.");
  }
  const auto& number_of_layers = topology.size();
  _layers = new std::vector<Layer>();

  // add the input layer
  auto layer = Layer::create_input_layer(topology[0], topology[1], _activation_method, _learning_rate);
  _layers->push_back(layer);
  
  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    const auto& previous_layer = _layers->back();
    layer = Layer::create_hidden_layer(num_neurons_current_layer, num_neurons_next_layer, previous_layer, _activation_method, _learning_rate);
    _layers->push_back(layer);
  }

  // finally, the output layer
  layer = Layer::create_output_layer(topology.back(), _layers->back(), _activation_method, _learning_rate);
  _layers->push_back(layer);
}

NeuralNetwork::NeuralNetwork(
  const std::vector<Layer>& layers, 
  const activation::method& activation,
  double learning_rate,
  double error
  ) :
  _error(error),
  _layers(nullptr),
  _activation_method(activation),
  _learning_rate(learning_rate)
{
  _layers = new std::vector<Layer>();
  for (auto layer : layers)
  {
    auto copy_layer = Layer(layer);
    _layers->push_back(copy_layer);
  }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& src) :
  _error(src._error),
  _layers(nullptr),
  _activation_method(src._activation_method),
  _learning_rate(src._learning_rate)
{
  _layers = new std::vector<Layer>();
  for (const auto& layer : *src._layers)
  {
    auto copy_layer = Layer(layer);
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

const std::vector<Layer>& NeuralNetwork::get_layers() const
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

std::vector<size_t> NeuralNetwork::get_suffled_indexes(size_t raw_size)
{
  std::vector<size_t> shuffled_indexes(raw_size);
  std::iota(shuffled_indexes.begin(), shuffled_indexes.end(), 0);

  std::random_device rd;
  std::mt19937 gen(rd()); // Mersenne Twister RNG
  std::shuffle(shuffled_indexes.begin(), shuffled_indexes.end(), gen);
  return shuffled_indexes;
}

std::vector<double> NeuralNetwork::think(const std::vector<double>& inputs) const
{
  auto layers = *_layers;
  return forward_feed(inputs, layers);
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

  // get the indexes
  auto suffled_indexes = get_suffled_indexes(training_inputs.size());
  assert(training_inputs.size() == suffled_indexes.size());

  // break the indexes into parts
  size_t total_size = suffled_indexes.size();
  size_t training_size = static_cast<size_t>(std::round(total_size * 0.80));
  size_t checking_size = static_cast<size_t>(std::round(total_size * 0.15));
  assert(training_size + checking_size < total_size); // make sure we don't get more than 100%
  size_t final_check_size = total_size - training_size - checking_size; // make sure nothing left over

  // then build the various indexes that will be used during testing.
  std::vector<size_t> training_indexes;
  training_indexes.assign(suffled_indexes.begin(), suffled_indexes.begin() + training_size);

  std::vector<size_t> checking_indexes;
  checking_indexes.assign(suffled_indexes.begin() + training_size, suffled_indexes.begin() + training_size + checking_size);

  std::vector<size_t> final_check_indexes;
  final_check_indexes.assign(suffled_indexes.begin() + training_size + checking_size, suffled_indexes.end());

  // build the trainning output batch so we can use it for error calculations
  std::vector<std::vector<double>> training_outputs_batch = {};
  training_outputs_batch.reserve(training_indexes.size());
  for (auto training_index : training_indexes)
  {
    const auto& outputs = training_outputs[training_index];
    training_outputs_batch.push_back(outputs);
  }  

  for (auto i = 0; i < number_of_epoch; ++i)
  {
    std::vector<std::vector<double>> predictions = {};
    for (auto training_index : training_indexes)
    {
      const auto& inputs = training_inputs[training_index];
      const auto& outputs = training_outputs[training_index];

      predictions.push_back(forward_feed(inputs, *_layers));
      back_propagation(outputs, *_layers);
    }

    _error = calculate_error(training_outputs_batch, predictions);

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

double NeuralNetwork::calculate_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
{
  return calculate_rmse_error(ground_truth, predictions);
  // return calculate_mae_error(ground_truth, predictions);
  //return calculate_huber_loss(ground_truth, predictions);
}

double NeuralNetwork::calculate_huber_loss(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta)
{
  if (ground_truth.size() != predictions.size())
  {
    std::cerr << "Mismatched number of samples" << std::endl;
    throw std::invalid_argument("Mismatched number of samples");
  }

  double total_loss = 0.0;
  size_t count = 0;

  for (size_t i = 0; i < ground_truth.size(); ++i)
  {
    if (ground_truth[i].size() != predictions[i].size())
    {
      std::cerr << "Mismatched vector sizes at index " << std::to_string(i) << std::endl;
      throw std::invalid_argument("Mismatched vector sizes at index " + std::to_string(i));
    }

    for (size_t j = 0; j < ground_truth[i].size(); ++j)
    {
      double error = ground_truth[i][j] - predictions[i][j];
      double abs_error = std::abs(error);

      if (abs_error <= delta)
      {
        total_loss += 0.5 * error * error;
      }
      else
      {
        total_loss += delta * (abs_error - 0.5 * delta);
      }
      ++count;
    }
  }
  return (count > 0) ? (total_loss / count) : 0.0;
}

double NeuralNetwork::calculate_mae_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
{
  if (ground_truth.size() != predictions.size())
  {
    std::cerr << "Mismatched number of samples" << std::endl;
    throw std::invalid_argument("Mismatched number of samples");
  }
  

  double total_abs_error = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < ground_truth.size(); ++i)
  {
    if (ground_truth[i].size() != predictions[i].size())
    {
      std::cerr << "Mismatched vector sizes at index " << std::to_string(i) << std::endl;
      throw std::invalid_argument("Mismatched vector sizes at index " + std::to_string(i));
    }
    for (size_t j = 0; j < ground_truth[i].size(); ++j)
    {
      total_abs_error += std::abs(ground_truth[i][j] - predictions[i][j]);
      ++count;
    }
  }
  return (count > 0) ? (total_abs_error / count) : 0.0;
}

double NeuralNetwork::calculate_rmse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
{
  auto mean_squared_error = calculate_mse_error(ground_truth, predictions);
  return std::sqrt(mean_squared_error); // RMSE
}

double NeuralNetwork::calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
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

void NeuralNetwork::back_propagation( const std::vector<double>& current_output, std::vector<Layer>& layers_src)
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
      hidden_layer.get_neuron(unsigned(n)).calculate_hidden_gradients(next_layer);
    }
  }

  // update the hidden layers
  for (auto layerNum = layers_src.size() - 1; layerNum > 0; --layerNum) 
  {
    auto& layer = layers_src[layerNum];
    auto& prevLayer = layers_src[layerNum - 1];

    for (unsigned n = 0; n < layer.size() - 1; ++n) 
    {
      layer.get_neuron(n).update_input_weights(prevLayer);
    }
  }
}

std::vector<std::vector<double>> NeuralNetwork::forward_feed(const std::vector<std::vector<double>>& inputs_batch, std::vector<Layer>& layers)
{
  std::vector<std::vector<double>> batch_predictions;
  batch_predictions.reserve(inputs_batch.size()); // Pre-allocate memory for efficiency

  // Loop through each individual input vector in the batch
  for (const auto& single_input_vector : inputs_batch)
  {
    // Call the existing single-example forward_feed for each input
    std::vector<double> prediction_for_single_example = forward_feed(single_input_vector, layers);
    batch_predictions.push_back(prediction_for_single_example);
  }

  return batch_predictions;
}

std::vector<double> NeuralNetwork::forward_feed( const std::vector<double>& inputs, std::vector<Layer>& layers)
{
  // Assign (latch) the input values into the input neurons
  auto& input_layer = layers.front();
  assert(inputs.size() == input_layer.size() - 1); //  last one is bias.
  for (unsigned i = 0; i < inputs.size(); ++i)
  {
    input_layer.get_neuron(i).set_output_value(inputs[i]);
  }

  // forward propagate
  for (size_t layer_number = 1; layer_number < layers.size(); ++layer_number)
  {
    const auto& previous_layer = layers[layer_number - 1];
    auto& this_layer = layers[layer_number];
    for (size_t n = 0; n < layers[layer_number].size() - 1; ++n)
    {
      this_layer.get_neuron(unsigned(n)).forward_feed(previous_layer);
    }
  }
  // return the output layers outputs.
  return layers.back().get_outputs();  
}

void NeuralNetwork::calculate_output_gradients( const std::vector<double>& targetVals, Layer& output_layer)
{
  for (size_t n = 0; n < output_layer.size() - 1; ++n)
  {
    output_layer.get_neuron(unsigned(n)).calculate_output_gradients(targetVals[n]);
  }
  output_layer.normalise_gradients();
}