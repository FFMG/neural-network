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
    std::cerr << "The topology is not value, you must have at least 2 layers." << std::endl;
    throw std::invalid_argument("The topology is not value, you must have at least 2 layers.");
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

std::vector<size_t> NeuralNetwork::get_shuffled_indexes(size_t raw_size)
{
  std::vector<size_t> shuffled_indexes(raw_size);
  std::iota(shuffled_indexes.begin(), shuffled_indexes.end(), 0);

  std::random_device rd;
  std::mt19937 gen(rd()); // Mersenne Twister RNG
  std::shuffle(shuffled_indexes.begin(), shuffled_indexes.end(), gen);
  return shuffled_indexes;
}

void NeuralNetwork::create_shuffled_indexes(size_t raw_size, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes)
{
  auto shuffled_indexes = get_shuffled_indexes(raw_size);
  assert(raw_size == shuffled_indexes.size());

  break_shuffled_indexes(shuffled_indexes, data_is_unique, training_indexes, checking_indexes, final_check_indexes);
}

void NeuralNetwork::break_shuffled_indexes(const std::vector<size_t>& shuffled_indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes)
{
  // break the indexes into parts
  size_t total_size = shuffled_indexes.size();
  size_t training_size = static_cast<size_t>(std::round(total_size * 0.80));
  size_t checking_size = static_cast<size_t>(std::round(total_size * 0.15));
  if(training_size+checking_size == total_size || checking_size == 0)
  {
    // in the case of small training models we might not have enough to split anything
    std::cout << "Training batch size does not allow for error checking batch!" << std::endl;
    training_indexes = shuffled_indexes;
    checking_indexes = {shuffled_indexes.front()};
    final_check_indexes = {shuffled_indexes.back()};
    return;
  }
  assert(training_size + checking_size < total_size); // make sure we don't get more than 100%
  if(training_size + checking_size > total_size) // make sure we don't get more than 100%
  {
    std::cerr << "Logic error, unable to do a final batch error check." << std::endl;
    throw std::invalid_argument("Logic error, unable to do a final batch error check.");
  }

  // then build the various indexes that will be used during testing.
  if(data_is_unique)
  {
    // because the data is uniqe we must use all of it for training
    // this is important in some cases where the NN needs all the data to train
    // otherwise we will ony train on some of the data.
    // the classic XOR example is a good use case ... 
    training_indexes = shuffled_indexes;
  }
  else
  {
  training_indexes.assign(shuffled_indexes.begin(), shuffled_indexes.begin() + training_size);
  }
  checking_indexes.assign(shuffled_indexes.begin() + training_size, shuffled_indexes.begin() + training_size + checking_size);
  final_check_indexes.assign(shuffled_indexes.begin() + training_size + checking_size, shuffled_indexes.end());
}

void NeuralNetwork::create_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs)
{
  shuffled_training_inputs = {};
  shuffled_training_outputs = {};
  shuffled_training_inputs.reserve(shuffled_indexes.size());
  shuffled_training_outputs.reserve(shuffled_indexes.size());
  for( auto shuffled_index : shuffled_indexes)
  {
    shuffled_training_inputs.push_back(training_inputs[shuffled_index]);
    shuffled_training_outputs.push_back(training_outputs[shuffled_index]);
  }
}

std::vector<double> NeuralNetwork::think(const std::vector<double>& inputs) const
{
  auto layers = *_layers;
  return calculate_forward_feed(inputs, layers).back();
}

long double NeuralNetwork::get_error() const
{
  return _error;
}

std::vector<std::vector<double>> NeuralNetwork::think(const std::vector<std::vector<double>>& inputs) const
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
  int batch_size,
  bool data_is_unique,
  const std::function<bool(int, NeuralNetwork&)>& progress_callback
)
{
  if(batch_size < -1 || batch_size > static_cast<int>(training_inputs.size()))
  {
    std::cerr << "The batch size if either -ve or too large for the training sample." << std::endl;
    throw std::invalid_argument("The batch size if either -ve or too large for the training sample.");
  }
  if(training_outputs.size() != training_inputs.size())
  {
    std::cerr << "The number of training samples does not match the number of expected outputs." << std::endl;
    throw std::invalid_argument("The number of training samples does not match the number of expected outputs.");
  }

  if(batch_size == -1)
  {
    // no batch training
    train( training_inputs, training_outputs,number_of_epoch, data_is_unique, progress_callback);
    return;
  }
  // run in batch
  train_in_batch( training_inputs, training_outputs,number_of_epoch, batch_size, data_is_unique, progress_callback);
}

void NeuralNetwork::train_in_batch( const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, int batch_size, bool data_is_unique, const std::function<bool(int, NeuralNetwork&)>& progress_callback)
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

  // get the indexes
  std::vector<size_t> training_indexes;
  std::vector<size_t> checking_indexes;
  std::vector<size_t> final_check_indexes;
  create_shuffled_indexes(training_inputs.size(), data_is_unique, training_indexes, checking_indexes, final_check_indexes);

  std::vector<std::vector<double>> checking_training_inputs = {};
  std::vector<std::vector<double>> checking_training_outputs = {};
  create_batch_from_indexes(checking_indexes, training_inputs, training_outputs, checking_training_inputs, checking_training_outputs);

  // build the training output batch so we can use it for error calculations
  std::vector<std::vector<double>> training_outputs_batch = {};
  training_outputs_batch.reserve(training_indexes.size());
  for (auto training_index : training_indexes)
  {
    const auto& outputs = training_outputs[training_index];
    training_outputs_batch.push_back(outputs);
  }

  const auto training_indexes_size = training_indexes.size();
  for (auto i = 0; i < number_of_epoch; ++i)
  {
    for( size_t j = 0; j < training_indexes_size; j += batch_size)
    {
      size_t end_size = j + batch_size > training_indexes_size ? training_indexes_size -j : j + batch_size;

      // create the batch input/outputs
      std::vector<std::vector<double>> batch_inputs = {};
      batch_inputs.insert(batch_inputs.end(), training_inputs.begin() + j, training_inputs.begin() + end_size );

      std::vector<std::vector<double>> batch_outputs = {};
      batch_outputs.insert(batch_outputs.end(), training_outputs.begin() + j, training_outputs.begin() + end_size );

      auto batch_predictions = calculate_forward_feed(batch_inputs, *_layers);
      batch_back_propagation(batch_outputs, batch_predictions, *_layers);
      //auto batch_predictions = calculate_forward_feed(batch_inputs.front(), *_layers);
      //back_propagation(batch_outputs.front(), batch_predictions, *_layers);
    }

    // do a batch check to see where we are at ...
    _error = calculate_error(checking_training_inputs, checking_training_outputs, *_layers);

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

  // final error checking
  std::vector<std::vector<double>> final_training_inputs = {};
  std::vector<std::vector<double>> final_training_outputs = {};
  create_batch_from_indexes(final_check_indexes, training_inputs, training_outputs, final_training_inputs, final_training_outputs);
  _error = calculate_error(final_training_inputs, final_training_outputs, *_layers);
  std::cout << "Final Test Error: " << _error << std::endl;

  // final callback if needed
  if (progress_callback != nullptr)
  {
    progress_callback(100, *this);
  }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int number_of_epoch, bool data_is_unique, const std::function<bool(int, NeuralNetwork&)>& progress_callback)
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

  // get the indexes
  std::vector<size_t> training_indexes;
  std::vector<size_t> checking_indexes;
  std::vector<size_t> final_check_indexes;
  create_shuffled_indexes(training_inputs.size(), data_is_unique, training_indexes, checking_indexes, final_check_indexes);

  std::vector<std::vector<double>> checking_training_inputs = {};
  std::vector<std::vector<double>> checking_training_outputs = {};
  create_batch_from_indexes(checking_indexes, training_inputs, training_outputs, checking_training_inputs, checking_training_outputs);

  // build the training output batch so we can use it for error calculations
  std::vector<std::vector<double>> training_outputs_batch = {};
  training_outputs_batch.reserve(training_indexes.size());
  for (auto training_index : training_indexes)
  {
    const auto& outputs = training_outputs[training_index];
    training_outputs_batch.push_back(outputs);
  }  

  for (auto i = 0; i < number_of_epoch; ++i)
  {
    for (auto training_index : training_indexes)
    {
      const auto& inputs = training_inputs[training_index];
      const auto& outputs = training_outputs[training_index];

      // calculate the feed forward just one item
      auto given_output = calculate_forward_feed(inputs, *_layers);
      back_propagation(outputs, given_output, *_layers);
    }

    // do a batch check to see where we are at ...
    _error = calculate_error(checking_training_inputs, checking_training_outputs, *_layers);

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

  // final error checking
  std::vector<std::vector<double>> final_training_inputs = {};
  std::vector<std::vector<double>> final_training_outputs = {};
  create_batch_from_indexes(final_check_indexes, training_inputs, training_outputs, final_training_inputs, final_training_outputs);
  _error = calculate_error(final_training_inputs, final_training_outputs, *_layers);
  std::cout << "Final Test Error: " << _error << std::endl;

  // final callback if needed
  if (progress_callback != nullptr)
  {
    progress_callback(100, *this);
  }
}

double NeuralNetwork::calculate_error(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<Layer>& layers)
{
  std::vector<std::vector<double>> predictions = {};
  int batch_size = training_inputs.size();
  for (auto index = 0; index < batch_size; ++index)
  {
    const auto& inputs = training_inputs[index];
    auto all_outputs = calculate_forward_feed(inputs, layers);
    predictions.push_back(all_outputs.back());
  }
  return calculate_rmse_error(training_outputs, predictions);
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
    std::cerr << "Mismatch in batch sizes." << std::endl;
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
      std::cerr << "Mismatch in output vector sizes at index " << i << std::endl;
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

NeuralNetwork::GradientsAndOutputs NeuralNetwork::average_batch_gradients(const std::vector<GradientsAndOutputs>& batch_activation_gradients)
{
  if (batch_activation_gradients.empty()) 
  {
    return GradientsAndOutputs();
  }

  size_t batch_size = batch_activation_gradients.size();
  
  // Prepare result vector with proper dimensions
  GradientsAndOutputs activation_gradients;

  // gradients
  size_t num_gradient_layers = batch_activation_gradients[0].num_gradient_layers();
  for (size_t layer = 0; layer < num_gradient_layers; ++layer)
  {
    size_t number_neurons = batch_activation_gradients[0].num_gradient_neurons(layer);
    for (size_t neuron = 0; neuron < number_neurons; ++neuron)
    {
      double sum = 0.0;
      for (size_t batch = 0; batch < batch_size; ++batch)
      {
        sum += batch_activation_gradients[batch].get_gradient(layer, neuron);
      }
      activation_gradients.set_gradient(layer, neuron, sum / static_cast<double>(batch_size));
    }
  }

  // we get the output to the connecting layer
  // so this layer will have then number of outputs to the _next_ layer.
  for (size_t layer = 0; layer < num_gradient_layers -1; ++layer)
  {
    // number of neurons to the next layer
    const size_t number_neurons = batch_activation_gradients[0].num_gradient_neurons(layer+1);
    for( size_t target_neuron = 0; target_neuron < number_neurons; ++target_neuron )
    {
      // bias is added by the get_outputs( ... ) method for _this_ layer.
      const size_t number_outputs = batch_activation_gradients[0].get_outputs(layer).size();
      std::vector<double> outputs_gradients(number_outputs, 0.0);
      for (size_t batch = 0; batch < batch_size; ++batch)
      {
        const auto& gradientsAndOutputs = batch_activation_gradients[batch];

        // get the gradient this neuron in the next layer
        const auto& next_layer_neuron_gradient = gradientsAndOutputs.get_gradient(layer+1, target_neuron);
        // get the output[layer] * gradient[layer+1]
        
        for (size_t output_number = 0; output_number < number_outputs; ++output_number)
        {
          const auto& layer_neuron_output = gradientsAndOutputs.get_output(layer, output_number);
          outputs_gradients[output_number] += (layer_neuron_output * next_layer_neuron_gradient);
        }
      }

      // finally ... average it all out
      for( auto& outputs_gradient : outputs_gradients)
      {
        outputs_gradient /= static_cast<double>(batch_size);
      }
      activation_gradients.set_outputs_gradients(layer, target_neuron, outputs_gradients);      
    }
  }
  return activation_gradients;
}

void NeuralNetwork::calculate_batch_back_propagation_gradients(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& layers_given_outputs, const std::vector<Layer>& layers)
{
  assert(target_outputs.size() == layers_given_outputs.size());
  for(size_t i = 0; i < target_outputs.size(); ++i)
  {
    calculate_back_propagation_gradients(target_outputs[i], layers_given_outputs[i], layers);
  }
}

void NeuralNetwork::calculate_back_propagation_gradients(const std::vector<double>& target_outputs, GradientsAndOutputs& layers_given_outputs, const std::vector<Layer>& layers)
{
  assert(target_outputs.size() == layers_given_outputs.back().size());

  // input layer is all 0, (bias is included)
  auto input_gradients = std::vector<double>(layers.front().size(), 0.0);
  layers_given_outputs.set_gradients(0, input_gradients);
  
  // set the output gradient
  const auto& output_layer = layers.back();
  auto next_activation_gradients = caclulate_output_gradients(target_outputs, layers_given_outputs.back(), output_layer);
  layers_given_outputs.set_gradients(layers.size()-1, next_activation_gradients);
  for (auto layer_number = layers.size() - 2; layer_number > 0; --layer_number)
  {
    const auto& hidden_layer = layers[layer_number];
    const auto& next_layer = layers[layer_number + 1];

    std::vector<double> current_activation_gradients;
    for (size_t n = 0; n < hidden_layer.size(); ++n)
    {
      const auto& neuron = hidden_layer.get_neuron(unsigned(n));
      const auto output_value = layers_given_outputs.get_output(layer_number, n);
      current_activation_gradients.push_back(neuron.calculate_hidden_gradients(next_layer, next_activation_gradients, output_value));
    }
    layers_given_outputs.set_gradients(layer_number, current_activation_gradients);
    next_activation_gradients = current_activation_gradients;
    current_activation_gradients = {};
  }
}

void NeuralNetwork::batch_back_propagation(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& batch_given_outputs, std::vector<Layer>& layers)
{
  // calculate all the gradients in the batch.
  calculate_batch_back_propagation_gradients(target_outputs, batch_given_outputs, layers);

  auto batch_size = batch_given_outputs.size();
  if(batch_size == 0)
  {
    std::cerr << "Batch size is 0 so we cannot do back propagation!" << std::endl;
    return;
  }
  
  // we now need to calculate the average of all the gradients
  // so we will sum them all up and devide by our batch size;
  auto activation_gradients = average_batch_gradients(batch_given_outputs);
  update_layers_with_gradients(activation_gradients, layers);
}

void NeuralNetwork::back_propagation(const std::vector<double>& target_outputs, GradientsAndOutputs& given_outputs, std::vector<Layer>& layers)
{
  // calculate all the gradients.
  calculate_back_propagation_gradients(target_outputs, given_outputs, layers);

  for (size_t layer_number = 0; layer_number < layers.size(); ++layer_number)
  {
    auto& layer = layers[layer_number];
    for (size_t neuron_number = 0; neuron_number < layer.size(); ++neuron_number)
    {
      auto& neuron = layer.get_neuron(unsigned(neuron_number));
      neuron.set_output_value(given_outputs.get_output(layer_number, neuron_number));
    }
  }
  
  given_outputs.update_outputs_gradients();
  update_layers_with_gradients(given_outputs, layers);
}

void NeuralNetwork::update_layers_with_gradients(GradientsAndOutputs& activation_gradients, std::vector<Layer>& layers)
{
  // set up the gradients that we just calculated.
  for (size_t layer_number = 0; layer_number < layers.size(); ++layer_number)
  {
    auto& layer = layers[layer_number];
    for (size_t neuron_number = 0; neuron_number < layer.size(); ++neuron_number)
    {
      auto& neuron = layer.get_neuron(unsigned(neuron_number));
      neuron.set_gradient_value(activation_gradients.get_gradient(layer_number, neuron_number));
    }
  }

  // update the weights in reverse
  for (auto layer_number = layers.size() - 1; layer_number > 0; --layer_number) 
  {
    auto& layer = layers[layer_number];
    auto& previous_layer = layers[layer_number - 1];

    for (unsigned neuron_number = 0; neuron_number < layer.size() - 1; ++neuron_number) 
    {
      auto& neuron = layer.get_neuron(neuron_number);
      auto weights_gradients = activation_gradients.get_outputs_gradients(layer_number-1, neuron_number);
      auto gradient = activation_gradients.get_gradient(layer_number, neuron_number);
      neuron.update_input_weights(previous_layer, gradient, weights_gradients);
    }
  }
}

std::vector<NeuralNetwork::GradientsAndOutputs> NeuralNetwork::calculate_forward_feed(const std::vector<std::vector<double>>& inputs, const std::vector<Layer>& layers)
{
  std::vector<GradientsAndOutputs> activations_per_layer_per_input = {};
  for(const auto& i : inputs)
  {
    activations_per_layer_per_input.push_back(calculate_forward_feed(i, layers));
  }
  return activations_per_layer_per_input;
}

NeuralNetwork::GradientsAndOutputs NeuralNetwork::calculate_forward_feed(const std::vector<double>& inputs, const std::vector<Layer>& layers)
{
  // the return value is the activation values per layers.
  GradientsAndOutputs activations_per_layer;

  //  the initial set of output values where we are starting from.
  activations_per_layer.set_outputs(0, inputs);

  // then forward propagate from the input to ... hopefully, the output.
  for (size_t layer_number = 1; layer_number < layers.size(); ++layer_number)
  {
    const auto& previous_layer = layers[layer_number - 1];
    auto& this_layer = layers[layer_number];
    auto previous_layer_output_values = activations_per_layer.back();
    previous_layer_output_values.push_back(1);  //  add the bias.
    for (size_t neuron_number = 0; neuron_number < this_layer.size() - 1; ++neuron_number)
    {
      const auto& neuron = this_layer.get_neuron(unsigned(neuron_number));
      activations_per_layer.set_output(
        layer_number,
        neuron_number,
        neuron.calculate_forward_feed(previous_layer, previous_layer_output_values));
    }
  }
  // return the output values per layer.
  return activations_per_layer;
}

std::vector<double> NeuralNetwork::caclulate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer)
{
  std::vector<double> activation_gradients = {};
  for (size_t n = 0; n < output_layer.size() - 1; ++n) // ignore bias
  {
    const auto& neuron = output_layer.get_neuron(unsigned(n));
    activation_gradients.push_back(neuron.calculate_output_gradients(
      target_outputs[n], given_outputs[n]));
  }
  activation_gradients.push_back(0);  //  add bias we ignored above
  return activation_gradients;
}

void NeuralNetwork::set_output_gradients(const std::vector<double>& current_outputs, Layer& output_layer)
{
  std::vector<double> given_outputs;
  for(const auto& output_layer_neuron : output_layer.get_neurons())
  {
    given_outputs.push_back(output_layer_neuron.get_output_value());
  }
  auto activation_gradients = caclulate_output_gradients(current_outputs, given_outputs, output_layer);
  assert(activation_gradients.size() == output_layer.size());
  for (unsigned n = 0; n < activation_gradients.size(); ++n)
  {
    auto& neuron = output_layer.get_neuron(unsigned(n));
    neuron.set_gradient_value(activation_gradients[n]);
  }
  output_layer.normalise_gradients();
}