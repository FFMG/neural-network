#include "neuralnetwork.h"
#include "threadpool.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <future>
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
  _topology(topology),
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

    // remove the bias Neuron.
    _topology.push_back(copy_layer.size() -1);
  }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& src) :
  _error(src._error),
  _topology(src._topology),
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

const std::vector<unsigned>& NeuralNetwork::get_topology() const
{
  return _topology;
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
  return calculate_forward_feed(inputs, layers).output_back();
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

std::vector<NeuralNetwork::GradientsAndOutputs> NeuralNetwork::train_single_batch(const std::vector<std::vector<double>>& batch_inputs, const std::vector<std::vector<double>>& batch_outputs) const
{
  auto batch_gradients_outputs = calculate_forward_feed(batch_inputs, *_layers);
  calculate_batch_back_propagation(batch_outputs, batch_gradients_outputs, *_layers);
  return batch_gradients_outputs;
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
  if(batch_size <=0 || batch_size > static_cast<int>(training_inputs.size()))
  {
    std::cerr << "The batch size if either -ve or too large for the training sample." << std::endl;
    throw std::invalid_argument("The batch size if either -ve or too large for the training sample.");
  }
  if(training_outputs.size() != training_inputs.size())
  {
    std::cerr << "The number of training samples does not match the number of expected outputs." << std::endl;
    throw std::invalid_argument("The number of training samples does not match the number of expected outputs.");
  }

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
  for (const auto& training_index : training_indexes)
  {
    const auto& outputs = training_outputs[training_index];
    training_outputs_batch.push_back(outputs);
  }

  ThreadPool threadpool;
  const auto training_indexes_size = training_indexes.size();
  for (auto epoch = 0; epoch < number_of_epoch; ++epoch)
  {
    size_t num_batches = (training_indexes_size + batch_size - 1) / batch_size;
    std::vector<std::future<std::vector<GradientsAndOutputs>>> futures;
    futures.reserve(num_batches);

    for( size_t j = 0; j < training_indexes_size; j += batch_size)
    {
      const size_t start = j;
      const size_t end_size = std::min(j + batch_size, training_indexes_size);

      futures.emplace_back(
        threadpool.enqueue( [=](){
          std::vector<std::vector<double>> batch_inputs(
              training_inputs.begin() + start,
              training_inputs.begin() + end_size
          );

          std::vector<std::vector<double>> batch_outputs(
              training_outputs.begin() + start,
              training_outputs.begin() + end_size
          );

          return train_single_batch(batch_inputs, batch_outputs);
      }));
    }

    // Collect the results
    std::vector<std::vector<GradientsAndOutputs>> epoch_gradients_outputs = {};
    epoch_gradients_outputs.reserve(num_batches);
    for (auto& f : futures)
    {
      epoch_gradients_outputs.emplace_back(std::move(f.get()));
    }
    update_layers_with_gradients(epoch_gradients_outputs, *_layers);
    
    if (progress_callback != nullptr)
    {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto elapsed_time = current_time - last_callback_time;
      auto percent = (int)(((float)epoch / number_of_epoch)*100);
      if (elapsed_time >= interval)
      {
        // do a batch check to see where we are at ...
        _error = calculate_error(checking_training_inputs, checking_training_outputs, *_layers);
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

NeuralNetwork::LayersAndNeurons<double> NeuralNetwork::recalculate_gradient_avergages(const std::vector<std::vector<GradientsAndOutputs>>& epoch_gradients_outputs) const
{
  LayersAndNeurons<double> averages(get_topology(), false, true);

  // this is the number of batches we had this epoch
  const size_t epoch_gradients_outputs_size = epoch_gradients_outputs.size();
  if( 0 == epoch_gradients_outputs_size)
  {
    return averages;
  }

  // go around all the layers/neurons to get the averages per batch
  const auto layers_size = averages.number_layers();
  for(unsigned layer_number = 0; layer_number < layers_size; ++layer_number)
  {
    // we get the number of neurons from the first batch.
    const auto neurons_size = averages.number_neurons(layer_number);

    for(unsigned neuron_number = 0; neuron_number < neurons_size; ++neuron_number)
    {
      double epoch_sum = 0.0;
      double total_epoch_size = 0.0;
      // then we look at all the batches to get the gradients.
      for(size_t epoch_gradients_outputs_number = 0; epoch_gradients_outputs_number < epoch_gradients_outputs_size; ++epoch_gradients_outputs_number)
      {
        // make sure that we get the actual size as they might not all be the same size.
        const size_t actual_batch_size = epoch_gradients_outputs[epoch_gradients_outputs_number].size();          
        for(size_t batch_index = 0; batch_index < actual_batch_size; ++batch_index)
        {
          epoch_sum += epoch_gradients_outputs[epoch_gradients_outputs_number][batch_index].get_gradient(layer_number, neuron_number);
        }
        total_epoch_size += static_cast<double>(actual_batch_size);
      }
      auto average_gradient = (epoch_sum / static_cast<double>(total_epoch_size));
      averages.set(layer_number, neuron_number, average_gradient);
    }
  }
  return averages;
}

double NeuralNetwork::calculate_error(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<Layer>& layers) const
{
  size_t batch_size = training_inputs.size();
  std::vector<std::vector<double>> predictions(batch_size);
  for (size_t index = 0; index < batch_size; ++index)
  {
    const auto& inputs = training_inputs[index];
    auto all_outputs = calculate_forward_feed(inputs, layers);
    predictions[index] = std::move(all_outputs.output_back());
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

NeuralNetwork::GradientsAndOutputs NeuralNetwork::average_batch_gradients_with_averages(const GradientsAndOutputs& activation_gradients, const LayersAndNeurons<double>& layers_neurons_averages) const
{
  // Prepare result vector with proper dimensions
  GradientsAndOutputs gradients_and_outputs(get_topology());

  // gradients
  gradients_and_outputs.set_gradients(layers_neurons_averages);

  // we get the output to the connecting layer
  // so this layer will have then number of outputs to the _next_ layer.
  const size_t layers_size = layers_neurons_averages.number_layers();
  for (size_t layer_number = 0; layer_number < layers_size -1; ++layer_number)
  {
    // number of neurons to the next layer
    const size_t neurons_size = layers_neurons_averages.number_neurons(layer_number+1);
    for( size_t neuron_number = 0; neuron_number < neurons_size; ++neuron_number )
    {
      // bias is added by the get_outputs( ... ) method for _this_ layer.
      const size_t number_outputs = activation_gradients.num_outputs(layer_number);
      std::vector<double> outputs_gradients(number_outputs, 0.0);

      // get the gradient this neuron in the next layer
      const auto& next_layer_neuron_gradient = activation_gradients.get_gradient(layer_number+1, neuron_number);
      for (size_t output_number = 0; output_number < number_outputs; ++output_number)
      {
        const auto& layer_neuron_output = activation_gradients.get_output(layer_number, output_number);
        outputs_gradients[output_number] += (layer_neuron_output * next_layer_neuron_gradient);
      }
      
      // we can now set the output + average gradient for that layer/neuron.
      gradients_and_outputs.set_gradients_and_outputs(layer_number, neuron_number, outputs_gradients);
    }
  }
  return gradients_and_outputs;
}

NeuralNetwork::GradientsAndOutputs NeuralNetwork::average_batch_gradients_with_averages(const std::vector<GradientsAndOutputs>& batch_activation_gradients, const std::vector<std::vector<double>>& averages) const
{
  if (batch_activation_gradients.empty()) 
  {
    return GradientsAndOutputs(get_topology());
  }

  size_t batch_size = batch_activation_gradients.size();
  
  // Prepare result vector with proper dimensions
  GradientsAndOutputs activation_gradients(get_topology());

  // gradients
  size_t num_gradient_layers = batch_activation_gradients[0].num_gradient_layers();
  for (size_t layer = 0; layer < num_gradient_layers; ++layer)
  {
    size_t number_neurons = batch_activation_gradients[0].num_gradient_neurons(layer);
    for (size_t neuron = 0; neuron < number_neurons; ++neuron)
    {
      activation_gradients.set_gradient(layer, neuron, averages[layer][neuron]);
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
      const size_t number_outputs = batch_activation_gradients[0].num_outputs(layer);
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
      activation_gradients.set_gradients_and_outputs(layer, target_neuron, outputs_gradients);      
    }
  }
  return activation_gradients;
}

NeuralNetwork::GradientsAndOutputs NeuralNetwork::average_batch_gradients(const std::vector<GradientsAndOutputs>& batch_activation_gradients) const
{
  if (batch_activation_gradients.empty()) 
  {
    return GradientsAndOutputs(get_topology());
  }

  size_t batch_size = batch_activation_gradients.size();
  
  // Prepare result vector with proper dimensions
  GradientsAndOutputs activation_gradients(get_topology());

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
      const size_t number_outputs = batch_activation_gradients[0].num_outputs(layer);
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
      activation_gradients.set_gradients_and_outputs(layer, target_neuron, outputs_gradients);      
    }
  }
  return activation_gradients;
}

void NeuralNetwork::calculate_batch_back_propagation_gradients(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& layers_given_outputs, const std::vector<Layer>& layers)
{
  assert(target_outputs.size() == layers_given_outputs.size());
  const auto& target_outputs_size = target_outputs.size();
  for(size_t i = 0; i < target_outputs_size; ++i)
  {
    calculate_back_propagation_gradients(target_outputs[i], layers_given_outputs[i], layers);
  }
}

void NeuralNetwork::calculate_back_propagation_gradients(const std::vector<double>& target_outputs, GradientsAndOutputs& layers_given_outputs, const std::vector<Layer>& layers)
{
  assert(target_outputs.size() == layers_given_outputs.output_back().size());

  // input layer is all 0, (bias is included)
  auto input_gradients = std::vector<double>(layers.front().size(), 0.0);
  layers_given_outputs.set_gradients(0, input_gradients);
  
  // set the output gradient
  const auto& output_layer = layers.back();
  auto next_activation_gradients = caclulate_output_gradients(target_outputs, layers_given_outputs.output_back(), output_layer);
  layers_given_outputs.set_gradients(layers.size()-1, next_activation_gradients);
  for (auto layer_number = layers.size() - 2; layer_number > 0; --layer_number)
  {
    const auto& hidden_layer = layers[layer_number];
    const auto& next_layer = layers[layer_number + 1];

    const auto& hidden_layer_size = hidden_layer.size();
    std::vector<double> current_activation_gradients;
    current_activation_gradients.resize(hidden_layer_size, 0.0);
    for (size_t hidden_layer_number = 0; hidden_layer_number < hidden_layer_size; ++hidden_layer_number)
    {
      const auto& neuron = hidden_layer.get_neuron(unsigned(hidden_layer_number));
      const auto output_value = layers_given_outputs.get_output(layer_number, hidden_layer_number);
      const auto& gradient = neuron.calculate_hidden_gradients(next_layer, next_activation_gradients, output_value);
      current_activation_gradients[hidden_layer_number] = gradient;
    }
    layers_given_outputs.set_gradients(layer_number, current_activation_gradients);
    next_activation_gradients = current_activation_gradients;
    current_activation_gradients = {};
  }
}

void NeuralNetwork::calculate_batch_back_propagation(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& batch_given_outputs, const std::vector<Layer>& layers)
{
  auto batch_size = batch_given_outputs.size();
  if(batch_size == 0)
  {
    std::cerr << "Batch size is 0 so we cannot do back propagation!" << std::endl;
    throw new std::invalid_argument("Batch size is 0 so we cannot do back propagation!");
    return;
  }

  // calculate all the gradients in the batch.
  calculate_batch_back_propagation_gradients(target_outputs, batch_given_outputs, layers);
}

void NeuralNetwork::update_layers_with_gradients(const std::vector<std::vector<GradientsAndOutputs>>& epoch_gradients_outputs, std::vector<Layer>& layers) const
{
  // get the average gradient for all the batches for that epoch
  auto averages = recalculate_gradient_avergages(epoch_gradients_outputs);

  for(const auto& this_epoch_gradients_outputs : epoch_gradients_outputs)
  {
    for(const auto& this_batch_gradients_outputs : this_epoch_gradients_outputs)
    {
      const auto& averaged_batch_gradients_outputs = average_batch_gradients_with_averages(this_batch_gradients_outputs, averages);
      update_layers_with_gradients(averaged_batch_gradients_outputs.get_gradients_and_outputs(), layers);
    }
  }
}

void NeuralNetwork::update_layers_with_gradients(const LayersAndNeurons<std::vector<double>>& activation_gradients, std::vector<Layer>& layers) const
{
  // update the weights in reverse
  for (auto layer_number = layers.size() - 1; layer_number > 0; --layer_number) 
  {
    auto& layer = layers[layer_number];
    auto& previous_layer = layers[layer_number - 1];

    for (unsigned neuron_number = 0; neuron_number < layer.size() - 1; ++neuron_number) 
    {
      auto& neuron = layer.get_neuron(neuron_number);
      const auto& weights_gradients = activation_gradients.get(layer_number-1, neuron_number);
      neuron.update_input_weights(previous_layer, weights_gradients);
    }
  }
}

std::vector<NeuralNetwork::GradientsAndOutputs> NeuralNetwork::calculate_forward_feed(const std::vector<std::vector<double>>& inputs, const std::vector<Layer>& layers) const
{
  const size_t inputs_size = inputs.size();
  std::vector<GradientsAndOutputs> activations_per_layer_per_input;
  activations_per_layer_per_input.reserve(inputs_size);

  for(size_t input_number = 0; input_number < inputs_size; ++input_number)
  {
    activations_per_layer_per_input.emplace_back(calculate_forward_feed(inputs[input_number], layers));
  }
  return activations_per_layer_per_input;
}

NeuralNetwork::GradientsAndOutputs NeuralNetwork::calculate_forward_feed(const std::vector<double>& inputs, const std::vector<Layer>& layers) const
{
  // the return value is the activation values per layers.
  GradientsAndOutputs activations_per_layer(get_topology());

  //  the initial set of output values where we are starting from.
  activations_per_layer.set_outputs(0, inputs);

  // then forward propagate from the input to ... hopefully, the output.
  auto previous_layer_output_values = inputs;
  previous_layer_output_values.push_back(1.0);
  for (size_t layer_number = 1; layer_number < layers.size(); ++layer_number)
  {
    const auto& previous_layer = layers[layer_number - 1];
    auto& this_layer = layers[layer_number];
    std::vector<double> this_output_values;
    this_output_values.reserve(this_layer.size() - 1);
    for (size_t neuron_number = 0; neuron_number < this_layer.size() - 1; ++neuron_number)
    {
      const auto& neuron = this_layer.get_neuron(unsigned(neuron_number));
      this_output_values.emplace_back(neuron.calculate_forward_feed(previous_layer, previous_layer_output_values));
    }

    activations_per_layer.set_outputs(
        layer_number,
        this_output_values
        );
    previous_layer_output_values = std::move(this_output_values);
    previous_layer_output_values.push_back(1.0);
  }
  // return the output values per layer.
  return activations_per_layer;
}

std::vector<double> NeuralNetwork::caclulate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer)
{
  const size_t output_layer_size = output_layer.size();
  std::vector<double> activation_gradients = {};
  activation_gradients.reserve(output_layer_size);
  for (size_t n = 0; n < output_layer_size - 1; ++n) // ignore bias
  {
    const auto& neuron = output_layer.get_neuron(unsigned(n));
    activation_gradients.emplace_back(neuron.calculate_output_gradients(
      target_outputs[n], given_outputs[n]));
  }
  activation_gradients.emplace_back(0);  //  add bias we ignored above
  return activation_gradients;
}