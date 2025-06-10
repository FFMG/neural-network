#include "adaptivelearningratescheduler.h"
#include "neuralnetwork.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <string>

static const double RecentAverageSmoothingFactor = 100.0;
static const long long IntervalErorCheckInSeconds = 15;

NeuralNetwork::NeuralNetwork(
  const std::vector<unsigned>& topology, 
  const activation::method& activation
  ) :
  _error(0.0),
  _topology(topology),
  _layers(nullptr),
  _activation_method(activation)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if(topology.size() < 2)
  {
    std::cerr << "The topology is not value, you must have at least 2 layers." << std::endl;
    throw std::invalid_argument("The topology is not value, you must have at least 2 layers.");
  }
  const auto& number_of_layers = topology.size();
  _layers = new std::vector<Layer>();

  // add the input layer
  auto layer = Layer::create_input_layer(topology[0], topology[1], _activation_method);
  _layers->push_back(layer);
  
  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    const auto& previous_layer = _layers->back();
    layer = Layer::create_hidden_layer(num_neurons_current_layer, num_neurons_next_layer, previous_layer, _activation_method);
    _layers->push_back(layer);
  }

  // finally, the output layer
  layer = Layer::create_output_layer(topology.back(), _layers->back(), _activation_method);
  _layers->push_back(layer);
}

NeuralNetwork::NeuralNetwork(
  const std::vector<Layer>& layers, 
  const activation::method& activation,
  double error
  ) :
  _error(error),
  _layers(nullptr),
  _activation_method(activation)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
  _activation_method(src._activation_method)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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

activation::method NeuralNetwork::get_activation_method() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _activation_method;
}

const std::vector<Layer>& NeuralNetwork::get_layers() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return *_layers;
}

const std::vector<unsigned>& NeuralNetwork::get_topology() const
{
  return _topology;
}

std::vector<size_t> NeuralNetwork::get_shuffled_indexes(size_t raw_size)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<size_t> shuffled_indexes(raw_size);
  std::iota(shuffled_indexes.begin(), shuffled_indexes.end(), 0);

  std::random_device rd;
  std::mt19937 gen(rd()); // Mersenne Twister RNG
  std::shuffle(shuffled_indexes.begin(), shuffled_indexes.end(), gen);
  return shuffled_indexes;
}

void NeuralNetwork::create_shuffled_indexes(size_t raw_size, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto shuffled_indexes = get_shuffled_indexes(raw_size);
  assert(raw_size == shuffled_indexes.size());

  break_shuffled_indexes(shuffled_indexes, data_is_unique, training_indexes, checking_indexes, final_check_indexes);
}

void NeuralNetwork::break_shuffled_indexes(const std::vector<size_t>& shuffled_indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto layers = *_layers;
  return calculate_forward_feed(inputs, layers).output_back();
}

long double NeuralNetwork::get_error() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _error;
}

std::vector<std::vector<double>> NeuralNetwork::think(const std::vector<std::vector<double>>& inputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<std::vector<double>> outputs(inputs.size(), std::vector<double>(1));
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    outputs[i] = think(inputs[i]);
  }
  return outputs;
}

std::vector<NeuralNetwork::GradientsAndOutputs> NeuralNetwork::train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  ) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto batch_gradients_outputs = calculate_forward_feed(inputs_begin, inputs_begin+size, *_layers);
  calculate_batch_back_propagation(outputs_begin, size, batch_gradients_outputs, *_layers);
  return batch_gradients_outputs;
}

void NeuralNetwork::train(
  const std::vector<std::vector<double>>& training_inputs,
  const std::vector<std::vector<double>>& training_outputs,
  double learning_rate,
  int number_of_epoch,
  int batch_size,
  bool data_is_unique,
  const std::function<bool(int, NeuralNetwork&)>& progress_callback
)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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

  std::cout << "Started trainning with " << training_inputs.size() << " inputs, " << number_of_epoch << " epoch and batch size " << batch_size << "." << std::endl;

  const auto interval = std::chrono::seconds(IntervalErorCheckInSeconds);
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

  TaskQueuePool<std::vector<std::vector<double>>>* errorPool = nullptr;
  if (batch_size > 1)
  {
    errorPool = new TaskQueuePool<std::vector<std::vector<double>>>();
    errorPool->start();
  }

  // get the indexes
  std::vector<size_t> training_indexes;
  std::vector<size_t> checking_indexes;
  std::vector<size_t> final_check_indexes;
  create_shuffled_indexes(training_inputs.size(), data_is_unique, training_indexes, checking_indexes, final_check_indexes);

  std::cout << "Tainning will use: " << std::endl;
  std::cout << "  " << training_indexes .size() << " training indexes." << std::endl;
  std::cout << "  " << checking_indexes.size() << " in training error check indexes." << std::endl;
  std::cout << "  " << final_check_indexes.size() << " final error check indexes." << std::endl;
  std::cout << "  " << "Learning rate:" << std::fixed << std::setprecision(15) << learning_rate << std::endl;

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

  auto task_pool = TaskQueuePool<std::vector<GradientsAndOutputs>>();
  if(batch_size > 1)
  {
    task_pool.start();
  }

  const auto training_indexes_size = training_indexes.size();

  size_t num_batches = (training_indexes_size + batch_size - 1) / batch_size;
  std::vector<std::vector<GradientsAndOutputs>> epoch_gradients_outputs;
  epoch_gradients_outputs.reserve(num_batches);

  AdaptiveLearningRateScheduler learning_rate_scheduler;
  
  for (auto epoch = 0; epoch < number_of_epoch; ++epoch)
  {
    // and the epoch outputs
    epoch_gradients_outputs.clear();

    // create the batches
    for( size_t start_index = 0; start_index < training_indexes_size; start_index += batch_size)
    {
      if(batch_size > 1)
      {
        const size_t end_size = std::min(start_index + batch_size, training_indexes_size);
        const size_t total_size = end_size - start_index;
        task_pool.enqueue([=]()
          {
            auto train = train_single_batch(
              training_inputs.begin() + start_index,
              training_outputs.begin() + start_index,
              total_size);
          return train;
        });
      }
      else
      {
        //  size is 1, it is faster to not use a thread.
        const size_t total_size = 1;
        epoch_gradients_outputs.emplace_back(
          train_single_batch(
            training_inputs.begin() + start_index,
            training_outputs.begin() + start_index,
            total_size)
        );
      }
    }

    // Collect the results
    if (batch_size > 1)
    {
      epoch_gradients_outputs = task_pool.get();
    }
    update_layers_with_gradients(epoch_gradients_outputs, *_layers, learning_rate);

    if (epoch % 10000 == 0)
    {
      double avg_ns = task_pool.average();
      auto total_epoch_duration_size = task_pool.total_tasks();
      std::cout << "Average time per call: " << std::fixed << std::setprecision(2) << avg_ns << " ns (" << total_epoch_duration_size << " calls)." << std::endl;
    }

    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = current_time - last_callback_time;
    if (elapsed_time >= interval)
    {
      // do an error check to see if we need to adapt.
      _error = calculate_error(checking_training_inputs, checking_training_outputs, batch_size, *_layers, errorPool);
      learning_rate = learning_rate_scheduler.update(_error, learning_rate);

      if (progress_callback != nullptr)
      {
        auto percent = (int)(((float)epoch / number_of_epoch)*100);
        if( !progress_callback(percent, *this))
        {
          return;
        }
      }
      last_callback_time = current_time;
    }
  }

  task_pool.stop();
  double avg_ns = task_pool.average();
  auto total_epoch_duration_size = task_pool.total_tasks();
  std::cout << "Average time per call: " << std::fixed << std::setprecision (2) << avg_ns << " ns (" << total_epoch_duration_size << " calls)." << std::endl;

  // final error checking
  std::vector<std::vector<double>> final_training_inputs = {};
  std::vector<std::vector<double>> final_training_outputs = {};
  create_batch_from_indexes(final_check_indexes, training_inputs, training_outputs, final_training_inputs, final_training_outputs);
  _error = calculate_error(final_training_inputs, final_training_outputs, batch_size, *_layers, errorPool);
  std::cout << "Final Test Error: " << std::fixed << std::setprecision (15) << _error << std::endl;

  if (errorPool != nullptr)
  {
    errorPool->stop();
    delete errorPool;
  }

  // final callback if needed
  if (progress_callback != nullptr)
  {
    progress_callback(100, *this);
  }
}

void NeuralNetwork::recalculate_gradient_avergages(const std::vector<std::vector<GradientsAndOutputs>>& epoch_gradients_outputs, LayersAndNeurons<double>& averages) const
{  
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  // this is the number of batches we had this epoch
  if( 0 == epoch_gradients_outputs.size())
  {
    return;
  }

  size_t total_epoch_size = std::accumulate(
    epoch_gradients_outputs.begin(), epoch_gradients_outputs.end(), size_t(0),
    [](size_t sum, const auto& v) {
        return sum + v.size();
    }
  );

  // go around all the layers/neurons to get the averages per batch
  const auto layers_size = averages.number_layers();
  for(unsigned layer_number = 0; layer_number < layers_size; ++layer_number)
  {
    // we get the number of neurons from the first batch.
    const auto neurons_size = averages.number_neurons(layer_number);
    for(unsigned neuron_number = 0; neuron_number < neurons_size; ++neuron_number)
    {
      double epoch_sum = 0.0;
      // then we look at all the batches to get the gradients.
      for(const auto& epoch_gradient_output : epoch_gradients_outputs)
      {
        // make sure that we get the actual size as they might not all be the same size.
        for(const auto& epoch_gradient_output_batch : epoch_gradient_output)
        {
          epoch_sum += epoch_gradient_output_batch.get_gradient(layer_number, neuron_number);
        }
      }
      auto average_gradient = (epoch_sum / static_cast<double>(total_epoch_size));
      averages.set(layer_number, neuron_number, average_gradient);
    }
  }
}

double NeuralNetwork::calculate_error(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int batch_size, std::vector<Layer>& layers, TaskQueuePool<std::vector<std::vector<double>>>* errorPool) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const size_t training_indexes_size = training_inputs.size();

  std::vector<std::vector<double>> predictions;
  predictions.reserve(training_indexes_size);

  for (size_t start_index = 0; start_index < training_indexes_size; start_index+= batch_size)
  {
    if(errorPool != nullptr)
    {
      const size_t end_index = std::min(start_index + batch_size, training_indexes_size);
      errorPool->enqueue(
        [=]()
        {
          auto outputs = calculate_forward_feed(
            training_inputs.begin() + start_index,
            training_inputs.begin() + end_index,
            layers);
          std::vector<std::vector<double>> predictions;
          predictions.reserve(outputs.size());
          for(const auto&output : outputs)
          {
            auto local_predictions = output.output_back();
            predictions.emplace_back(std::move(local_predictions));
          }
          return predictions;
        });
    }
    else
    {
      const auto& inputs = training_inputs[start_index];
      auto all_outputs = calculate_forward_feed(inputs, layers);
      predictions.emplace_back(all_outputs.output_back());
    }
  }

  if(errorPool != nullptr)
  {
    auto task_predictions = errorPool->get();
    for (auto& task_prediction : task_predictions)
    {
      predictions.insert(predictions.end(), 
        std::make_move_iterator(task_prediction.begin()), 
        std::make_move_iterator(task_prediction.end()));
    }
  }

  return calculate_rmse_error(training_outputs, predictions);
  // return calculate_mae_error(ground_truth, predictions);
  //return calculate_huber_loss(ground_truth, predictions);
}

double NeuralNetwork::calculate_huber_loss(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto mean_squared_error = calculate_mse_error(ground_truth, predictions);
  return std::sqrt(mean_squared_error); // RMSE
}

double NeuralNetwork::calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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

void NeuralNetwork::average_batch_gradients_with_averages(const GradientsAndOutputs& activation_gradients, const LayersAndNeurons<double>& layers_neurons_averages, LayersAndNeurons<std::vector<double>>& gradients_and_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<double> outputs_gradients;
  
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
      const size_t number_outputs = activation_gradients.num_outputs(static_cast<unsigned>(layer_number));
      outputs_gradients.reserve(number_outputs);
      
      // get the gradient this neuron in the next layer
      const auto& next_layer_neuron_gradient = activation_gradients.get_gradient(static_cast<unsigned>(layer_number+1), static_cast<unsigned>(neuron_number));
      for (size_t output_number = 0; output_number < number_outputs; ++output_number)
      {
        const auto& layer_neuron_output = activation_gradients.get_output(static_cast<unsigned>(layer_number), static_cast<unsigned>(output_number));
        outputs_gradients.emplace_back(layer_neuron_output * next_layer_neuron_gradient);
      }
      
      // we can now set the output + average gradient for that layer/neuron.
      gradients_and_outputs.set(static_cast<unsigned>(layer_number), static_cast<unsigned>(neuron_number), std::move(outputs_gradients));
      outputs_gradients.clear();
    }
  }
}

void NeuralNetwork::calculate_batch_back_propagation_gradients(
    const std::vector<std::vector<double>>::const_iterator outputs_begin, 
    const size_t outputs_size, 
    std::vector<GradientsAndOutputs>& layers_given_outputs, 
    const std::vector<Layer>& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  assert(outputs_size == layers_given_outputs.size());
  for(size_t i = 0; i < outputs_size; ++i)
  {
    const auto& output = outputs_begin + i;
    calculate_back_propagation_gradients(*output, layers_given_outputs[i], layers);
  }
}

void NeuralNetwork::calculate_batch_back_propagation_gradients(const std::vector<std::vector<double>>& target_outputs, std::vector<GradientsAndOutputs>& layers_given_outputs, const std::vector<Layer>& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  assert(target_outputs.size() == layers_given_outputs.size());
  const auto& target_outputs_size = target_outputs.size();
  for(size_t i = 0; i < target_outputs_size; ++i)
  {
    calculate_back_propagation_gradients(target_outputs[i], layers_given_outputs[i], layers);
  }
}

void NeuralNetwork::calculate_back_propagation_gradients(const std::vector<double>& target_outputs, GradientsAndOutputs& layers_given_outputs, const std::vector<Layer>& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  assert(target_outputs.size() == layers_given_outputs.output_back().size());

  // input layer is all 0, (bias is included)
  auto input_gradients = std::vector<double>(layers.front().size(), 0.0);
  layers_given_outputs.set_gradients(0, input_gradients);
  
  // set the output gradient
  const auto& output_layer = layers.back();
  auto next_activation_gradients = caclulate_output_gradients(target_outputs, layers_given_outputs.output_back(), output_layer);
  layers_given_outputs.set_gradients(static_cast<unsigned>(layers.size()-1), next_activation_gradients);
  for (auto layer_number = layers.size() - 2; layer_number > 0; --layer_number)
  {
    const auto& hidden_layer = layers[layer_number];
    const auto& next_layer = layers[layer_number + 1];

    const auto& hidden_layer_size = hidden_layer.size();
    std::vector<double> current_activation_gradients;
    current_activation_gradients.resize(hidden_layer_size, 0.0);
    for (size_t hidden_layer_number = 0; hidden_layer_number < hidden_layer_size; ++hidden_layer_number)
    {
      const auto& neuron = hidden_layer.get_neuron(static_cast<unsigned>(hidden_layer_number));
      const auto output_value = layers_given_outputs.get_output(static_cast<unsigned>(layer_number), static_cast<unsigned>(hidden_layer_number));
      const auto& gradient = neuron.calculate_hidden_gradients(next_layer, next_activation_gradients, output_value);
      current_activation_gradients[hidden_layer_number] = gradient;
    }
    layers_given_outputs.set_gradients(static_cast<unsigned>(layer_number), current_activation_gradients);
    next_activation_gradients = current_activation_gradients;
    current_activation_gradients = {};
  }
}

void NeuralNetwork::calculate_batch_back_propagation(
    const std::vector<std::vector<double>>::const_iterator outputs_begin, 
    const size_t outputs_size,
    std::vector<GradientsAndOutputs>& batch_given_outputs, 
    const std::vector<Layer>& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto batch_size = batch_given_outputs.size();
  if(batch_size == 0)
  {
    std::cerr << "Batch size is 0 so we cannot do back propagation!" << std::endl;
    throw new std::invalid_argument("Batch size is 0 so we cannot do back propagation!");
    return;
  }

  // calculate all the gradients in the batch.
  calculate_batch_back_propagation_gradients(outputs_begin, outputs_size, batch_given_outputs, layers);
}

void NeuralNetwork::update_layers_with_gradients(const std::vector<std::vector<GradientsAndOutputs>>& epoch_gradients_outputs, std::vector<Layer>& layers, double learning_rate) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  // Prepare result vector with proper dimensions
  LayersAndNeurons<std::vector<double>> gradients_and_outputs(get_topology(), true, true);
  LayersAndNeurons<double> averages(get_topology(), false, true);

  // get the average gradient for all the batches for that epoch
  recalculate_gradient_avergages(epoch_gradients_outputs, averages);
  for(const auto& this_epoch_gradients_outputs : epoch_gradients_outputs)
  {
    for(const auto& this_batch_gradients_outputs : this_epoch_gradients_outputs)
    {
      average_batch_gradients_with_averages(this_batch_gradients_outputs, averages, gradients_and_outputs);
      update_layers_with_gradients(gradients_and_outputs, layers, learning_rate);      
    }
  }
}

void NeuralNetwork::update_layers_with_gradients(const LayersAndNeurons<std::vector<double>>& activation_gradients, std::vector<Layer>& layers, double learning_rate) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  // update the weights in reverse
  for (size_t layer_number = layers.size() - 1; layer_number > 0; --layer_number) 
  {
    auto& layer = layers[layer_number];
    auto& previous_layer = layers[layer_number - 1];

    const size_t layer_size = layer.size() - 1; //  exclude bia
    for (size_t neuron_number = 0; neuron_number < layer_size; ++neuron_number) 
    {
      auto& neuron = layer.get_neuron(static_cast<unsigned>(neuron_number));
      const auto& weights_gradients = activation_gradients.get(static_cast<unsigned>(layer_number-1), static_cast<unsigned>(neuron_number));
      neuron.update_input_weights(previous_layer, weights_gradients, learning_rate);
    }
  }
}

std::vector<NeuralNetwork::GradientsAndOutputs> NeuralNetwork::calculate_forward_feed(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator inputs_end, 
    const std::vector<Layer>& layers) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  size_t size = inputs_end - inputs_begin;
  std::vector<GradientsAndOutputs> activations_per_layer_per_input;
  activations_per_layer_per_input.reserve(size);

  for(size_t input_number = 0; input_number < size; ++input_number)
  {
    const auto& input = inputs_begin+input_number;
    activations_per_layer_per_input.emplace_back(calculate_forward_feed(*input, layers));
  }
  return activations_per_layer_per_input;
}

NeuralNetwork::GradientsAndOutputs NeuralNetwork::calculate_forward_feed(const std::vector<double>& inputs, const std::vector<Layer>& layers) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

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
      static_cast<unsigned>(layer_number),
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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