#include "adaptivelearningratescheduler.h"
#include "neuralnetwork.h"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>

static const double RecentAverageSmoothingFactor = 100.0;
static const long long IntervalErorCheckInSeconds = 15;

NeuralNetworkHelper::NeuralNetworkHelperMetrics NeuralNetworkHelper::calculate_forecast_metric(NeuralNetworkOptions::ErrorCalculation error_type) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  return _neural_network->calculate_forecast_metric(error_type);
}

std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> NeuralNetworkHelper::calculate_forecast_metrics(const std::vector<NeuralNetworkOptions::ErrorCalculation>& error_types) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
  return _neural_network->calculate_forecast_metrics(error_types);
}

NeuralNetwork::NeuralNetwork(const NeuralNetworkOptions options) :
  _learning_rate(0.0),
  _options(options),
  _neural_network_helper(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const auto& topology = options.topology();
  const auto& number_of_layers = topology.size();
  _layers.reserve(number_of_layers);

  // add the input layer
  auto layer = Layer::create_input_layer(topology[0], topology[1], options.logger());
  _layers.emplace_back(std::move(layer));
  
  // then the hidden layers
  for (size_t layer_number = 1; layer_number < number_of_layers -1; ++layer_number)
  {
    auto num_neurons_current_layer = topology[layer_number];
    auto num_neurons_next_layer = topology[layer_number + 1];
    const auto& previous_layer = _layers.back();
    layer = Layer::create_hidden_layer(num_neurons_current_layer, num_neurons_next_layer, previous_layer, options.hidden_activation_method(), options.optimiser_type(), options.logger());
    _layers.emplace_back(std::move(layer));
  }

  // finally, the output layer
  layer = Layer::create_output_layer(topology.back(), _layers.back(), options.output_activation_method(), options.optimiser_type(), options.logger());
  _layers.emplace_back(std::move(layer));
}

NeuralNetwork::NeuralNetwork(
  const std::vector<unsigned>& topology, 
  const activation::method& hidden_layer_activation, 
  const activation::method& output_layer_activation,
  const Logger& logger
  ) :
  NeuralNetwork(NeuralNetworkOptions::create(topology)
    .with_hidden_activation_method(hidden_layer_activation)
    .with_output_activation_method(output_layer_activation)
    .with_logger(logger)
    .build())
{
}

NeuralNetwork::NeuralNetwork(
  const std::vector<Layer>& layers, 
  const activation::method& hidden_layer_activation, 
  const activation::method& output_layer_activation,
  const Logger& logger
  ) :
  _learning_rate(0.0),
  _options(NeuralNetworkOptions::create(layers)    
    .with_hidden_activation_method(hidden_layer_activation)
    .with_output_activation_method(output_layer_activation)
    .with_logger(logger)
    .build()),
  _neural_network_helper(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  _layers.reserve(layers.size());
  for (const auto& layer : layers)
  {
    auto copy_layer = Layer(layer);
    _layers.emplace_back(std::move(copy_layer));
  }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& src) :
  _learning_rate(src._learning_rate),
  _options(src._options),
  _neural_network_helper(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (src._neural_network_helper != nullptr)
  {
    _neural_network_helper = new NeuralNetworkHelper(*src._neural_network_helper);
  }

  _layers.reserve(src._layers.size());
  for (const auto& layer : src._layers)
  {
    auto copy_layer = Layer(layer);
    _layers.emplace_back(std::move(copy_layer));
  }
}

NeuralNetwork::~NeuralNetwork()
{
  delete _neural_network_helper;
  _neural_network_helper = nullptr;
}

const activation::method& NeuralNetwork::get_output_activation_method() const
{
  return _options.output_activation_method();
}

const activation::method& NeuralNetwork::get_hidden_activation_method() const
{
  return _options.hidden_activation_method();
}

const std::vector<Layer>& NeuralNetwork::get_layers() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _layers;
}

const std::vector<unsigned>& NeuralNetwork::get_topology() const
{
  return _options.topology();
}

std::vector<size_t> NeuralNetwork::get_shuffled_indexes(size_t raw_size) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<size_t> shuffled_indexes(raw_size);
  std::iota(shuffled_indexes.begin(), shuffled_indexes.end(), 0);

  std::random_device rd;
  std::mt19937 gen(rd()); // Mersenne Twister RNG
  std::shuffle(shuffled_indexes.begin(), shuffled_indexes.end(), gen);
  return shuffled_indexes;
}

void NeuralNetwork::create_shuffled_indexes(NeuralNetworkHelper& neural_network_helper, size_t raw_size, bool data_is_unique) const
{
  std::vector<size_t> training_indexes;
  std::vector<size_t> checking_indexes;
  std::vector<size_t> final_check_indexes;

  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto shuffled_indexes = get_shuffled_indexes(raw_size);
  assert(raw_size == shuffled_indexes.size());

  break_shuffled_indexes(shuffled_indexes, data_is_unique, training_indexes, checking_indexes, final_check_indexes);

  neural_network_helper.move_indexes(std::move(training_indexes), std::move(checking_indexes), std::move(final_check_indexes));
}

void NeuralNetwork::break_shuffled_indexes(const std::vector<size_t>& shuffled_indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  // break the indexes into parts
  size_t total_size = shuffled_indexes.size();
  size_t training_size = static_cast<size_t>(std::round(total_size * 0.80));
  size_t checking_size = static_cast<size_t>(std::round(total_size * 0.15));
  if(training_size+checking_size == total_size || checking_size == 0)
  {
    // in the case of small training models we might not have enough to split anything
    _options.logger().log_warning("Training batch size does not allow for error checking batch!");
    training_indexes = shuffled_indexes;
    checking_indexes = {shuffled_indexes.front()};
    final_check_indexes = {shuffled_indexes.back()};
    return;
  }
  assert(training_size + checking_size < total_size); // make sure we don't get more than 100%
  if(training_size + checking_size > total_size) // make sure we don't get more than 100%
  {
    _options.logger().log_error("Logic error, unable to do a final batch error check.");
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

void NeuralNetwork::recreate_batch_from_indexes(NeuralNetworkHelper& neural_network_helper, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const
{
  auto indexes = neural_network_helper.training_indexes();
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::shuffle(indexes.begin(), indexes.end(), gen);
  neural_network_helper.move_training_indexes(std::move(indexes));
  create_batch_from_indexes(neural_network_helper.training_indexes(), training_inputs, training_outputs, shuffled_training_inputs, shuffled_training_outputs);
}

void NeuralNetwork::create_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  shuffled_training_inputs.clear();
  shuffled_training_outputs.clear();
  shuffled_training_inputs.reserve(shuffled_indexes.size());
  shuffled_training_outputs.reserve(shuffled_indexes.size());
  for( auto shuffled_index : shuffled_indexes)
  {
    shuffled_training_inputs.emplace_back(training_inputs[shuffled_index]);
    shuffled_training_outputs.emplace_back(training_outputs[shuffled_index]);
  }
}

std::vector<double> NeuralNetwork::think(const std::vector<double>& inputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  GradientsAndOutputs gradients(get_topology());
  {
    std::shared_lock lock(_mutex);
    calculate_forward_feed(gradients, inputs, _layers);
  }
  return gradients.output_back();
}

double NeuralNetwork::get_learning_rate() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _learning_rate;
}

NeuralNetworkHelper::NeuralNetworkHelperMetrics NeuralNetwork::calculate_forecast_metric(NeuralNetworkOptions::ErrorCalculation error_type) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto results = calculate_forecast_metrics({ error_type }, false);
  return results.front();
}

std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> NeuralNetwork::calculate_forecast_metrics(const std::vector<NeuralNetworkOptions::ErrorCalculation>& error_types) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return calculate_forecast_metrics(error_types, false);
}

std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> NeuralNetwork::calculate_forecast_metrics(const std::vector<NeuralNetworkOptions::ErrorCalculation>& error_types, bool final_check) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> errors = {};
  errors.reserve(errors.size());

  if (nullptr == _neural_network_helper)
  {
    for (size_t index = 0; index < error_types.size(); ++index)
    {
      errors.emplace_back( NeuralNetworkHelper::NeuralNetworkHelperMetrics(0.0, error_types[index]));
    }
    _options.logger().log_warning("Trying to get training metrics when no training was done!");
    return errors;
  }

  const NeuralNetworkHelper& helper = *_neural_network_helper;
  const auto& training_inputs = helper.training_inputs();
  const auto& taining_outputs = helper.training_outputs();

  const std::vector<size_t>* checks_indexes = 0;
  size_t prediction_size = 0;
  if (final_check)
  {
    checks_indexes = &helper.final_check_indexes();
  }
  else
  {
    checks_indexes = &helper.checking_indexes();
  }
  prediction_size = checks_indexes->size();
  if (prediction_size == 0)
  {
    for (size_t index = 0; index < error_types.size(); ++index)
    {
      errors.emplace_back(NeuralNetworkHelper::NeuralNetworkHelperMetrics(0.0, error_types[index]));
    }
    return errors;
  }

  std::vector<std::vector<double>> predictions;
  std::vector<std::vector<double>> checking_outputs;
  predictions.reserve(prediction_size);
  checking_outputs.reserve(prediction_size);

  GradientsAndOutputs gradients(get_topology());
  std::shared_lock lock(_mutex);
  {
    for (size_t index = 0; index < prediction_size; ++index)
    {
      const auto& checks_index = (*checks_indexes)[index];
      const auto& inputs = training_inputs[checks_index];
      calculate_forward_feed(gradients, inputs, _layers);
      predictions.emplace_back(gradients.output_back());
      gradients.zero();

      // set the output we will need it just now.
      checking_outputs.emplace_back(taining_outputs[checks_index]);
    }
  }// release the lock


  for (size_t index = 0; index < error_types.size(); ++index)
  {
    errors.emplace_back(
      NeuralNetworkHelper::NeuralNetworkHelperMetrics(
      calculate_error(error_types[index], checking_outputs, predictions),
        error_types[index]));
  }
  return errors;
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

NeuralNetwork::GradientsAndOutputs NeuralNetwork::train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  ) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  GradientsAndOutputs gradients(_options.topology(), static_cast<unsigned>(size));
  if(size == 1)
  {
    calculate_forward_feed(gradients, *inputs_begin, _layers);
    calculate_back_propagation(gradients, *outputs_begin, _layers);
    return gradients;
  }

  for(size_t index = 0; index < size; ++index)
  {
    GradientsAndOutputs this_gradients(_options.topology());
    calculate_forward_feed(gradients, *inputs_begin, _layers);
    calculate_back_propagation(gradients, *outputs_begin, _layers);
    gradients.add(this_gradients);
  } 
  return gradients;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& training_inputs,const std::vector<std::vector<double>>& training_outputs)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  const auto& number_of_epoch = _options.number_of_epoch();
  _learning_rate = _options.learning_rate();
  const auto& progress_callback = _options.progress_callback();
  const auto& batch_size = _options.batch_size();

  dump_layer_info();

  if(batch_size <=0 || batch_size > static_cast<int>(training_inputs.size()))
  {
    _options.logger().log_error("The batch size if either -ve or too large for the training sample.");
    throw std::invalid_argument("The batch size if either -ve or too large for the training sample.");
  }
  if(training_outputs.size() != training_inputs.size())
  {
    _options.logger().log_error("The number of training samples does not match the number of expected outputs.");
    throw std::invalid_argument("The number of training samples does not match the number of expected outputs.");
  }

  _options.logger().log_info("Started training with ", training_inputs.size(), " inputs, ", number_of_epoch, " epoch and batch size ", batch_size, ".");

  TaskQueuePool<GradientsAndOutputs>* task_pool = nullptr;
  if (batch_size > 1)
  {
    task_pool = new TaskQueuePool<GradientsAndOutputs>(
      _options.logger(),
      _options.number_of_threads());
  }

  delete _neural_network_helper;
  _neural_network_helper = new NeuralNetworkHelper(*this, _learning_rate, number_of_epoch, training_inputs, training_outputs);
  if (progress_callback != nullptr)
  {
    if (!progress_callback(*_neural_network_helper))
    {
      _options.logger().log_warning("Progress callback function returned false before training started, closing now!");
      return;
    }
  }
  create_shuffled_indexes(*_neural_network_helper, training_inputs.size(), _options.data_is_unique());

  // with the indexes, create the check training 
  std::vector<std::vector<double>> checking_training_inputs = {};
  std::vector<std::vector<double>> checking_training_outputs = {};
  create_batch_from_indexes(_neural_network_helper->checking_indexes(), training_inputs, training_outputs, checking_training_inputs, checking_training_outputs);

  // create the batch training
  std::vector<std::vector<double>> batch_training_inputs = {};
  std::vector<std::vector<double>> batch_training_outputs = {};
  create_batch_from_indexes(_neural_network_helper->training_indexes(), training_inputs, training_outputs, batch_training_inputs, batch_training_outputs);

  // final error checking
  std::vector<std::vector<double>> final_training_inputs = {};
  std::vector<std::vector<double>> final_training_outputs = {};
  create_batch_from_indexes(_neural_network_helper->final_check_indexes(), training_inputs, training_outputs, final_training_inputs, final_training_outputs);

  // add a log message.
  log_training_info(training_inputs, training_outputs);

  // build the training output batch so we can use it for error calculations
  const auto training_indexes_size = _neural_network_helper->training_indexes().size();
  std::vector<std::vector<double>> training_outputs_batch = {};
  training_outputs_batch.reserve(training_indexes_size);
  for (const auto& training_index : _neural_network_helper->training_indexes())
  {
    const auto& outputs = training_outputs[training_index];
    training_outputs_batch.push_back(outputs);
  }

  size_t num_batches = (training_indexes_size + batch_size - 1) / batch_size;
  std::vector<GradientsAndOutputs> epoch_gradients;
  epoch_gradients.reserve(num_batches);

  // learning rate decay
  const auto initial_learning_rate = _neural_network_helper->learning_rate();
  const auto learning_rate_decay_rate = _options.learning_rate_decay_rate() == 0 ? 0 : std::log(1.0 / _options.learning_rate_decay_rate()) / number_of_epoch;

  // learning rate boost.
  const auto learning_rate_restart_rate = static_cast<int>(_options.learning_rate_restart_rate() / 100.0 * number_of_epoch); // every 10%

  // the current learning rate base.
  double learning_rate_base = initial_learning_rate;

  AdaptiveLearningRateScheduler learning_rate_scheduler(_options.logger());
  
  for (auto epoch = 0; epoch < number_of_epoch; ++epoch)
  {
    _neural_network_helper->set_epoch(epoch);
    _learning_rate = _neural_network_helper->learning_rate();

    // create the batches
    for (size_t start_index = 0; start_index < training_indexes_size; start_index += batch_size)
    {
      if (task_pool != nullptr)
      {
        const size_t end_size = std::min(start_index + batch_size, training_indexes_size);
        const size_t total_size = end_size - start_index;
        task_pool->enqueue([=]()
          {
            auto train = train_single_batch(
              batch_training_inputs.begin() + start_index,
              batch_training_outputs.begin() + start_index,
              total_size);
            return train;
          });
      }
      else
      {
        //  size is 1, it is faster to not use a thread.
        const size_t total_size = 1;
        const auto& single_batch = train_single_batch(
            batch_training_inputs.begin() + start_index,
            batch_training_outputs.begin() + start_index,
            total_size);

        apply_weight_gradients(_layers, single_batch, _neural_network_helper->learning_rate(), epoch);
      }
    }
    MYODDWEB_PROFILE_MARK();

    // Collect the results
    if (task_pool != nullptr)
    {
      epoch_gradients = task_pool->get();
      apply_weight_gradients(_layers, epoch_gradients, _neural_network_helper->learning_rate(), epoch);

      // then re-shuffle everything
      recreate_batch_from_indexes(*_neural_network_helper, training_inputs, training_outputs, batch_training_inputs, batch_training_outputs);

      // clear it for next time
      epoch_gradients.clear();
    }

    // decay the learning rate.
    if(learning_rate_decay_rate != 0 )
    {
      _neural_network_helper->set_learning_rate(learning_rate_base * std::exp(-learning_rate_decay_rate * epoch));
    }
    
    // Boost the baseline every N epochs
    if (epoch != 0 && epoch % learning_rate_restart_rate == 0 && _options.learning_rate_restart_boost() != 1.0)
    {
      learning_rate_base *= _options.learning_rate_restart_boost();
      _options.logger().log_debug("Learning rate boost to ", std::fixed, std::setprecision(15), learning_rate_base);
    }

    // then get the scheduler if we can improve it further.
    if (_options.adaptive_learning_rate())
    {
      auto metric = calculate_forecast_metrics(
        {
          NeuralNetworkOptions::ErrorCalculation::rmse,
        }, false);
      _neural_network_helper->set_learning_rate(learning_rate_scheduler.update(metric[0].error(), _neural_network_helper->learning_rate(), epoch, number_of_epoch));
    }    
    if (progress_callback != nullptr)
    {
      if (!progress_callback(*_neural_network_helper))
      {
        _options.logger().log_warning("Progress callback function returned false during training, closing now!");
        return;
      }
    }
    MYODDWEB_PROFILE_MARK();
  }

  if(task_pool != nullptr)
  {
    task_pool->stop();
    double avg_ns = task_pool->average();
    auto total_epoch_duration_size = task_pool->total_tasks();
    _options.logger().log_debug("Average time per call: ", std::fixed, std::setprecision (2), avg_ns, " ns (", total_epoch_duration_size, " calls).");
    delete task_pool;
  }

  auto metrics = calculate_forecast_metrics(
    {
      NeuralNetworkOptions::ErrorCalculation::rmse,
      NeuralNetworkOptions::ErrorCalculation::mape 
    }, true);

  _options.logger().log_info("Final RMSE Error: ", std::fixed, std::setprecision (15), metrics[0].error());
  _options.logger().log_info("Final Forecast accuracy (MAPE): ", std::fixed, std::setprecision (15), metrics[1].error());

  // finaly learning rate
  _options.logger().log_info("Final Learning rate: ", std::fixed, std::setprecision(15), _neural_network_helper->learning_rate());

  // final callback to show 100% done.
  if (progress_callback != nullptr)
  {
    progress_callback(*_neural_network_helper);
  }

  dump_layer_info();

  MYODDWEB_PROFILE_MARK();
}

double NeuralNetwork::calculate_error(NeuralNetworkOptions::ErrorCalculation error_type, const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  switch (error_type)
  {
  case NeuralNetworkOptions::ErrorCalculation::none:
    return 0.0;

  case NeuralNetworkOptions::ErrorCalculation::huber_loss:
    return calculate_huber_loss_error(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::mae:
    return calculate_mae_error(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::mse:
    return calculate_mse_error(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::rmse:
    return calculate_rmse_error(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::mape:
    return calculate_forecast_accuracy_mape(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::smape:
    return calculate_forecast_accuracy_smape(ground_truth, predictions);
  }

  _options.logger().log_error("Unknown ErrorCalculation type!");
  throw std::invalid_argument("Unknown ErrorCalculation type!");
}

double NeuralNetwork::calculate_huber_loss_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (ground_truth.size() != predictions.size())
  {
    _options.logger().log_error("Mismatched number of samples");
    throw std::invalid_argument("Mismatched number of samples");
  }

  double total_loss = 0.0;
  size_t count = 0;

  for (size_t i = 0; i < ground_truth.size(); ++i)
  {
    if (ground_truth[i].size() != predictions[i].size())
    {
      _options.logger().log_error("Mismatched vector sizes at index ", i);
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

double NeuralNetwork::calculate_mae_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (ground_truth.size() != predictions.size())
  {
    _options.logger().log_error("Mismatched number of samples");
    throw std::invalid_argument("Mismatched number of samples");
  }
  

  double total_abs_error = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < ground_truth.size(); ++i)
  {
    if (ground_truth[i].size() != predictions[i].size())
    {
      _options.logger().log_error("Mismatched vector sizes at index ", i);
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

double NeuralNetwork::calculate_rmse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto mean_squared_error = calculate_mse_error(ground_truth, predictions);
  return std::sqrt(mean_squared_error); // RMSE
}

double NeuralNetwork::calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (ground_truth.size() != predictions.size()) 
  {
    _options.logger().log_error("Mismatch in batch sizes.");
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
      _options.logger().log_warning("Mismatch in output vector sizes at index ",i);
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

std::vector<double> NeuralNetwork::calculate_weight_gradients(unsigned layer_number, unsigned neuron_number, const GradientsAndOutputs& source) const
{
  if (layer_number == 0 || layer_number > source.num_gradient_layers())
  {
    return {};
  }
  const auto batch_size = source.batch_size();
  const auto number_outputs = source.num_outputs(layer_number - 1);
  std::vector<double> gradients(number_outputs, 0.0);
  const auto gradient = source.get_gradient(layer_number, neuron_number);
  for (unsigned output_neuron_number = 0; output_neuron_number < number_outputs; ++output_neuron_number)
  {
    const auto& weight = source.get_output(layer_number-1, output_neuron_number);
    gradients[output_neuron_number] = (weight * gradient) / batch_size;
  }
  return gradients;
}

// multiple batches
void NeuralNetwork::apply_weight_gradients(std::vector<Layer>& layers, const std::vector<GradientsAndOutputs>& batch_activation_gradients, double learning_rate, unsigned epoch) const
{
  for(const auto& batch_activation_gradient : batch_activation_gradients)  
  {
    apply_weight_gradients(layers, batch_activation_gradient, learning_rate, epoch);
  }
}

// single batch
void NeuralNetwork::apply_weight_gradients(std::vector<Layer>& layers, const GradientsAndOutputs& batch_activation_gradient, double learning_rate, unsigned epoch) const
{
  const auto& layer_size = batch_activation_gradient.num_gradient_layers();

  // we need to obtain write lock here as we are about to change the neurons
  std::unique_lock lock(_mutex);
  for (auto layer_number = layer_size-1; layer_number > 0; --layer_number)
  {
    const auto& neuron_size = batch_activation_gradient.num_gradient_neurons(layer_number) -1; // exclude bias
    for (unsigned neuron_number = 0; neuron_number < neuron_size; ++neuron_number)
    {
      auto& neuron = layers[layer_number].get_neuron(neuron_number);
      const auto& gradients = calculate_weight_gradients(layer_number, neuron_number, batch_activation_gradient);
      neuron.apply_weight_gradients(layers[layer_number - 1], gradients, learning_rate, epoch);
    }
  }
}

void NeuralNetwork::calculate_back_propagation(GradientsAndOutputs& gradients, const std::vector<double>& outputs, const std::vector<Layer>& layers) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  assert(outputs.size() == gradients.output_back().size());

  // input layer is all 0, (bias is included)
  auto input_gradients = std::vector<double>(layers.front().number_neurons(), 0.0);
  gradients.set_gradients(0, input_gradients);
  
  // set the output gradient
  const auto& output_layer = layers.back();
  auto next_activation_gradients = caclulate_output_gradients(outputs, gradients.output_back(), output_layer);
  gradients.set_gradients(static_cast<unsigned>(layers.size()-1), next_activation_gradients);
  for (auto layer_number = layers.size() - 2; layer_number > 0; --layer_number)
  {
    const auto& hidden_layer = layers[layer_number];
    const auto& next_layer = layers[layer_number + 1];

    const auto& hidden_layer_size = hidden_layer.number_neurons();
    std::vector<double> current_activation_gradients;
    current_activation_gradients.resize(hidden_layer_size, 0.0);
    for (size_t hidden_layer_number = 0; hidden_layer_number < hidden_layer_size; ++hidden_layer_number)
    {
      const auto& neuron = hidden_layer.get_neuron(static_cast<unsigned>(hidden_layer_number));
      const auto output_value = gradients.get_output(static_cast<unsigned>(layer_number), static_cast<unsigned>(hidden_layer_number));
      const auto& gradient = neuron.calculate_hidden_gradients(next_layer, next_activation_gradients, output_value);
      current_activation_gradients[hidden_layer_number] = gradient;
    }
    gradients.set_gradients(static_cast<unsigned>(layer_number), current_activation_gradients);
    next_activation_gradients = std::move(current_activation_gradients);
    current_activation_gradients = {};
  }
}

void NeuralNetwork::calculate_forward_feed(
  NeuralNetwork::GradientsAndOutputs& gradients_outputs,
  const std::vector<double>& inputs, 
  const std::vector<Layer>& layers) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  //  the initial set of output values where we are starting from.
  gradients_outputs.set_outputs(0, inputs);

  // then forward propagate from the input to ... hopefully, the output.
  auto previous_layer_output_values = inputs;
  previous_layer_output_values.push_back(1.0);
  for (size_t layer_number = 1; layer_number < layers.size(); ++layer_number)
  {
    const auto& previous_layer = layers[layer_number - 1];
    auto& current_layer = layers[layer_number];
    
    std::vector<double> this_output_values;
    this_output_values.reserve(current_layer.number_neurons() - 1);
    for (size_t neuron_number = 0; neuron_number < current_layer.number_neurons() - 1; ++neuron_number)
    {
      const auto& neuron = current_layer.get_neuron(unsigned(neuron_number));
      this_output_values.emplace_back(neuron.calculate_forward_feed(previous_layer, previous_layer_output_values));
    }

    gradients_outputs.set_outputs(
      static_cast<unsigned>(layer_number),
      this_output_values
    );
    previous_layer_output_values = std::move(this_output_values);
    previous_layer_output_values.push_back(1.0);
  }
}

std::vector<double> NeuralNetwork::caclulate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const size_t output_layer_size = output_layer.number_neurons();
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

double NeuralNetwork::calculate_forecast_accuracy_smape(const std::vector<double>& ground_truth, const std::vector<double>& predictions) const
{
  if (predictions.size() != ground_truth.size() || predictions.empty())
  {
    _options.logger().log_error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  size_t non_zero_count = 0;
  const double epsilon = 1e-8; // To avoid divide-by-zero
  double sum_of_percentage_errors = 0.0;

  for (size_t i = 0; i < predictions.size(); ++i)
  {
    const auto denominator = (std::abs(predictions[i]) + std::abs(ground_truth[i])) / 2.0;

    if (denominator < epsilon) 
    {
      continue; // Optionally skip (or count as 0 error)
    }
    ++non_zero_count;
    const auto numerator = std::abs(predictions[i] - ground_truth[i]);
    sum_of_percentage_errors += numerator / denominator;
  }
  return non_zero_count == 0 ? 0.0 : (sum_of_percentage_errors / non_zero_count);
}

double NeuralNetwork::calculate_forecast_accuracy_smape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions) const
{
  if (predictions.size() != ground_truths.size() || predictions.empty())
  {
    _options.logger().log_error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  double average_total = 0;
  double average_size = 0;
  for (size_t index = 0; index < ground_truths.size(); ++index)
  {
    auto percentage = calculate_forecast_accuracy_smape(ground_truths[index], predictions[index]);
    if (percentage == 0)
    {
      continue;
    }
    average_total += percentage;
    ++average_size;
  }
  return average_size > 0 ? average_total / average_size : 0.0;
}

double NeuralNetwork::calculate_forecast_accuracy_mape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions) const
{
  if (predictions.size() != ground_truths.size() || predictions.empty()) 
  {
    _options.logger().log_error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  double average_total = 0;
  double average_size = 0;
  for (size_t index = 0; index < ground_truths.size(); ++index)
  {
    auto percentage = calculate_forecast_accuracy_mape(ground_truths[index], predictions[index]);
    if(percentage == 0)
    {
      continue;
    }
    average_total += percentage;
    ++average_size;
  }
  return average_size > 0 ? average_total / average_size : 0.0;
}

double NeuralNetwork::calculate_forecast_accuracy_mape(const std::vector<double>& ground_truth, const std::vector<double>& predictions) const
{
  if (predictions.size() != ground_truth.size() || predictions.empty()) 
  {
    _options.logger().log_error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  size_t non_zero_count = 0;
  double sum_of_percentage_errors = 0.0;
  const double epsilon = 1e-8; // Small value to avoid division by zero

  for (size_t i = 0; i < predictions.size(); ++i) 
  {
    if (std::abs(ground_truth[i]) < epsilon) 
    {
      // If actual value is very close to zero, this pair is often skipped
      // or handled differently to avoid astronomical percentage errors.
      // Here, we'll just skip it to not skew the result.
      continue; 
    }
    ++non_zero_count;
    sum_of_percentage_errors += std::abs((ground_truth[i] - predictions[i]) / ground_truth[i]);
  }
  return non_zero_count == 0 ? 0.0 : sum_of_percentage_errors / non_zero_count;
}

void NeuralNetwork::log_training_info(
  const std::vector<std::vector<double>>& training_inputs,
  const std::vector<std::vector<double>>& training_outputs) const
{
  assert(_neural_network_helper != nullptr);
  _options.logger().log_info("Tainning will use: ");
  _options.logger().log_info(_neural_network_helper->training_indexes().size(), " training samples.");
  _options.logger().log_info(_neural_network_helper->checking_indexes().size(), " in training error check samples.");
  _options.logger().log_info(_neural_network_helper->final_check_indexes().size(), " final error check samples.");
  _options.logger().log_info("Learning rate:", std::fixed, std::setprecision(15), _options.learning_rate());
  _options.logger().log_info("Learning rate decay rate:", std::fixed, std::setprecision(15), _options.learning_rate_decay_rate());
  _options.logger().log_info("Hidden activation method:", activation::method_to_string(get_hidden_activation_method()));
  _options.logger().log_info("Output activation method:", activation::method_to_string(get_output_activation_method()));
  _options.logger().log_info("Input size:", training_inputs.front().size());
  _options.logger().log_info("Output size:", training_outputs.front().size());
  _options.logger().log_info("Optimiser:", optimiser_type_to_string(_options.optimiser_type()));
  std::string hidden_layer_message = "Hidden layers: {";
  for (size_t layer = 1; layer < _layers.size() - 1; ++layer)
  {
    hidden_layer_message += std::to_string(_layers[layer].number_neurons() - 1); // remove the bias
    if (layer < _layers.size() - 2)
    {
      hidden_layer_message += ", ";
    }
  }
  hidden_layer_message += "}";
  _options.logger().log_info(hidden_layer_message);

  _options.logger().log_info("Batch size: ", _options.batch_size());
  if (_options.batch_size() > 1)
  {
    if (_options.number_of_threads() <= 0)
    {
      _options.logger().log_info("Number of threads: ", (std::thread::hardware_concurrency() - 1));
    }
    else
    {
      _options.logger().log_info("Number of threads: ", _options.number_of_threads());
    }
  }
}

void NeuralNetwork::dump_layer_info() const
{
  for(size_t layer_number = 0; layer_number < _layers.size(); ++layer_number)
  {
    _options.logger().log_debug("Layer ", layer_number );

    auto& layer = _layers[layer_number];
    for(size_t neuron_number = 0; neuron_number < layer.number_neurons(); ++neuron_number )
    {
      auto& neuron = layer.get_neuron(neuron_number);
      _options.logger().log_debug("  → Neuron ", neuron_number, neuron.is_bias() ? " (bias)" : "" );

      auto& wp = neuron.get_weight_params();
      for(size_t index_number = 0; index_number < wp.size(); ++index_number )
      {
        _options.logger().log_debug(
                     std::fixed, std::setprecision(15),
                     "    → weight[", index_number,
                     "] = ",
                     wp[index_number].value());
      }
    }
  }
}
