#include "neuralnetwork.h"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>

#ifndef M_PI
# define M_PI   3.141592653589793238462643383279502884
#endif

static const double RecentAverageSmoothingFactor = 100.0;
static const long long IntervalErorCheckInSeconds = 15;

NeuralNetwork::NeuralNetwork(const NeuralNetworkOptions& options) :
  _learning_rate(0.0),
  _layers(
    options.topology(), 
    options.dropout(),
    options.hidden_activation_method(), 
    options.output_activation_method(),
    options.optimiser_type(),
    options.residual_layer_jump(),
    options.logger()),
  _options(options),
  _neural_network_helper(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
}

NeuralNetwork::NeuralNetwork(
  const std::vector<Layer>& layers, 
  const NeuralNetworkOptions& options
  ) :
  _learning_rate(options.learning_rate()),
  _layers(layers),
  _options(options),
  _neural_network_helper(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& src) :
  _learning_rate(src._learning_rate),
  _layers(src._layers),
  _options(src._options),
  _neural_network_helper(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (src._neural_network_helper != nullptr)
  {
    _neural_network_helper = new NeuralNetworkHelper(*src._neural_network_helper);
  }
}

NeuralNetwork::~NeuralNetwork()
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  delete _neural_network_helper;
  _neural_network_helper = nullptr;
}

const Logger& NeuralNetwork::logger() const
{
  return _options.logger();
}

const activation::method& NeuralNetwork::get_output_activation_method() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _options.output_activation_method();
}

const activation::method& NeuralNetwork::get_hidden_activation_method() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _options.hidden_activation_method();
}

const std::vector<Layer>& NeuralNetwork::get_layers() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _layers.get_layers();
}

const std::vector<unsigned>& NeuralNetwork::get_topology() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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

void NeuralNetwork::create_shuffled_indexes(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<size_t> training_indexes;
  std::vector<size_t> checking_indexes;
  std::vector<size_t> final_check_indexes;

  auto sample_size = neural_network_helper.sample_size();

  auto shuffled_indexes = get_shuffled_indexes(sample_size);
  assert(sample_size == shuffled_indexes.size());

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
    logger().log_warning("Training batch size does not allow for error checking batch!");
    training_indexes = shuffled_indexes;
    checking_indexes = {shuffled_indexes.front()};
    final_check_indexes = {shuffled_indexes.back()};
    return;
  }
  assert(training_size + checking_size < total_size); // make sure we don't get more than 100%
  if(training_size + checking_size > total_size) // make sure we don't get more than 100%
  {
    logger().log_error("Logic error, unable to do a final batch error check.");
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
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
    calculate_forward_feed(gradients, inputs, _layers, false);
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
    logger().log_warning("Trying to get training metrics when no training was done!");
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
      calculate_forward_feed(gradients, inputs, _layers, false);
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
    calculate_forward_feed(gradients, *inputs_begin, _layers, true);
    calculate_back_propagation(gradients, *outputs_begin, _layers);
    return gradients;
  }

  for(size_t index = 0; index < size; ++index)
  {
    GradientsAndOutputs this_gradients(_options.topology());
    calculate_forward_feed(gradients, *inputs_begin, _layers, true);
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
  if (_options.learning_rate_warmup_target() > 0.0)
  {
    _learning_rate = _options.learning_rate_warmup_start();
    logger().log_info("Using learning rate warmup, starting at ", std::setprecision(15),  _learning_rate, " and ending at ", _options.learning_rate(), " (at ", std::setprecision(4), (_options.learning_rate_warmup_target()*100.0), "%)", ".");
  }
  const auto& progress_callback = _options.progress_callback();
  const auto& batch_size = _options.batch_size();

  dump_layer_info();

  if(batch_size <=0 || batch_size > static_cast<int>(training_inputs.size()))
  {
    logger().log_error("The batch size if either -ve or too large for the training sample.");
    throw std::invalid_argument("The batch size if either -ve or too large for the training sample.");
  }
  if(training_outputs.size() != training_inputs.size())
  {
    logger().log_error("The number of training samples does not match the number of expected outputs.");
    throw std::invalid_argument("The number of training samples does not match the number of expected outputs.");
  }

  logger().log_info("Started training with ", training_inputs.size(), " inputs, ", number_of_epoch, " epoch and batch size ", batch_size, ".");

  // create the thread pool if we need one ...
  TaskQueuePool<GradientsAndOutputs>* task_pool = nullptr;
  if (batch_size > 1)
  {
    task_pool = new TaskQueuePool<GradientsAndOutputs>(logger(),_options.number_of_threads());
  }

  // create the neural network helper.
  delete _neural_network_helper;
  _neural_network_helper = new NeuralNetworkHelper(*this, _learning_rate, number_of_epoch, training_inputs, training_outputs);
  create_shuffled_indexes(*_neural_network_helper, _options.data_is_unique());

  // create the callback task if we need one.
  SingleTaskQueue<bool>* callback_task = nullptr;
  if (progress_callback != nullptr)
  {
    callback_task = new SingleTaskQueue<bool>();
    callback_task->call(progress_callback , std::ref(*_neural_network_helper));
    if (!callback_task->get())
     {
      logger().log_warning("Progress callback function returned false before training started, closing now!");
      return;
    }
  }

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

  // for the learning rate decay we need to set the target learning rate and the decay rate.
  auto number_of_epoch_after_decay = number_of_epoch - static_cast<int>(std::round(_options.learning_rate_warmup_target() * number_of_epoch));
  const auto learning_rate_decay_rate = number_of_epoch_after_decay == 0 || _options.learning_rate_decay_rate() == 0 ? 0 : std::log(1.0 / _options.learning_rate_decay_rate()) / number_of_epoch_after_decay;

  // because we boost the rate from time to time, the base, (or target rate), we will use is different.
  double learning_rate_base = _options.learning_rate();

  AdaptiveLearningRateScheduler learning_rate_scheduler(logger());

  for (auto epoch = 0; epoch < number_of_epoch; ++epoch)
  {
    _neural_network_helper->set_epoch(epoch);
    _learning_rate = _neural_network_helper->learning_rate();

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

    // Learning rate
    //
    auto learning_rate = calculate_learning_rate(learning_rate_base, learning_rate_decay_rate, epoch, number_of_epoch, learning_rate_scheduler);

    // boost the learning rate base if we need to.
    learning_rate = calculate_smooth_learning_rate_boost(epoch, number_of_epoch, learning_rate_base);
    _neural_network_helper->set_learning_rate(learning_rate);

    // callback
    // 
    if (!CallCallback(progress_callback, callback_task))
    {
      logger().log_warning("Progress callback function returned false during training, closing now!");
      break;
    }

    MYODDWEB_PROFILE_MARK();
  }

  if(task_pool != nullptr)
  {
    task_pool->stop();
    delete task_pool;
  }

  auto metrics = calculate_forecast_metrics(
    {
      NeuralNetworkOptions::ErrorCalculation::rmse,
      NeuralNetworkOptions::ErrorCalculation::mape 
    }, true);

  logger().log_info("Final RMSE Error: ", std::fixed, std::setprecision (15), metrics[0].error());
  logger().log_info("Final Forecast accuracy (MAPE): ", std::fixed, std::setprecision (15), metrics[1].error()*100.0);

  // finaly learning rate
  logger().log_info("Final Learning rate: ", std::fixed, std::setprecision(15), _neural_network_helper->learning_rate());

  // final callback to show 100% done.
  if (progress_callback != nullptr)
  {
    // wait for the future to complete if running
    // we are out of the loop already, so we no longer care about the result.
    callback_task->stop();
    delete callback_task;

    // then do one final call, again, we don't care about the result.
    progress_callback(*_neural_network_helper);
  }

  dump_layer_info();

  MYODDWEB_PROFILE_MARK();
}

bool NeuralNetwork::CallCallback(const std::function<bool(NeuralNetworkHelper&)>& callback, SingleTaskQueue<bool>* callback_task) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (callback == nullptr || callback_task == nullptr)
  {
    // no callback, nothing to do, continue training.
    return true;
  }

  // if it running?
  if (!callback_task->busy())
  {
    if (callback_task->has_result() && !callback_task->get())
    {
      return false; // stop training if the callback returns false.
    }

    // it was not running at all, so we start it.
    if (!callback_task->call(callback, std::ref(*_neural_network_helper)))
    {
      logger().log_error("Trying to call Progress callback function but an error was returned.");
    }
  }
  else
  {
    logger().log_debug("Progress callback function is still running, continuing to next epoch.");
  }
  return true; // continue training
}

double NeuralNetwork::calculate_clipping_scale(const Layer& layer, unsigned int layer_number) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto gradient_clip_threshold = options().clip_threshold();

  double total_sq_sum = 0.0;

  // Accumulate squared gradients for all weights and biases
  for (size_t neuron_number = 0; neuron_number < layer.number_neurons(); ++neuron_number)
  {
    const auto& neuron = layer.get_neuron(static_cast<unsigned>(neuron_number));
    for (const auto& weight_param : neuron.get_weight_params())
    {
      auto gradient = weight_param.gradient();
      total_sq_sum += gradient * gradient;
    }
  }

  double norm = std::sqrt(total_sq_sum);
  if (!std::isfinite(norm))
  {
    logger().log_error("Layer Number:", layer_number, ", Gradient norm is NaN / Inf � resetting optimizer buffers and skipping batch.");
    return 0.0; // Skip updates entirely
  }

  if (norm <= gradient_clip_threshold || norm <= 0.0)
  {
    return 1.0; // No clipping needed, use identity scale
  }

  auto clipping_scale = gradient_clip_threshold / norm; // scale factor < 1 to reduce gradient norm
  logger().log_warningf([&]
    {
      auto lr = get_learning_rate();
      std::string warning = Logger::log_factory(std::setprecision(4), "Layer Number:", layer_number, ", Clipping gradients: norm = ", norm, " scale = ", clipping_scale, " (learnign rate: ", lr, ")");
      return warning;
    });
  return clipping_scale;
}

double NeuralNetwork::calculate_smooth_learning_rate_boost(
  int epoch,
  int total_epochs,
  double base_learning_rate) const  // base learning rate
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  const auto& boost_every_percent = options().learning_rate_restart_rate();  // e.g., 7.5 means every 7.5% progress
  const auto& boost_ratio = options().learning_rate_restart_boost();         // e.g., 1.10 means +10%

  assert(boost_every_percent >= 0.0 && boost_every_percent <= 1.0);
  assert(boost_ratio >= 0.0 && boost_ratio <= 1.0);
  if(boost_ratio == 0 || boost_every_percent == 0)
  {
    // no boost, return the base learning rate
    return base_learning_rate;
  }

  auto boost_interval = static_cast<int>(std::round(boost_every_percent * total_epochs)); // e.g., 7.5% of 100 epochs is 7.5 epochs
  if (boost_interval <= 0)
  {
    // no boost interval, return the base learning rate
    return base_learning_rate;
  }
  // Calculate the number of boosts that have occurred so far
  auto total_boosts = total_epochs / boost_interval;
  auto per_boost_ratio = boost_ratio / total_boosts; // e.g., if 3 boosts, each boost is 1.10^(1/3)

  auto cycle_position = epoch % boost_interval; // e.g., if epoch 8 and interval is 7, position is 1
  auto progress = static_cast<double>(cycle_position) / boost_interval; // e.g., 1/7 = 0.142857

  // Apply cosine boost: start at 0, peak at 1, then back to 0
  auto cosine_multiplier = (1.0 - std::cos(progress * M_PI)) / 2.0;
  auto current_boost = per_boost_ratio * cosine_multiplier;

  auto completed_cycles = epoch / boost_interval; // e.g., 8/7 = 1 full cycle
  auto cumulative_boost = completed_cycles * per_boost_ratio;

  return base_learning_rate * (1.0 + cumulative_boost + current_boost);
}

double NeuralNetwork::calculate_learning_rate(double learning_rate_base, double learning_rate_decay_rate, int epoch, int number_of_epoch, AdaptiveLearningRateScheduler& learning_rate_scheduler) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto learning_rate = learning_rate_base;

  // the completed percent
  auto completed_percent = (static_cast<double>(epoch) / number_of_epoch);

  // do warmup first.
  if (completed_percent < _options.learning_rate_warmup_target())
  {
    learning_rate = calculate_learning_rate_warmup(epoch, completed_percent);
  }
  // are we decaying the learning rate?
  // this is done after warmup
  else if (learning_rate_decay_rate != 0)
  {
    learning_rate = learning_rate_base * std::exp(-learning_rate_decay_rate * epoch);
    logger().log_tracef([=] 
      {
        return Logger::log_factory("Learning rate to ", std::fixed, std::setprecision(15), learning_rate, " at epoch ", epoch, " (", std::setprecision(4), completed_percent * 100.0, "%)");
      });
  }

  // then get the scheduler if we can improve it further.
  if (_options.adaptive_learning_rate())
  {
    auto metric = calculate_forecast_metrics(
      {
        NeuralNetworkOptions::ErrorCalculation::rmse,
      }, false);
    learning_rate = learning_rate_scheduler.update(metric[0].error(), learning_rate, epoch, number_of_epoch);
    logger().log_trace("Adaptive learning rate to ", std::fixed, std::setprecision(15), learning_rate, " at epoch ", epoch, " (", std::setprecision(4), completed_percent * 100.0, "%)");
  }
  return learning_rate;
}

double NeuralNetwork::calculate_learning_rate_warmup(int epoch, double completed_percent) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const auto& target_percent = _options.learning_rate_warmup_target();
  const auto& start = _options.learning_rate_warmup_start();
  const auto& target = _options.learning_rate();

  // linear
  // auto warmup_learning_rate = start + (target - start) * (completed_percent / target_percent);

  double ratio = std::min(1.0, completed_percent / target_percent); // both as [0..1]
  double warmup_learning_rate;
  if (start <= 0.0) 
  {
    // fall back to linear if start is zero (avoid division by zero)
    warmup_learning_rate = start + (target - start) * ratio;
  }
  else 
  {
    double geom = std::pow(target / start, ratio); // (target/start)^ratio
    warmup_learning_rate = start * geom;
  }
  logger().log_tracef([=] {
      return Logger::log_factory("Learning rate warmup to ", std::fixed, std::setprecision(15), warmup_learning_rate, " at epoch ", epoch, " (", std::setprecision(4), completed_percent * 100.0, "%)");
    });
  return warmup_learning_rate;
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
    return calculate_forecast_mape(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::smape:
    return calculate_forecast_smape(ground_truth, predictions);
  }

  logger().log_error("Unknown ErrorCalculation type!");
  throw std::invalid_argument("Unknown ErrorCalculation type!");
}

double NeuralNetwork::calculate_huber_loss_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (ground_truth.size() != predictions.size())
  {
    logger().log_error("Mismatched number of samples");
    throw std::invalid_argument("Mismatched number of samples");
  }

  double total_loss = 0.0;
  size_t count = 0;

  for (size_t i = 0; i < ground_truth.size(); ++i)
  {
    if (ground_truth[i].size() != predictions[i].size())
    {
      logger().log_error("Mismatched vector sizes at index ", i);
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
    logger().log_error("Mismatched number of samples");
    throw std::invalid_argument("Mismatched number of samples");
  }
  

  double total_abs_error = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < ground_truth.size(); ++i)
  {
    if (ground_truth[i].size() != predictions[i].size())
    {
      logger().log_error("Mismatched vector sizes at index ", i);
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
    logger().log_error("Mismatch in batch sizes.");
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
      logger().log_warning("Mismatch in output vector sizes at index ",i);
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

std::vector<double> NeuralNetwork::calculate_residual_projection_gradients(
    unsigned layer_number,
    unsigned neuron_number,
    const GradientsAndOutputs& source
) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const int residual_layer_number = _layers[layer_number].residual_layer_number();
  if (residual_layer_number == -1)
  {
    return {};
  }

  const auto batch_size = source.batch_size();
  const auto residual_output_count = source.num_outputs(static_cast<unsigned>(residual_layer_number));

  // gradient of loss with respect to the neuron's output
  const double delta = source.get_gradient(layer_number, neuron_number);

  std::vector<double> gradients;
  gradients.reserve(residual_output_count);

  // regular residual outputs
  for (unsigned i = 0; i < residual_output_count -1; ++i) // exclude bias
  {
    const double residual_output = source.get_output(static_cast<unsigned>(residual_layer_number), i);
    gradients.push_back((residual_output * delta) / batch_size);
  }

  // residual bias contribution
  gradients.push_back((1.0 * delta) / batch_size);

  return gradients;
}

std::vector<double> NeuralNetwork::calculate_weight_gradients(unsigned layer_number, unsigned neuron_number, const GradientsAndOutputs& source) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
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
void NeuralNetwork::apply_weight_gradients(Layers& layers, const std::vector<GradientsAndOutputs>& batch_activation_gradients, double learning_rate, unsigned epoch) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  for(const auto& batch_activation_gradient : batch_activation_gradients)  
  {
    apply_weight_gradients(layers, batch_activation_gradient, learning_rate, epoch);
  }
}

// single batch
void NeuralNetwork::apply_weight_gradients(Layers& layers, const GradientsAndOutputs& batch_activation_gradient, double learning_rate, unsigned epoch) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const auto& layer_size = batch_activation_gradient.num_gradient_layers();

  // we need to obtain write lock here as we are about to change the neurons
  std::unique_lock lock(_mutex);
  for (auto layer_number = layer_size-1; layer_number > 0; --layer_number)
  {
    auto& previous_layer = layers[layer_number - 1];
    auto& current_layer = layers[layer_number];

    std::vector<double> residual_output_values = {};
    auto* residual_layer = get_residual_layer(layers, batch_activation_gradient, residual_output_values, current_layer);

    auto clipping_scale = calculate_clipping_scale(current_layer, layer_number);
    const auto& neuron_size = batch_activation_gradient.num_gradient_neurons(layer_number) -1; // exclude bias
    for (unsigned neuron_number = 0; neuron_number < neuron_size; ++neuron_number)
    {
      auto& neuron = current_layer.get_neuron(neuron_number);
      const auto& gradients = calculate_weight_gradients(layer_number, neuron_number, batch_activation_gradient);
      neuron.apply_weight_gradients(previous_layer, gradients, learning_rate, epoch, clipping_scale);

      if(residual_layer != nullptr)
      {
        const auto residual_gradients = calculate_residual_projection_gradients(
            layer_number, neuron_number, batch_activation_gradient
          );
        neuron.apply_residual_weight_gradients(
          current_layer, 
          *residual_layer,
          residual_output_values, 
          residual_gradients, 
          learning_rate, 
          clipping_scale);
      }
    }
  }
}

Layer* NeuralNetwork::get_residual_layer(Layers& layers, const GradientsAndOutputs& batch_activation_gradient, std::vector<double>& residual_output_values, const Layer& current_layer) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const auto residual_layer_number = current_layer.residual_layer_number();
  if (residual_layer_number == -1)
  {
    return nullptr; // no residual layer
  }
  assert(residual_output_values.size() == 0);
  auto* residual_layer = &(layers[static_cast<unsigned>(residual_layer_number)]);
  auto residual_layer_neuron_size = residual_layer->number_neurons();
  residual_output_values.reserve(residual_layer_neuron_size);
  for (unsigned neuron_number = 0; neuron_number < residual_layer_neuron_size; ++neuron_number)
  {
    const auto output = batch_activation_gradient.get_output(residual_layer_number, neuron_number);
    residual_output_values.emplace_back(output);
  }
  return residual_layer;
}

void NeuralNetwork::calculate_back_propagation(GradientsAndOutputs& gradients, const std::vector<double>& outputs, const Layers& layers) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  assert(outputs.size() == gradients.output_back().size());

  // input layer is all 0, (bias is included)
  const auto& input_gradients = std::vector<double>(layers.input_layer().number_neurons(), 0.0);
  gradients.set_gradients(0, input_gradients);
  
  // set the output gradient
  const auto& output_layer = layers.output_layer();
  auto next_activation_gradients = calculate_output_gradients(outputs, gradients.output_back(), output_layer);
  gradients.set_gradients(static_cast<unsigned>(layers.size()-1), next_activation_gradients);
  for (auto layer_number = layers.size() - 2; layer_number > 0; --layer_number)
  {
    const auto& hidden_layer = layers[static_cast<unsigned>(layer_number)];
    const auto& next_layer = layers[static_cast<unsigned>(layer_number + 1)];

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
  const Layers& layers,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  //  the initial set of output values where we are starting from.
  gradients_outputs.set_outputs(0, inputs);

  // then forward propagate from the input to ... hopefully, the output.
  std::vector<std::vector<double>> layer_output_values;
  layer_output_values.reserve(layers.size());
  layer_output_values.emplace_back(inputs);
  layer_output_values.back().push_back(1.0);
  for (size_t layer_number = 1; layer_number < layers.size(); ++layer_number)
  {
    const auto& previous_layer = layers[static_cast<unsigned>(layer_number - 1)];
    const auto& current_layer = layers[static_cast<unsigned>(layer_number)];

    // previous output values.
    const auto& previous_layer_output_values = layer_output_values[layer_number -1];

    // the residual layer output values.
    std::vector<double> residual_output_values = {};
    const auto residual_layer_mumber = current_layer.residual_layer_number();
    if(residual_layer_mumber != -1)
    {
      // get that layer output values.
      residual_output_values = current_layer.residual_output_values(layer_output_values[residual_layer_mumber]);
    }
    
    std::vector<double> this_output_values;
    this_output_values.reserve(current_layer.number_neurons() - 1);
    for (size_t neuron_number = 0; neuron_number < current_layer.number_neurons() - 1; ++neuron_number)
    {
      const auto& neuron = current_layer.get_neuron(static_cast<unsigned>(neuron_number));
      this_output_values.emplace_back(neuron.calculate_forward_feed(previous_layer, previous_layer_output_values, residual_output_values, is_training));
    }

    // log explosion
    if (logger().can_log_trace())
    {
      double sum = 0.0;
      double max_abs = 0.0;
      for (double val : this_output_values)
      {
        sum += val;
        max_abs = std::max(max_abs, std::fabs(val));
      }
      double mean = sum / this_output_values.size();
      if (std::fabs(mean) < 1e-6 || std::fabs(mean) > 10 || std::fabs(max_abs) > 50)
      {
        logger().log_tracef([=]
          {
            return Logger::log_factory("[ACT] Layer ", layer_number, ": mean=", mean, ", max=", max_abs);
          });
      }
    }

    gradients_outputs.set_outputs( static_cast<unsigned>(layer_number),this_output_values );

    // add output value
    this_output_values.push_back(1.0);
    layer_output_values.emplace_back(std::move(this_output_values));
  }
}

std::vector<double> NeuralNetwork::calculate_output_gradients(const std::vector<double>& target_outputs, const std::vector<double>& given_outputs, const Layer& output_layer) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const auto output_layer_size = output_layer.number_neurons();
  std::vector<double> activation_gradients = {};
  activation_gradients.reserve(output_layer_size);
  unsigned neuron_number_count = 0;
  for (unsigned neuron_number = 0; neuron_number < output_layer_size; ++neuron_number)
  {
    const auto& neuron = output_layer.get_neuron(unsigned(neuron_number));
    auto gradient = 0.0;
    if (!neuron.is_bias())
    {
      gradient = neuron.calculate_output_gradients(target_outputs[neuron_number_count], given_outputs[neuron_number_count]);
      ++neuron_number_count;
    }
    activation_gradients.emplace_back(gradient);
  }
  return activation_gradients;
}

double NeuralNetwork::calculate_forecast_smape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty()) 
  {
    logger().log_error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  double total_smape = 0.0;
  size_t sequence_count = 0;

  for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx) 
  {
    const auto& gt = ground_truths[seq_idx];
    const auto& pred = predictions[seq_idx];

    if (gt.size() != pred.size() || gt.empty()) {
      continue; // skip empty or mismatched
    }

    double seq_error_sum = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < gt.size(); ++i) 
    {
      double denom = (std::abs(gt[i]) + std::abs(pred[i])) / 2.0;
      if (denom < epsilon) continue; // skip both near-zero
      seq_error_sum += std::abs(gt[i] - pred[i]) / denom;
      ++count;
    }

    if (count > 0) 
    {
      total_smape += seq_error_sum / count;
      ++sequence_count;
    }
  }
  return (sequence_count == 0) ? 0.0 : (total_smape / sequence_count);
}

double NeuralNetwork::calculate_forecast_mape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty()) 
  {
    logger().log_error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  double total_mape = 0.0;
  size_t sequence_count = 0;

  for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx) 
  {
    const auto& gt = ground_truths[seq_idx];
    const auto& pred = predictions[seq_idx];

    if (gt.size() != pred.size() || gt.empty()) 
    {
      continue; // skip empty or mismatched
    }

    double seq_error_sum = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < gt.size(); ++i) 
    {
      double denom = std::abs(gt[i]);
      if (denom < epsilon) continue; // skip tiny values
      seq_error_sum += std::abs((gt[i] - pred[i]) / denom);
      ++count;
    }

    if (count > 0) 
    {
      total_mape += seq_error_sum / count;
      ++sequence_count;
    }
  }
  return (sequence_count == 0) ? 0.0 : (total_mape / sequence_count);
}

void NeuralNetwork::log_training_info(
  const std::vector<std::vector<double>>& training_inputs,
  const std::vector<std::vector<double>>& training_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const char* tab = "  ";
  assert(_neural_network_helper != nullptr);
  logger().log_info("Training will use: ");
  logger().log_info(tab, _neural_network_helper->training_indexes().size(), " training samples.");
  logger().log_info(tab, _neural_network_helper->checking_indexes().size(), " in training error check samples.");
  logger().log_info(tab, _neural_network_helper->final_check_indexes().size(), " final error check samples.");
  logger().log_info(tab, "Learning rate              : ", std::fixed, std::setprecision(15), _options.learning_rate());
  logger().log_info(tab, "Learning rate decay rate   : ", std::fixed, std::setprecision(15), _options.learning_rate_decay_rate());
  logger().log_info(tab, "Learning rate warmup start : ", std::fixed, std::setprecision(15), _options.learning_rate_warmup_start());
  logger().log_info(tab, "Learning rate warmup target: ", std::fixed, std::setprecision(4), _options.learning_rate_warmup_target()*100, "%");
  logger().log_info(tab, "Gradient clip threshold    : ", std::fixed, std::setprecision(4), _options.clip_threshold());
  logger().log_info(tab, "Hidden activation method   : ", activation::method_to_string(get_hidden_activation_method()));
  logger().log_info(tab, "Output activation method   : ", activation::method_to_string(get_output_activation_method()));
  logger().log_info(tab, "Residual layerjump         : ", _options.residual_layer_jump());
  logger().log_info(tab, "Input size                 : ", training_inputs.front().size());
  logger().log_info(tab, "Output size                : ", training_outputs.front().size());
  logger().log_info(tab, "Optimiser                  : ", optimiser_type_to_string(_options.optimiser_type()));
  std::string hidden_layer_message = 
                                "  Hidden layers              : {";
  for (size_t layer = 1; layer < _layers.size() - 1; ++layer)
  {
    hidden_layer_message += std::to_string(_layers[static_cast<unsigned>(layer)].number_neurons() - 1); // remove the bias
    if (layer < _layers.size() - 2)
    {
      hidden_layer_message += ", ";
    }
  }
  hidden_layer_message += "}";
  logger().log_info(hidden_layer_message);

  std::string dropout_layer_message = 
                                "  Hidden layers dropout rate : {";
  for( auto& dropout : options().dropout())
  {
    dropout_layer_message += std::to_string(dropout);
    dropout_layer_message += ", ";
  }
  dropout_layer_message = dropout_layer_message.substr(0, dropout_layer_message.size() - 2); // remove the last ", "
  dropout_layer_message += "}";
  logger().log_info(dropout_layer_message);

  logger().log_info(tab, "Batch size                 : ", _options.batch_size());
  if (_options.batch_size() > 1)
  {
    if (_options.number_of_threads() <= 0)
    {
      logger().log_info(tab, "Number of threads          : ", (std::thread::hardware_concurrency() - 1));
    }
    else
    {
      logger().log_info(tab, "Number of threads          : ", _options.number_of_threads());
    }
  }
}

void NeuralNetwork::dump_layer_info() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (!logger().can_log_trace())
  {
    return;
  }

#ifndef NDEBUG
  for (size_t layer_number = 0; layer_number < _layers.size(); ++layer_number)
  {
    logger().log_trace("Layer ", layer_number, " (residual layer: ", _layers[static_cast<unsigned>(layer_number)].residual_layer_number(), ").");

    auto& layer = _layers[static_cast<unsigned>(layer_number)];
    for (unsigned neuron_number = 0; neuron_number < layer.number_neurons(); ++neuron_number)
    {
      auto& neuron = layer.get_neuron(neuron_number);
      if (neuron.is_dropout())
      {
        logger().log_trace("  -> Neuron ", neuron_number, " (dropout ", neuron.get_dropout_rate()*100.f, "%)");
      }
      else
      {
        logger().log_trace("  -> Neuron ", neuron_number, neuron.is_bias() ? " (bias)" : "");
      }

      auto& wp = neuron.get_weight_params();
      for (size_t index_number = 0; index_number < wp.size(); ++index_number)
      {
        logger().log_trace(
                     std::fixed, std::setprecision(15),
          "    -> weight[", index_number,
                     "] = ",
                     wp[index_number].value());
      }
    } 
  }
#endif
}