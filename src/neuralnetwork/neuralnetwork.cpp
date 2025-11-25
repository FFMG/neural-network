#include "logger.h"
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
    options.recurrent_layers(),
    options.dropout(),
    options.hidden_activation_method(), 
    options.output_activation_method(),
    options.optimiser_type(),
    options.residual_layer_jump()),
  _options(options),
  _neural_network_helper(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
}

NeuralNetwork::NeuralNetwork(
  const std::vector<unsigned>& topology, 
  const activation::method& hidden_layer_activation, 
  const activation::method& output_layer_activation
  ) :
  NeuralNetwork(NeuralNetworkOptions::create(topology)
    .with_hidden_activation_method(hidden_layer_activation)
    .with_output_activation_method(output_layer_activation)
    .build())
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
}

NeuralNetwork::NeuralNetwork(
  const std::vector<Layer>& layers, 
  const NeuralNetworkOptions& options,
  const std::map<NeuralNetworkOptions::ErrorCalculation, double>& errors
  ) :
  _learning_rate(options.learning_rate()),
  _layers(layers),
  _options(options),
  _neural_network_helper(nullptr),
  _saved_errors(errors)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& src) :
  _layers(src._layers),
  _options(src._options),
  _neural_network_helper(nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  *this = src;
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& src)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (this != &src)
  {
    // to remain c++99 compliant we will not use std::scoped_lock
    std::unique_lock<std::shared_mutex> lhs_lock(_mutex, std::defer_lock);
    std::unique_lock<std::shared_mutex> rhs_lock(src._mutex, std::defer_lock);
    std::lock(lhs_lock, rhs_lock);

    _learning_rate = src._learning_rate;
    _layers = src._layers;
    _options = src._options;
    _saved_errors = src._saved_errors;
    delete _neural_network_helper;
    _neural_network_helper = nullptr;

    if (src._neural_network_helper != nullptr)
    {
      _neural_network_helper = new NeuralNetworkHelper(*src._neural_network_helper);
    }
  }
  return *this;
}

NeuralNetwork::~NeuralNetwork()
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  delete _neural_network_helper;
  _neural_network_helper = nullptr;
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

void NeuralNetwork::create_indexes(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<size_t> training_indexes;
  std::vector<size_t> checking_indexes;
  std::vector<size_t> final_check_indexes;

  auto sample_size = neural_network_helper.sample_size();

  std::vector<size_t> indexes(sample_size);
  std::iota(indexes.begin(), indexes.end(), 0);

  assert(sample_size == indexes.size());

  break_indexes(indexes, data_is_unique, training_indexes, checking_indexes, final_check_indexes);

  neural_network_helper.move_indexes(std::move(training_indexes), std::move(checking_indexes), std::move(final_check_indexes));
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

  break_indexes(shuffled_indexes, data_is_unique, training_indexes, checking_indexes, final_check_indexes);

  neural_network_helper.move_indexes(std::move(training_indexes), std::move(checking_indexes), std::move(final_check_indexes));
}

void NeuralNetwork::break_indexes(const std::vector<size_t>& indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  // break the indexes into parts
  size_t total_size = indexes.size();
  size_t training_size = static_cast<size_t>(std::round(total_size * 0.80));
  size_t checking_size = static_cast<size_t>(std::round(total_size * 0.15));
  if(training_size+checking_size == total_size || checking_size == 0)
  {
    // in the case of small training models we might not have enough to split anything
    Logger::warning("Training batch size does not allow for error checking batch!");
    training_indexes = indexes;
    checking_indexes = {indexes.front()};
    final_check_indexes = {indexes.back()};
    return;
  }
  assert(training_size + checking_size < total_size); // make sure we don't get more than 100%
  if(training_size + checking_size > total_size) // make sure we don't get more than 100%
  {
    Logger::error("Logic error, unable to do a final batch error check.");
    throw std::invalid_argument("Logic error, unable to do a final batch error check.");
  }

  // then build the various indexes that will be used during testing.
  if(data_is_unique)
  {
    // because the data is uniqe we must use all of it for training
    // this is important in some cases where the NN needs all the data to train
    // otherwise we will ony train on some of the data.
    // the classic XOR example is a good use case ... 
    training_indexes = indexes;
  }
  else
  {
    training_indexes.assign(indexes.begin(), indexes.begin() + training_size);
  }
  checking_indexes.assign(indexes.begin() + training_size, indexes.begin() + training_size + checking_size);
  final_check_indexes.assign(indexes.begin() + training_size + checking_size, indexes.end());
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
  // TODO validate the input size matches out topology
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  std::vector<HiddenStates> hidden_states;
  hidden_states.resize(1, HiddenStates(get_topology(), options().recurrent_layers()));
  std::vector<GradientsAndOutputs> gradients;
  gradients.push_back(GradientsAndOutputs(get_topology()));
  {
    std::shared_lock<std::shared_mutex> read(_mutex);
    calculate_forward_feed(gradients, hidden_states, { inputs }, _layers, false);
  }
  return gradients.front().output_back();
}

double NeuralNetwork::get_learning_rate() const noexcept
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

bool NeuralNetwork::has_training_data() const
{
  std::shared_lock read(_mutex);
  if (nullptr != _neural_network_helper)
  {
    return true; // we are currently training.
  }

  // do we have saved error, (from file).
  return !_saved_errors.empty();
}

std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> NeuralNetwork::calculate_forecast_metrics(const std::vector<NeuralNetworkOptions::ErrorCalculation>& error_types, bool final_check) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> errors = {};
  errors.reserve(errors.size());

  {
    std::shared_lock read(_mutex);
    if (nullptr == _neural_network_helper)
    {
      for (size_t index = 0; index < error_types.size(); ++index)
      {
        const auto& saved_error = _saved_errors.find(error_types[index]);
        if (saved_error == _saved_errors.end())
        {
          errors.emplace_back(NeuralNetworkHelper::NeuralNetworkHelperMetrics(0.0, error_types[index]));
          Logger::warning("Trying to get training metrics:", (int)error_types[index], " when no training was done!");
          continue;
        }
        errors.emplace_back(NeuralNetworkHelper::NeuralNetworkHelperMetrics(saved_error->second, error_types[index]));
      }
      return errors;
    }
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

  {
    std::shared_lock read(_mutex);
    for (size_t index = 0; index < prediction_size; ++index)
    {
      std::vector<GradientsAndOutputs> gradients;
      gradients.push_back(GradientsAndOutputs(get_topology()));
      std::vector<HiddenStates> hidden_states;
      hidden_states.resize(1, HiddenStates(get_topology(), options().recurrent_layers()));

      const auto& checks_index = (*checks_indexes)[index];
      const auto& inputs = training_inputs[checks_index];
      calculate_forward_feed(gradients, hidden_states, { inputs }, _layers, false);
      predictions.emplace_back(gradients.front().output_back());

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

std::vector<NeuralNetwork::GradientsAndOutputs> NeuralNetwork::train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  ) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  std::vector<HiddenStates> hidden_states;
  hidden_states.resize(size, HiddenStates(get_topology(), options().recurrent_layers()));

  std::vector<GradientsAndOutputs> gradients;
  gradients.resize(size, GradientsAndOutputs(get_topology()));

  if(size == 1)
  {
    calculate_forward_feed(gradients, hidden_states, { *inputs_begin }, _layers, true);
    calculate_back_propagation(gradients, { *outputs_begin }, _layers);
    return gradients;
  }

  const std::vector<std::vector<double>> inputs_slice(
    inputs_begin,
    inputs_begin + size
  );
  const std::vector<std::vector<double>> outputs_slice(
    outputs_begin,
    outputs_begin + size
  );

  calculate_forward_feed(gradients, hidden_states, inputs_slice, _layers, true);
  calculate_back_propagation(gradients, outputs_slice, _layers);
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
    Logger::info("Using learning rate warmup, starting at ", std::setprecision(15),  _learning_rate, " and ending at ", _options.learning_rate(), " (at ", std::setprecision(4), (_options.learning_rate_warmup_target()*100.0), "%)", ".");
  }
  const auto& progress_callback = _options.progress_callback();
  const auto& batch_size = _options.batch_size();

  if(batch_size <=0 || batch_size > static_cast<int>(training_inputs.size()))
  {
    Logger::error("The batch size if either -ve or too large for the training sample.");
    throw std::invalid_argument("The batch size if either -ve or too large for the training sample.");
  }
  if(training_outputs.size() != training_inputs.size())
  {
    Logger::error("The number of training samples does not match the number of expected outputs.");
    throw std::invalid_argument("The number of training samples does not match the number of expected outputs.");
  }

  Logger::info("Started training with ", training_inputs.size(), " inputs, ", number_of_epoch, " epoch and batch size ", batch_size, ".");

  // create the thread pool if we need one ...
  TaskQueuePool<std::vector<GradientsAndOutputs>>* task_pool = nullptr;
//  if (batch_size > 1)
//  {
//    task_pool = new TaskQueuePool<std::vector<GradientsAndOutputs>>(_options.number_of_threads());
//  }

  // create the neural network helper.
  delete _neural_network_helper;
  _neural_network_helper = new NeuralNetworkHelper(*this, _learning_rate, number_of_epoch, training_inputs, training_outputs);

  // set all the indexes in the helper, either shiffled or not.
  if (options().shuffle_training_data())
  {
    create_shuffled_indexes(*_neural_network_helper, _options.data_is_unique());
  }
  else
  {
    create_indexes(*_neural_network_helper, _options.data_is_unique());
  }

  // create the callback task if we need one.
  SingleTaskQueue<bool>* callback_task = nullptr;
  if (progress_callback != nullptr)
  {
    callback_task = new SingleTaskQueue<bool>();
    callback_task->call(progress_callback , std::ref(*_neural_network_helper));
    if (!callback_task->get())
     {
      Logger::warning("Progress callback function returned false before training started, closing now!");
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
  std::vector<std::vector<GradientsAndOutputs>> epoch_gradients;
  epoch_gradients.reserve(num_batches);

  // for the learning rate decay we need to set the target learning rate and the decay rate.
  auto number_of_epoch_after_decay = number_of_epoch - static_cast<int>(std::round(_options.learning_rate_warmup_target() * number_of_epoch));
  const auto learning_rate_decay_rate = number_of_epoch_after_decay == 0 || _options.learning_rate_decay_rate() == 0 ? 0 : std::log(1.0 / _options.learning_rate_decay_rate()) / number_of_epoch_after_decay;

  // because we boost the rate from time to time, the base, (or target rate), we will use is different.
  double learning_rate_base = _options.learning_rate();

  AdaptiveLearningRateScheduler learning_rate_scheduler;

  for (auto epoch = 0; epoch < number_of_epoch; ++epoch)
  {
    _neural_network_helper->set_epoch(epoch);
    _learning_rate = _neural_network_helper->learning_rate();

    for (size_t start_index = 0; start_index < training_indexes_size; start_index += batch_size)
    {
      const size_t end_size = std::min(start_index + batch_size, training_indexes_size);
      const size_t total_size = end_size - start_index;
      if (task_pool != nullptr)
      {
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

      if (options().shuffle_training_data())
      {
        recreate_batch_from_indexes(*_neural_network_helper, training_inputs, training_outputs, batch_training_inputs, batch_training_outputs);
      }

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
      Logger::warning("Progress callback function returned false during training, closing now!");
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
      NeuralNetworkOptions::ErrorCalculation::mape,
      NeuralNetworkOptions::ErrorCalculation::wape
    }, true);

  Logger::info("Final RMSE Error: ", std::fixed, std::setprecision (15), metrics[0].error());
  Logger::info("Final mean absolute error (MAPE): ", std::fixed, std::setprecision (15), metrics[1].error()*100.0);
  Logger::info("Final weighted absolute error (WAPE): ", std::fixed, std::setprecision(15), metrics[2].error() * 100.0);

  // finaly learning rate
  Logger::info("Final Learning rate: ", std::fixed, std::setprecision(15), _neural_network_helper->learning_rate());

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
      Logger::error("Trying to call Progress callback function but an error was returned.");
    }
  }
  else
  {
    Logger::debug("Progress callback function is still running, continuing to next epoch.");
  }
  return true; // continue training
}

double NeuralNetwork::calculate_global_clipping_scale() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  const double gradient_clip_threshold = options().clip_threshold();
  double total_sq_sum = 0.0;

  for (auto layer_number =0; layer_number < _layers.size(); ++layer_number)
  {
    const auto& layer = _layers[layer_number];

    // 1. Accumulate weights (W_ih for RNN, W for FNN)
    for (const auto& input_row : layer.get_weight_params()) 
    {
      for (const auto& wparam : input_row)
      {
        total_sq_sum += wparam.get_unclipped_gradient() * wparam.get_unclipped_gradient();
      }
    }

    // 2. Accumulate biases (B)
    for (const auto& bparam : layer.get_bias_weight_params()) 
    {
      total_sq_sum += bparam.get_unclipped_gradient() * bparam.get_unclipped_gradient();
    }

    // 3. Accumulate Residual/Recurrent weights (W_s, W_hh, etc.)
    if (!layer.get_residual_weight_params().empty())
    {
      for (const auto& row : layer.get_residual_weight_params())
      {
        for (const auto& wparam : row)
        {
          double g = wparam.get_unclipped_gradient();
          total_sq_sum += g * g;
        }
      }
    }

    // --- IMPORTANT: ADD RECURRENT WEIGHTS HERE ---
    // if (RecurrentLayer* rnn_layer = dynamic_cast<RecurrentLayer*>(&layer)) {
    //   for (const auto& row : rnn_layer->get_recurrent_weights()) {
    //     // ... accumulate squared gradients for W_hh
    //   }
    // }
  }

  // 4. Compute global gradient norm
  double norm = std::sqrt(total_sq_sum);

  if (!std::isfinite(norm))
  {
    Logger::error("Layers gradient norm is NaN/Inf. Resetting optimizer buffers and skipping batch.");
    return 0.0;
  }

  if (norm <= gradient_clip_threshold || norm <= 0.0)
  {
    return 1.0;
  }

  // Scale factor < 1
  double clipping_scale = gradient_clip_threshold / norm;
  Logger::warning([&]
    {
      auto lr = get_learning_rate();
      std::ostringstream ss;
      ss << std::setprecision(4)
        << "Layers gradient clipping: norm=" << norm
        << " scale=" << clipping_scale
        << " (learning rate: " << lr << ")";
      return ss.str();
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
    Logger::trace([=] 
      {
        return Logger::factory("Learning rate to ", std::fixed, std::setprecision(15), learning_rate, " at epoch ", epoch, " (", std::setprecision(4), completed_percent * 100.0, "%)");
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
    Logger::trace("Adaptive learning rate to ", std::fixed, std::setprecision(15), learning_rate, " at epoch ", epoch, " (", std::setprecision(4), completed_percent * 100.0, "%)");
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
  Logger::trace([=] {
      return Logger::factory("Learning rate warmup to ", std::fixed, std::setprecision(15), warmup_learning_rate, " at epoch ", epoch, " (", std::setprecision(4), completed_percent * 100.0, "%)");
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

  case NeuralNetworkOptions::ErrorCalculation::nrmse:
    return calculate_nrmse_error(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::mape:
    return calculate_forecast_mape(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::wape:
    return calculate_forecast_wape(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::smape:
    return calculate_forecast_smape(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::directional_accuracy:
    return calculate_directional_accuracy(ground_truth, predictions);

  case NeuralNetworkOptions::ErrorCalculation::bce_loss:
      return calculate_bce_loss(ground_truth, predictions);
  }

  Logger::error("Unknown ErrorCalculation type!");
  throw std::invalid_argument("Unknown ErrorCalculation type!");
}

double NeuralNetwork::calculate_huber_loss_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
#if VALIDATE_DATA == 1
  if (ground_truth.size() != predictions.size())
  {
    Logger::error("Mismatched number of samples");
    throw std::invalid_argument("Mismatched number of samples");
  }
#endif

  double total_loss = 0.0;
  size_t count = 0;

  for (size_t i = 0; i < ground_truth.size(); ++i)
  {
    if (ground_truth[i].size() != predictions[i].size())
    {
      Logger::error("Mismatched vector sizes at index ", i);
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
    Logger::error("Mismatched number of samples");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }
  

  double total_abs_error = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < ground_truth.size(); ++i)
  {
    if (ground_truth[i].size() != predictions[i].size())
    {
      Logger::error("Mismatched vector sizes at index ", i);
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

double NeuralNetwork::calculate_rmse_error(
  const std::vector<std::vector<double>>& ground_truths,
  const std::vector<std::vector<double>>& predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty())
  {
    Logger::error("Mismatched number of samples");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  double total_rmse = 0.0;
  size_t sequence_count = 0;

  for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
  {
    const auto& gt = ground_truths[seq_idx];
    const auto& pred = predictions[seq_idx];

    if (gt.size() != pred.size() || gt.empty())
      continue;

    double mse = 0.0;
    for (size_t i = 0; i < gt.size(); ++i)
    {
      double diff = gt[i] - pred[i];
      mse += diff * diff;
    }

    mse /= gt.size();
    total_rmse += std::sqrt(mse);
    ++sequence_count;
  }

  return (sequence_count == 0) ? 0.0 : (total_rmse / sequence_count);
}

double NeuralNetwork::calculate_bce_loss(
  const std::vector<std::vector<double>>& ground_truths,
  const std::vector<std::vector<double>>& predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty()) 
  {
    Logger::error("Mismatched number of samples");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  double total_bce = 0.0;
  size_t sequence_count = 0;

  // small epsilon to avoid log(0)
  const double eps = 1e-12;

  for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx) 
  {
    const auto& gt = ground_truths[seq_idx];
    const auto& pred = predictions[seq_idx];

    if (gt.size() != pred.size() || gt.empty())
    {
      continue;
    }

    double bce = 0.0;
    for (size_t i = 0; i < gt.size(); ++i) 
    {
      // clip predictions to [eps, 1 - eps]
      double p = std::max(eps, std::min(1.0 - eps, pred[i]));
      double y = gt[i];

      bce += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    }

    bce /= gt.size();    // average over outputs in sequence
    total_bce += bce;
    ++sequence_count;
  }

  return (sequence_count == 0) ? 0.0 : (total_bce / sequence_count);
}

double NeuralNetwork::calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (ground_truth.size() != predictions.size()) 
  {
    Logger::error("Mismatch in batch sizes.");
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
      Logger::warning("Mismatch in output vector sizes at index ",i);
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
  const unsigned batch_size,
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

  assert(batch_size != 0);
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

std::vector<double> NeuralNetwork::calculate_weight_gradients(
  const unsigned batch_size,
  unsigned layer_number, 
  unsigned neuron_number, 
  const GradientsAndOutputs& source) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (layer_number == 0 || layer_number > source.num_gradient_layers())
  {
    return {};
  }
  assert(batch_size != 0);
  
  const unsigned num_inputs = source.num_outputs(layer_number - 1); // prev layer outputs
  const double   delta = source.get_gradient(layer_number, neuron_number);

  std::vector<double> gradients(num_inputs, 0.0);
  
  // input weight gradients
  for (unsigned i = 0; i < num_inputs; ++i)
  {
    const double prev_activation = source.get_output(layer_number - 1, i);
    gradients[i] = (prev_activation * delta);
  }

  return gradients;
}

void NeuralNetwork::apply_weight_gradients(
  Layers& layers, 
  const std::vector<std::vector<GradientsAndOutputs>>& batch_activation_gradients, 
  double learning_rate, 
  unsigned epoch) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::size_t total_size = 0;
  for (const auto& v : batch_activation_gradients)
  {
    total_size += v.size();
  }

  std::vector<GradientsAndOutputs> flat_batch_activation_gradients;
  flat_batch_activation_gradients.reserve(total_size);
  for (const auto& gradients : batch_activation_gradients)
  {
    for (const auto& inner_gradients : gradients)
    {
      flat_batch_activation_gradients.emplace_back(inner_gradients);
    }
  }

  // pass the flattened vector.
  apply_weight_gradients(layers, flat_batch_activation_gradients, learning_rate, epoch);
}

void NeuralNetwork::apply_weight_gradients(
  Layers& layers,
  const std::vector<GradientsAndOutputs>& batch_activation_gradients,
  double learning_rate,
  unsigned epoch) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  if (batch_activation_gradients.empty())
    return;

  const unsigned batch_size = static_cast<unsigned>(batch_activation_gradients.size());
  const unsigned num_layers = batch_activation_gradients[0].num_gradient_layers();

  // Lock network for writing
  std::unique_lock<std::shared_mutex> write(_mutex);

  // ----------------------------------------------------------------
  // Compute clipping scale using current (unclipped) gradients
  // ----------------------------------------------------------------
  double global_clipping_scale = calculate_global_clipping_scale();

  // Walk backwards: output -> first hidden, skip input layer (0)
  for (int layer_number = num_layers - 1; layer_number > 0; --layer_number)
  {
    auto& previous_layer = layers[layer_number - 1];
    auto& current_layer = layers[layer_number];

    const unsigned num_outputs = current_layer.number_neurons();            // this layer size
    const unsigned num_inputs = current_layer.number_input_neurons(false); // inputs (no bias)
    const bool has_bias = current_layer.has_bias();
    const unsigned num_inputs_with_bias = static_cast<unsigned>(num_inputs + (has_bias ? 1 : 0));

    // --- accumulate gradients across batch (weights only, no bias)
    std::vector<double> accumulated_weight_gradients(num_inputs, 0.0);
    std::vector<double> accumulated_residual_gradients;

    // For every neuron index in THIS layer
    for (unsigned neuron_index = 0; neuron_index < num_outputs; ++neuron_index)
    {
      bool residual_initialized = false;

      for (unsigned b = 0; b < batch_size; ++b)
      {
        const auto& g = batch_activation_gradients[b];

        // normal weight gradients (returns vector sized num_inputs)
        const auto weight_grads =
          calculate_weight_gradients(batch_size, layer_number, neuron_index, g);

        if (weight_grads.size() != num_inputs)
        {
          Logger::error("calculate_weight_gradients returned size ", weight_grads.size(),
            " but expected ", num_inputs);
          throw std::runtime_error("calculate_weight_gradients size mismatch");
        }

        for (size_t w = 0; w < weight_grads.size(); ++w)
          accumulated_weight_gradients[w] += weight_grads[w];

        // residual gradients (if any)
        std::vector<double> residual_output_values;
        auto* residual_layer = get_residual_layer(layers, g, residual_output_values, current_layer);

        if (residual_layer)
        {
          const auto residual_grads =
            calculate_residual_projection_gradients(batch_size, layer_number, neuron_index, g);

          if (!residual_initialized)
          {
            accumulated_residual_gradients.assign(residual_grads.size(), 0.0);
            residual_initialized = true;
          }

          if (residual_grads.size() != accumulated_residual_gradients.size())
          {
            Logger::panic("Mismatched residual_grad sizes");
          }

          for (size_t r = 0; r < residual_grads.size(); ++r)
          {
            accumulated_residual_gradients[r] += residual_grads[r];
          }
        }
      } // end batch

      // Average gradients across batch
      for (double& v : accumulated_weight_gradients)
      {
        v /= static_cast<double>(batch_size);
      }

      if (!accumulated_residual_gradients.empty())
      {
        for (double& v : accumulated_residual_gradients)
        {
          v /= static_cast<double>(batch_size);
        }
      }

      // ----------------------------------------------------------------
      // 1) STORE unclipped gradients into WeightParam objects
      // ----------------------------------------------------------------
      for (unsigned w = 0; w < num_inputs; ++w)
      {
        auto& wp = current_layer.get_weight_param(w, neuron_index);
        wp.set_unclipped_gradient(accumulated_weight_gradients[w]);
      }

      if (has_bias)
      {
        auto& bp = current_layer.get_bias_weight_param(neuron_index);

        // if you compute bias gradient separately, place it into the last slot of accumulated_weight_gradients
        double bias_grad = 0.0;
        // compute bias gradient as average delta:
        {
          double delta_sum = 0.0;
          for (unsigned b = 0; b < batch_size; ++b)
          {
            delta_sum += batch_activation_gradients[b].get_gradient(layer_number, neuron_index);
          }
          bias_grad = delta_sum / static_cast<double>(batch_size);
        }
        bp.set_unclipped_gradient(bias_grad);
      }

      // If residual grads exist and residual weights are WeightParam, store them too
      if (!accumulated_residual_gradients.empty() && !current_layer.get_residual_weight_params().empty())
      {
        auto& residual_params = current_layer.get_residual_weight_params(neuron_index);
#if VALIDATE_DATA == 1
        if (residual_params.size() != accumulated_residual_gradients.size())
        {
          Logger::panic("residual param size mismatch");
        }
#endif
        for (size_t r = 0; r < accumulated_residual_gradients.size(); ++r)
        {
          // assume residual_params[r] is vector<WeightParam> sized N_this
          auto& rwp = residual_params[r];
          rwp.set_unclipped_gradient(accumulated_residual_gradients[r]);
        }
      }
    }

    for (unsigned neuron_index = 0; neuron_index < num_outputs; ++neuron_index)
    {
      // ----------------------------------------------------------------
      // Apply clipped gradients through layer's optimizer helper
      // ----------------------------------------------------------------
      // For each input weight:
      for (unsigned w = 0; w < num_inputs; ++w)
      {
        auto& wp = current_layer.get_weight_param(w, neuron_index);
        auto unclipped_gradient = wp.get_unclipped_gradient();

        // Apply the GLOBAL clipping scale
        double clipped_gradient = (global_clipping_scale <= 0.0) ? 
          wp.clip_gradient(unclipped_gradient) : 
          unclipped_gradient * global_clipping_scale;

        // apply via layer helper (this will set the gradient used by optimizer inside WeightParam)
        current_layer.apply_weight_gradient(clipped_gradient, learning_rate, false, wp, global_clipping_scale);
        wp.clear_unclipped_gradient();
      }

      // Bias:
      if (has_bias)
      {
        auto& bp = current_layer.get_bias_weight_param(neuron_index);
        auto unclipped_gradient = bp.get_unclipped_gradient();
        double clipped_gradient = (global_clipping_scale <= 0.0) ? 
          bp.clip_gradient(unclipped_gradient) : 
          unclipped_gradient * global_clipping_scale;

        current_layer.apply_weight_gradient(clipped_gradient, learning_rate, true, bp, global_clipping_scale);
        bp.clear_unclipped_gradient();
      }

      // Residuals: apply if stored as WeightParam
      if (!accumulated_residual_gradients.empty() && !current_layer.get_residual_weight_params().empty())
      {
        auto& residual_params = current_layer.get_residual_weight_params(neuron_index);
        for (size_t r = 0; r < accumulated_residual_gradients.size(); ++r)
        {
          auto& rwp = residual_params[r];
          double unclipped = rwp.get_unclipped_gradient();
          double clipped = (global_clipping_scale <= 0.0) ? 
            rwp.clip_gradient(unclipped) : 
            unclipped * global_clipping_scale;
          current_layer.apply_weight_gradient(clipped, learning_rate, false, rwp, global_clipping_scale);
          rwp.clear_unclipped_gradient();
        }
      }
    } // end neuron loop
  } // end layer loop
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

void NeuralNetwork::calculate_back_propagation(
  std::vector<GradientsAndOutputs>& gradients, 
  const std::vector<std::vector<double>>& outputs, 
  const Layers& layers) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  
  calculate_back_propagation_output_layer(gradients, outputs, layers);
  calculate_back_propagation_hidden_layers(gradients, layers);
  calculate_back_propagation_input_layer(gradients, layers);
}

void NeuralNetwork::calculate_back_propagation_input_layer(
  std::vector<GradientsAndOutputs>& gradients,
  const Layers& layers) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  
  // input layer is all 0, (bias is included)
  const auto& input_gradients = std::vector<double>(layers.input_layer().number_neurons(), 0.0);
  std::vector<std::vector<double>> full_input_gradients;
  full_input_gradients.resize(gradients.size(), input_gradients);
  set_gradients_for_layer(gradients, 0, full_input_gradients);
}

void NeuralNetwork::calculate_back_propagation_output_layer(
  std::vector<GradientsAndOutputs>& gradients,
  const std::vector<std::vector<double>>& outputs,
  const Layers& layers) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
#if VALIDATE_DATA == 1
  assert(outputs.size() == gradients.size());
  for (auto gradients_index = 0; gradients_index < gradients.size(); ++gradients_index)
  {
    // the given output gradient does not have bias.
    assert(outputs[gradients_index].size() == gradients[gradients_index].output_back().size());
  }
#endif

  // set the output gradient
  const auto& output_layer_number = static_cast<unsigned>(layers.size() - 1);
  const auto& output_layer = layers.output_layer();
  std::vector<std::vector<double>> given_outputs;
  given_outputs.reserve(gradients.size());

  for (auto gradients_index = 0; gradients_index < gradients.size(); ++gradients_index)
  {
    given_outputs.emplace_back(gradients[gradients_index].output_back());
  }
  auto next_activation_gradients = output_layer.calculate_output_gradients(outputs, given_outputs);

  set_gradients_for_layer(gradients, output_layer_number, next_activation_gradients);
}

void NeuralNetwork::calculate_back_propagation_hidden_layers(
    std::vector<GradientsAndOutputs>& gradients,
    const Layers& layers) const
{
  // get the last calculated gradients, (in this case from the output layer).
  const unsigned output_layer_number = static_cast<unsigned>(layers.size() - 1);
  std::vector<std::vector<double>> next_activation_gradients;
  next_activation_gradients.reserve(gradients.size());
  for (auto& gradient : gradients)
  {
    next_activation_gradients.emplace_back(gradient.get_gradients(output_layer_number));
  }

  // we are going backward from output to input
  for (auto layer_number = layers.size() - 2; layer_number > 0; --layer_number)
  {
    // Input => Hidden(0) => Hidden(1) => Output
    const auto& hidden_0 = layers[static_cast<unsigned>(layer_number)];
    const auto& hidden_1 = layers[static_cast<unsigned>(layer_number + 1)];

    const auto output_values = get_outputs_for_layer(gradients, layer_number);
    const auto& next_gradients = get_gradients_for_layer(gradients, layer_number+1);

    auto current_activation_gradients = hidden_0.calculate_hidden_gradients(hidden_1, next_gradients, output_values);

    set_gradients_for_layer(gradients, static_cast<unsigned>(layer_number), current_activation_gradients);
    next_activation_gradients = std::move(current_activation_gradients);
    current_activation_gradients.clear();
  }
}

std::vector<std::vector<double>> NeuralNetwork::get_outputs_for_layer(const std::vector<GradientsAndOutputs>& source, unsigned layer_number) const
{
  std::vector<std::vector<double>> outputs;
  outputs.reserve(source.size());
  for (const auto& gradient : source)
  {
    outputs.emplace_back(gradient.get_outputs(layer_number));
  }
  return outputs;
}

std::vector<std::vector<double>> NeuralNetwork::get_gradients_for_layer(const std::vector<GradientsAndOutputs>& source, unsigned layer_number) const
{
  std::vector<std::vector<double>> gradients;
  gradients.reserve(source.size());
  for (const auto& gradient : source)
  {
    gradients.emplace_back(gradient.get_gradients(layer_number));
  }
  return gradients;
}

void NeuralNetwork::set_gradients_for_layer(std::vector<GradientsAndOutputs>& source, unsigned layer_number, const std::vector<std::vector<double>>& gradients) const
{
  assert(source.size() == gradients.size());
  for (auto gradients_index = 0; gradients_index < gradients.size(); ++gradients_index)
  {
    source[gradients_index].set_gradients(layer_number, gradients[gradients_index]);
  }
}

void NeuralNetwork::calculate_forward_feed(
  std::vector<NeuralNetwork::GradientsAndOutputs>& gradients_and_output,
  std::vector<HiddenStates>& hidden_states,
  const std::vector<std::vector<double>>& inputs,
  const Layers& layers,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  const size_t batch_size = inputs.size();
  assert(gradients_and_output.size() == batch_size);

  // --- 1. Store input layer outputs (no bias appended here) ---
  for (size_t i = 0; i < batch_size; ++i)
  {
    gradients_and_output[i].set_outputs(0, inputs[i]);
  }

  // --- 2. Prepare per-layer outputs container (layer 0 = input) ---
  std::vector<std::vector<std::vector<double>>> layer_outputs;
  layer_outputs.push_back(inputs);

  // --- 3. Forward propagate from first hidden layer onward ---
  for (size_t layer_number = 1; layer_number < layers.size(); ++layer_number)
  {
    const auto& previous_layer = layers[static_cast<unsigned>(layer_number - 1)];
    const auto& current_layer = layers[static_cast<unsigned>(layer_number)];

    const auto& previous_layer_output_values = layer_outputs.back(); // batch x N_prev

    // --- 3a. Prepare residual outputs if current layer uses residuals ---
    std::vector<std::vector<double>> residual_outputs;
    const int residual_layer_number = current_layer.residual_layer_number();
    if (residual_layer_number != -1)
    {
#if VALIDATE_DATA == 1
      if (static_cast<size_t>(residual_layer_number) >= layer_outputs.size())
      {
        Logger::error("Residual layer number out of range");
        throw std::runtime_error("Residual layer number out of range");
      }
#endif

      residual_outputs = layer_outputs[static_cast<size_t>(residual_layer_number)];

#if VALIDATE_DATA == 1
      if (residual_outputs.size() != batch_size)
      {
        Logger::error("Residual outputs batch size mismatch");
        throw std::runtime_error("Residual outputs batch size mismatch");
      }
#endif
    }

    // --- 3b. Build hidden-state slices for this layer ---
    std::vector<std::vector<HiddenState>> hs;
    hs.reserve(hidden_states.size());
    for (auto& state : hidden_states)
    {
      hs.push_back(state.at(layer_number)); // keep original behaviour
    }

    // --- 3c. Call layer forward: correct signature is (previous_layer, previous_inputs, residuals, hs, is_training)
    auto forward_feed = current_layer.calculate_forward_feed(
      previous_layer,                // const Layer& previous_layer
      previous_layer_output_values,  // const std::vector<std::vector<double>>& previous_layer_inputs
      residual_outputs,              // const std::vector<std::vector<double>>& residual_output_values
      hs,                            // std::vector<std::vector<HiddenState>>& hidden_states
      is_training);                  // bool

    // --- 3d. Log activations (diagnostic) ---
    if (Logger::can_trace())
    {
      double sum = 0.0;
      double max_abs = 0.0;
      for (const auto& row : forward_feed)
      {
        for (const auto& val : row)
        {
          sum += val;
          max_abs = std::max(max_abs, std::fabs(val));
        }
      }
      const double total_values = forward_feed.empty() ? 1.0 : static_cast<double>(forward_feed.size() * forward_feed[0].size());
      double mean = sum / total_values;
      if (std::fabs(mean) < 1e-6 || std::fabs(mean) > 10 || std::fabs(max_abs) > 50)
      {
        Logger::trace([=]
          {
            return Logger::factory("[ACT] Layer ", layer_number, ": mean=", mean, ", max=", max_abs);
          });
      }
    }

    // --- 3e. Store outputs into gradients_and_output ---
    for (size_t b = 0; b < batch_size; ++b)
    {
      gradients_and_output[b].set_outputs(static_cast<unsigned>(layer_number), forward_feed[b]);
    }

    // --- 3f. Save forward outputs for next layer / residuals ---
    layer_outputs.emplace_back(std::move(forward_feed));
  }
}

double NeuralNetwork::calculate_forecast_smape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty()) 
  {
    Logger::error("Input vectors must have the same, non-zero size.");
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

double NeuralNetwork::calculate_directional_accuracy( const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double neutral_tolerance) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty())
  {
    Logger::error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  size_t correct = 0;
  size_t total = 0;

  for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
  {
    const auto& gt = ground_truths[seq_idx];
    const auto& pred = predictions[seq_idx];

    if (gt.size() != pred.size() || gt.size() < 2)
    {
      continue; // Skip sequences that are too short
    }

    for (size_t i = 1; i < gt.size(); ++i)
    {
      double gt_diff = gt[i] - gt[i - 1];
      double pred_diff = pred[i] - pred[i - 1];

      // Ignore negligible ground truth movements (noise)
      if (std::abs(gt_diff) < neutral_tolerance)
      {
        continue;
      }

      if ((gt_diff * pred_diff) > 0.0)
      {
        ++correct;
      }
      ++total;
    }
  }
  return (total == 0) ? 0.0 : (static_cast<double>(correct) / total);
}

double NeuralNetwork::calculate_nrmse_error(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty())
  {
    Logger::error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  double total_nrmse = 0.0;
  size_t sequence_count = 0;
  const double eps = 1e-12; // small value to avoid division by zero

  for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
  {
    const auto& gt = ground_truths[seq_idx];
    const auto& pred = predictions[seq_idx];

    if (gt.size() != pred.size() || gt.empty())
      continue;

    double mse = 0.0;
    double min_val = gt[0], max_val = gt[0], mean_abs = 0.0;

    for (size_t i = 0; i < gt.size(); ++i)
    {
      double diff = gt[i] - pred[i];
      mse += diff * diff;

      min_val = std::min(min_val, gt[i]);
      max_val = std::max(max_val, gt[i]);
      mean_abs += std::abs(gt[i]);
    }

    mse /= gt.size();
    double rmse = std::sqrt(mse);
    mean_abs /= gt.size();

    // Auto-select normalization
    double denom = max_val - min_val;         // primary: range
    if (denom < eps) denom = mean_abs;        // fallback: mean magnitude
    if (denom < eps) continue;                // skip if both tiny

    total_nrmse += rmse / denom;
    ++sequence_count;
  }

  return (sequence_count == 0) ? 0.0 : (total_nrmse / sequence_count);
}

double NeuralNetwork::calculate_forecast_wape(const std::vector<std::vector<double>>&ground_truths, const std::vector<std::vector<double>>&predictions) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty())
  {
    Logger::error("Input vectors must have the same, non-zero size.");
    throw std::invalid_argument("Input vectors must have the same, non-zero size.");
  }

  double total_absolute_error = 0.0;
  double total_absolute_actuals = 0.0;

  for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
  {
    const auto& gt = ground_truths[seq_idx];
    const auto& pred = predictions[seq_idx];

    // Skip mismatched or empty sequences
    if (gt.size() != pred.size() || gt.empty())
    {
      continue;
    }

    // Sum the errors and actuals for this sequence
    for (size_t i = 0; i < gt.size(); ++i)
    {
      total_absolute_error += std::abs(gt[i] - pred[i]);
      total_absolute_actuals += std::abs(gt[i]);
    }
  }

  // Perform a single division at the end
  // Check if the total sum of actuals is zero
  if (total_absolute_actuals == 0.0)
  {
    // If total actuals are 0, error is 0 only if total error is also 0.
    // Otherwise, it's undefined. We can return 0 if both are 0, 
    // or 1.0 (100% error) if we predicted non-zero values for all-zero actuals.
    return (total_absolute_error == 0.0) ? 0.0 : 1.0;
  }

  // WAPE formula
  return total_absolute_error / total_absolute_actuals;
}

double NeuralNetwork::calculate_forecast_mape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (predictions.size() != ground_truths.size() || predictions.empty()) 
  {
    Logger::error("Input vectors must have the same, non-zero size.");
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
  Logger::info("Training will use: ");
  Logger::info(tab, _neural_network_helper->training_indexes().size(), " training samples.");
  Logger::info(tab, _neural_network_helper->checking_indexes().size(), " in training error check samples.");
  Logger::info(tab, _neural_network_helper->final_check_indexes().size(), " final error check samples.");
  Logger::info(tab, "Data is shuffled           : ", options().shuffle_training_data() ? "true" : "false");
  Logger::info(tab, "Learning rate              : ", std::fixed, std::setprecision(15), _options.learning_rate());
  Logger::info(tab, "  Decay rate               : ", std::fixed, std::setprecision(15), _options.learning_rate_decay_rate());
  Logger::info(tab, "  Warmup start             : ", std::fixed, std::setprecision(15), _options.learning_rate_warmup_start());
  Logger::info(tab, "  Warmup target            : ", std::fixed, std::setprecision(4), _options.learning_rate_warmup_target()*100, "%");
  Logger::info(tab, "  Restart rate             : ", _options.learning_rate_restart_rate());
  Logger::info(tab, "  Restart boost            : ", _options.learning_rate_restart_boost());
  Logger::info(tab, "Gradient clip threshold    : ", std::fixed, std::setprecision(4), _options.clip_threshold());
  Logger::info(tab, "Hidden activation method   : ", activation::method_to_string(get_hidden_activation_method()));
  Logger::info(tab, "Output activation method   : ", activation::method_to_string(get_output_activation_method()));
  Logger::info(tab, "Residual layerjump         : ", _options.residual_layer_jump());
  Logger::info(tab, "Input size                 : ", training_inputs.front().size());
  Logger::info(tab, "Output size                : ", training_outputs.front().size());
  Logger::info(tab, "Optimiser                  : ", optimiser_type_to_string(_options.optimiser_type()));

  std::string hidden_layer_message = 
                                "  Hidden layers              : {";
  for (size_t layer = 1; layer < _layers.size() - 1; ++layer)
  {
    hidden_layer_message += std::to_string(_layers[static_cast<unsigned>(layer)].number_neurons());
    if (layer < _layers.size() - 2)
    {
      hidden_layer_message += ", ";
    }
  }
  hidden_layer_message += "}";
  Logger::info(hidden_layer_message);

  const auto& rl =_options.recurrent_layers();
  std::string recurrent_layer_message =
    "  Recurrent layers           : {";
  for (size_t rl_index = 1; rl_index < rl.size() - 1; ++rl_index)
  {
    recurrent_layer_message += std::to_string(rl[rl_index]);
    if (rl_index < rl.size() - 2)
    {
      recurrent_layer_message += ", ";
    }
  }
  recurrent_layer_message += "}";
  Logger::info(recurrent_layer_message);

  std::string dropout_layer_message = 
                                "  Hidden layers dropout rate : {";
  for( auto& dropout : options().dropout())
  {
    dropout_layer_message += std::to_string(dropout);
    dropout_layer_message += ", ";
  }
  dropout_layer_message = dropout_layer_message.substr(0, dropout_layer_message.size() - 2); // remove the last ", "
  dropout_layer_message += "}";
  Logger::info(dropout_layer_message);

  Logger::info(tab, "Batch size                 : ", _options.batch_size());
  if (_options.batch_size() > 1)
  {
    if (_options.number_of_threads() <= 0)
    {
      Logger::info(tab, "Number of threads          : ", (std::thread::hardware_concurrency() - 1));
    }
    else
    {
      Logger::info(tab, "Number of threads          : ", _options.number_of_threads());
    }
  }
}
