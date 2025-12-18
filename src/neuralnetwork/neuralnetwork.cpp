#include "elmanrnnlayer.h"
#include "logger.h"
#include "neuralnetwork.h"
#include "layergradients.h"

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
    options.weight_decay(),
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

const std::vector<std::unique_ptr<Layer>>& NeuralNetwork::get_layers() const
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
  hidden_states.resize(1, HiddenStates(get_topology()));
  std::vector<GradientsAndOutputs> gradients;
  std::vector<unsigned> rnn_topology_vec;
  for (unsigned rnn_layer_idx : _layers.recurrent_layers()) 
  {
    rnn_topology_vec.push_back(_layers[rnn_layer_idx].get_number_neurons());
  }
  gradients.push_back(GradientsAndOutputs(get_topology(), rnn_topology_vec));
  {
    std::shared_lock<std::shared_mutex> read(_mutex);
    const std::vector<std::vector<double>> all_inputs = { inputs };
    calculate_forward_feed(gradients, all_inputs.begin(), 1, _layers, hidden_states, false);
  }
  return gradients.front().output_back();
}

double NeuralNetwork::get_learning_rate() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _learning_rate;
}

NeuralNetworkHelper::NeuralNetworkHelperMetrics NeuralNetwork::calculate_forecast_metric(ErrorCalculation::type error_type) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto results = calculate_forecast_metrics({ error_type }, false);
  return results.front();
}

std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> NeuralNetwork::calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types) const
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

std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> NeuralNetwork::calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types, bool final_check) const
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
  std::vector<unsigned> rnn_topology_vec;
  for (unsigned rnn_layer_idx : _layers.recurrent_layers()) {
      rnn_topology_vec.push_back(_layers[rnn_layer_idx].get_number_neurons());
  }
  {
    std::shared_lock read(_mutex);
    for (size_t index = 0; index < prediction_size; ++index)
    {
      std::vector<GradientsAndOutputs> gradients;
      gradients.push_back(GradientsAndOutputs(get_topology(), rnn_topology_vec));
      std::vector<HiddenStates> hidden_states;
      hidden_states.resize(1, HiddenStates(get_topology()));

      const auto& checks_index = (*checks_indexes)[index];
      const auto& inputs = training_inputs[checks_index];
      const std::vector<std::vector<double>> all_inputs = { inputs };
      calculate_forward_feed(gradients, all_inputs.begin(), 1, _layers, hidden_states, false);
      predictions.emplace_back(gradients[0].output_back());

      // set the output we will need it just now.
      checking_outputs.emplace_back(taining_outputs[checks_index]);
    }
  }// release the lock

  for (size_t index = 0; index < error_types.size(); ++index)
  {
    errors.emplace_back(
      NeuralNetworkHelper::NeuralNetworkHelperMetrics(
        ErrorCalculation::calculate_error(error_types[index], checking_outputs, predictions),
        error_types[index]));
  }
  return errors;
}

std::vector<std::vector<double>> NeuralNetwork::think(const std::vector<std::vector<double>>& inputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<std::vector<double>> outputs;
  outputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    outputs.emplace_back(think(inputs[i]));
  }
  return outputs;
}

void NeuralNetwork::train_single_batch(
    std::vector<std::vector<double>>::const_iterator inputs_begin, 
    std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  )
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  std::vector<HiddenStates> hidden_states;
  hidden_states.resize(size, HiddenStates(get_topology()));

  std::vector<unsigned> rnn_topology_vec;
  for (unsigned rnn_layer_idx : _layers.recurrent_layers()) {
      rnn_topology_vec.push_back(_layers[rnn_layer_idx].get_number_neurons());
  }

  std::vector<GradientsAndOutputs> gradients;
  gradients.resize(size, GradientsAndOutputs(get_topology(), rnn_topology_vec));

  calculate_forward_feed(gradients, inputs_begin, size, _layers, hidden_states, true);
  calculate_back_propagation(gradients, outputs_begin, size, _layers, hidden_states);
  std::vector<LayerGradients> layer_gradients;
  apply_weight_gradients(_layers, gradients, _learning_rate, _neural_network_helper->epoch(), hidden_states, get_topology().size(), layer_gradients);
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

// TODO - re-add queues
// create the thread pool if we need one ...
//  TaskQueuePool<std::vector<GradientsAndOutputs>>* task_pool = nullptr;
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

      // TODO re-add queue
      /*
      if (task_pool != nullptr)
      {
        task_pool->enqueue([=]()
          {
            train_single_batch(
              batch_training_inputs.begin() + start_index,
              batch_training_outputs.begin() + start_index,
              total_size);
          });
      }
      else
      */
      {
        //  size is 1, it is faster to not use a thread.
        train_single_batch(
            batch_training_inputs.begin() + start_index,
            batch_training_outputs.begin() + start_index,
            total_size);
      }
    }
    MYODDWEB_PROFILE_MARK();

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

  // TODO re-add queue
  /*
  if(task_pool != nullptr)
  {
    task_pool->stop();
    delete task_pool;
  }
  */

  auto metrics = calculate_forecast_metrics(
    {
      ErrorCalculation::type::rmse,
      ErrorCalculation::type::mape,
      ErrorCalculation::type::wape
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

double NeuralNetwork::calculate_global_clipping_scale(const std::vector<LayerGradients>& layer_gradients) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  const double gradient_clip_threshold = options().clip_threshold();
  if (gradient_clip_threshold <= 0.0) // treat <=0 as "no clipping"
  {
    return 1.0;
  }

  if (layer_gradients.empty())
  {
    return 1.0;
  }

  // Accumulate squared sum in a local accumulator to help the compiler optimize.
  double total_sq_sum = 0.0;

  for (size_t li = 0, ln = layer_gradients.size(); li < ln; ++li)
  {
    const auto& layer_grad = layer_gradients[li];

    // weights: vector<vector<double>>
    for (size_t i = 0, in = layer_grad.weights.size(); i < in; ++i)
    {
      const auto& row = layer_grad.weights[i];
      const double* p = row.empty() ? nullptr : row.data();
      for (size_t j = 0, jn = row.size(); j < jn; ++j)
      {
        const double g = p[j];
        total_sq_sum += g * g;
      }
    }

    // biases: vector<double>
    {
      const auto& b = layer_grad.biases;
      const double* p = b.empty() ? nullptr : b.data();
      for (size_t j = 0, jn = b.size(); j < jn; ++j)
      {
        const double g = p[j];
        total_sq_sum += g * g;
      }
    }

    // recurrent_weights: vector<vector<double>>
    for (size_t r = 0, rn = layer_grad.recurrent_weights.size(); r < rn; ++r)
    {
      const auto& row = layer_grad.recurrent_weights[r];
      const double* p = row.empty() ? nullptr : row.data();
      for (size_t j = 0, jn = row.size(); j < jn; ++j)
      {
        const double g = p[j];
        total_sq_sum += g * g;
      }
    }
  }

  // Validate accumulation
  if (!std::isfinite(total_sq_sum))
  {
    Logger::error("Layers gradient accumulation produced NaN/Inf. Resetting optimizer buffers and skipping batch.");
    return 0.0;
  }

  if (total_sq_sum == 0.0)
  {
    return 1.0;
  }

  const double norm = std::sqrt(total_sq_sum);
  if (!std::isfinite(norm))
  {
    Logger::error("Layers gradient norm is NaN/Inf. Resetting optimizer buffers and skipping batch.");
    return 0.0;
  }

  if (norm <= gradient_clip_threshold)
  {
    return 1.0;
  }

  const double clipping_scale = gradient_clip_threshold / norm;

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
        ErrorCalculation::type::rmse,
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

void NeuralNetwork::apply_weight_gradients(
  Layers& layers, 
  const std::vector<GradientsAndOutputs>& batch_activation_gradients, 
  double learning_rate, 
  unsigned epoch,
  const std::vector<HiddenStates>& hidden_states,
  unsigned num_layers_param,
  std::vector<LayerGradients>& layer_gradients)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  if (batch_activation_gradients.empty())
  {
    return;
  }

  const unsigned batch_size = static_cast<unsigned>(batch_activation_gradients.size()); // This is now correct: outer vector size
  const unsigned num_layers = num_layers_param;

  // Lock network for writing
  std::unique_lock<std::shared_mutex> write(_mutex);
  layer_gradients.resize(num_layers);

  // --- Store unclipped gradients ---
  for (int layer_number = num_layers - 1; layer_number > 0; --layer_number)
  {
    auto& current_layer = layers[static_cast<unsigned>(layer_number)];
    
    auto* rnn_layer = dynamic_cast<ElmanRNNLayer*>(&current_layer);
    const unsigned num_outputs = current_layer.get_number_neurons();
    const unsigned num_inputs = current_layer.get_number_input_neurons();
    
    layer_gradients[layer_number].weights.resize(num_inputs, std::vector<double>(num_outputs, 0.0));
    layer_gradients[layer_number].biases.resize(num_outputs, 0.0);
    if (rnn_layer != nullptr)
    {
        layer_gradients[layer_number].recurrent_weights.resize(num_outputs, std::vector<double>(num_outputs, 0.0));
    }

    if (rnn_layer != nullptr) // Recurrent Layer (ElmanRNNLayer)
    {
      const auto& previous_layer = layers[static_cast<unsigned>(layer_number - 1)];
      const unsigned num_time_steps = hidden_states[0].at(layer_number).size();
      const unsigned num_prev_inputs = previous_layer.get_number_neurons();

      const double time_scale = (num_time_steps > 0) ? static_cast<double>(num_time_steps) : 1.0;
      const double denom = static_cast<double>(batch_size) * time_scale;

      // 1. Gradients for Input-to-Hidden Weights (_weights)
      for (unsigned i = 0; i < num_inputs; ++i)
      {
        for (unsigned j = 0; j < num_outputs; ++j)
        {
          double grad = 0.0;
          for (unsigned b = 0; b < batch_size; ++b)
          {
            const auto rnn_grads = batch_activation_gradients[b].get_rnn_gradients(layer_number);
            const auto& prev_outputs = batch_activation_gradients[b].get_outputs(layer_number - 1);
            for (unsigned t = 0; t < num_time_steps; ++t)
            {
              // dE/dz_j(t) * x_i(t)
              grad += rnn_grads[t * num_outputs + j] * prev_outputs[t * num_prev_inputs + i];
            }
          }
          // average over batch AND time
          layer_gradients[layer_number].weights[i][j] = grad / denom;
        }
      }

      // 2. Gradients for Recurrent Weights (_recurrent_weights)
      for (unsigned i = 0; i < num_outputs; ++i) // previous hidden state neuron
      {
        for (unsigned j = 0; j < num_outputs; ++j) // current neuron
        {
          double grad = 0.0;
          for (unsigned b = 0; b < batch_size; ++b)
          {
            const auto rnn_grads = batch_activation_gradients[b].get_rnn_gradients(layer_number);
            for (unsigned t = 1; t < num_time_steps; ++t) // Start from t=1 as h(t-1) is used
            {
              // dE/dz_j(t) * h_i(t-1)
              grad += rnn_grads[t * num_outputs + j] * hidden_states[b].at(layer_number)[t - 1].get_hidden_state_value_at_neuron(i);
            }
          }
          // average over batch AND time (note: t loop starts at 1, so denom_t = num_time_steps > 1 ? num_time_steps - 1 : 1)
          const double time_denom_rec = (num_time_steps > 1) ? static_cast<double>(num_time_steps - 1) : 1.0;
          layer_gradients[layer_number].recurrent_weights[i][j] = grad / (static_cast<double>(batch_size) * time_denom_rec);
        }
      }

      // 3. Gradients for Bias Weights (_bias_weights)
      for (unsigned j = 0; j < num_outputs; ++j)
      {
        double bias_grad = 0.0;
        for (unsigned b = 0; b < batch_size; ++b)
        {
          const auto rnn_grads = batch_activation_gradients[b].get_rnn_gradients(layer_number);
          for (unsigned t = 0; t < num_time_steps; ++t)
          {
            // dE/dz_j(t)
            bias_grad += rnn_grads[t * num_outputs + j];
          }
        }
        layer_gradients[layer_number].biases[j] = bias_grad / denom;
      }
    }
    else // FeedForward Layer (FFLayer)
    {
      // 1. Gradients for Weights (_weights)
      for (unsigned i = 0; i < num_inputs; ++i) 
      {
        for (unsigned j = 0; j < num_outputs; ++j) 
        {
          double grad = 0.0;
          for(unsigned b = 0; b < batch_size; ++b)
          {
            if (Logger::can_trace())
            {
              Logger::trace([&]
              {
                std::ostringstream ss;
                ss << "[GRAD_DEBUG] b=" << b << ", layer=" << layer_number << ", i=" << i << ", j=" << j
                   << ", get_gradient=" << batch_activation_gradients[b].get_gradient(layer_number, j)
                   << ", get_outputs=" << batch_activation_gradients[b].get_outputs(layer_number-1)[i];
                return ss.str();
              });
            }
            // dE/dz_j * x_i
            grad += batch_activation_gradients[b].get_gradient(layer_number, j) * batch_activation_gradients[b].get_outputs(layer_number-1)[i];
          }
          if (Logger::can_trace())
          {
            Logger::trace([&]
            {
              std::ostringstream ss;
              ss << "[GRAD_DEBUG] Calculated grad=" << grad << " for W[" << i << "][" << j << "]";
              return ss.str();
            });
          }
          layer_gradients[layer_number].weights[i][j] = grad / static_cast<double>(batch_size);
          if (Logger::can_trace())
          {
            Logger::trace([&]
            {
              std::ostringstream ss;
              ss << "[GRAD] FF Layer " << layer_number << " W[" << i << "][" << j << "]: " << grad / static_cast<double>(batch_size);
              return ss.str();
            });
          }
        }
      }

      // 2. Gradients for Bias Weights (_bias_weights)
      for(unsigned j=0; j<num_outputs; ++j)
      {
        double bias_grad = 0.0;
        for(unsigned b = 0; b < batch_size; ++b)
        {
            // dE/dz_j
            bias_grad += batch_activation_gradients[b].get_gradient(layer_number, j);
        }
        layer_gradients[layer_number].biases[j] = bias_grad / static_cast<double>(batch_size);
        if (Logger::can_trace())
        {
          Logger::trace([&]
          {
            std::ostringstream ss;
            ss << "[GRAD] FF Layer " << layer_number << " Bias[" << j << "]: " << bias_grad / static_cast<double>(batch_size);
            return ss.str();
          });
        }
      }
    }
    // Residual gradients (if any)
    std::vector<double> accumulated_residual_gradients;
    // Add residual gradient calculation here if needed
  }

  // --- Apply gradients ---
  double global_clipping_scale = calculate_global_clipping_scale(layer_gradients);
  for (int layer_number = num_layers - 1; layer_number > 0; --layer_number)
  {
    auto& current_layer = layers[layer_number];
    auto* rnn_layer = dynamic_cast<ElmanRNNLayer*>(&current_layer);

    const unsigned num_outputs = current_layer.get_number_neurons();
    const unsigned num_inputs = current_layer.get_number_input_neurons();

    for(unsigned j=0; j<num_outputs; ++j)
    {
      // Apply input-to-hidden weights
      for(unsigned i=0; i<num_inputs; ++i)
      {
        auto& wp = current_layer.get_weight_param(i,j);
        current_layer.apply_weight_gradient(layer_gradients[layer_number].weights[i][j], learning_rate, false, wp, global_clipping_scale, _options.clip_threshold());
      }

      // Apply bias weights
      auto& bp = current_layer.get_bias_weight_param(j);
      current_layer.apply_weight_gradient(layer_gradients[layer_number].biases[j], learning_rate, true, bp, global_clipping_scale, _options.clip_threshold());
      
      // Apply recurrent weights (if applicable)
      if (rnn_layer != nullptr)
      {
        // For each recurrent connection from previous hidden state neuron 'k' to current neuron 'j'
        for (unsigned k = 0; k < num_outputs; ++k) 
        {
            auto& rwp = rnn_layer->get_recurrent_weight_params()[k][j]; // W_rec from k to j
            current_layer.apply_weight_gradient(layer_gradients[layer_number].recurrent_weights[k][j], learning_rate, false, rwp, global_clipping_scale, _options.clip_threshold());
        }
      }
    }
    // Apply residual weights (if any)
  }
}


Layer* NeuralNetwork::get_residual_layer(Layers& layers, const GradientsAndOutputs& batch_activation_gradient, std::vector<double>& residual_output_values, unsigned current_layer_index) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const auto residual_layer_number = layers.get_residual_layer_number(current_layer_index);
  if (residual_layer_number == -1)
  {
    return nullptr; // no residual layer
  }
  assert(residual_output_values.size() == 0);
  auto* residual_layer = &(layers[static_cast<unsigned>(residual_layer_number)]);
  auto residual_layer_neuron_size = residual_layer->get_number_neurons();
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
  std::vector<std::vector<double>>::const_iterator outputs_begin,
  size_t batch_size,
  const Layers& layers,
  const std::vector<HiddenStates>& hidden_states)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  
  calculate_back_propagation_output_layer(gradients, outputs_begin, batch_size, layers, hidden_states);
  calculate_back_propagation_hidden_layers(gradients, layers, hidden_states);
  calculate_back_propagation_input_layer(gradients, layers);
}

void NeuralNetwork::calculate_back_propagation_input_layer(
  std::vector<GradientsAndOutputs>& gradients,
  const Layers& layers)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  
  // input layer is all 0, (bias is included)
  const auto& input_gradients = std::vector<double>(layers.input_layer().get_number_neurons(), 0.0);
  std::vector<std::vector<double>> full_input_gradients;
  full_input_gradients.resize(gradients.size(), input_gradients);
  set_gradients_for_layer(gradients, 0, full_input_gradients);
}

void NeuralNetwork::calculate_back_propagation_output_layer(
  std::vector<GradientsAndOutputs>& gradients,
  std::vector<std::vector<double>>::const_iterator outputs_begin,
  size_t batch_size,
  const Layers& layers,
  const std::vector<HiddenStates>& hidden_states)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
#if VALIDATE_DATA == 1
  assert(batch_size == gradients.size());
  for (auto gradients_index = 0; gradients_index < batch_size; ++gradients_index)
  {
    // the given output gradient does not have bias.
    assert((*(outputs_begin + gradients_index)).size() == gradients[gradients_index].output_back().size());
  }
#endif

  // set the output gradient
  const auto& output_layer_number = static_cast<unsigned>(layers.size() - 1);
  auto& output_layer = layers.output_layer();
  
  // build hs for output layer
  std::vector<std::vector<HiddenState>> hs;
  hs.reserve(hidden_states.size());
  for (const auto& state : hidden_states)
  {
      hs.push_back(state.at(output_layer_number));
  }

  for(size_t i = 0; i < batch_size; ++i)
  {
    output_layer.calculate_output_gradients(gradients[i], *(outputs_begin + i), hidden_states[i].at(output_layer_number), _options.clip_threshold(), _options.error_calculation_type());
  }
}

void NeuralNetwork::calculate_back_propagation_hidden_layers(
    std::vector<GradientsAndOutputs>& gradients,
    const Layers& layers,
    const std::vector<HiddenStates>& hidden_states)
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
    auto& hidden_0 = layers[static_cast<unsigned>(layer_number)];
    const auto& hidden_1 = layers[static_cast<unsigned>(layer_number + 1)];

    const auto output_values = get_outputs_for_layer(gradients, layer_number);
    const auto& next_gradients = get_gradients_for_layer(gradients, layer_number+1);

    // build hs for this layer
    std::vector<std::vector<HiddenState>> hs;
    hs.reserve(hidden_states.size());
    for(const auto& state : hidden_states)
    {
        hs.push_back(state.at(layer_number));
    }

    for(size_t i = 0; i < gradients.size(); ++i)
    {
      hidden_0.calculate_hidden_gradients(gradients[i], hidden_1, next_gradients[i], output_values[i], hidden_states[i].at(layer_number), _options.clip_threshold());
    }
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
  std::vector<GradientsAndOutputs>& gradients_and_output,
  std::vector<std::vector<double>>::const_iterator inputs_begin,
  size_t batch_size,
  const Layers& layers_container,
  std::vector<HiddenStates>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  assert(gradients_and_output.size() == batch_size);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& current_input = *(inputs_begin + b);
    // --- 1. Store input layer outputs (no bias appended here) ---
    gradients_and_output[b].set_outputs(0, current_input);

    // --- 2. Forward propagate from first hidden layer onward ---
    for (size_t layer_number = 1; layer_number < layers_container.size(); ++layer_number)
    {
      const auto& previous_layer = layers_container[static_cast<unsigned>(layer_number - 1)];
      const auto& current_layer = layers_container[static_cast<unsigned>(layer_number)];

      const auto previous_layer_input_for_batch_item = gradients_and_output[b].get_outputs(static_cast<unsigned>(layer_number - 1));

      // --- 2a. Prepare residual outputs if current layer uses residuals ---
      std::vector<double> residual_output_values;
      const auto* residual_projector = layers_container.get_residual_layer_projector(static_cast<unsigned>(layer_number));
      if (residual_projector != nullptr)
      {
        auto residual_layer_number = layers_container.get_residual_layer_number(static_cast<unsigned>(layer_number));
        std::vector<double> residual_layer_outputs = gradients_and_output[b].get_outputs(static_cast<unsigned>(residual_layer_number));

        residual_output_values = residual_projector->project(residual_layer_outputs);
      }

      // --- 2b. Build hidden-state slices for this layer ---
      std::vector<std::vector<HiddenState>> hs_batch;
      hs_batch.push_back(hidden_states[b].at(layer_number));

      // --- 2c. Call layer forward
      std::vector<HiddenState>& layer_hidden_states = hidden_states[b].at(layer_number);
      // Calculate num_time_steps for the current layer's inputs
      const size_t N_prev = previous_layer.get_number_neurons();
      const size_t current_num_time_steps = N_prev > 0 ? previous_layer_input_for_batch_item.size() / N_prev : 0;
      layer_hidden_states.resize(current_num_time_steps); // Resize to num_time_steps
      
      std::vector<double> forward_feed_result = current_layer.calculate_forward_feed(
        gradients_and_output[b], // Pass gradients_and_output for this batch item
        previous_layer,
        previous_layer_input_for_batch_item,
        residual_output_values,
        layer_hidden_states, // Pass the resized vector
        is_training);

      // --- 2d. Log activations (diagnostic) ---
      if (Logger::can_trace())
      {
        double sum = 0.0;
        double max_abs = 0.0;
        for (const auto& val : forward_feed_result)
        {
          sum += val;
          max_abs = std::max(max_abs, std::fabs(val));
        }
        const double total_values = static_cast<double>(forward_feed_result.size());
        double mean = sum / total_values;
        if (std::fabs(mean) < 1e-6 || std::fabs(mean) > 10 || std::fabs(max_abs) > 50)
        {
          Logger::trace([=]
            {
              return Logger::factory("[ACT] Batch ", b, " Layer ", layer_number, ": mean=", mean, ", max=", max_abs);
            });
        }
      }

      // --- 2e. Store outputs into gradients_and_output ---
      // This is now handled internally by calculate_forward_feed for RNNs
      // and directly by the return value for FFLayers.
      // For FFLayers, forward_feed_result contains the output of current batch item
      if (layers_container.recurrent_layers()[layer_number] == 0) // Not RNN, so it's FFLayer or similar
      {
          gradients_and_output[b].set_outputs(static_cast<unsigned>(layer_number), forward_feed_result);
      }
      // If it's an RNN, set_outputs and set_rnn_outputs are handled inside ElmanRNNLayer::calculate_forward_feed.
    }
  }
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
  Logger::info(tab, "Weight Decay               : ", std::fixed, std::setprecision(5), _options.weight_decay());
  Logger::info(tab, "Input size                 : ", training_inputs.front().size());
  Logger::info(tab, "Output size                : ", training_outputs.front().size());
  Logger::info(tab, "Optimiser                  : ", optimiser_type_to_string(_options.optimiser_type()));

  std::string hidden_layer_message = 
                                "  Hidden layers              : {";
  for (size_t layer = 1; layer < _layers.size() - 1; ++layer)
  {
    hidden_layer_message += std::to_string(_layers[static_cast<unsigned>(layer)].get_number_neurons());
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
