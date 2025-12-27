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
#include <future>
#include <thread>
#include <algorithm>

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
  gradients.push_back(GradientsAndOutputs(get_topology()));
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



std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> NeuralNetwork::calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types, bool final_check, size_t limit) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> errors;
  errors.reserve(error_types.size());

  {
    std::shared_lock read(_mutex);
    if (nullptr == _neural_network_helper)
    {
      for (const auto& error_type : error_types)
      {
        const auto& saved_error = _saved_errors.find(error_type);
        if (saved_error == _saved_errors.end())
        {
          errors.emplace_back(0.0, error_type);
          Logger::warning("Trying to get training metrics:", (int)error_type, " when no training was done!");
        }
        else
        {
          errors.emplace_back(saved_error->second, error_type);
        }
      }
      return errors;
    }
  }

  const NeuralNetworkHelper& helper = *_neural_network_helper;
  const auto& training_inputs = helper.training_inputs();
  const auto& taining_outputs = helper.training_outputs();

  const std::vector<size_t>* checks_indexes = final_check ? &helper.final_check_indexes() : &helper.checking_indexes();
  size_t prediction_size = checks_indexes->size();

  if (limit > 0 && limit < prediction_size)
  {
    prediction_size = limit;
  }

  if (prediction_size == 0)
  {
    for (const auto& error_type : error_types)
    {
      errors.emplace_back(0.0, error_type);
    }
    return errors;
  }

  std::vector<std::vector<double>> predictions;
  std::vector<std::vector<double>> checking_outputs;
  predictions.reserve(prediction_size);
  checking_outputs.reserve(prediction_size);

  {
    std::shared_lock read(_mutex);
    std::vector<GradientsAndOutputs> batch_gradients(prediction_size, GradientsAndOutputs(get_topology()));
    std::vector<HiddenStates> batch_hidden_states(prediction_size, HiddenStates(get_topology()));
    std::vector<size_t> sub_indices(checks_indexes->begin(), checks_indexes->begin() + prediction_size);

    calculate_forward_feed(batch_gradients, training_inputs, sub_indices, _layers, batch_hidden_states, false);

    for (size_t i = 0; i < prediction_size; ++i)
    {
      predictions.push_back(batch_gradients[i].output_back());
      checking_outputs.push_back(taining_outputs[sub_indices[i]]);
    }
  }

  for (const auto& error_type : error_types)
  {
    errors.emplace_back(
      ErrorCalculation::calculate_error(error_type, checking_outputs, predictions),
      error_type);
  }
  return errors;
}

std::vector<std::vector<double>> NeuralNetwork::think(const std::vector<std::vector<double>>& inputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const size_t batch_size = inputs.size();
  if (batch_size == 0) return {};

  std::vector<GradientsAndOutputs> batch_gradients;
  batch_gradients.resize(batch_size, GradientsAndOutputs(get_topology()));
  std::vector<HiddenStates> batch_hidden_states;
  batch_hidden_states.resize(batch_size, HiddenStates(get_topology()));

  {
    std::shared_lock<std::shared_mutex> read(_mutex);
    calculate_forward_feed(batch_gradients, inputs.begin(), batch_size, _layers, batch_hidden_states, false);
  }

  std::vector<std::vector<double>> outputs;
  outputs.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i)
  {
    outputs.push_back(batch_gradients[i].output_back());
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

  std::vector<GradientsAndOutputs> gradients;
  gradients.resize(size, GradientsAndOutputs(get_topology()));

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

// --- Replace calculate_global_clipping_scale with a diagnostic, per-term breakdown ---
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

  double total_sq_sum = 0.0;

  Logger::trace([&] { 
    return Logger::factory("Calculating global clipping scale for ", layer_gradients.size(), " layers (clip_threshold=", gradient_clip_threshold, ")."); 
  });

  for (size_t li = 0, ln = layer_gradients.size(); li < ln; ++li)
  {
    const auto& layer_grad = layer_gradients[li];

    double weight_sq = 0.0;
    double bias_sq = 0.0;
    double rec_sq = 0.0;
    double residual_sq = 0.0;

    // weights
    for (double g : layer_grad.weights) { weight_sq += g * g; }
    // biases
    for (double g : layer_grad.biases) { bias_sq += g * g; }
    // recurrent
    for (double g : layer_grad.recurrent_weights) { rec_sq += g * g; }
    // residual
    for (double g : layer_grad.residual_weights) { residual_sq += g * g; }

    double layer_sq_sum = weight_sq + bias_sq + rec_sq + residual_sq;

    if (!std::isfinite(layer_sq_sum))
    {
      Logger::error("Layer ", li, " gradient accumulation produced NaN/Inf. Resetting optimizer buffers and skipping batch.");
      return 0.0;
    }

    total_sq_sum += layer_sq_sum;

    Logger::trace([=] 
    {
      const double layer_norm = std::sqrt(layer_sq_sum);
      std::ostringstream ss;
      ss << "Grad breakdown layer " << li
          << " norm=" << std::setprecision(6) << layer_norm
          << " (w=" << std::sqrt(weight_sq)
          << ", b=" << std::sqrt(bias_sq)
          << ", r=" << std::sqrt(rec_sq)
          << ", res=" << std::sqrt(residual_sq) << ")";
      return ss.str();
    });
  }

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
      ss << std::fixed << std::setprecision(15)
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
  if (total_boosts == 0)
  {
    // Interval is larger than total epochs, cannot apply boost strategy as defined.
    return base_learning_rate;
  }

  auto per_boost_ratio = boost_ratio / total_boosts; // e.g., if 3 boosts, each boost is 1.10^(1/3)

  auto cycle_position = epoch % boost_interval; // e.g., if epoch 8 and interval is 7, position is 1
  auto progress = static_cast<double>(cycle_position) / boost_interval; // e.g., 1/7 = 0.142857

  // Apply cosine boost: start at 0, peak at 1 (smooth step up)
  // This creates a smooth staircase effect when combined with cumulative_boost.
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
  else
  {
    // boost the learning rate base if we need to.
    learning_rate = calculate_smooth_learning_rate_boost(epoch, number_of_epoch, learning_rate);

    // are we decaying the learning rate?
    // this is done after warmup
    if (learning_rate_decay_rate != 0)
    {
      learning_rate = learning_rate * std::exp(-learning_rate_decay_rate * epoch);
      Logger::trace([=] 
        {
          return Logger::factory("Learning rate to ", std::fixed, std::setprecision(15), learning_rate, " at epoch ", epoch, " (", std::setprecision(4), completed_percent * 100.0, "%)");
        });
    }
  }

  // then get the scheduler if we can improve it further.
  // This is done after warmup and is throttled to run every 5 epochs
  if (_options.adaptive_learning_rate() && completed_percent >= _options.learning_rate_warmup_target() && epoch % 5 == 0)
  {
    auto metric = calculate_forecast_metrics(
      {
        ErrorCalculation::type::rmse,
      }, false, 100 /*limit*/);
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
  auto compute_layer_gradients = [&](int layer_number)
  {
    auto& current_layer = layers[static_cast<unsigned>(layer_number)];
    
    auto* rnn_layer = dynamic_cast<ElmanRNNLayer*>(&current_layer);
    const unsigned num_outputs = current_layer.get_number_neurons();
    const unsigned num_inputs = current_layer.get_number_input_neurons();
    
    layer_gradients[layer_number].weights.resize(num_inputs * num_outputs, 0.0);
    layer_gradients[layer_number].biases.resize(num_outputs, 0.0);
    if (rnn_layer != nullptr)
    {
        layer_gradients[layer_number].recurrent_weights.resize(num_outputs * num_outputs, 0.0);
    }

    if (rnn_layer != nullptr) // Recurrent Layer (ElmanRNNLayer)
    {
      const auto& previous_layer = layers[static_cast<unsigned>(layer_number - 1)];
      const unsigned num_time_steps = (unsigned)hidden_states[0].at(layer_number).size();
      const unsigned num_prev_inputs = previous_layer.get_number_neurons();

      const int t_start = static_cast<int>(num_time_steps) - 1;
      const int t_end = options().enable_bptt() ? std::max(0, t_start - options().bptt_max_ticks() + 1) : t_start;
      const int active_ticks = t_start - t_end + 1;

      const double time_scale = (active_ticks > 0) ? static_cast<double>(active_ticks) : 1.0;
      const double denom = static_cast<double>(batch_size) * time_scale;

      for (unsigned b = 0; b < batch_size; ++b)
      {
        const auto& rnn_grads = batch_activation_gradients[b].get_rnn_gradients(layer_number);
        const auto& prev_outputs_rnn = batch_activation_gradients[b].get_rnn_outputs(static_cast<unsigned>(layer_number - 1));
        const auto& prev_outputs_std = batch_activation_gradients[b].get_outputs(static_cast<unsigned>(layer_number - 1));
        const auto& prev_outputs = !prev_outputs_rnn.empty() ? prev_outputs_rnn : prev_outputs_std;

        if (rnn_grads.empty() || prev_outputs.empty()) continue;

        // 1. Gradients for Input-to-Hidden Weights
        for (unsigned i = 0; i < num_inputs; ++i)
        {
          for (unsigned j = 0; j < num_outputs; ++j)
          {
            double grad = 0.0;
            // Only loop over active time steps
            for (int t = t_start; t >= t_end; --t)
            {
              grad += rnn_grads[t * num_outputs + j] * ((prev_outputs.size() == num_prev_inputs) ? prev_outputs[i] : prev_outputs[t * num_prev_inputs + i]);
            }
            layer_gradients[layer_number].weights[i * num_outputs + j] += grad;
          }
        }

        // 2. Gradients for Recurrent Weights
        for (unsigned i = 0; i < num_outputs; ++i)
        {
          for (unsigned j = 0; j < num_outputs; ++j)
          {
            double grad = 0.0;
            // t=0 has no previous hidden state for recurrence contribution
            int rec_t_end = std::max(1, t_end);
            for (int t = t_start; t >= rec_t_end; --t)
            {
              grad += rnn_grads[t * num_outputs + j] * hidden_states[b].at(layer_number)[t - 1].get_hidden_state_value_at_neuron(i);
            }
            layer_gradients[layer_number].recurrent_weights[i * num_outputs + j] += grad;
          }
        }

        // 3. Gradients for Bias Weights
        for (unsigned j = 0; j < num_outputs; ++j)
        {
          double bias_grad = 0.0;
          for (int t = t_start; t >= t_end; --t)
          {
            bias_grad += rnn_grads[t * num_outputs + j];
          }
          layer_gradients[layer_number].biases[j] += bias_grad;
        }
      }

      // Final normalization
      for (double& w : layer_gradients[layer_number].weights) w /= denom;
      const double time_denom_rec = (active_ticks > 1) ? static_cast<double>(active_ticks - 1) : 1.0;
      for (double& rw : layer_gradients[layer_number].recurrent_weights) rw /= (static_cast<double>(batch_size) * time_denom_rec);
      for (double& b_val : layer_gradients[layer_number].biases) b_val /= denom;
    }
    else // FeedForward Layer (FFLayer)
    {
      const auto& previous_layer_num = static_cast<unsigned>(layer_number - 1);
      for (unsigned b = 0; b < batch_size; ++b)
      {
        const auto& layer_outputs = batch_activation_gradients[b].get_outputs(previous_layer_num);
        const auto& layer_grads = batch_activation_gradients[b].get_gradients(layer_number);

        for (unsigned i = 0; i < num_inputs; ++i) 
        {
          const double input_val = layer_outputs[i];
          for (unsigned j = 0; j < num_outputs; ++j) 
          {
            layer_gradients[layer_number].weights[i * num_outputs + j] += layer_grads[j] * input_val;
          }
        }

        for(unsigned j=0; j<num_outputs; ++j)
        {
          layer_gradients[layer_number].biases[j] += layer_grads[j];
        }
      }
      for (double& w : layer_gradients[layer_number].weights) w /= static_cast<double>(batch_size);
      for (double& b_val : layer_gradients[layer_number].biases) b_val /= static_cast<double>(batch_size);
    }
    // Residual gradients (if any)
    const auto* residual_projector = current_layer.get_residual_projector();
    if (residual_projector != nullptr)
    {
      auto residual_layer_number = current_layer.get_residual_layer_number();
      const unsigned proj_output_size = residual_projector->output_size();
      const unsigned proj_input_size = residual_projector->input_size();

      layer_gradients[layer_number].residual_weights.assign(proj_output_size * proj_input_size, 0.0);

      for (unsigned b = 0; b < batch_size; ++b)
      {
        const auto& next_gradients = batch_activation_gradients[b].get_gradients(layer_number);
        const auto& residual_layer_outputs = batch_activation_gradients[b].get_outputs(static_cast<unsigned>(residual_layer_number));

        for (unsigned in = 0; in < proj_input_size; ++in)
        {
          for (unsigned out = 0; out < proj_output_size; ++out)
          {
            layer_gradients[layer_number].residual_weights[in * proj_output_size + out] += next_gradients[out] * residual_layer_outputs[in];
          }
        }
      }

      for (unsigned in = 0; in < proj_input_size; ++in)
      {
        for (unsigned out = 0; out < proj_output_size; ++out)
        {
          layer_gradients[layer_number].residual_weights[in * proj_output_size + out] /= static_cast<double>(batch_size);
        }
      }
    }
  };

  std::vector<std::future<void>> futures;
  for (int layer_number = num_layers - 1; layer_number > 0; --layer_number)
  {
      futures.push_back(std::async(std::launch::async, compute_layer_gradients, layer_number));
  }
  for (auto& f : futures) f.get();
  futures.clear();

  // --- Apply gradients ---
  double global_clipping_scale = calculate_global_clipping_scale(layer_gradients);
  
  auto apply_layer_gradients = [&](int layer_number)
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
        unsigned weight_index = i * num_outputs + j;
        current_layer.apply_weight_gradient(layer_gradients[layer_number].weights[weight_index], learning_rate, false, weight_index, global_clipping_scale);
      }

      // Apply bias weights
      if (current_layer.has_bias())
      {
        current_layer.apply_weight_gradient(layer_gradients[layer_number].biases[j], learning_rate, true, j, global_clipping_scale);
      }
      
      // Apply recurrent weights (if applicable)
      if (rnn_layer != nullptr)
      {
        // For each recurrent connection from previous hidden state neuron 'k' to current neuron 'j'
        for (unsigned k = 0; k < num_outputs; ++k) 
        {
            rnn_layer->apply_recurrent_weight_gradient(k, j, layer_gradients[layer_number].recurrent_weights[k * num_outputs + j], learning_rate, global_clipping_scale);
        }
      }
    }
    // Apply residual weights (if any)
    auto* residual_projector = current_layer.get_residual_projector();
    if (residual_projector != nullptr && !layer_gradients[layer_number].residual_weights.empty())
    {
      const unsigned proj_output_size = residual_projector->output_size();
      const unsigned proj_input_size = residual_projector->input_size();

      for (unsigned in = 0; in < proj_input_size; ++in)
      {
        for (unsigned out = 0; out < proj_output_size; ++out)
        {
          double gradient = layer_gradients[layer_number].residual_weights[in * proj_output_size + out];
          residual_projector->apply_weight_gradient(gradient, learning_rate, in, out, global_clipping_scale);
        }
      }
    }
  };

  for (int layer_number = num_layers - 1; layer_number > 0; --layer_number)
  {
      futures.push_back(std::async(std::launch::async, apply_layer_gradients, layer_number));
  }
  for (auto& f : futures) f.get();
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
  for (size_t i = 0; i < gradients.size(); ++i)
  {
      gradients[i].set_gradients(0, input_gradients);
  }
}

void NeuralNetwork::calculate_back_propagation_output_layer(
  std::vector<GradientsAndOutputs>& gradients,
  std::vector<std::vector<double>>::const_iterator outputs_begin,
  size_t batch_size,
  const Layers& layers,
  const std::vector<HiddenStates>& hidden_states)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  const auto& output_layer_number = static_cast<unsigned>(layers.size() - 1);
  auto& output_layer = layers.output_layer();
  
  output_layer.calculate_output_gradients(gradients, outputs_begin, hidden_states, _options.error_calculation_type());
}

void NeuralNetwork::calculate_back_propagation_hidden_layers(
    std::vector<GradientsAndOutputs>& gradients,
    const Layers& layers,
    const std::vector<HiddenStates>& hidden_states)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  // we are going backward from output to input
  for (auto layer_number = (int)layers.size() - 2; layer_number > 0; --layer_number)
  {
    auto& hidden_0 = layers[static_cast<unsigned>(layer_number)];
    const auto& hidden_1 = layers[static_cast<unsigned>(layer_number + 1)];

    std::vector<std::vector<double>> batch_next_gradients;
    batch_next_gradients.reserve(gradients.size());
    for (const auto& g : gradients)
    {
      std::vector<double> grad;
      if (_options.enable_bptt() && hidden_1.use_bptt())
      {
        grad = g.get_rnn_gradients(static_cast<unsigned>(layer_number + 1));
      }
      if (grad.empty())
      {
        grad = g.get_gradients(static_cast<unsigned>(layer_number + 1));
      }
      batch_next_gradients.emplace_back(std::move(grad));
    }

    hidden_0.calculate_hidden_gradients(gradients, hidden_1, batch_next_gradients, hidden_states, _options.bptt_max_ticks());
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

  // --- 1. Store input layer outputs for the entire batch ---
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& current_input = *(inputs_begin + b);
    gradients_and_output[b].set_outputs(0, current_input);

    if (options().enable_bptt() && options().bptt_max_ticks() > 1)
    {
      const size_t input_size = layers_container[0].get_number_neurons();
      if (current_input.size() == input_size)
      {
        std::vector<double> expanded;
        const int ticks = options().bptt_max_ticks();
        expanded.reserve(input_size * ticks);
        for (int t = 0; t < ticks; ++t) expanded.insert(expanded.end(), current_input.begin(), current_input.end());
        gradients_and_output[b].set_rnn_outputs(0, expanded);
      }
    }
  }

  // --- 2. Forward propagate layer by layer for the entire batch ---
  for (size_t layer_number = 1; layer_number < layers_container.size(); ++layer_number)
  {
    const auto& previous_layer = layers_container[static_cast<unsigned>(layer_number - 1)];
    const auto& current_layer = layers_container[static_cast<unsigned>(layer_number)];

    // Prepare batched residual outputs if needed
    std::vector<std::vector<double>> batch_residual_values;
    const auto* residual_projector = layers_container.get_residual_layer_projector(static_cast<unsigned>(layer_number));
    if (residual_projector != nullptr)
    {
      auto residual_layer_number = layers_container.get_residual_layer_number(static_cast<unsigned>(layer_number));
      std::vector<std::vector<double>> batch_residual_inputs;
      batch_residual_values.reserve(batch_size);
      for (size_t b = 0; b < batch_size; ++b)
      {
        batch_residual_inputs.emplace_back(gradients_and_output[b].get_outputs(static_cast<unsigned>(residual_layer_number)));
      }
      batch_residual_values = residual_projector->project_batch(batch_residual_inputs);
    }

    // Ensure hidden state vectors are sized correctly
    for (size_t b = 0; b < batch_size; ++b)
    {
        if (current_layer.use_bptt())
        {
            std::vector<double> prev_rnn_out = gradients_and_output[b].get_rnn_outputs(previous_layer.get_layer_index());
            if (prev_rnn_out.empty()) prev_rnn_out = gradients_and_output[b].get_outputs(previous_layer.get_layer_index());
            const size_t n_prev = previous_layer.get_number_neurons();
            const size_t num_time_steps = n_prev > 0 ? prev_rnn_out.size() / n_prev : 0;
            hidden_states[b].at(layer_number).assign(num_time_steps, HiddenState(current_layer.get_number_neurons()));
        }
        else
        {
            hidden_states[b].at(layer_number).assign(1, HiddenState(current_layer.get_number_neurons()));
        }
    }

    // Call batched forward feed
    current_layer.calculate_forward_feed(
      gradients_and_output,
      previous_layer,
      batch_residual_values,
      hidden_states,
      is_training
    );
  }
}
void NeuralNetwork::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& gradients_and_output,
  const std::vector<std::vector<double>>& all_inputs,
  const std::vector<size_t>& indices,
  const Layers& layers_container,
  std::vector<HiddenStates>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  size_t batch_size = indices.size();
  assert(gradients_and_output.size() == batch_size);

  // --- 1. Store input layer outputs for the entire batch ---
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& current_input = all_inputs[indices[b]];
    gradients_and_output[b].set_outputs(0, current_input);

    if (options().enable_bptt() && options().bptt_max_ticks() > 1)
    {
      const size_t input_size = layers_container[0].get_number_neurons();
      if (current_input.size() == input_size)
      {
        std::vector<double> expanded;
        const int ticks = options().bptt_max_ticks();
        expanded.reserve(input_size * ticks);
        for (int t = 0; t < ticks; ++t) expanded.insert(expanded.end(), current_input.begin(), current_input.end());
        gradients_and_output[b].set_rnn_outputs(0, expanded);
      }
    }
  }

  // --- 2. Forward propagate layer by layer for the entire batch ---
  for (size_t layer_number = 1; layer_number < layers_container.size(); ++layer_number)
  {
    const auto& previous_layer = layers_container[static_cast<unsigned>(layer_number - 1)];
    const auto& current_layer = layers_container[static_cast<unsigned>(layer_number)];

    // Prepare batched residual outputs if needed
    std::vector<std::vector<double>> batch_residual_values;
    const auto* residual_projector = layers_container.get_residual_layer_projector(static_cast<unsigned>(layer_number));
    if (residual_projector != nullptr)
    {
      auto residual_layer_number = layers_container.get_residual_layer_number(static_cast<unsigned>(layer_number));
      std::vector<std::vector<double>> batch_residual_inputs;
      batch_residual_values.reserve(batch_size);
      for (size_t b = 0; b < batch_size; ++b)
      {
        batch_residual_inputs.emplace_back(gradients_and_output[b].get_outputs(static_cast<unsigned>(residual_layer_number)));
      }
      batch_residual_values = residual_projector->project_batch(batch_residual_inputs);
    }

    // Ensure hidden state vectors are sized correctly
    for (size_t b = 0; b < batch_size; ++b)
    {
        if (current_layer.use_bptt())
        {
            std::vector<double> prev_rnn_out = gradients_and_output[b].get_rnn_outputs(previous_layer.get_layer_index());
            if (prev_rnn_out.empty()) prev_rnn_out = gradients_and_output[b].get_outputs(previous_layer.get_layer_index());
            const size_t n_prev = previous_layer.get_number_neurons();
            const size_t num_time_steps = n_prev > 0 ? prev_rnn_out.size() / n_prev : 0;
            hidden_states[b].at(layer_number).assign(num_time_steps, HiddenState(current_layer.get_number_neurons()));
        }
        else
        {
            hidden_states[b].at(layer_number).assign(1, HiddenState(current_layer.get_number_neurons()));
        }
    }

    // Call batched forward feed
    current_layer.calculate_forward_feed(
      gradients_and_output,
      previous_layer,
      batch_residual_values,
      hidden_states,
      is_training
    );
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
  Logger::info(tab, "BPTT Enabled               : ", _options.enable_bptt() ? "true" : "false");
  Logger::info(tab, "BPTT Max Ticks             : ", _options.bptt_max_ticks());

  std::string hidden_layer_message = "  Hidden layers              : {";
  // Log hidden layers details
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

  // Log recurrent layers details
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

  // Log dropout rates for hidden layers
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
