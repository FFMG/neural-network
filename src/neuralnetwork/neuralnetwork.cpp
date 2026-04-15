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
  _layers(options),
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
    .with_output_layer_details(OutputLayerDetails(topology.back(), activation(output_layer_activation, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.05))
    .build())
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
}

NeuralNetwork::NeuralNetwork(
  const Layers& layers,
  const NeuralNetworkOptions& options,
  const std::vector<std::map<ErrorCalculation::type, double>>& errors
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
    std::shared_lock<std::shared_mutex> rhs_lock(src._mutex, std::defer_lock);
    std::lock(lhs_lock, rhs_lock);

    _learning_rate = src._learning_rate;
    _layers = src._layers;
    _options = src._options;
    _saved_errors = src._saved_errors;
    _last_metrics = src._last_metrics;
    _gradients_pool = src._gradients_pool;
    _hidden_states_pool = src._hidden_states_pool;

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

const Layers& NeuralNetwork::get_layers() const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _layers;
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

void NeuralNetwork::create_indexes_in_lock(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const
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

void NeuralNetwork::create_shuffled_indexes_in_lock(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const
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
    Logger::panic("Logic error, unable to do a final batch error check.");
  }

  // then build the various indexes that will be used during testing.
  if(data_is_unique)
  {
    // because the data is unique we must use all of it for training
    // this is important in some cases where the NN needs all the data to train
    // otherwise we will only train on some of the data.
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

void NeuralNetwork::create_batch_from_indexes(
  const std::vector<size_t>& indexes, 
  const std::vector<std::vector<double>>& training_inputs, 
  const std::vector<std::vector<double>>& training_outputs, 
  std::vector<std::vector<double>>& training_inputs_data, 
  std::vector<std::vector<double>>& training_outputs_data) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  training_inputs_data.clear();
  training_outputs_data.clear();
  training_inputs_data.reserve(indexes.size());
  training_outputs_data.reserve(indexes.size());
  for( auto shuffled_index : indexes)
  {
    training_inputs_data.emplace_back(training_inputs[shuffled_index]);
    training_outputs_data.emplace_back(training_outputs[shuffled_index]);
  }
}

std::vector<std::vector<double>> NeuralNetwork::think(const std::vector<std::vector<double>>& inputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _layers.think(_options, inputs);
}

std::vector<double> NeuralNetwork::think(const std::vector<double>& inputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  if (inputs.size() != get_topology().front())
  {
    Logger::error("The input size, '", inputs.size(),"' does not match the topology!");
    return {};
  }
  return _layers.think(_options, inputs);
}

double NeuralNetwork::get_learning_rate() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  return _learning_rate;
}

double NeuralNetwork::get_percent_complete() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::shared_lock<std::shared_mutex> read(_mutex);
  if (nullptr == _neural_network_helper)
  {
    return 1.0;
  }
  return _neural_network_helper->percent_complete();
}

bool NeuralNetwork::has_training_data() const
{
  std::shared_lock<std::shared_mutex> read(_mutex);
  if (nullptr != _neural_network_helper)
  {
    return true; // we are currently training.
  }

  // do we have saved error, (from file).
  return !_saved_errors.empty();
}

NeuralNetworkHelperMetrics NeuralNetwork::calculate_forecast_metric(ErrorCalculation::type error_type) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto results = calculate_forecast_metrics({ error_type }, false);
  return results.front();
}

std::vector<NeuralNetworkHelperMetrics> NeuralNetwork::calculate_forecast_metric_all_layers(ErrorCalculation::type error_type) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto results = calculate_forecast_metrics_all_layers({ error_type }, false);
  return results.front();
}

std::vector<NeuralNetworkHelperMetrics> NeuralNetwork::calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types, bool final_check) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  auto results = calculate_forecast_metrics_all_layers(error_types, false);
  return results.front();
}

std::vector<std::vector<NeuralNetworkHelperMetrics>> NeuralNetwork::calculate_forecast_metrics_all_layers(const std::vector<ErrorCalculation::type>& error_types, bool final_check) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  {
    std::shared_lock read(_mutex);
    if (nullptr == _neural_network_helper)
    {
      std::vector<std::vector<NeuralNetworkHelperMetrics>> errors;
      errors.reserve(_saved_errors.size());

      std::vector<NeuralNetworkHelperMetrics> layer_errors;
      layer_errors.reserve(error_types.size());

      for (const auto& layer_saved_errors : _saved_errors)
      {
        for (const auto& error_type : error_types)
        {
          const auto& saved_error = layer_saved_errors.find(error_type);
          if (saved_error == layer_saved_errors.end())
          {
            layer_errors.emplace_back(0.0, error_type);
            Logger::warning("Trying to get training metrics:", (int)error_type, " when no training was done!");
          }
          else
          {
            layer_errors.emplace_back(saved_error->second, error_type);
          }
        }
        errors.emplace_back(layer_errors);
        layer_errors.clear();
      }
      return errors;
    }
  }

  const NeuralNetworkHelper& helper = *_neural_network_helper;
  const auto& training_inputs = helper.training_inputs();
  const auto& training_outputs = helper.training_outputs();

  const std::vector<size_t>* checks_indexes = final_check ? &helper.final_check_indexes() : &helper.checking_indexes();
  size_t prediction_size = checks_indexes->size();

  if (prediction_size == 0)
  {
    return {};
  }

  std::vector<std::vector<double>> predictions;
  std::vector<std::vector<double>> checking_outputs;
  predictions.reserve(prediction_size);
  checking_outputs.reserve(prediction_size);

  {
    std::shared_lock read(_mutex);

    // Use fresh, local vectors instead of the pool to ensure isolation.
    std::vector<GradientsAndOutputs> temp_gradients;
    std::vector<HiddenStates> temp_hidden_states;
    temp_gradients.reserve(prediction_size);
    temp_hidden_states.reserve(prediction_size);
    while (temp_gradients.size() < prediction_size)
    {
      temp_gradients.emplace_back(get_topology());
      temp_hidden_states.emplace_back(get_topology());
    }

    std::vector<size_t> sub_indices(checks_indexes->begin(), checks_indexes->begin() + prediction_size);

    calculate_forward_feed_for_forecast_metrics(temp_gradients, training_inputs, sub_indices, _layers, temp_hidden_states, false);

    for (size_t i = 0; i < prediction_size; ++i)
    {
      predictions.push_back(temp_gradients[i].output_back());
      checking_outputs.push_back(training_outputs[sub_indices[i]]);
    }
  }

  return _layers.output_layer().calculate_output_metrics(error_types, checking_outputs, predictions);
}

void NeuralNetwork::train_single_batch(
    std::vector<std::vector<double>>::const_iterator inputs_begin, 
    std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t batch_size
  )
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  _layers.train(_options, _learning_rate, inputs_begin, outputs_begin, batch_size);
}

void NeuralNetwork::create_bptt_batches(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, std::vector<std::vector<std::vector<double>>>& bptt_inputs, std::vector<std::vector<std::vector<double>>>& bptt_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  bptt_inputs.clear();
  bptt_outputs.clear();

  const size_t total_samples = inputs.size();
  if (total_samples == 0)
  {
    return;
  }
  if (total_samples != outputs.size())
  {
    Logger::panic("The training input data size does not match the output data size!");
  }

  const auto& bptt_size = _options.bptt_max_ticks();
  const size_t batch_size = static_cast<size_t>(_options.batch_size());

  // If BPTT is disabled or sequence length is 1, we just need to split the data into batches of 'batch_size'
  if (bptt_size <= 1 || !_options.enable_bptt())
  {
    for (size_t i = 0; i < total_samples; i += batch_size)
    {
      size_t end_idx = std::min(i + batch_size, total_samples);
      bptt_inputs.emplace_back(inputs.begin() + i, inputs.begin() + end_idx);
      bptt_outputs.emplace_back(outputs.begin() + i, outputs.begin() + end_idx);
    }
    return;
  }

  const auto& is_shuffled = _options.shuffle_bptt_batches();

  // Create sequences
  std::vector<std::vector<std::vector<double>>> sequences_inputs;
  std::vector<std::vector<std::vector<double>>> sequences_outputs;

  for (size_t start_idx = 0; start_idx < total_samples; start_idx += bptt_size)
  {
    size_t end_idx = std::min(start_idx + bptt_size, total_samples);

    std::vector<std::vector<double>> seq_input(inputs.begin() + start_idx, inputs.begin() + end_idx);
    std::vector<std::vector<double>> seq_output(outputs.begin() + start_idx, outputs.begin() + end_idx);

    // Optionally skip incomplete sequences (last one)
    if (seq_input.size() < bptt_size)
    {
      continue;
    }

    sequences_inputs.push_back(std::move(seq_input));
    sequences_outputs.push_back(std::move(seq_output));
  }

  // Shuffle sequences if needed
  if (!is_shuffled)
  {
    bptt_inputs = std::move(sequences_inputs);
    bptt_outputs = std::move(sequences_outputs);
    return;
  }

  std::vector<size_t> indices(sequences_inputs.size());
  for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  std::vector<std::vector<std::vector<double>>> shuffled_inputs;
  std::vector<std::vector<std::vector<double>>> shuffled_outputs;

  for (size_t idx : indices)
  {
    shuffled_inputs.push_back(std::move(sequences_inputs[idx]));
    shuffled_outputs.push_back(std::move(sequences_outputs[idx]));
  }

  bptt_inputs = std::move(shuffled_inputs);
  bptt_outputs = std::move(shuffled_outputs);
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
    Logger::panic("The batch size if either -ve or too large for the training sample.");
  }
  if(training_outputs.size() != training_inputs.size())
  {
    Logger::panic("The number of training samples does not match the number of expected outputs.");
  }

  Logger::info("Started training with ", training_inputs.size(), " inputs, ", number_of_epoch, " epoch and batch size ", batch_size, ".");

  // create the neural network helper.
  recreate_neural_network_helper(number_of_epoch, training_inputs, training_outputs);

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

  std::vector<std::vector<std::vector<double>>> bptt_in;
  std::vector<std::vector<std::vector<double>>> bptt_out;
  for (auto epoch = 0; epoch < number_of_epoch; ++epoch)
  {
    // set the values
    _neural_network_helper->set_epoch(epoch);
    _learning_rate = _neural_network_helper->learning_rate();

    // (re) create the bptt batches
    create_bptt_batches(batch_training_inputs, batch_training_outputs, bptt_in, bptt_out);

    const auto bptt_indexes_size = bptt_out.size();
    for (size_t bptt_index = 0; bptt_index < bptt_indexes_size; ++bptt_index)
    {
      const auto total_size = bptt_out[bptt_index].size();
      train_single_batch( bptt_in[bptt_index].begin(), bptt_out[bptt_index].begin(), total_size);
    }
    MYODDWEB_PROFILE_MARK();

    // Learning rate
    //
    auto learning_rate = calculate_learning_rate(learning_rate_base, learning_rate_decay_rate, epoch, number_of_epoch, learning_rate_scheduler);
    _neural_network_helper->set_learning_rate(learning_rate);

    // update the training monitor metrics
    if (_neural_network_helper->is_at_epoch_interval(_options.update_training_monitor_percent()))
    {
      Logger::trace([=] {
        return Logger::factory("Updating training monitor at epoch #", epoch, " of ", number_of_epoch);
        });
    }

    // callback
    // 
    if (!CallCallback(progress_callback, callback_task))
    {
      Logger::warning("Progress callback function returned false during training, closing now!");
      break;
    }

    MYODDWEB_PROFILE_MARK();
  }

  if (Logger::can_info() && options().final_error_calculation_types().size() > 0)
  {
    const auto& metrics = calculate_forecast_metrics_all_layers(options().final_error_calculation_types(), true);
    std::string message = "";
    unsigned output_layer_number = 0;
    for (const auto metric : metrics)
    {
      for (const auto metric_error : metric)
      {
        if (!message.empty())
        {
          message += "\n";
        }
        message += Logger::factory("Final: Layer = ", output_layer_number, ", ", ErrorCalculation::type_to_string(metric_error.error_type()), " error: ", std::fixed, std::setprecision(15), metric_error.error());
      }
      ++output_layer_number;
    }
    Logger::info(message);
  }

  // finally learning rate
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

void NeuralNetwork::recreate_neural_network_helper(int number_of_epoch, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs)
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
  std::unique_lock<std::shared_mutex> lock(_mutex);
  delete _neural_network_helper;
  _neural_network_helper = new NeuralNetworkHelper(*this, _learning_rate, number_of_epoch, training_inputs, training_outputs);

  // set all the indexes in the helper, either shuffled or not.
  if (options().shuffle_training_data())
  {
    create_shuffled_indexes_in_lock(*_neural_network_helper, _options.data_is_unique());
  }
  else
  {
    create_indexes_in_lock(*_neural_network_helper, _options.data_is_unique());
  }
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
    if (!_adaptive_lr_task.busy())
    {
      if (_adaptive_lr_task.has_result())
      {
        _last_metrics = _adaptive_lr_task.get();
      }

      // start a new task
      _adaptive_lr_task.call([this]() {
        return calculate_forecast_metrics({ ErrorCalculation::type::rmse }, false);
        });
    }

    if (!_last_metrics.empty())
    {
      learning_rate = learning_rate_scheduler.update(_last_metrics[0].error(), learning_rate, epoch, number_of_epoch);
      Logger::trace("Adaptive learning rate to ", std::fixed, std::setprecision(15), learning_rate, " at epoch ", epoch, " (", std::setprecision(4), completed_percent * 100.0, "%)");
    }
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

void NeuralNetwork::calculate_forward_feed_for_forecast_metrics(
  std::vector<GradientsAndOutputs>& gradients_and_output,
  const std::vector<std::vector<double>>& all_inputs,
  const std::vector<size_t>& indices,
  const Layers& layers_container,
  std::vector<HiddenStates>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");

  size_t batch_size = indices.size();
  if (batch_size == 0)
  {
    return;
  }

#if VALIDATE_DATA == 1
  if (gradients_and_output.size() != batch_size)
  {
    Logger::panic("The gradient vector size does not match the batch size!");
  }
#endif

  // --- 1. Store input layer outputs for the entire batch ---
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& current_input = all_inputs[indices[b]];
    if (b < 2)
    {
      Logger::trace([=]
        {
          return Logger::factory("DEBUG: [b=", b, "] Forecast Input[0]=", (current_input.empty() ? 0.0 : current_input[0]));
        });
    }
    const size_t input_size = layers_container[0].get_number_neurons();

    if (current_input.size() == input_size)
    {
      gradients_and_output[b].set_outputs(0, current_input);
      if (options().enable_bptt() && options().bptt_max_ticks() > 1)
      {
        const int ticks = options().bptt_max_ticks();
        std::vector<double> expanded;
        expanded.reserve(input_size * ticks);
        for (int t = 0; t < ticks; ++t)
        {
        expanded.insert(expanded.end(), current_input.begin(), current_input.end());
        }
        gradients_and_output[b].set_rnn_outputs(0, expanded);
      }
    }
    else if (options().enable_bptt() && input_size > 0 && current_input.size() % input_size == 0)
    {
       // Sequence input provided!
       std::vector<double> last_step(current_input.end() - input_size, current_input.end());
       gradients_and_output[b].set_outputs(0, last_step);
       gradients_and_output[b].set_rnn_outputs(0, current_input);
    }
    else
    {
       gradients_and_output[b].set_outputs(0, current_input);
    }
  }

  // --- 2. Forward propagate layer by layer for the entire batch ---
  for (size_t layer_number = 1; layer_number < layers_container.size(); ++layer_number)
  {
    const auto& previous_layer = layers_container[static_cast<unsigned>(layer_number - 1)];
    const auto& current_layer = layers_container[static_cast<unsigned>(layer_number)];

    // Prepare batched residual outputs
    std::vector<std::vector<double>> batch_residual_values;
    const auto* residual_projector = layers_container.get_residual_layer_projector(static_cast<unsigned>(layer_number));
    if (residual_projector != nullptr)
    {
      auto residual_layer_number = layers_container.get_residual_layer_number(static_cast<unsigned>(layer_number));
      std::vector<std::vector<double>> batch_residual_inputs;
      batch_residual_inputs.reserve(batch_size);
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
            const std::vector<double> prev_rnn_out = gradients_and_output[b].get_rnn_outputs(previous_layer.get_layer_index());
            const std::vector<double> prev_std_out = gradients_and_output[b].get_outputs(previous_layer.get_layer_index());
            const size_t seq_size = !prev_rnn_out.empty() ? prev_rnn_out.size() : prev_std_out.size();
            
            const size_t n_prev = previous_layer.get_number_neurons();
            const size_t num_time_steps = n_prev > 0 ? seq_size / n_prev : 0;
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
      batch_size,
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

  // Hidden
  Logger::info(tab, "Hidden layer(s)            : ");
  const auto& hl = _options.hidden_layers();
  
  // Log recurrent layers details
  for (size_t hl_index = 0; hl_index < hl.size(); ++hl_index)
  {
    const auto& this_hl = hl[hl_index];
    Logger::info(tab, tab, "[", hl_index, "] Type    : ", this_hl.get_type_string(), "\n",
                 tab, tab, tab, "Size                   : ", this_hl.get_size(), "\n",
                 tab, tab, tab, "Activation method      : ", activation::method_to_string(this_hl.get_activation().get_method()), "\n",
                 tab, tab, tab, "Activation alpha       : ", std::fixed, std::setprecision(5), this_hl.get_activation().get_alpha(), "\n",
                 tab, tab, tab, "Dropout                : ", std::fixed, std::setprecision(5), this_hl.get_dropout(), "\n",
                 tab, tab, tab, "Weight Decay           : ", std::fixed, std::setprecision(5), this_hl.get_weight_decay());
  }

  // Output
  const auto& output_layer_details = _options.output_layer_details();
  std::string output_layer_details_string;
  
  output_layer_details_string += Logger::factory(tab, "Output layer(s)            :", "\n");
  auto output_layer_index = 0;
  for (const auto& details : output_layer_details)
  {
    output_layer_details_string += Logger::factory(tab, tab, "[", output_layer_index, "]\n",
      tab, tab, tab, "Size                   : ", details.get_size(), "\n",
      tab, tab, tab, "Optimizer type         : ", optimiser_type_to_string(details.get_optimiser_type()), "\n",
      tab, tab, tab, "Weight decay           : ", std::fixed, std::setprecision(7), details.get_weight_decay(), "\n",
      tab, tab, tab, "Activation method      : ", activation::method_to_string(details.get_activation().get_method()), "\n",
      tab, tab, tab, "Activation alpha       : ", std::fixed, std::setprecision(5), details.get_activation().get_alpha(), "\n",
      tab, tab, tab, "Error calculation type : ", ErrorCalculation::type_to_string(details.get_output_error_calculation_type()), "\n",
      tab, tab, tab, "Error evaluation config: ", std::fixed, std::setprecision(5), "\n",
      tab, tab, tab, tab, "confidence-threshold : ", details.get_error_evaluation_config().confidence_threshold(), "\n",
      tab, tab, tab, tab, "neutral-tolerance    : ", details.get_error_evaluation_config().neutral_tolerance(), "\n",
      tab, tab, tab, tab, "huber delta          : ", details.get_error_evaluation_config().huber_delta(), "\n",
      tab, tab, tab, tab, "lambda\n",
      tab, tab, tab, tab, tab, "direction          : ", details.get_error_evaluation_config().direction_lambda(), "\n",
      tab, tab, tab, tab, tab, "cross-entropy      : ", details.get_error_evaluation_config().cross_entropy_lambda(), "\n",
      tab, tab, tab, tab, "use direction penalty: ", details.get_error_evaluation_config().use_direction_penalty() ? "true" : "false"); 
    ++output_layer_index;
    if (output_layer_index < output_layer_details.size())
    {
      output_layer_details_string += "\n";
    }
  }
  Logger::info(output_layer_details_string);

  Logger::info(tab, "Residual layerjump         : ", _options.residual_layer_jump());
  Logger::info(tab, "Input size                 : ", training_inputs.front().size());
  Logger::info(tab, "Output size                : ", training_outputs.front().size());
  Logger::info(tab, "Optimiser                  : ", optimiser_type_to_string(_options.optimiser_type()));
  Logger::info(tab, "BPTT Enabled               : ", _options.enable_bptt() ? "true" : "false");
  Logger::info(tab, "BPTT Max Ticks             : ", _options.bptt_max_ticks());
  Logger::info(tab, "BPTT Batches are shuffled  : ", _options.shuffle_bptt_batches() ? "true" : "false");

  Logger::info(tab, "Batch size                 : ", _options.batch_size());
  if (_options.final_error_calculation_types().size() == 0)
  {
    Logger::info(tab, "Final error(s) metrics     :\n", tab, tab, "None");
  }
  else
  {
    std::string message = tab;
    message += "Final error(s) metrics     :";
    for (const auto& type : _options.final_error_calculation_types())
    {
      message += Logger::factory("\n", tab, tab, ErrorCalculation::type_to_string(type));
    }
    Logger::info(message);
  }

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
