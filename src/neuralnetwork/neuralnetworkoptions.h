#pragma once
#include <cassert>
#include <functional>
#include <vector>

#include "libraries/instrumentor.h"

#include "activation.h"
#include "errorcalculation.h"
#include "layer.h"
#include "logger.h"
#include "neuralnetworkhelper.h"
#include "optimiser.h"

class NeuralNetworkHelper;
class NeuralNetworkOptions
{
private:
  NeuralNetworkOptions(const std::vector<unsigned>& topology) :
    _topology(topology),
    _dropout({}),
    _hidden_activation(activation::method::sigmoid),
    _output_activation(activation::method::sigmoid),
    _learning_rate(0.15),
    _number_of_epoch(1000),
    _batch_size(1),
    _data_is_unique(true),
    _progress_callback(nullptr),
    _log_level(Logger::LogLevel::None),
    _number_of_threads(0),
    _learning_rate_decay_rate(0.0),
    _adaptive_learning_rate(false),
    _optimiser_type(OptimiserType::SGD),
    _learning_rate_restart_rate(0),
    _learning_rate_restart_boost(0),
    _residual_layer_jump(-1),
    _clip_threshold(1.0),
    _learning_rate_warmup_start(0.0),
    _learning_rate_warmup_target(0.0),
    _shuffle_training_data(true),
    _recurrent_layers({}),
    _weight_decay(0.0),
    _error_calculation_type(ErrorCalculation::type::mse),
    _enable_bptt(true),
    _bptt_max_ticks(0)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
  }

public:
  NeuralNetworkOptions(const NeuralNetworkOptions& nno) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    *this = nno;
  }

  NeuralNetworkOptions(NeuralNetworkOptions&& nno) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    *this = std::move(nno);
  }

  NeuralNetworkOptions& operator=(const NeuralNetworkOptions& nno) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    if (this != &nno)
    {
      _topology = nno._topology;
      _dropout = nno._dropout;
      _hidden_activation = nno._hidden_activation;
      _output_activation = nno._output_activation;
      _learning_rate = nno._learning_rate;
      _number_of_epoch = nno._number_of_epoch;
      _batch_size = nno._batch_size;
      _data_is_unique = nno._data_is_unique;
      _progress_callback = nno._progress_callback;
      _log_level = nno._log_level;
      _number_of_threads = nno._number_of_threads;
      _learning_rate_decay_rate = nno._learning_rate_decay_rate;
      _adaptive_learning_rate = nno._adaptive_learning_rate;
      _optimiser_type = nno._optimiser_type;
      _learning_rate_restart_rate = nno._learning_rate_restart_rate;
      _learning_rate_restart_boost = nno._learning_rate_restart_boost;
      _residual_layer_jump = nno._residual_layer_jump;
      _clip_threshold = nno._clip_threshold;
      _learning_rate_warmup_start = nno._learning_rate_warmup_start;
      _learning_rate_warmup_target = nno._learning_rate_warmup_target;
      _shuffle_training_data = nno._shuffle_training_data;
      _recurrent_layers = nno._recurrent_layers;
      _weight_decay = nno._weight_decay;
      _error_calculation_type = nno._error_calculation_type;
      _enable_bptt = nno._enable_bptt;
      _bptt_max_ticks = nno._bptt_max_ticks;
    }
    return *this;
  }

  NeuralNetworkOptions& operator=(NeuralNetworkOptions&& nno) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    if (this != &nno)
    {
      _topology = std::move(nno._topology);
      _dropout = std::move(nno._dropout);
      _hidden_activation = nno._hidden_activation;
      _output_activation = nno._output_activation;
      _learning_rate = nno._learning_rate;
      _number_of_epoch = nno._number_of_epoch;
      _batch_size = nno._batch_size;
      _data_is_unique = nno._data_is_unique;
      _progress_callback = nno._progress_callback;
      _log_level = nno._log_level;
      _number_of_threads = nno._number_of_threads;
      _learning_rate_decay_rate = nno._learning_rate_decay_rate;
      _adaptive_learning_rate = nno._adaptive_learning_rate;
      _optimiser_type = nno._optimiser_type;
      _learning_rate_restart_rate = nno._learning_rate_restart_rate;
      _learning_rate_restart_boost = nno._learning_rate_restart_boost;
      _residual_layer_jump = nno._residual_layer_jump;
      _clip_threshold = nno._clip_threshold;
      _learning_rate_warmup_start = nno._learning_rate_warmup_start;
      _learning_rate_warmup_target = nno._learning_rate_warmup_target;
      _shuffle_training_data = nno._shuffle_training_data;
      _recurrent_layers = std::move(nno._recurrent_layers);
      _weight_decay = nno._weight_decay;
      _error_calculation_type = nno._error_calculation_type;
      _enable_bptt = nno._enable_bptt;
      _bptt_max_ticks = nno._bptt_max_ticks;

      nno._log_level = Logger::LogLevel::None;
      nno._number_of_epoch = 0;
      nno._batch_size = 0;
      nno._learning_rate = 0.00;
      nno._data_is_unique = false;
      nno._optimiser_type = OptimiserType::None;
      nno._residual_layer_jump = -1;
      nno._clip_threshold = 1.0;
      nno._learning_rate_warmup_start = 0.0;
      nno._learning_rate_warmup_target = 0.0;
      nno._shuffle_training_data = true;
      nno._weight_decay = 0.0;
      nno._error_calculation_type = ErrorCalculation::type::mse; // Reset moved-from value
      nno._bptt_max_ticks = 0;
    }
    return *this;
  }
  NeuralNetworkOptions& with_hidden_activation_method(const activation::method& activation)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _hidden_activation = activation;
    return *this;
  }
  NeuralNetworkOptions& with_output_activation_method(const activation::method& activation)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _output_activation = activation;
    return *this;
  }
  NeuralNetworkOptions& with_number_of_epoch(int number_of_epoch)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _number_of_epoch = number_of_epoch;
    return *this;
  }
  NeuralNetworkOptions& with_batch_size(int batch_size)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _batch_size = batch_size;
    return *this;
  }
  NeuralNetworkOptions& with_data_is_unique(bool data_is_unique)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    // unique training data means that we cannot have
    // data split for epoch error checking and final error checking.
    _data_is_unique = data_is_unique;
    return *this;
  }

  NeuralNetworkOptions& with_progress_callback(const std::function<bool(NeuralNetworkHelper&)>& progress_callback)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _progress_callback = progress_callback;
    return *this;
  }
  NeuralNetworkOptions& with_log_level(const Logger::LogLevel& log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _log_level = log_level;
    return *this;
  }
  NeuralNetworkOptions& with_number_of_threads(int number_of_threads)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _number_of_threads = number_of_threads <= 0 ? 0 : number_of_threads;
    return *this;
  }
  NeuralNetworkOptions& with_dropout(const std::vector<double>& dropout)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _dropout = dropout;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate(double learning_rate)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _learning_rate = learning_rate;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate_decay_rate(double learning_rate_decay_rate)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _learning_rate_decay_rate = learning_rate_decay_rate;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate_boost_rate(double every_percent, double restart_boost)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _learning_rate_restart_rate = every_percent;
    _learning_rate_restart_boost = restart_boost;
    return *this;
  }
  NeuralNetworkOptions& with_adaptive_learning_rates(bool adaptive_learning_rate)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _adaptive_learning_rate = adaptive_learning_rate;
    return *this;
  }
  NeuralNetworkOptions& with_optimiser_type(OptimiserType optimiser_type)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _optimiser_type = optimiser_type;
    return *this;
  }
  NeuralNetworkOptions& with_residual_layer_jump(int residual_layer_jump)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _residual_layer_jump = residual_layer_jump;
    return *this;
  }
  NeuralNetworkOptions& with_clip_threshold(double clip_threshold)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _clip_threshold = clip_threshold;
    return *this;
  }
  NeuralNetworkOptions& with_shuffle_training_data(bool shuffle_training_data)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _shuffle_training_data = shuffle_training_data;
    return *this;
  }
  NeuralNetworkOptions& with_recurrent_layers(std::vector<unsigned> recurrent_layers)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _recurrent_layers = recurrent_layers;
    return *this;
  }

  NeuralNetworkOptions& with_learning_rate_warmup(double learning_rate_warmup_start, double learning_rate_warmup_target)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _learning_rate_warmup_start = learning_rate_warmup_start;
    _learning_rate_warmup_target = learning_rate_warmup_target;
    return *this;
  }
  NeuralNetworkOptions& with_weight_decay(double weight_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _weight_decay = weight_decay;
    return *this;
  }
  NeuralNetworkOptions& with_error_calculation_type(ErrorCalculation::type error_calculation_type)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _error_calculation_type = error_calculation_type;
    return *this;
  }
  NeuralNetworkOptions& with_enable_bptt(bool enable_bptt)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _enable_bptt = enable_bptt;
    return *this;
  }
  NeuralNetworkOptions& with_bptt_max_ticks(int bptt_max_ticks)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    _bptt_max_ticks = bptt_max_ticks;
    return *this;
  }
  
  NeuralNetworkOptions& build()
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    // set the log level first
    Logger::set_level(log_level());

    if (topology().size() < 2)
    {
      Logger::panic("The topology is not value, you must have at least 2 layers.");
    }
    if (dropout().size() == 0)
    {
      _dropout = std::vector<double>(topology().size() - 2, 0.0);
    }
    if (dropout().size() != topology().size() -2)
    {
      Logger::panic("The dropout size must be exactly the topology size less the input and outpout layers.");
    }
    for (auto& dropout : dropout())
    {
      if(dropout < 0.0 || dropout > 1.0)
      {
        Logger::panic("The dropout rate must be between 0 and 1!");
      }
    }
    if (number_of_threads() > 0 && batch_size() <= 1)
    {
      Logger::warning("Because the batch size is 1, the number of threads is ignored.");
    }
    if (learning_rate_decay_rate() < 0)
    {
      Logger::panic("The learning rate decay rate cannot be negative!");
    }
    if (learning_rate_decay_rate() > 1.0)
    {
      Logger::panic("The learning rate decay rate cannot be more than 1!");
    }
    if (learning_rate_restart_rate() < 0.0 || learning_rate_restart_rate() > 1.0)
    {
      Logger::panic("The learning rate restart rate has to be between 0.0 and 1.0!");
    }
    if (learning_rate_restart_boost() < 0.0|| learning_rate_restart_boost() > 1.0)
    {
      Logger::panic("The learning rate restart boost has to be between 0.0 and 1.0!");
    }
    if(residual_layer_jump() < -1 || residual_layer_jump() == 0)
    {
      _residual_layer_jump = -1;
      Logger::warning("The residual layer jump must be positive or -1, setting it to -1.");
    }
    if (clip_threshold() <= 0.0)
    {
      Logger::panic("A gradient clip threshold smaller or equal to zero does not make sense!");
    }
    if (learning_rate_warmup_start() < 0.0)
    {
      Logger::panic("The learning rate warm up start value cannot be less than zero.");
    }
    if (learning_rate_warmup_start() > learning_rate())
    {
      Logger::panic("The learning rate warm up start value cannot be greater than the target rate.");
    }
    if (learning_rate_warmup_target() < 0.0 || learning_rate_warmup_target() > 1.0)
    {
      Logger::panic("The learning rate warm up target must range between 0.0 and 1.0.");
    }
    if (_recurrent_layers.size() == 0)
    {
      _recurrent_layers.resize(_topology.size(), 0);
    }
    else if (_recurrent_layers.size() == (_topology.size() - 2))
    {
      // add 0 in front and back for input/output
      _recurrent_layers.insert(_recurrent_layers.begin(), 0);
      _recurrent_layers.push_back(0);
    }
    else if (_recurrent_layers.size() == _topology.size())
    {
      if (_recurrent_layers.front() != 0 || _recurrent_layers.back() != 0)
      {
        Logger::panic("The input/output recurrent layer must be zero!");
      }
    }
    else if (_recurrent_layers.size() != _topology.size())
    {
      Logger::panic("The recurrent layer must be the same size as the topology, (or size -2).");
    }
    if (_weight_decay < 0)
    {
      Logger::panic("The weight decay cannot be -ve!");
    }
    return *this;
  }

  static NeuralNetworkOptions create(const std::vector<Layer>& layers)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    auto topology = std::vector<unsigned>();
    topology.reserve(layers.size());
    for (const auto& layer : layers)
    {
      // remove the bias Neuron.
      topology.emplace_back(layer.get_number_neurons() - 1);
    }
    return create(topology);
  }

  static NeuralNetworkOptions create(const std::vector<unsigned>& topology)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions");
    auto clip_threshold = 1.0;
    std::vector<double> dropout = {};
    if(topology.size() > 2)
    {
      dropout.resize(topology.size() - 2, 0.0);
      if (topology.size() > 8)  //  6 hidden
      {
        clip_threshold = 2.0;
      }
      else if (topology.size() > 2) //  2 hidden
      {
        clip_threshold = 0.5;
      }
    }
    else
    {
      dropout = {};
    }
    return NeuralNetworkOptions(topology)
      .with_dropout(dropout)
      .with_learning_rate(0.1)
      .with_learning_rate_warmup(0.0, 0.0)
      .with_hidden_activation_method(activation::method::sigmoid)
      .with_output_activation_method(activation::method::sigmoid)
      .with_number_of_epoch(1000)
      .with_batch_size(1)
      .with_data_is_unique(true)
      .with_progress_callback(nullptr)
      .with_learning_rate_decay_rate(0.0)
      .with_adaptive_learning_rates(false)
      .with_optimiser_type(OptimiserType::SGD)
      .with_learning_rate_boost_rate(0.0, 0.0)
      .with_residual_layer_jump(-1)
      .with_clip_threshold(clip_threshold)
      .with_shuffle_training_data(true)
      .with_weight_decay(0.0)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(0);
  }

  inline const std::vector<unsigned>& topology() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _topology; }
  inline const std::vector<double>& dropout() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _dropout; }
  inline const activation::method& hidden_activation_method() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _hidden_activation; }
  inline const activation::method& output_activation_method() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _output_activation; }
  inline double learning_rate() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _learning_rate; }
  inline int number_of_epoch() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _number_of_epoch; }
  inline int batch_size() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _batch_size; }
  inline bool data_is_unique() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _data_is_unique; }
  inline const std::function<bool(NeuralNetworkHelper&)>& progress_callback() const { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _progress_callback; }
  inline Logger::LogLevel log_level() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _log_level; }
  inline int number_of_threads() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _number_of_threads; }
  inline double learning_rate_decay_rate() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _learning_rate_decay_rate; }
  inline bool adaptive_learning_rate() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _adaptive_learning_rate; }
  inline OptimiserType optimiser_type() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _optimiser_type; }
  inline double learning_rate_restart_rate() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _learning_rate_restart_rate; }
  inline double learning_rate_restart_boost() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _learning_rate_restart_boost; }
  inline int residual_layer_jump() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _residual_layer_jump; }
  inline double clip_threshold() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _clip_threshold; }
  inline double learning_rate_warmup_start() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _learning_rate_warmup_start; }
  inline double learning_rate_warmup_target() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _learning_rate_warmup_target; }
  inline bool shuffle_training_data() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _shuffle_training_data; }
  inline const std::vector<unsigned>& recurrent_layers() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _recurrent_layers; }
  inline double weight_decay() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _weight_decay; }
  inline ErrorCalculation::type error_calculation_type() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _error_calculation_type; }
  inline bool enable_bptt() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _enable_bptt; }
  inline int bptt_max_ticks() const noexcept { MYODDWEB_PROFILE_FUNCTION("NeuralNetworkOptions"); return _bptt_max_ticks; }

private:
  std::vector<unsigned> _topology;
  std::vector<double> _dropout;
  activation::method _hidden_activation;
  activation::method _output_activation;
  double _learning_rate;
  int _number_of_epoch;
  int _batch_size;
  bool _data_is_unique;
  std::function<bool(NeuralNetworkHelper&)> _progress_callback;
  Logger::LogLevel _log_level;
  int _number_of_threads;
  double _learning_rate_decay_rate;
  bool _adaptive_learning_rate;
  OptimiserType _optimiser_type;
  double _learning_rate_restart_rate;
  double _learning_rate_restart_boost;
  int _residual_layer_jump;
  double _clip_threshold;
  double _learning_rate_warmup_start;  //  initial learning rate for warmup
  double _learning_rate_warmup_target; //  the percentage of the epoch to reach during warmup
  bool _shuffle_training_data;
  std::vector<unsigned> _recurrent_layers;
  double _weight_decay;
  ErrorCalculation::type _error_calculation_type;
  bool _enable_bptt;
  int _bptt_max_ticks;
};
      