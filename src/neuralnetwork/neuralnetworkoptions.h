#pragma once
#include <cassert>
#include <functional>
#include <vector>

#include "activation.h"
#include "layer.h"
#include "logger.h"
#include "optimiser.h"
#include "neuralnetworkhelper.h"

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
    _logger(Logger::LogLevel::None),
    _number_of_threads(0),
    _learning_rate_decay_rate(0.0),
    _adaptive_learning_rate(false),
    _optimiser_type(OptimiserType::SGD),
    _learning_rate_restart_rate(1),
    _learning_rate_restart_boost(1),
    _residual_layer_jump(-1)
  {
  }

public:
  enum class ErrorCalculation
  {
    none,
    huber_loss,
    mae,
    mse,
    rmse,
    mape,
    smape,
  };

  NeuralNetworkOptions(const NeuralNetworkOptions& nno) noexcept
  {
    *this = nno;
  }

  NeuralNetworkOptions(NeuralNetworkOptions&& nno) noexcept
  {
    *this = std::move(nno);
  }

  NeuralNetworkOptions& operator=(const NeuralNetworkOptions& nno) noexcept
  {
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
      _logger = nno._logger;
      _number_of_threads = nno._number_of_threads;
      _learning_rate_decay_rate = nno._learning_rate_decay_rate;
      _adaptive_learning_rate = nno._adaptive_learning_rate;
      _optimiser_type = nno._optimiser_type;
      _learning_rate_restart_rate = nno._learning_rate_restart_rate;
      _learning_rate_restart_boost = nno._learning_rate_restart_boost;
      _residual_layer_jump = nno._residual_layer_jump;
    }
    return *this;
  }

  NeuralNetworkOptions& operator=(NeuralNetworkOptions&& nno) noexcept
  {
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
      _logger = nno._logger;
      _number_of_threads = nno._number_of_threads;
      _learning_rate_decay_rate = nno._learning_rate_decay_rate;
      _adaptive_learning_rate = nno._adaptive_learning_rate;
      _optimiser_type = nno._optimiser_type;
      _learning_rate_restart_rate = nno._learning_rate_restart_rate;
      _learning_rate_restart_boost = nno._learning_rate_restart_boost;
      _residual_layer_jump = nno._residual_layer_jump;

      nno._number_of_epoch = 0;
      nno._batch_size = 0;
      nno._learning_rate = 0.00;
      nno._data_is_unique = false;
      nno._optimiser_type = OptimiserType::None;
      nno._residual_layer_jump = -1;
    }
    return *this;
  }
  NeuralNetworkOptions& with_hidden_activation_method(const activation::method& activation)
  {
    _hidden_activation = activation;
    return *this;
  }
  NeuralNetworkOptions& with_output_activation_method(const activation::method& activation)
  {
    _output_activation = activation;
    return *this;
  }
  NeuralNetworkOptions& with_number_of_epoch(int number_of_epoch)
  {
    _number_of_epoch = number_of_epoch;
    return *this;
  }
  NeuralNetworkOptions& with_batch_size(int batch_size)
  {
    _batch_size = batch_size;
    return *this;
  }
  NeuralNetworkOptions& with_data_is_unique(bool data_is_unique)
  {
    // unique training data means that we cannot have
    // data split for epoch error checking and final error checking.
    _data_is_unique = data_is_unique;
    return *this;
  }

  NeuralNetworkOptions& with_progress_callback(const std::function<bool(NeuralNetworkHelper&)>& progress_callback)
  {
    _progress_callback = progress_callback;
    return *this;
  }
  NeuralNetworkOptions& with_logger(const Logger& logger)
  {
    _logger = logger;
    return *this;
  }
  NeuralNetworkOptions& with_number_of_threads(int number_of_threads)
  {
    _number_of_threads = number_of_threads <= 0 ? 0 : number_of_threads;
    return *this;
  }
  NeuralNetworkOptions& with_dropout(const std::vector<double>& dropout)
  {
    _dropout = dropout;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate(double learning_rate)
  {
    _learning_rate = learning_rate;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate_decay_rate(double learning_rate_decay_rate)
  {
    _learning_rate_decay_rate = learning_rate_decay_rate;
    return *this;
  }
  NeuralNetworkOptions& with_learning_rate_boost_rate(double every_percent, double restart_boost)
  {
    _learning_rate_restart_rate = every_percent;
    _learning_rate_restart_boost = restart_boost;
    return *this;
  }
  NeuralNetworkOptions& with_adaptive_learning_rates(bool adaptive_learning_rate)
  {
    _adaptive_learning_rate = adaptive_learning_rate;
    return *this;
  }
  NeuralNetworkOptions& with_optimiser_type(OptimiserType optimiser_type)
  {
    _optimiser_type = optimiser_type;
    return *this;
  }

  NeuralNetworkOptions& with_residual_layer_jump(int residual_layer_jump)
  {
    _residual_layer_jump = residual_layer_jump;
    return *this;
  }

  NeuralNetworkOptions& build()
  {
    if (topology().size() < 2)
    {
      logger().log_error("The topology is not value, you must have at least 2 layers.");
      throw std::invalid_argument("The topology is not value, you must have at least 2 layers.");
    }
    if (dropout().size() == 0)
    {
      _dropout = std::vector<double>(topology().size() - 2, 0.0);
    }
    if (dropout().size() != topology().size() -2)
    {
      logger().log_error("The dropout size must be exactly the topology size less the input and outpout layers.");
      throw std::invalid_argument("The dropout size must be exactly the topology size less the input and outpout layers.");
    }
    for (auto& dropout : dropout())
    {
      if(dropout < 0.0 || dropout > 1.0)
      {
        logger().log_error("The dropout rate must be between 0 and 1!");
        throw std::invalid_argument("The dropout rate must be between 0 and 1!");
      }
    }
    if (number_of_threads() > 0 && batch_size() <= 1)
    {
      logger().log_warning("Because the batch size is 1, the number of threads is ignored.");
    }
    if (learning_rate_decay_rate() < 0)
    {
      logger().log_error("The learning rate decay rate cannot be negative!");
      throw std::invalid_argument("The learning rate decay rate cannot be negative!");
    }
    if (learning_rate_decay_rate() >= 1.0)
    {
      logger().log_error("The learning rate decay rate cannot be more than 1!");
      throw std::invalid_argument("The learning rate decay rate cannot be more than 1!");
    }
    if (learning_rate_restart_rate() <= 0.0 || learning_rate_restart_rate() > 100)
    {
      logger().log_error("The learning rate has to be between 0% and 100%!");
      throw std::invalid_argument("The learning rate has to be between 0% and 100%!");
    }
    if (learning_rate_restart_boost() < 1.0)
    {
      logger().log_error("The learning rate restart boost cannot be less than 1!");
      throw std::invalid_argument("The learning rate restart boost cannot be less than 1!");
    }
    if(residual_layer_jump() < -1 || residual_layer_jump() == 0)
    {
      logger().log_warning("The residual_layer_jump must be positive or -1");
    }
    return *this;
  }

  static NeuralNetworkOptions create(const std::vector<Layer>& layers)
  {
    auto topology = std::vector<unsigned>();
    topology.reserve(layers.size());
    for (const auto& layer : layers)
    {
      // remove the bias Neuron.
      topology.emplace_back(layer.number_neurons() - 1);
    }
    return create(topology);
  }

  static NeuralNetworkOptions create(const std::vector<unsigned>& topology)
  {
    std::vector<double> dropout = {};
    if(topology.size() > 2)
    {
      dropout.resize(topology.size() - 2, 0.0);
    }
    else
    {
      dropout = {};
    }
    return NeuralNetworkOptions(topology)
      .with_dropout(dropout)
      .with_learning_rate(0.1)
      .with_hidden_activation_method(activation::method::sigmoid)
      .with_output_activation_method(activation::method::sigmoid)
      .with_number_of_epoch(1000)
      .with_batch_size(1)
      .with_data_is_unique(true)
      .with_progress_callback(nullptr)
      .with_learning_rate_decay_rate(0.0)
      .with_adaptive_learning_rates(false)
      .with_optimiser_type(OptimiserType::SGD)
      .with_learning_rate_boost_rate(1.0, 1.0)
      .with_residual_layer_jump(-1);
  }

  inline const std::vector<unsigned>& topology() const { return _topology; }
  inline const std::vector<double>& dropout() const { return _dropout; }
  inline const activation::method& hidden_activation_method() const { return _hidden_activation; }
  inline const activation::method& output_activation_method() const { return _output_activation; }
  inline double learning_rate() const { return _learning_rate; }
  inline int number_of_epoch() const { return _number_of_epoch; }
  inline int batch_size() const { return _batch_size; }
  inline bool data_is_unique() const { return _data_is_unique; }
  inline const std::function<bool(NeuralNetworkHelper&)>& progress_callback() const { return _progress_callback; }
  inline const Logger& logger() const { return _logger; }
  inline int number_of_threads() const { return _number_of_threads; }
  inline double learning_rate_decay_rate() const { return _learning_rate_decay_rate; }
  inline bool adaptive_learning_rate() const { return _adaptive_learning_rate; }
  inline OptimiserType optimiser_type() const { return _optimiser_type; }
  inline double learning_rate_restart_rate() const { return _learning_rate_restart_rate; }
  inline double learning_rate_restart_boost() const { return _learning_rate_restart_boost; }
  inline int residual_layer_jump() const { return _residual_layer_jump; }

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
  Logger _logger;
  int _number_of_threads;
  double _learning_rate_decay_rate;
  bool _adaptive_learning_rate;
  OptimiserType _optimiser_type;
  double _learning_rate_restart_rate;
  double _learning_rate_restart_boost;
  int _residual_layer_jump;
};