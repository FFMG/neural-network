#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include <functional>
#include <map>
#include <shared_mutex>
#include <vector>

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "adaptivelearningratescheduler.h"
#include "errorcalculation.h"
#include "gradientsandoutputs.h"
#include "hiddenstates.h"
#include "layers.h"
#include "neuralnetworkhelper.h"
#include "neuralnetworkhelpermetrics.h"
#include "neuralnetworkoptions.h"
#include "rng.h"
#include "taskqueue.h"

class NeuralNetwork
{
public:
  NeuralNetwork(const NeuralNetworkOptions& options);
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation);
  NeuralNetwork(const Layers& layers, const NeuralNetworkOptions& options, const std::vector<std::map<ErrorCalculation::type, double>>& errors);

  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&);

  virtual ~NeuralNetwork();

  void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs);
  void calibrate_temperature(const std::vector<std::vector<double>>& validation_inputs, const std::vector<std::vector<double>>& validation_outputs);
  std::vector<std::vector<double>> think(const std::vector<std::vector<double>>& inputs) const;
  std::vector<double> think(const std::vector<double>& inputs) const;

  const std::vector<unsigned>& get_topology() const;
  void scale_temperature(unsigned output_layer_index, double factor) noexcept;
  [[nodiscard]] const Layers& get_layers() const;
  [[nodiscard]] const Layer& get_layer(unsigned index) const;

  // Output layer 0 only (common use case)
  NeuralNetworkHelperMetrics calculate_forecast_metric(ErrorCalculation::type error_type) const;
  std::vector<NeuralNetworkHelperMetrics> calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types, bool final_check = false) const;

  // Multiple Output layers
  std::vector<NeuralNetworkHelperMetrics> calculate_forecast_metric_all_layers(ErrorCalculation::type error_type) const;
  std::vector<std::vector<NeuralNetworkHelperMetrics>> calculate_forecast_metrics_all_layers(const std::vector<ErrorCalculation::type>& error_types, bool final_check = false) const;

  double get_learning_rate() const noexcept;
  double get_temperature() const noexcept;
  double get_temperature(unsigned output_layer_index) const noexcept;
  double get_percent_complete() const noexcept;
  bool has_training_data() const;

  inline NeuralNetworkOptions& options() noexcept { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    return _options;
  }
  inline const NeuralNetworkOptions& options() const noexcept { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetwork");
    return _options;
  }

private:
  void train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t batch_size
  );

  void calculate_forward_feed_for_forecast_metrics(
    std::vector<GradientsAndOutputs>& gradients_and_output,
    const std::vector<std::vector<double>>& all_inputs,
    const std::vector<size_t>& indices,
    const Layers& layers,
    std::vector<HiddenStates>& hidden_states,
    bool is_training) const;

  void create_bptt_batches(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, std::vector<std::vector<std::vector<double>>>& bptt_inputs, std::vector<std::vector<std::vector<double>>>& bptt_outputs) const;
  void recreate_batch_from_indexes(NeuralNetworkHelper& neural_network_helper, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void create_batch_from_indexes(const std::vector<size_t>& indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& training_inputs_data, std::vector<std::vector<double>>& training_outputs_data) const;
  void break_indexes(const std::vector<size_t>& indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const;
  void create_shuffled_indexes_in_lock(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const;
  void create_indexes_in_lock(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const;

  double calculate_learning_rate(double learning_rate_base, double learning_rate_decay_rate, int epoch, int number_of_epoch, AdaptiveLearningRateScheduler& learning_rate_scheduler) const;
  double calculate_smooth_learning_rate_boost(int epoch, int total_epochs, double base_learning_rate) const;
  double calculate_learning_rate_warmup(int epoch, double completed_percent) const;

  void recreate_neural_network_helper(int number_of_epoch, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs);

  bool CallCallback(const std::function<bool(NeuralNetworkHelper&)>& callback, SingleTaskQueue<bool>* callback_task) const;

  void log_training_info(
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs) const;

  std::vector<size_t> get_shuffled_indexes(size_t raw_size) const;

  mutable std::shared_mutex _mutex;

  double _learning_rate;
  Layers _layers;
  NeuralNetworkOptions _options;
  NeuralNetworkHelper* _neural_network_helper;
  std::vector<std::map<ErrorCalculation::type, double>> _saved_errors;
  
  Rng _rng;

  mutable SingleTaskQueue<std::vector<NeuralNetworkHelperMetrics>> _adaptive_lr_task;
  mutable std::vector<NeuralNetworkHelperMetrics> _last_metrics;
  mutable std::vector<GradientsAndOutputs> _gradients_pool;
  mutable std::vector<HiddenStates> _hidden_states_pool;
};