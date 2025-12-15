#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include <cassert>
#include <functional>
#include <map>
#include <shared_mutex>
#include <vector>

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "adaptivelearningratescheduler.h"
#include "gradientsandoutputs.h" // Include the new header
#include "errorcalculation.h"
#include "hiddenstates.h"
#include "layers.h"
#include "neuron.h"
#include "optimiser.h"
#include "rng.h"
#include "taskqueue.h"
#include "neuralnetworkhelper.h"
#include "neuralnetworkoptions.h"
#include "layergradients.h"
#include "layer.h"

class NeuralNetwork;

class NeuralNetwork
{
public:
  NeuralNetwork(const NeuralNetworkOptions& options);
  NeuralNetwork(const std::vector<unsigned>& topology, const activation::method& hidden_layer_activation, const activation::method& output_layer_activation);

  NeuralNetwork(const NeuralNetwork& src);
  NeuralNetwork& operator=(const NeuralNetwork&);

  virtual ~NeuralNetwork();

  void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs);

  std::vector<std::vector<double>> think(const std::vector<std::vector<double>>& inputs) const;
  std::vector<double> think(const std::vector<double>& inputs) const;

  const std::vector<unsigned>& get_topology() const;
  const std::vector<std::unique_ptr<Layer>>& get_layers() const;
  const activation::method& get_output_activation_method() const;
  const activation::method& get_hidden_activation_method() const;

  NeuralNetworkHelper::NeuralNetworkHelperMetrics calculate_forecast_metric(ErrorCalculation::type error_type) const;
  std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types) const;
  double get_learning_rate() const noexcept;

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
  void calculate_back_propagation(
    std::vector<GradientsAndOutputs>& gradients,
    std::vector<std::vector<double>>::const_iterator outputs_begin,
    size_t batch_size,
    const Layers& layers,
    const std::vector<HiddenStates>& hidden_states);

  void calculate_back_propagation_input_layer(
    std::vector<GradientsAndOutputs>& gradients,
    const Layers& layers);

  void calculate_back_propagation_output_layer(
    std::vector<GradientsAndOutputs>& gradients,
    std::vector<std::vector<double>>::const_iterator outputs_begin,
    size_t batch_size,
    const Layers& layers,
    const std::vector<HiddenStates>& hidden_states);

  void calculate_back_propagation_hidden_layers(
    std::vector<GradientsAndOutputs>& gradients,
    const Layers& layers,
    const std::vector<HiddenStates>& hidden_states);
  void train_single_batch(
    const std::vector<std::vector<double>>::const_iterator inputs_begin, 
    const std::vector<std::vector<double>>::const_iterator outputs_begin,
    const size_t size
  );

  void set_gradients_for_layer(std::vector<GradientsAndOutputs>& source, unsigned layer_number, const std::vector<std::vector<double>>& gradients) const;
  std::vector<std::vector<double>> get_outputs_for_layer(const std::vector<GradientsAndOutputs>& source, unsigned layer_number) const;
  std::vector<std::vector<double>> get_gradients_for_layer(const std::vector<GradientsAndOutputs>& source, unsigned layer_number) const;

  void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& gradients_and_output,
    std::vector<std::vector<double>>::const_iterator inputs_begin,
    size_t batch_size,
    const Layers& layers, 
    std::vector<HiddenStates>& hidden_states,
    bool is_training) const;

  void apply_weight_gradients(Layers& layers, const std::vector<GradientsAndOutputs>& batch_activation_gradients, double learning_rate, unsigned epoch, const std::vector<HiddenStates>& hidden_states, unsigned num_layers_param, std::vector<LayerGradients>& layer_gradients);

  Layer* get_residual_layer(Layers& layers, const GradientsAndOutputs& batch_activation_gradient, std::vector<double>& residual_output_values, unsigned current_layer_index) const;

  std::vector<NeuralNetworkHelper::NeuralNetworkHelperMetrics> calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types, bool final_check) const;

  void recreate_batch_from_indexes(NeuralNetworkHelper& neural_network_helper, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void create_batch_from_indexes(const std::vector<size_t>& shuffled_indexes, const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, std::vector<std::vector<double>>& shuffled_training_inputs, std::vector<std::vector<double>>& shuffled_training_outputs) const;
  void break_indexes(const std::vector<size_t>& indexes, bool data_is_unique, std::vector<size_t>& training_indexes, std::vector<size_t>& checking_indexes, std::vector<size_t>& final_check_indexes) const;
  void create_shuffled_indexes(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const;
  void create_indexes(NeuralNetworkHelper& neural_network_helper, bool data_is_unique) const;

  double calculate_global_clipping_scale(const std::vector<LayerGradients>& layer_gradients) const;

  double calculate_learning_rate(double learning_rate_base, double learning_rate_decay_rate, int epoch, int number_of_epoch, AdaptiveLearningRateScheduler& learning_rate_scheduler) const;
  double calculate_smooth_learning_rate_boost(int epoch, int total_epochs, double base_learning_rate) const;
  double calculate_learning_rate_warmup(int epoch, double completed_percent) const;

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
  std::map<ErrorCalculation::type, double> _saved_errors;

  Rng _rng;
};