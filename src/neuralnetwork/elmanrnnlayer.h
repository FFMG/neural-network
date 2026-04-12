#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include "errorcalculation.h"
#include "hiddenstate.h"
#include "layer.h"

#include <vector>

class ElmanRNNLayer final : public Layer
{
protected:
  friend class Layers;

public:
  ElmanRNNLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    double weight_decay,
    LayerType layer_type, 
    const activation& activation_method, 
    const OptimiserType& optimiser_type, 
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads,
    bool has_bias);

  ElmanRNNLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer,
    unsigned num_neurons_in_this_layer,
    const std::vector<double>& weight_decays,
    LayerType layer_type,
    const activation& activation_method,
    const OptimiserType& optimiser_type,
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads,
    bool has_bias);

  ElmanRNNLayer(
    unsigned layer_index,
    const LayerType layer_type,
    const activation& activation_method,
    const OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned number_input_neurons,
    unsigned number_output_neurons,
    const std::vector<Neuron>& neurons,
    const std::vector<double>& w_values,
    const std::vector<double>& w_grads,
    const std::vector<double>& w_velocities,
    const std::vector<double>& w_m1,
    const std::vector<double>& w_m2,
    const std::vector<long long>& w_timesteps,
    const std::vector<double>& w_decays,
    const std::vector<double>& b_values,
    const std::vector<double>& b_grads,
    const std::vector<double>& b_velocities,
    const std::vector<double>& b_m1,
    const std::vector<double>& b_m2,
    const std::vector<long long>& b_timesteps,
    const std::vector<double>& b_decays,
    const std::vector<double>& rw_values,
    const std::vector<double>& rw_grads,
    const std::vector<double>& rw_velocities,
    const std::vector<double>& rw_m1,
    const std::vector<double>& rw_m2,
    const std::vector<long long>& rw_timesteps,
    const std::vector<double>& rw_decays,
    const ResidualProjector* residual_projector,
    int number_of_threads
  ) noexcept;

  ElmanRNNLayer(const ElmanRNNLayer& src) noexcept;
  ElmanRNNLayer(ElmanRNNLayer&& src) noexcept;
  ElmanRNNLayer& operator=(const ElmanRNNLayer& src) noexcept;
  ElmanRNNLayer& operator=(ElmanRNNLayer&& src) noexcept;
  virtual ~ElmanRNNLayer();

public:
  bool use_bptt() const noexcept override {
    return true;
  }

  void calculate_forward_feed(
      std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
      const Layer &previous_layer,
      const std::vector<std::vector<double>> &batch_residual_output_values,
      std::vector<HiddenStates> &batch_hidden_states,
      size_t batch_size,
      bool is_training) const override;

  void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates> &batch_hidden_states,
    size_t batch_size) const  override;

  void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    int bptt_max_ticks) const override;

  double get_recurrent_weight_value(unsigned from_neuron, unsigned to_neuron) const;

  Layer* clone() const override;

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override;

  double get_gradient_norm_sq() const override;

  void apply_stored_gradients(double learning_rate, double clipping_scale) override;

  inline const std::vector<double>& get_rw_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return _rw_values;
  }
  inline const std::vector<double>& get_rw_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return _rw_grads;
  }
  inline const std::vector<double>& get_rw_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return _rw_velocities;
  }
  inline const std::vector<double>& get_rw_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return _rw_m1;
  }
  inline const std::vector<double>& get_rw_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return _rw_m2;
  }
  inline const std::vector<long long>& get_rw_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return _rw_timesteps;
  }
  inline const std::vector<double>& get_rw_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return _rw_decays;
  }

private:
  // Hoisted buffers for performance
  mutable std::vector<double> _flattened_batch_inputs_buffer;
  mutable std::vector<double> _batch_output_sequences_buffer;
  mutable std::vector<double> _flattened_next_grads_buffer;
  mutable std::vector<double> _flattened_inputs_buffer;
  mutable std::vector<double> _flattened_rnn_grads_buffer;
  mutable std::vector<double> _flattened_prev_h_buffer;

  void initialize_recurrent_weights(double weight_decay);
  
  // SoA for recurrent weights
  std::vector<double> _rw_values;
  std::vector<double> _rw_grads;
  std::vector<double> _rw_velocities;
  std::vector<double> _rw_m1;
  std::vector<double> _rw_m2;
  std::vector<long long> _rw_timesteps;
  std::vector<double> _rw_decays;
};
