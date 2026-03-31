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

class FFLayer : public Layer
{
protected:
  friend class Layers;

public:
  FFLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer, 
    unsigned num_neurons_in_this_layer, 
    double weight_decay,
    LayerType layer_type, 
    const activation& activation_method, 
    const OptimiserType& optimiser_type, 
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads);

  FFLayer(
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
    const ResidualProjector* residual_projector,
    int number_of_threads
  ) noexcept;

  FFLayer(const FFLayer& src) noexcept;
  FFLayer(FFLayer&& src) noexcept;
  FFLayer& operator=(const FFLayer& src) noexcept;
  FFLayer& operator=(FFLayer&& src) noexcept;
  virtual ~FFLayer();

public:
  void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer &previous_layer,
    const std::vector<std::vector<double>> &batch_residual_output_values,
    std::vector<HiddenStates> &batch_hidden_states,
    bool is_training) const override;

  void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates> &batch_hidden_states) const override;

  void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer &next_layer,
    const std::vector<std::vector<double>> &batch_next_grad_matrix,
    const std::vector<HiddenStates> &batch_hidden_states,
    int bptt_max_ticks) const override;

  bool has_bias() const noexcept override;
  
  Layer* clone() const override;

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    int bptt_max_ticks) override;

  double get_gradient_norm_sq() const override;

  void apply_stored_gradients(double learning_rate, double clipping_scale) override;

private:
  // Hoisted buffers for performance
  mutable std::vector<double> _batch_inputs_buffer;
  mutable std::vector<double> _batch_pre_activation_sums_buffer;
  mutable std::vector<double> _batch_grads_buffer;
  mutable std::vector<double> _flattened_next_grads_buffer;
  mutable std::vector<double> _flattened_this_grads_buffer;
};