#pragma once
#include "../helpers/errorcalculation.h"
#include "../common/hiddenstate.h"
#include "../common/aligned_allocator.h"
#include "layer.h"

#include <vector>
#include <span>


namespace myoddweb::nn
{
class FFLayer : public Layer
{
public:
  FFLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer,
    unsigned num_neurons_in_this_layer,
    double weight_decay,
    const Role layer_role,
    const activation& activation_method,
    const OptimiserType& optimiser_type,
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads,
    bool has_bias,
    double momentum,
    std::optional<uint32_t> seed);

  FFLayer(unsigned layer_index,
    unsigned num_neurons_in_previous_layer,
    unsigned num_neurons_in_this_layer,
    const std::vector<double>& weight_decays,
    const Role layer_role,
    const activation& activation_method,
    const OptimiserType& optimiser_type,
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads,
    bool has_bias,
    double momentum,
    std::optional<uint32_t> seed);

  FFLayer(
    unsigned layer_index,
    const Role layer_role,
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
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum
  ) noexcept;

  FFLayer(const FFLayer& src) noexcept;
  FFLayer(FFLayer&& src) noexcept;
  FFLayer& operator=(const FFLayer& src) noexcept;
  FFLayer& operator=(FFLayer&& src) noexcept;
  virtual ~FFLayer();

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return Architecture::FF;
  }

public:
  using AlignedVector = myoddweb::nn::AlignedVector<double, 32>;

  // Multiplier = 1: Standard pre-activation sum (z)
  static constexpr unsigned Multiplier = 1;
  static constexpr unsigned GateCount = 1;

  [[nodiscard]] unsigned get_pre_activation_multiplier() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return Multiplier;
  }

public:
  void calculate_forward_feed(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& previous_layer,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    bool is_training) const override;

  void calculate_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    std::vector<std::vector<double>>::const_iterator target_outputs_begin,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size) const override;

  void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    int bptt_max_ticks) const override;

  void calculate_hidden_gradients_from_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<std::vector<double>>& batch_output_gradients,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    int bptt_max_ticks) const override;

  Layer* clone() const override;

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override;

  virtual double get_gradient_norm_sq() const override;

  virtual void accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count) override;
  virtual void update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha) override;

  virtual void apply_stored_gradients(double learning_rate, double clipping_scale) override;

  void cache_recurrent_weights() override;

  [[nodiscard]] bool is_ff_layer() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return true;
  }

  [[nodiscard]] inline const AlignedVector& get_w_values_T() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    return _w_values_T;
  }

  void set_w_values(const std::vector<double>& v) override
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    Layer::set_w_values(v);
    cache_recurrent_weights();
  }

protected:
  FFLayer(unsigned layer_index,
    const std::vector<double>& weight_decays,
    const Role layer_role,
    const layer_activation_helper& lah,
    const OptimiserType& optimiser_type,
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads,
    bool has_bias,
    double momentum,
    std::optional<uint32_t> seed);

  void run_gemm(
    size_t b_start,
    size_t b_end,
    size_t N_prev,
    size_t N_this,
    const double* batch_inputs,
    double* batch_pre_activation_sums) const;

  inline void run_gemm(
    size_t b_start,
    size_t b_end,
    size_t N_prev,
    size_t N_this,
    const std::vector<double>& batch_inputs_buffer,
    std::vector<double>& batch_pre_activation_sums_buffer) const
  {
    run_gemm(b_start, b_end, N_prev, N_this, batch_inputs_buffer.data(), batch_pre_activation_sums_buffer.data());
  }

  virtual void run_post_gemm(
    size_t start,
    size_t end,
    size_t num_time_steps,
    size_t N_this,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    const double* batch_inputs,
    double* batch_pre_activation_sums,
    bool is_training) const;

  virtual void run_post_gemm(
    size_t start,
    size_t end,
    size_t num_time_steps,
    size_t N_this,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    const std::vector<double>& batch_inputs_buffer,
    std::vector<double>& batch_pre_activation_sums_buffer,
    bool is_training) const
  {
    run_post_gemm(start, end, num_time_steps, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, batch_inputs_buffer.data(), batch_pre_activation_sums_buffer.data(), is_training);
  }

private:
  void run_gemm_backward(
    size_t b_start,
    size_t b_end,
    size_t N_next,
    size_t N_this,
    const double* W_next,
    const double* flattened_next_grads,
    double* flattened_this_grads) const;

  void run_gemm_backward_fast(
    size_t b_start,
    size_t b_end,
    size_t N_next,
    size_t N_this,
    const double* W_next_T,
    const double* flattened_next_grads,
    double* flattened_this_grads) const;

  void run_post_gemm_backward(
    size_t start,
    size_t end,
    size_t N_this,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& batch_hidden_states,
    const double* flattened_this_grads) const;

  void calculate_and_store_gradients_chunk(
    size_t start,
    size_t end,
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    unsigned prev_layer_index,
    unsigned this_layer_index,
    unsigned num_inputs,
    unsigned num_outputs,
    size_t num_time_steps,
    std::span<double> local_w_grads,
    std::span<double> local_b_grads) const;

  struct thread_ff_grad_accumulators
  {
    AlignedVector w_grads;
    AlignedVector b_grads;
  };

  mutable std::vector<thread_ff_grad_accumulators> _thread_grad_accumulators;
  AlignedVector _w_values_T;
};
} // namespace myoddweb::nn
