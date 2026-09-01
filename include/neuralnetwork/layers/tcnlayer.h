#pragma once
#include "layer.h"

#include <vector>


namespace myoddweb::nn
{
// Dilated causal 1D convolution ("Temporal Convolutional Network" block):
// consumes the full T-timestep window of the preceding layer (via
// Layer::calculate_forward_feed's previous_layer.get_rnn_outputs) and, for
// every output timestep t, gathers the kernel_size dilated input taps
// {X[t - j*dilation] : j = 0..kernel_size-1} (zero-padded where
// t - j*dilation < 0) into one flat [kernel_size * in_channels] vector, then
// applies a single dense affine map + activation - structurally identical to
// a per-timestep FF layer fed a gathered (non-contiguous) input, which is why
// this layer reuses the inherited base-class _w_*/_b_* weight family (sized
// [kernel_size * in_channels] x [layer_size]) rather than declaring a new one.
// Always strictly causal - never looks ahead - with no configuration flag to
// disable this, since this project has a documented label-leakage history.
class TcnLayer final : public Layer
{
public:
  TcnLayer(
    unsigned layer_index,
    unsigned num_neurons_in_previous_layer,
    unsigned layer_size,
    unsigned kernel_size,
    unsigned dilation,
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

  TcnLayer(
    unsigned layer_index,
    const Role layer_role,
    const OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned kernel_size,
    unsigned dilation,
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
    double momentum) noexcept;

  TcnLayer(const TcnLayer& src) noexcept;
  TcnLayer(TcnLayer&& src) noexcept;
  TcnLayer& operator=(const TcnLayer& src) noexcept;
  TcnLayer& operator=(TcnLayer&& src) noexcept;
  virtual ~TcnLayer();

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("TcnLayer");
    return Architecture::Tcn;
  }

  [[nodiscard]] inline unsigned get_kernel_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TcnLayer");
    return _kernel_size;
  }

  [[nodiscard]] inline unsigned get_dilation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TcnLayer");
    return _dilation;
  }

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

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override;

  double get_gradient_norm_sq() const override;

  void accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count) override;
  void update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha) override;

  void apply_stored_gradients(double learning_rate, double clipping_scale) override;

  Layer* clone() const override;

private:
  void process_forward_range(
    size_t b_start,
    size_t b_end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    unsigned prev_layer_index,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    bool is_training) const;

  // Shared tail of both calculate_hidden_gradients entry points: given
  // `raw_delta_all[b]` (T*N_out per batch item, gradient w.r.t. this layer's
  // own ACTIVATED output, before undoing this layer's own activation
  // derivative/dropout), undoes them to get this layer's own per-timestep
  // pre-activation delta (deposited via set_gradients [last timestep only]
  // and set_rnn_gate_gradients [full T*N_out sequence] for this layer's own
  // later calculate_and_store_gradients call), then backprops that delta
  // through this layer's own weight matrix and scatter-adds the result across
  // the kernel_size dilated source timesteps to produce the T*in_channels
  // gradient the preceding layer consumes directly (deposited via
  // set_rnn_gradients - the "direct gradient injection" mechanism also used
  // by AttentionPoolLayer).
  void finish_hidden_gradients_range(
    size_t b_start,
    size_t b_end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& batch_hidden_states,
    const std::vector<std::vector<double>>& raw_delta_all) const;

  void accumulate_gradients_range(
    size_t b_start,
    size_t b_end,
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    unsigned prev_layer_index,
    std::vector<double>& local_w_grads,
    std::vector<double>& local_b_grads,
    size_t& local_contributing) const;

  struct thread_tcn_grad_accumulators
  {
    std::vector<double> w_grads;
    std::vector<double> b_grads;
    size_t contributing = 0;
  };

  struct tcn_forward_task
  {
    const TcnLayer* layer;
    size_t start;
    size_t end;
    std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
    unsigned prev_layer_index;
    const std::vector<std::vector<double>>* batch_residual_output_values;
    std::vector<HiddenStates>* batch_hidden_states;
    bool is_training;

    void operator()() const;
  };

  struct tcn_finish_hidden_gradients_task
  {
    const TcnLayer* layer;
    size_t start;
    size_t end;
    std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
    const std::vector<HiddenStates>* batch_hidden_states;
    const std::vector<std::vector<double>>* raw_delta_all;

    void operator()() const;
  };

  struct tcn_grad_calc_task
  {
    const TcnLayer* layer;
    size_t start;
    size_t end;
    const std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
    unsigned prev_layer_index;
    std::vector<double>* local_w_grads;
    std::vector<double>* local_b_grads;
    size_t* local_contributing;

    void operator()() const;
  };

  unsigned _kernel_size;
  unsigned _dilation;
};

} // namespace myoddweb::nn
