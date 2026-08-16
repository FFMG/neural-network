#pragma once
#include "layer.h"

#include <vector>


namespace myoddweb::nn
{
// Additive (Bahdanau-style) attention pooling over the BPTT window of a
// preceding Gru/Lstm hidden layer: consumes the full T-timestep hidden-state
// sequence (via Layer::calculate_forward_feed's previous_layer.get_rnn_outputs)
// and produces a single pooled context vector, letting the network learn
// which past timesteps matter most instead of only ever seeing the last one.
// Output size always equals input size (pooling never changes dimensionality).
class AttentionPoolLayer final : public Layer
{
public:
  AttentionPoolLayer(
    unsigned layer_index,
    unsigned num_neurons_in_previous_layer,
    unsigned attention_hidden_size,
    double weight_decay,
    const Role layer_role,
    const activation& activation_method,
    const OptimiserType& optimiser_type,
    double dropout_rate,
    int number_of_threads,
    bool has_bias,
    double momentum,
    std::optional<uint32_t> seed);

  AttentionPoolLayer(
    unsigned layer_index,
    const Role layer_role,
    const OptimiserType optimiser_type,
    unsigned num_neurons,
    unsigned attention_hidden_size,
    const std::vector<Neuron>& neurons,
    const std::vector<double>& b_values,
    const std::vector<double>& b_grads,
    const std::vector<double>& b_velocities,
    const std::vector<double>& b_m1,
    const std::vector<double>& b_m2,
    const std::vector<long long>& b_timesteps,
    const std::vector<double>& b_decays,
    const std::vector<double>& wa_values,
    const std::vector<double>& wa_grads,
    const std::vector<double>& wa_velocities,
    const std::vector<double>& wa_m1,
    const std::vector<double>& wa_m2,
    const std::vector<long long>& wa_timesteps,
    const std::vector<double>& wa_decays,
    const std::vector<double>& ba_values,
    const std::vector<double>& ba_grads,
    const std::vector<double>& ba_velocities,
    const std::vector<double>& ba_m1,
    const std::vector<double>& ba_m2,
    const std::vector<long long>& ba_timesteps,
    const std::vector<double>& ba_decays,
    const std::vector<double>& v_values,
    const std::vector<double>& v_grads,
    const std::vector<double>& v_velocities,
    const std::vector<double>& v_m1,
    const std::vector<double>& v_m2,
    const std::vector<long long>& v_timesteps,
    const std::vector<double>& v_decays,
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum) noexcept;

  AttentionPoolLayer(const AttentionPoolLayer& src) noexcept;
  AttentionPoolLayer(AttentionPoolLayer&& src) noexcept;
  AttentionPoolLayer& operator=(const AttentionPoolLayer& src) noexcept;
  AttentionPoolLayer& operator=(AttentionPoolLayer&& src) noexcept;
  virtual ~AttentionPoolLayer();

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return Architecture::AttentionPool;
  }

  [[nodiscard]] inline unsigned get_attention_hidden_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _attention_hidden_size;
  }

  [[nodiscard]] inline const std::vector<double>& get_wa_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _wa_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_wa_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _wa_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_wa_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _wa_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_wa_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _wa_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_wa_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _wa_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_wa_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _wa_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_wa_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _wa_decays;
  }

  [[nodiscard]] inline const std::vector<double>& get_ba_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _ba_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_ba_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _ba_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_ba_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _ba_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_ba_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _ba_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_ba_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _ba_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_ba_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _ba_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_ba_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _ba_decays;
  }

  [[nodiscard]] inline const std::vector<double>& get_v_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _v_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_v_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _v_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_v_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _v_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_v_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _v_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_v_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _v_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_v_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _v_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_v_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    return _v_decays;
  }

  inline void set_wa_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _wa_values = v;
  }
  inline void set_wa_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _wa_grads = v;
  }
  inline void set_wa_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _wa_velocities = v;
  }
  inline void set_wa_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _wa_m1 = v;
  }
  inline void set_wa_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _wa_m2 = v;
  }
  inline void set_wa_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _wa_timesteps = v;
  }
  inline void set_wa_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _wa_decays = v;
  }

  inline void set_ba_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _ba_values = v;
  }
  inline void set_ba_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _ba_grads = v;
  }
  inline void set_ba_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _ba_velocities = v;
  }
  inline void set_ba_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _ba_m1 = v;
  }
  inline void set_ba_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _ba_m2 = v;
  }
  inline void set_ba_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _ba_timesteps = v;
  }
  inline void set_ba_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _ba_decays = v;
  }

  inline void set_v_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _v_values = v;
  }
  inline void set_v_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _v_grads = v;
  }
  inline void set_v_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _v_velocities = v;
  }
  inline void set_v_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _v_m1 = v;
  }
  inline void set_v_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _v_m2 = v;
  }
  inline void set_v_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _v_timesteps = v;
  }
  inline void set_v_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
    _v_decays = v;
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

  void apply_stored_gradients(double learning_rate, double clipping_scale) override;

  void zero_gradients() override;

  Layer* clone() const override;

private:
  // Recomputes the additive-attention forward pass (score/softmax/context)
  // for one batch item's T-timestep sequence from `h_seq` (T*N) and the
  // layer's current wa/ba/v weights, then backpropagates the already-known
  // local delta `delta` (dL/d(context), width N) through it: writes
  // dL/dh_t for every t into `out_dh` (T*N, fresh write, not accumulated),
  // and - when non-null - accumulates dL/dWa/dL/dba/dL/dv into the given
  // buffers (added, not overwritten, matching this codebase's existing
  // gate-weight-gradient accumulation convention).
  void attention_backward_one(
    const double* h_seq,
    size_t T,
    const double* delta,
    double* out_dh,
    double* wa_grad_accum,
    double* ba_grad_accum,
    double* v_grad_accum) const;

  // Shared tail end of both calculate_hidden_gradients entry points: given
  // `raw_delta_all` (batch_size * N, gradient w.r.t. this layer's own pooled
  // output, before undoing this layer's own activation/dropout), undoes
  // them to get the true per-batch-item delta (stored via set_gradients for
  // this layer's own later calculate_and_store_gradients call), then runs
  // the attention backward math to produce and store (via set_rnn_gradients)
  // the T*N-wide gradient the preceding recurrent layer consumes directly.
  void finish_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    const double* raw_delta_all) const;

  unsigned _attention_hidden_size;

  std::vector<double> _wa_values, _wa_grads, _wa_velocities, _wa_m1, _wa_m2;
  mutable std::vector<long long> _wa_timesteps;
  std::vector<double> _wa_decays;

  std::vector<double> _ba_values, _ba_grads, _ba_velocities, _ba_m1, _ba_m2;
  mutable std::vector<long long> _ba_timesteps;
  std::vector<double> _ba_decays;

  std::vector<double> _v_values, _v_grads, _v_velocities, _v_m1, _v_m2;
  mutable std::vector<long long> _v_timesteps;
  std::vector<double> _v_decays;
};

} // namespace myoddweb::nn
