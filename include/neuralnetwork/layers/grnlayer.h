#pragma once
#include "layer.h"

#include <array>
#include <vector>

namespace myoddweb::nn
{
class GrnLayer final : public Layer
{
public:
  static constexpr double LayerNormEpsilon = 1e-5;

  GrnLayer(
    unsigned layer_index,
    unsigned num_neurons_in_previous_layer,
    unsigned layer_size,
    unsigned feed_forward_hidden_size,
    double weight_decay,
    const Role layer_role,
    const activation& activation_method,
    const OptimiserType& optimiser_type,
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads,
    bool has_bias,
    bool use_layer_normalisation,
    double momentum,
    std::optional<uint32_t> seed);

  GrnLayer(
    unsigned layer_index,
    const Role layer_role,
    const OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned num_neurons_in_previous_layer,
    unsigned layer_size,
    unsigned feed_forward_hidden_size,
    bool use_layer_normalisation,
    const std::vector<Neuron>& neurons,
    const std::vector<double>& w1_values, const std::vector<double>& w1_grads, const std::vector<double>& w1_velocities, const std::vector<double>& w1_m1, const std::vector<double>& w1_m2, const std::vector<long long>& w1_timesteps, const std::vector<double>& w1_decays,
    const std::vector<double>& b1_values, const std::vector<double>& b1_grads, const std::vector<double>& b1_velocities, const std::vector<double>& b1_m1, const std::vector<double>& b1_m2, const std::vector<long long>& b1_timesteps, const std::vector<double>& b1_decays,
    const std::vector<double>& w2_values, const std::vector<double>& w2_grads, const std::vector<double>& w2_velocities, const std::vector<double>& w2_m1, const std::vector<double>& w2_m2, const std::vector<long long>& w2_timesteps, const std::vector<double>& w2_decays,
    const std::vector<double>& b2_values, const std::vector<double>& b2_grads, const std::vector<double>& b2_velocities, const std::vector<double>& b2_m1, const std::vector<double>& b2_m2, const std::vector<long long>& b2_timesteps, const std::vector<double>& b2_decays,
    const std::vector<double>& w_skip_values, const std::vector<double>& w_skip_grads, const std::vector<double>& w_skip_velocities, const std::vector<double>& w_skip_m1, const std::vector<double>& w_skip_m2, const std::vector<long long>& w_skip_timesteps, const std::vector<double>& w_skip_decays,
    const std::vector<double>& b_skip_values, const std::vector<double>& b_skip_grads, const std::vector<double>& b_skip_velocities, const std::vector<double>& b_skip_m1, const std::vector<double>& b_skip_m2, const std::vector<long long>& b_skip_timesteps, const std::vector<double>& b_skip_decays,
    const std::vector<double>& ln_gain_values, const std::vector<double>& ln_gain_grads, const std::vector<double>& ln_gain_velocities, const std::vector<double>& ln_gain_m1, const std::vector<double>& ln_gain_m2, const std::vector<long long>& ln_gain_timesteps, const std::vector<double>& ln_gain_decays,
    const std::vector<double>& ln_bias_values, const std::vector<double>& ln_bias_grads, const std::vector<double>& ln_bias_velocities, const std::vector<double>& ln_bias_m1, const std::vector<double>& ln_bias_m2, const std::vector<long long>& ln_bias_timesteps, const std::vector<double>& ln_bias_decays,
    const ResidualProjector* residual_projector,
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum) noexcept;

  GrnLayer(const GrnLayer& src) noexcept;
  GrnLayer(GrnLayer&& src) noexcept;
  GrnLayer& operator=(const GrnLayer& src) noexcept;
  GrnLayer& operator=(GrnLayer&& src) noexcept;
  virtual ~GrnLayer();

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("GrnLayer");
    return Architecture::Grn;
  }

  [[nodiscard]] inline unsigned get_feed_forward_hidden_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GrnLayer");
    return _feed_forward_hidden_size;
  }

  [[nodiscard]] inline bool get_use_layer_normalisation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GrnLayer");
    return _use_layer_normalisation;
  }

  [[nodiscard]] inline bool has_skip_projection() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GrnLayer");
    return get_number_input_neurons() != get_number_output_neurons();
  }

  // Weight family accessors for serialization
  [[nodiscard]] inline const std::vector<double>& get_w1_values() const noexcept { return _w1.values; }
  [[nodiscard]] inline const std::vector<double>& get_w1_grads() const noexcept { return _w1.grads; }
  [[nodiscard]] inline const std::vector<double>& get_w1_velocities() const noexcept { return _w1.velocities; }
  [[nodiscard]] inline const std::vector<double>& get_w1_m1() const noexcept { return _w1.m1; }
  [[nodiscard]] inline const std::vector<double>& get_w1_m2() const noexcept { return _w1.m2; }
  [[nodiscard]] inline const std::vector<long long>& get_w1_timesteps() const noexcept { return _w1.timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_w1_decays() const noexcept { return _w1.decays; }
  inline void set_w1_values(const std::vector<double>& v) { _w1.values = v; }

  [[nodiscard]] inline const std::vector<double>& get_b1_values() const noexcept { return _b1.values; }
  [[nodiscard]] inline const std::vector<double>& get_b1_grads() const noexcept { return _b1.grads; }
  [[nodiscard]] inline const std::vector<double>& get_b1_velocities() const noexcept { return _b1.velocities; }
  [[nodiscard]] inline const std::vector<double>& get_b1_m1() const noexcept { return _b1.m1; }
  [[nodiscard]] inline const std::vector<double>& get_b1_m2() const noexcept { return _b1.m2; }
  [[nodiscard]] inline const std::vector<long long>& get_b1_timesteps() const noexcept { return _b1.timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_b1_decays() const noexcept { return _b1.decays; }
  inline void set_b1_values(const std::vector<double>& v) { _b1.values = v; }

  [[nodiscard]] inline const std::vector<double>& get_w2_values() const noexcept { return _w2.values; }
  [[nodiscard]] inline const std::vector<double>& get_w2_grads() const noexcept { return _w2.grads; }
  [[nodiscard]] inline const std::vector<double>& get_w2_velocities() const noexcept { return _w2.velocities; }
  [[nodiscard]] inline const std::vector<double>& get_w2_m1() const noexcept { return _w2.m1; }
  [[nodiscard]] inline const std::vector<double>& get_w2_m2() const noexcept { return _w2.m2; }
  [[nodiscard]] inline const std::vector<long long>& get_w2_timesteps() const noexcept { return _w2.timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_w2_decays() const noexcept { return _w2.decays; }
  inline void set_w2_values(const std::vector<double>& v) { _w2.values = v; }

  [[nodiscard]] inline const std::vector<double>& get_b2_values() const noexcept { return _b2.values; }
  [[nodiscard]] inline const std::vector<double>& get_b2_grads() const noexcept { return _b2.grads; }
  [[nodiscard]] inline const std::vector<double>& get_b2_velocities() const noexcept { return _b2.velocities; }
  [[nodiscard]] inline const std::vector<double>& get_b2_m1() const noexcept { return _b2.m1; }
  [[nodiscard]] inline const std::vector<double>& get_b2_m2() const noexcept { return _b2.m2; }
  [[nodiscard]] inline const std::vector<long long>& get_b2_timesteps() const noexcept { return _b2.timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_b2_decays() const noexcept { return _b2.decays; }
  inline void set_b2_values(const std::vector<double>& v) { _b2.values = v; }

  [[nodiscard]] inline const std::vector<double>& get_w_skip_values() const noexcept { return _w_skip.values; }
  [[nodiscard]] inline const std::vector<double>& get_w_skip_grads() const noexcept { return _w_skip.grads; }
  [[nodiscard]] inline const std::vector<double>& get_w_skip_velocities() const noexcept { return _w_skip.velocities; }
  [[nodiscard]] inline const std::vector<double>& get_w_skip_m1() const noexcept { return _w_skip.m1; }
  [[nodiscard]] inline const std::vector<double>& get_w_skip_m2() const noexcept { return _w_skip.m2; }
  [[nodiscard]] inline const std::vector<long long>& get_w_skip_timesteps() const noexcept { return _w_skip.timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_w_skip_decays() const noexcept { return _w_skip.decays; }
  inline void set_w_skip_values(const std::vector<double>& v) { _w_skip.values = v; }

  [[nodiscard]] inline const std::vector<double>& get_b_skip_values() const noexcept { return _b_skip.values; }
  [[nodiscard]] inline const std::vector<double>& get_b_skip_grads() const noexcept { return _b_skip.grads; }
  [[nodiscard]] inline const std::vector<double>& get_b_skip_velocities() const noexcept { return _b_skip.velocities; }
  [[nodiscard]] inline const std::vector<double>& get_b_skip_m1() const noexcept { return _b_skip.m1; }
  [[nodiscard]] inline const std::vector<double>& get_b_skip_m2() const noexcept { return _b_skip.m2; }
  [[nodiscard]] inline const std::vector<long long>& get_b_skip_timesteps() const noexcept { return _b_skip.timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_b_skip_decays() const noexcept { return _b_skip.decays; }
  inline void set_b_skip_values(const std::vector<double>& v) { _b_skip.values = v; }

  [[nodiscard]] inline const std::vector<double>& get_ln_gain_values() const noexcept { return _ln_gain.values; }
  [[nodiscard]] inline const std::vector<double>& get_ln_gain_grads() const noexcept { return _ln_gain.grads; }
  [[nodiscard]] inline const std::vector<double>& get_ln_gain_velocities() const noexcept { return _ln_gain.velocities; }
  [[nodiscard]] inline const std::vector<double>& get_ln_gain_m1() const noexcept { return _ln_gain.m1; }
  [[nodiscard]] inline const std::vector<double>& get_ln_gain_m2() const noexcept { return _ln_gain.m2; }
  [[nodiscard]] inline const std::vector<long long>& get_ln_gain_timesteps() const noexcept { return _ln_gain.timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_ln_gain_decays() const noexcept { return _ln_gain.decays; }
  inline void set_ln_gain_values(const std::vector<double>& v) { _ln_gain.values = v; }

  [[nodiscard]] inline const std::vector<double>& get_ln_bias_values() const noexcept { return _ln_bias.values; }
  [[nodiscard]] inline const std::vector<double>& get_ln_bias_grads() const noexcept { return _ln_bias.grads; }
  [[nodiscard]] inline const std::vector<double>& get_ln_bias_velocities() const noexcept { return _ln_bias.velocities; }
  [[nodiscard]] inline const std::vector<double>& get_ln_bias_m1() const noexcept { return _ln_bias.m1; }
  [[nodiscard]] inline const std::vector<double>& get_ln_bias_m2() const noexcept { return _ln_bias.m2; }
  [[nodiscard]] inline const std::vector<long long>& get_ln_bias_timesteps() const noexcept { return _ln_bias.timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_ln_bias_decays() const noexcept { return _ln_bias.decays; }
  inline void set_ln_bias_values(const std::vector<double>& v) { _ln_bias.values = v; }

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

  void zero_gradients() override;

  Layer* clone() const override;

private:
  struct forward_scratch
  {
    std::vector<double> z1;
    std::vector<double> eta2;
    std::vector<double> eta1;
    std::vector<double> glu;
    std::vector<double> glu_drop;
    std::vector<double> mask;
    std::vector<double> x_skip;
    std::vector<double> y_res;
    std::vector<double> y_norm;
    std::vector<double> inv_std;
    std::vector<double> out_seq;
  };

  void process_forward_range(
    size_t b_start,
    size_t b_end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    unsigned prev_layer_index,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<HiddenStates>& batch_hidden_states,
    bool is_training,
    forward_scratch& scratch) const;

  struct grad_accumulators
  {
    double* w1 = nullptr; double* b1 = nullptr;
    double* w2 = nullptr; double* b2 = nullptr;
    double* w_skip = nullptr; double* b_skip = nullptr;
    double* ln_gain = nullptr; double* ln_bias = nullptr;
  };

  struct backward_scratch
  {
    std::vector<double> z1;
    std::vector<double> eta2;
    std::vector<double> eta1;
    std::vector<double> y_norm;
    std::vector<double> d_ln_gain;
    std::vector<double> d_ln_bias;
    std::vector<double> d_y_res;
    std::vector<double> d_glu;
    std::vector<double> d_eta1;
    std::vector<double> d_eta2;
    std::vector<double> d_z1;
    std::vector<double> d_x_dense;
    std::vector<double> d_x_skip;
    std::vector<double> d_x_total;
    std::vector<double> delta;
  };

  void finish_hidden_gradients_range(
    size_t b_start,
    size_t b_end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& batch_hidden_states,
    const double* raw_delta_all,
    backward_scratch& scratch) const;

  void accumulate_gradients_range(
    size_t b_start,
    size_t b_end,
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& batch_hidden_states,
    unsigned prev_layer_index,
    unsigned this_layer_index,
    const grad_accumulators& accum,
    size_t& local_contributing,
    backward_scratch& scratch) const;

  struct thread_grad_accumulators
  {
    std::vector<double> w1, b1;
    std::vector<double> w2, b2;
    std::vector<double> w_skip, b_skip;
    std::vector<double> ln_gain, ln_bias;
    size_t contributing = 0;
  };

  struct grn_forward_task
  {
    const GrnLayer* layer;
    size_t start;
    size_t end;
    std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
    unsigned prev_layer_index;
    const std::vector<std::vector<double>>* batch_residual_output_values;
    std::vector<HiddenStates>* batch_hidden_states;
    bool is_training;
    forward_scratch* scratch;

    void operator()() const;
  };

  struct grn_finish_hidden_gradients_task
  {
    const GrnLayer* layer;
    size_t start;
    size_t end;
    std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
    const std::vector<HiddenStates>* batch_hidden_states;
    const double* raw_delta_all;
    backward_scratch* scratch;

    void operator()() const;
  };

  struct grn_grad_calc_task
  {
    const GrnLayer* layer;
    size_t start;
    size_t end;
    const std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
    const std::vector<HiddenStates>* batch_hidden_states;
    unsigned prev_layer_index;
    unsigned this_layer_index;
    grad_accumulators accum;
    size_t* local_contributing;
    backward_scratch* scratch;

    void operator()() const;
  };

  struct WeightFamily
  {
    std::vector<double> values, grads, velocities, m1, m2;
    mutable std::vector<long long> timesteps;
    std::vector<double> decays;
  };

  struct FamilyRef
  {
    WeightFamily* family;
    bool is_bias;
  };
  [[nodiscard]] std::array<FamilyRef, 8> all_families() noexcept;
  [[nodiscard]] std::array<const WeightFamily*, 8> all_families() const noexcept;

  static void init_family(
    WeightFamily& family,
    size_t num_in,
    size_t num_out,
    double weight_decay,
    const activation& init_activation,
    unsigned block_tag,
    std::optional<uint32_t> seed);

  static void init_bias_family(WeightFamily& family, size_t n, double fill_value);

  void grn_backward_one(
    const double* x_seq,
    size_t T,
    const double* delta,
    double* out_dx,
    const std::vector<HiddenState>& hidden_state_row,
    const grad_accumulators& accum,
    backward_scratch& scratch) const;

  void finish_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    const double* raw_delta_all) const;

  unsigned _feed_forward_hidden_size;
  bool _use_layer_normalisation;

  WeightFamily _w1, _b1;
  WeightFamily _w2, _b2;
  WeightFamily _w_skip, _b_skip;
  WeightFamily _ln_gain, _ln_bias;
};

} // namespace myoddweb::nn
