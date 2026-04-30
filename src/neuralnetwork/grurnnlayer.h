#pragma once
#include "aligned_allocator.h"
#include "errorcalculation.h"
#include "hiddenstate.h"
#include "layer.h"

#include <vector>

class GRURNNLayer final : public Layer
{
protected:
  friend class Layers;

public:
  GRURNNLayer(unsigned layer_index,
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
    double momentum);

  GRURNNLayer(unsigned layer_index,
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
    double momentum);

  GRURNNLayer(
    unsigned layer_index,
    const Role layer_role,
    const OptimiserType optimiser_type,
    int residual_layer_number,
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
    // Update Gate (z)
    const std::vector<double>& z_w_values,
    const std::vector<double>& z_w_grads,
    const std::vector<double>& z_w_velocities,
    const std::vector<double>& z_w_m1,
    const std::vector<double>& z_w_m2,
    const std::vector<long long>& z_w_timesteps,
    const std::vector<double>& z_w_decays,
    const std::vector<double>& z_rw_values,
    const std::vector<double>& z_rw_grads,
    const std::vector<double>& z_rw_velocities,
    const std::vector<double>& z_rw_m1,
    const std::vector<double>& z_rw_m2,
    const std::vector<long long>& z_rw_timesteps,
    const std::vector<double>& z_rw_decays,
    const std::vector<double>& z_b_values,
    const std::vector<double>& z_b_grads,
    const std::vector<double>& z_b_velocities,
    const std::vector<double>& z_b_m1,
    const std::vector<double>& z_b_m2,
    const std::vector<long long>& z_b_timesteps,
    const std::vector<double>& z_b_decays,
    // Reset Gate (r)
    const std::vector<double>& r_w_values,
    const std::vector<double>& r_w_grads,
    const std::vector<double>& r_w_velocities,
    const std::vector<double>& r_w_m1,
    const std::vector<double>& r_w_m2,
    const std::vector<long long>& r_w_timesteps,
    const std::vector<double>& r_w_decays,
    const std::vector<double>& r_rw_values,
    const std::vector<double>& r_rw_grads,
    const std::vector<double>& r_rw_velocities,
    const std::vector<double>& r_rw_m1,
    const std::vector<double>& r_rw_m2,
    const std::vector<long long>& r_rw_timesteps,
    const std::vector<double>& r_rw_decays,
    const std::vector<double>& r_b_values,
    const std::vector<double>& r_b_grads,
    const std::vector<double>& r_b_velocities,
    const std::vector<double>& r_b_m1,
    const std::vector<double>& r_b_m2,
    const std::vector<long long>& r_b_timesteps,
    const std::vector<double>& r_b_decays,
    const ResidualProjector* residual_projector,
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum
  ) noexcept;

  GRURNNLayer(const GRURNNLayer& src) noexcept;
  GRURNNLayer(GRURNNLayer&& src) noexcept;
  GRURNNLayer& operator=(const GRURNNLayer& src) noexcept;
  GRURNNLayer& operator=(GRURNNLayer&& src) noexcept;
  virtual ~GRURNNLayer();

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return Architecture::Gru;
  }

public:
  [[nodiscard]] inline bool use_bptt() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return true;
  }

  /* 
   * Multiplier = 4:
   * 1. Update gate (z) pre-activation
   * 2. Reset gate (r) pre-activation
   * 3. Candidate state (h_hat) pre-activation
   * 4. Dropout mask (stored to ensure consistency between forward and BPTT passes)
   */
  static constexpr unsigned Multiplier = 5;
  static constexpr unsigned GateCount = 3; // Update, Reset, Candidate

  [[nodiscard]] unsigned get_pre_activation_multiplier() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return Multiplier;
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

  void calculate_hidden_gradients_from_output_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<std::vector<double>>& batch_output_gradients,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    int bptt_max_ticks) const override;

  [[nodiscard]] double get_recurrent_weight_value(unsigned from_neuron, unsigned to_neuron) const;

  [[nodiscard]] Layer* clone() const override;

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override;

  [[nodiscard]] double get_gradient_norm_sq() const override;

  void zero_gradients() override;

  void apply_stored_gradients(double learning_rate, double clipping_scale) override;

  void cache_recurrent_weights() override;

  [[nodiscard]] inline const std::vector<double>& get_rw_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_rw_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_decays;
  }

  // Update Gate Accessor
  [[nodiscard]] inline const std::vector<double>& get_z_w_values() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_w_values; }
  [[nodiscard]] inline const std::vector<double>& get_z_w_grads() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_w_grads; }
  [[nodiscard]] inline const std::vector<double>& get_z_w_velocities() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_w_velocities; }
  [[nodiscard]] inline const std::vector<double>& get_z_w_m1() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_w_m1; }
  [[nodiscard]] inline const std::vector<double>& get_z_w_m2() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_w_m2; }
  [[nodiscard]] inline const std::vector<long long>& get_z_w_timesteps() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_w_timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_z_w_decays() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_w_decays; }

  [[nodiscard]] inline const std::vector<double>& get_z_rw_values() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_rw_values; }
  [[nodiscard]] inline const std::vector<double>& get_z_rw_grads() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_rw_grads; }
  [[nodiscard]] inline const std::vector<double>& get_z_rw_velocities() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_rw_velocities; }
  [[nodiscard]] inline const std::vector<double>& get_z_rw_m1() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_rw_m1; }
  [[nodiscard]] inline const std::vector<double>& get_z_rw_m2() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_rw_m2; }
  [[nodiscard]] inline const std::vector<long long>& get_z_rw_timesteps() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_rw_timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_z_rw_decays() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_rw_decays; }

  [[nodiscard]] inline const std::vector<double>& get_z_b_values() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_b_values; }
  [[nodiscard]] inline const std::vector<double>& get_z_b_grads() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_b_grads; }
  [[nodiscard]] inline const std::vector<double>& get_z_b_velocities() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_b_velocities; }
  [[nodiscard]] inline const std::vector<double>& get_z_b_m1() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_b_m1; }
  [[nodiscard]] inline const std::vector<double>& get_z_b_m2() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_b_m2; }
  [[nodiscard]] inline const std::vector<long long>& get_z_b_timesteps() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_b_timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_z_b_decays() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _z_b_decays; }

  // Reset Gate Accessor
  [[nodiscard]] inline const std::vector<double>& get_r_w_values() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_w_values; }
  [[nodiscard]] inline const std::vector<double>& get_r_w_grads() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_w_grads; }
  [[nodiscard]] inline const std::vector<double>& get_r_w_velocities() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_w_velocities; }
  [[nodiscard]] inline const std::vector<double>& get_r_w_m1() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_w_m1; }
  [[nodiscard]] inline const std::vector<double>& get_r_w_m2() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_w_m2; }
  [[nodiscard]] inline const std::vector<long long>& get_r_w_timesteps() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_w_timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_r_w_decays() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_w_decays; }

  [[nodiscard]] inline const std::vector<double>& get_r_rw_values() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_rw_values; }
  [[nodiscard]] inline const std::vector<double>& get_r_rw_grads() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_rw_grads; }
  [[nodiscard]] inline const std::vector<double>& get_r_rw_velocities() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_rw_velocities; }
  [[nodiscard]] inline const std::vector<double>& get_r_rw_m1() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_rw_m1; }
  [[nodiscard]] inline const std::vector<double>& get_r_rw_m2() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_rw_m2; }
  [[nodiscard]] inline const std::vector<long long>& get_r_rw_timesteps() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_rw_timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_r_rw_decays() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_rw_decays; }

  [[nodiscard]] inline const std::vector<double>& get_r_b_values() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_b_values; }
  [[nodiscard]] inline const std::vector<double>& get_r_b_grads() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_b_grads; }
  [[nodiscard]] inline const std::vector<double>& get_r_b_velocities() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_b_velocities; }
  [[nodiscard]] inline const std::vector<double>& get_r_b_m1() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_b_m1; }
  [[nodiscard]] inline const std::vector<double>& get_r_b_m2() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_b_m2; }
  [[nodiscard]] inline const std::vector<long long>& get_r_b_timesteps() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_b_timesteps; }
  [[nodiscard]] inline const std::vector<double>& get_r_b_decays() const noexcept { MYODDWEB_PROFILE_FUNCTION("GRURNNLayer"); return _r_b_decays; }

  void set_w_values(const std::vector<double>& v) override;
  void set_w_grads(const std::vector<double>& v) override;
  void set_w_velocities(const std::vector<double>& v) override;
  void set_w_m1(const std::vector<double>& v) override;
  void set_w_m2(const std::vector<double>& v) override;
  void set_w_timesteps(const std::vector<long long>& v) override;
  void set_w_decays(const std::vector<double>& v) override;

  void set_b_values(const std::vector<double>& v) override;
  void set_b_grads(const std::vector<double>& v) override;
  void set_b_velocities(const std::vector<double>& v) override;
  void set_b_m1(const std::vector<double>& v) override;
  void set_b_m2(const std::vector<double>& v) override;
  void set_b_timesteps(const std::vector<long long>& v) override;
  void set_b_decays(const std::vector<double>& v) override;

  void set_rw_values(const std::vector<double>& v) override;
  void set_rw_grads(const std::vector<double>& v) override;
  void set_rw_velocities(const std::vector<double>& v) override;
  void set_rw_m1(const std::vector<double>& v) override;
  void set_rw_m2(const std::vector<double>& v) override;
  void set_rw_timesteps(const std::vector<long long>& v) override;
  void set_rw_decays(const std::vector<double>& v) override;

  void set_z_w_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_w_values = v;
  }
  void set_z_w_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_w_grads = v;
  }
  void set_z_w_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_w_velocities = v;
  }
  void set_z_w_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_w_m1 = v;
  }
  void set_z_w_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_w_m2 = v;
  }
  void set_z_w_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_w_timesteps = v;
  }
  void set_z_w_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_w_decays = v;
  }

  void set_z_rw_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_rw_values = v;
  }
  void set_z_rw_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_rw_grads = v;
  }
  void set_z_rw_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_rw_velocities = v;
  }
  void set_z_rw_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_rw_m1 = v;
  }
  void set_z_rw_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_rw_m2 = v;
  }
  void set_z_rw_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_rw_timesteps = v;
  }
  void set_z_rw_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_rw_decays = v;
  }

  void set_z_b_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_b_values = v;
  }
  void set_z_b_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_b_grads = v;
  }
  void set_z_b_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_b_velocities = v;
  }
  void set_z_b_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_b_m1 = v;
  }
  void set_z_b_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_b_m2 = v;
  }
  void set_z_b_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_b_timesteps = v;
  }
  void set_z_b_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _z_b_decays = v;
  }

  void set_r_w_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_w_values = v;
  }
  void set_r_w_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_w_grads = v;
  }
  void set_r_w_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_w_velocities = v;
  }
  void set_r_w_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_w_m1 = v;
  }
  void set_r_w_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_w_m2 = v;
  }
  void set_r_w_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_w_timesteps = v;
  }
  void set_r_w_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_w_decays = v;
  }

  void set_r_rw_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_rw_values = v;
  }
  void set_r_rw_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_rw_grads = v;
  }
  void set_r_rw_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_rw_velocities = v;
  }
  void set_r_rw_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_rw_m1 = v;
  }
  void set_r_rw_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_rw_m2 = v;
  }
  void set_r_rw_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_rw_timesteps = v;
  }
  void set_r_rw_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_rw_decays = v;
  }

  void set_r_b_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_b_values = v;
  }
  void set_r_b_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_b_grads = v;
  }
  void set_r_b_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_b_velocities = v;
  }
  void set_r_b_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_b_m1 = v;
  }
  void set_r_b_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_b_m2 = v;
  }
  void set_r_b_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_b_timesteps = v;
  }
  void set_r_b_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    _r_b_decays = v;
  }

private:

  void run_forward_pass(
    const size_t start,
    const size_t end,
    const size_t N_this,
    const size_t N_prev,
    const size_t num_time_steps,
    const std::vector<double>& flattened_batch_inputs,
    const std::vector<std::vector<double>>& batch_residual_output_values,
    std::vector<double>& batch_output_sequences,
    std::vector<HiddenStates>& batch_hidden_states,
    bool is_training
  ) const;

  struct BPTTWorkspace 
  {
    using AlignedVector = std::vector<double, AlignedAllocator<double, 32>>;
    AlignedVector grad_from_next_all_t;
    AlignedVector d_next_h;
    AlignedVector rnn_grad_matrix; // Stores gate gradients [Batch x T x 3N]
    AlignedVector dx_matrix;      // Stores input gradients [Batch x T x N_prev]
    AlignedVector chunk_dz;
    AlignedVector chunk_dr;
    AlignedVector chunk_dh_hat;
    AlignedVector chunk_dh_prev_accum;
    AlignedVector h_hat_vals;
    AlignedVector temp_Uh_T_dh_hat;

    void resize(size_t n, size_t n_prev, size_t batch_chunk_size, size_t num_time_steps)
    {
      grad_from_next_all_t.assign(batch_chunk_size * num_time_steps * n, 0.0);
      d_next_h.assign(batch_chunk_size * n, 0.0);
      rnn_grad_matrix.assign(batch_chunk_size * num_time_steps * GateCount * n, 0.0);
      dx_matrix.assign(batch_chunk_size * num_time_steps * n_prev, 0.0);
      chunk_dz.assign(batch_chunk_size * n, 0.0);
      chunk_dr.assign(batch_chunk_size * n, 0.0);
      chunk_dh_hat.assign(batch_chunk_size * n, 0.0);
      chunk_dh_prev_accum.assign(batch_chunk_size * n, 0.0);
      h_hat_vals.assign(n, 0.0);
      temp_Uh_T_dh_hat.assign(batch_chunk_size * n, 0.0);
    }
  };

  BPTTWorkspace& get_workspace(size_t thread_idx) const;
  void allocate_workspace(unsigned int num_threads);
  void allocate_workspace();

  void calculate_bptt_batch_chunk(
    size_t start,
    size_t end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    int bptt_max_ticks,
    BPTTWorkspace& workspace,
    const BPTTWorkspace::AlignedVector& rw_values_T,
    const BPTTWorkspace::AlignedVector& z_rw_values_T,
    const BPTTWorkspace::AlignedVector& r_rw_values_T) const;

  void initialize_recurrent_weights(double weight_decay);

  void init_weights(
    std::vector<double>& values, std::vector<double>& grads,
    std::vector<double>& velocities, std::vector<double>& m1,
    std::vector<double>& m2, std::vector<long long>& timesteps,
    std::vector<double>& decays, size_t size, bool is_input,
    double weight_decay) const;

  void init_bias(
    std::vector<double>& values, std::vector<double>& grads,
    std::vector<double>& velocities, std::vector<double>& m1,
    std::vector<double>& m2, std::vector<long long>& timesteps,
    std::vector<double>& decays) const;
  
  // SoA for recurrent weights (Candidate State)
  std::vector<double> _rw_values;
  std::vector<double> _rw_grads;
  std::vector<double> _rw_velocities;
  std::vector<double> _rw_m1;
  std::vector<double> _rw_m2;
  std::vector<long long> _rw_timesteps;
  std::vector<double> _rw_decays;

  // --- Update Gate (z) ---
  // Input weights
  std::vector<double> _z_w_values;
  std::vector<double> _z_w_grads;
  std::vector<double> _z_w_velocities;
  std::vector<double> _z_w_m1;
  std::vector<double> _z_w_m2;
  std::vector<long long> _z_w_timesteps;
  std::vector<double> _z_w_decays;
  // Recurrent weights
  std::vector<double> _z_rw_values;
  std::vector<double> _z_rw_grads;
  std::vector<double> _z_rw_velocities;
  std::vector<double> _z_rw_m1;
  std::vector<double> _z_rw_m2;
  std::vector<long long> _z_rw_timesteps;
  std::vector<double> _z_rw_decays;
  // Biases
  std::vector<double> _z_b_values;
  std::vector<double> _z_b_grads;
  std::vector<double> _z_b_velocities;
  std::vector<double> _z_b_m1;
  std::vector<double> _z_b_m2;
  std::vector<long long> _z_b_timesteps;
  std::vector<double> _z_b_decays;

  // --- Reset Gate (r) ---
  // Input weights
  std::vector<double> _r_w_values;
  std::vector<double> _r_w_grads;
  std::vector<double> _r_w_velocities;
  std::vector<double> _r_w_m1;
  std::vector<double> _r_w_m2;
  std::vector<long long> _r_w_timesteps;
  std::vector<double> _r_w_decays;
  // Recurrent weights
  std::vector<double> _r_rw_values;
  std::vector<double> _r_rw_grads;
  std::vector<double> _r_rw_velocities;
  std::vector<double> _r_rw_m1;
  std::vector<double> _r_rw_m2;
  std::vector<long long> _r_rw_timesteps;
  std::vector<double> _r_rw_decays;
  // Biases
  std::vector<double> _r_b_values;
  std::vector<double> _r_b_grads;
  std::vector<double> _r_b_velocities;
  std::vector<double> _r_b_m1;
  std::vector<double> _r_b_m2;
  std::vector<long long> _r_b_timesteps;
  std::vector<double> _r_b_decays;

  // Cached transposed recurrent weights
  BPTTWorkspace::AlignedVector _rw_values_T;
  BPTTWorkspace::AlignedVector _z_rw_values_T;
  BPTTWorkspace::AlignedVector _r_rw_values_T;

  // Per-thread workspaces for BPTT
  std::vector<std::unique_ptr<BPTTWorkspace>> _thread_workspaces;
};
