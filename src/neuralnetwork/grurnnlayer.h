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
#include "aligned_allocator.h"

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
    LayerType layer_type, 
    const activation& activation_method, 
    const OptimiserType& optimiser_type, 
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads
  ) noexcept;

  GRURNNLayer(
    unsigned layer_index,
    const LayerType layer_type,
    const activation activation,
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
    int number_of_threads
  ) noexcept;

  GRURNNLayer(const GRURNNLayer& src) noexcept;
  GRURNNLayer(GRURNNLayer&& src) noexcept;
  GRURNNLayer& operator=(const GRURNNLayer& src) noexcept;
  GRURNNLayer& operator=(GRURNNLayer&& src) noexcept;
  virtual ~GRURNNLayer();

public:
  bool use_bptt() const noexcept override {
    return true;
  }

  void calculate_forward_feed(
      std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
      const Layer &previous_layer,
      const std::vector<std::vector<double>> &batch_residual_output_values,
      std::vector<HiddenStates> &batch_hidden_states,
      bool is_training) const override;

  void calculate_output_gradients(
      std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
      std::vector<std::vector<double>>::const_iterator target_outputs_begin,
      const std::vector<HiddenStates> &batch_hidden_states,
      ErrorCalculation::type error_calculation_type,
      const ErrorCalculation::EvaluationConfig& evaluation_config) const  override;

  void calculate_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    int bptt_max_ticks) const override;



  double get_recurrent_weight_value(unsigned from_neuron, unsigned to_neuron) const;

  bool has_bias() const noexcept override;

  Layer* clone() const override;

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    int bptt_max_ticks) override;

  double get_gradient_norm_sq() const override;

  void apply_stored_gradients(double learning_rate, double clipping_scale) override;

  inline const std::vector<double>& get_rw_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_values;
  }
  inline const std::vector<double>& get_rw_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_grads;
  }
  inline const std::vector<double>& get_rw_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_velocities;
  }
  inline const std::vector<double>& get_rw_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_m1;
  }
  inline const std::vector<double>& get_rw_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_m2;
  }
  inline const std::vector<long long>& get_rw_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_timesteps;
  }
  inline const std::vector<double>& get_rw_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GRURNNLayer");
    return _rw_decays;
  }

  // Update Gate Accessors
  inline const std::vector<double>& get_z_w_values() const noexcept { return _z_w_values; }
  inline const std::vector<double>& get_z_w_grads() const noexcept { return _z_w_grads; }
  inline const std::vector<double>& get_z_w_velocities() const noexcept { return _z_w_velocities; }
  inline const std::vector<double>& get_z_w_m1() const noexcept { return _z_w_m1; }
  inline const std::vector<double>& get_z_w_m2() const noexcept { return _z_w_m2; }
  inline const std::vector<long long>& get_z_w_timesteps() const noexcept { return _z_w_timesteps; }
  inline const std::vector<double>& get_z_w_decays() const noexcept { return _z_w_decays; }

  inline const std::vector<double>& get_z_rw_values() const noexcept { return _z_rw_values; }
  inline const std::vector<double>& get_z_rw_grads() const noexcept { return _z_rw_grads; }
  inline const std::vector<double>& get_z_rw_velocities() const noexcept { return _z_rw_velocities; }
  inline const std::vector<double>& get_z_rw_m1() const noexcept { return _z_rw_m1; }
  inline const std::vector<double>& get_z_rw_m2() const noexcept { return _z_rw_m2; }
  inline const std::vector<long long>& get_z_rw_timesteps() const noexcept { return _z_rw_timesteps; }
  inline const std::vector<double>& get_z_rw_decays() const noexcept { return _z_rw_decays; }

  inline const std::vector<double>& get_z_b_values() const noexcept { return _z_b_values; }
  inline const std::vector<double>& get_z_b_grads() const noexcept { return _z_b_grads; }
  inline const std::vector<double>& get_z_b_velocities() const noexcept { return _z_b_velocities; }
  inline const std::vector<double>& get_z_b_m1() const noexcept { return _z_b_m1; }
  inline const std::vector<double>& get_z_b_m2() const noexcept { return _z_b_m2; }
  inline const std::vector<long long>& get_z_b_timesteps() const noexcept { return _z_b_timesteps; }
  inline const std::vector<double>& get_z_b_decays() const noexcept { return _z_b_decays; }

  // Reset Gate Accessors
  inline const std::vector<double>& get_r_w_values() const noexcept { return _r_w_values; }
  inline const std::vector<double>& get_r_w_grads() const noexcept { return _r_w_grads; }
  inline const std::vector<double>& get_r_w_velocities() const noexcept { return _r_w_velocities; }
  inline const std::vector<double>& get_r_w_m1() const noexcept { return _r_w_m1; }
  inline const std::vector<double>& get_r_w_m2() const noexcept { return _r_w_m2; }
  inline const std::vector<long long>& get_r_w_timesteps() const noexcept { return _r_w_timesteps; }
  inline const std::vector<double>& get_r_w_decays() const noexcept { return _r_w_decays; }

  inline const std::vector<double>& get_r_rw_values() const noexcept { return _r_rw_values; }
  inline const std::vector<double>& get_r_rw_grads() const noexcept { return _r_rw_grads; }
  inline const std::vector<double>& get_r_rw_velocities() const noexcept { return _r_rw_velocities; }
  inline const std::vector<double>& get_r_rw_m1() const noexcept { return _r_rw_m1; }
  inline const std::vector<double>& get_r_rw_m2() const noexcept { return _r_rw_m2; }
  inline const std::vector<long long>& get_r_rw_timesteps() const noexcept { return _r_rw_timesteps; }
  inline const std::vector<double>& get_r_rw_decays() const noexcept { return _r_rw_decays; }

  inline const std::vector<double>& get_r_b_values() const noexcept { return _r_b_values; }
  inline const std::vector<double>& get_r_b_grads() const noexcept { return _r_b_grads; }
  inline const std::vector<double>& get_r_b_velocities() const noexcept { return _r_b_velocities; }
  inline const std::vector<double>& get_r_b_m1() const noexcept { return _r_b_m1; }
  inline const std::vector<double>& get_r_b_m2() const noexcept { return _r_b_m2; }
  inline const std::vector<long long>& get_r_b_timesteps() const noexcept { return _r_b_timesteps; }
  inline const std::vector<double>& get_r_b_decays() const noexcept { return _r_b_decays; }

private:
  struct BPTTWorkspace {
    using AlignedVector = std::vector<double, AlignedAllocator<double, 32>>;
    AlignedVector grad_from_next_all_t;
    AlignedVector d_next_h;
    AlignedVector rnn_grad_matrix;
    AlignedVector chunk_dz;
    AlignedVector chunk_dr;
    AlignedVector chunk_dh_hat;
    AlignedVector chunk_dh_prev_accum;
    AlignedVector h_hat_vals;
    AlignedVector temp_Uh_T_dh_hat;
  };
  mutable std::vector<BPTTWorkspace> _workspaces;
  mutable BPTTWorkspace::AlignedVector _rw_values_T;
  mutable BPTTWorkspace::AlignedVector _z_rw_values_T;
  mutable BPTTWorkspace::AlignedVector _r_rw_values_T;

  void calculate_bptt_batch_chunk(
    size_t start,
    size_t end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    int bptt_max_ticks,
    BPTTWorkspace& workspace) const;

  void initialize_recurrent_weights(double weight_decay);
  
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
};
