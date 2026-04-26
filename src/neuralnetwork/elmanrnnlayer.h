#pragma once
#include "errorcalculation.h"
#include "hiddenstate.h"
#include "layer.h"

#include <shared_mutex>
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
    const Role layer_role,
    const activation& activation_method, 
    const OptimiserType& optimiser_type, 
    int residual_layer_number,
    double dropout_rate,
    ResidualProjector* residual_projector,
    int number_of_threads,
    bool has_bias,
    double momentum);

  ElmanRNNLayer(unsigned layer_index,
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

  ElmanRNNLayer(
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
    const ResidualProjector* residual_projector,
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum
  ) noexcept;

  ElmanRNNLayer(const ElmanRNNLayer& src) noexcept;
  ElmanRNNLayer(ElmanRNNLayer&& src) noexcept;
  ElmanRNNLayer& operator=(const ElmanRNNLayer& src) noexcept;
  ElmanRNNLayer& operator=(ElmanRNNLayer&& src) noexcept;
  virtual ~ElmanRNNLayer();

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return Architecture::Elman;
  }

public:
  bool use_bptt() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return true;
  }

  [[nodiscard]] unsigned get_pre_activation_multiplier() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    return 1;
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

  double get_recurrent_weight_value(unsigned from_neuron, unsigned to_neuron) const;

  Layer* clone() const override;

  void calculate_and_store_gradients(
    const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& hidden_states,
    const Layer& previous_layer,
    size_t batch_size,
    int bptt_max_ticks) override;

  virtual double get_gradient_norm_sq() const override;

  virtual void zero_gradients() override;

  virtual void apply_stored_gradients(double learning_rate, double clipping_scale) override;

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

  void set_rw_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    _rw_values = v;
  }
  void set_rw_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    _rw_grads = v;
  }
  void set_rw_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    _rw_velocities = v;
  }
  void set_rw_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    _rw_m1 = v;
  }
  void set_rw_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    _rw_m2 = v;
  }
  void set_rw_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    _rw_timesteps = v;
  }
  void set_rw_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
    _rw_decays = v;
  }

  void cache_recurrent_weights() override;

private:
  struct BPTTWorkspace 
  {
    using AlignedVector = std::vector<double, AlignedAllocator<double, 32>>;
    AlignedVector grad_from_next_all_t;
    AlignedVector d_next_h;
    AlignedVector rnn_grad_matrix;
    AlignedVector chunk_dh_hat;

    void resize(size_t n, size_t batch_chunk_size, size_t num_time_steps)
    {
      grad_from_next_all_t.assign(batch_chunk_size * num_time_steps * n, 0.0);
      d_next_h.assign(batch_chunk_size * n, 0.0);
      rnn_grad_matrix.assign(batch_chunk_size * num_time_steps * n, 0.0);
      chunk_dh_hat.assign(batch_chunk_size * n, 0.0);
    }
  };

  BPTTWorkspace& get_workspace(size_t thread_idx) const;

  void calculate_bptt_batch_chunk(
    size_t start,
    size_t end,
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const Layer& next_layer,
    const std::vector<std::vector<double>>& batch_next_grad_matrix,
    const std::vector<HiddenStates>& batch_hidden_states,
    int bptt_max_ticks,
    BPTTWorkspace& workspace,
    const BPTTWorkspace::AlignedVector& rw_values_T) const;

  void initialize_recurrent_weights(double weight_decay);
  
  // SoA for recurrent weights
  std::vector<double> _rw_values;
  std::vector<double> _rw_grads;
  std::vector<double> _rw_velocities;
  std::vector<double> _rw_m1;
  std::vector<double> _rw_m2;
  std::vector<long long> _rw_timesteps;
  std::vector<double> _rw_decays;

  // Cached transposed recurrent weights
  BPTTWorkspace::AlignedVector _rw_values_T;

  // Per-thread workspaces for BPTT
  mutable std::vector<std::unique_ptr<BPTTWorkspace>> _thread_workspaces;
  mutable std::shared_mutex _workspace_mutex;
};
