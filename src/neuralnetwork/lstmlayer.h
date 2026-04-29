#pragma once
#include "aligned_allocator.h"
#include "errorcalculation.h"
#include "hiddenstate.h"
#include "layer.h"

#include <vector>

class LSTMLayer final : public Layer
{
protected:
  friend class Layers;

public:
  LSTMLayer(unsigned layer_index,
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

  LSTMLayer(unsigned layer_index,
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

  LSTMLayer(
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
    // Forget Gate (f)
    const std::vector<double>& f_w_values,
    const std::vector<double>& f_w_grads,
    const std::vector<double>& f_w_velocities,
    const std::vector<double>& f_w_m1,
    const std::vector<double>& f_w_m2,
    const std::vector<long long>& f_w_timesteps,
    const std::vector<double>& f_w_decays,
    const std::vector<double>& f_rw_values,
    const std::vector<double>& f_rw_grads,
    const std::vector<double>& f_rw_velocities,
    const std::vector<double>& f_rw_m1,
    const std::vector<double>& f_rw_m2,
    const std::vector<long long>& f_rw_timesteps,
    const std::vector<double>& f_rw_decays,
    const std::vector<double>& f_b_values,
    const std::vector<double>& f_b_grads,
    const std::vector<double>& f_b_velocities,
    const std::vector<double>& f_b_m1,
    const std::vector<double>& f_b_m2,
    const std::vector<long long>& f_b_timesteps,
    const std::vector<double>& f_b_decays,
    // Input Gate (i)
    const std::vector<double>& i_w_values,
    const std::vector<double>& i_w_grads,
    const std::vector<double>& i_w_velocities,
    const std::vector<double>& i_w_m1,
    const std::vector<double>& i_w_m2,
    const std::vector<long long>& i_w_timesteps,
    const std::vector<double>& i_w_decays,
    const std::vector<double>& i_rw_values,
    const std::vector<double>& i_rw_grads,
    const std::vector<double>& i_rw_velocities,
    const std::vector<double>& i_rw_m1,
    const std::vector<double>& i_rw_m2,
    const std::vector<long long>& i_rw_timesteps,
    const std::vector<double>& i_rw_decays,
    const std::vector<double>& i_b_values,
    const std::vector<double>& i_b_grads,
    const std::vector<double>& i_b_velocities,
    const std::vector<double>& i_b_m1,
    const std::vector<double>& i_b_m2,
    const std::vector<long long>& i_b_timesteps,
    const std::vector<double>& i_b_decays,
    // Output Gate (o)
    const std::vector<double>& o_w_values,
    const std::vector<double>& o_w_grads,
    const std::vector<double>& o_w_velocities,
    const std::vector<double>& o_w_m1,
    const std::vector<double>& o_w_m2,
    const std::vector<long long>& o_w_timesteps,
    const std::vector<double>& o_w_decays,
    const std::vector<double>& o_rw_values,
    const std::vector<double>& o_rw_grads,
    const std::vector<double>& o_rw_velocities,
    const std::vector<double>& o_rw_m1,
    const std::vector<double>& o_rw_m2,
    const std::vector<long long>& o_rw_timesteps,
    const std::vector<double>& o_rw_decays,
    const std::vector<double>& o_b_values,
    const std::vector<double>& o_b_grads,
    const std::vector<double>& o_b_velocities,
    const std::vector<double>& o_b_m1,
    const std::vector<double>& o_b_m2,
    const std::vector<long long>& o_b_timesteps,
    const std::vector<double>& o_b_decays,
    const ResidualProjector* residual_projector,
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum
  ) noexcept;

  LSTMLayer(const LSTMLayer& src) noexcept;
  LSTMLayer(LSTMLayer&& src) noexcept;
  LSTMLayer& operator=(const LSTMLayer& src) noexcept;
  LSTMLayer& operator=(LSTMLayer&& src) noexcept;
  virtual ~LSTMLayer();

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return Architecture::Lstm;
  }

public:
  [[nodiscard]] inline bool use_bptt() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return true;
  }

  [[nodiscard]] unsigned get_pre_activation_multiplier() const noexcept override
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return 5;
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
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _rw_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _rw_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _rw_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _rw_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _rw_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_rw_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _rw_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_rw_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _rw_decays;
  }

  // Forget Gate Accessors
  [[nodiscard]] inline const std::vector<double>& get_f_w_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_w_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_w_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_w_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_w_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_w_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_w_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_w_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_w_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_w_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_f_w_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_w_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_w_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_w_decays;
  }

  [[nodiscard]] inline const std::vector<double>& get_f_rw_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_rw_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_rw_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_rw_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_rw_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_rw_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_rw_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_rw_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_rw_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_rw_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_f_rw_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_rw_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_rw_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_rw_decays;
  }

  [[nodiscard]] inline const std::vector<double>& get_f_b_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_b_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_b_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_b_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_b_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_b_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_b_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_b_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_b_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_b_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_f_b_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_b_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_f_b_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _f_b_decays;
  }

  // Input Gate Accessors
  [[nodiscard]] inline const std::vector<double>& get_i_w_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_w_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_w_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_w_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_w_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_w_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_w_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_w_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_w_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_w_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_i_w_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_w_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_w_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_w_decays;
  }

  [[nodiscard]] inline const std::vector<double>& get_i_rw_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_rw_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_rw_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_rw_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_rw_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_rw_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_rw_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_rw_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_rw_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_rw_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_i_rw_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_rw_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_rw_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_rw_decays;
  }

  [[nodiscard]] inline const std::vector<double>& get_i_b_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_b_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_b_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_b_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_b_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_b_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_b_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_b_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_b_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_b_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_i_b_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_b_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_i_b_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _i_b_decays;
  }

  // Output Gate Accessors
  [[nodiscard]] inline const std::vector<double>& get_o_w_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_w_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_w_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_w_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_w_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_w_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_w_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_w_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_w_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_w_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_o_w_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_w_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_w_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_w_decays;
  }

  [[nodiscard]] inline const std::vector<double>& get_o_rw_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_rw_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_rw_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_rw_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_rw_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_rw_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_rw_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_rw_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_rw_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_rw_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_o_rw_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_rw_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_rw_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_rw_decays;
  }

  [[nodiscard]] inline const std::vector<double>& get_o_b_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_b_values;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_b_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_b_grads;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_b_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_b_velocities;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_b_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_b_m1;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_b_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_b_m2;
  }
  [[nodiscard]] inline const std::vector<long long>& get_o_b_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_b_timesteps;
  }
  [[nodiscard]] inline const std::vector<double>& get_o_b_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    return _o_b_decays;
  }

  void set_rw_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _rw_values = v;
  }
  void set_rw_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _rw_grads = v;
  }
  void set_rw_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _rw_velocities = v;
  }
  void set_rw_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _rw_m1 = v;
  }
  void set_rw_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _rw_m2 = v;
  }
  void set_rw_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _rw_timesteps = v;
  }
  void set_rw_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _rw_decays = v;
  }

  void set_f_w_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_w_values = v;
  }
  void set_f_w_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_w_grads = v;
  }
  void set_f_w_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_w_velocities = v;
  }
  void set_f_w_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_w_m1 = v;
  }
  void set_f_w_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_w_m2 = v;
  }
  void set_f_w_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_w_timesteps = v;
  }
  void set_f_w_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_w_decays = v;
  }

  void set_f_rw_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_rw_values = v;
  }
  void set_f_rw_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_rw_grads = v;
  }
  void set_f_rw_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_rw_velocities = v;
  }
  void set_f_rw_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_rw_m1 = v;
  }
  void set_f_rw_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_rw_m2 = v;
  }
  void set_f_rw_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_rw_timesteps = v;
  }
  void set_f_rw_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_rw_decays = v;
  }

  void set_f_b_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_b_values = v;
  }
  void set_f_b_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_b_grads = v;
  }
  void set_f_b_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_b_velocities = v;
  }
  void set_f_b_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_b_m1 = v;
  }
  void set_f_b_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_b_m2 = v;
  }
  void set_f_b_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_b_timesteps = v;
  }
  void set_f_b_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _f_b_decays = v;
  }

  void set_i_w_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_w_values = v;
  }
  void set_i_w_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_w_grads = v;
  }
  void set_i_w_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_w_velocities = v;
  }
  void set_i_w_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_w_m1 = v;
  }
  void set_i_w_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_w_m2 = v;
  }
  void set_i_w_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_w_timesteps = v;
  }
  void set_i_w_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_w_decays = v;
  }

  void set_i_rw_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_rw_values = v;
  }
  void set_i_rw_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_rw_grads = v;
  }
  void set_i_rw_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_rw_velocities = v;
  }
  void set_i_rw_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_rw_m1 = v;
  }
  void set_i_rw_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_rw_m2 = v;
  }
  void set_i_rw_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_rw_timesteps = v;
  }
  void set_i_rw_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_rw_decays = v;
  }

  void set_i_b_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_b_values = v;
  }
  void set_i_b_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_b_grads = v;
  }
  void set_i_b_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_b_velocities = v;
  }
  void set_i_b_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_b_m1 = v;
  }
  void set_i_b_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_b_m2 = v;
  }
  void set_i_b_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_b_timesteps = v;
  }
  void set_i_b_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _i_b_decays = v;
  }

  void set_o_w_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_w_values = v;
  }
  void set_o_w_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_w_grads = v;
  }
  void set_o_w_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_w_velocities = v;
  }
  void set_o_w_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_w_m1 = v;
  }
  void set_o_w_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_w_m2 = v;
  }
  void set_o_w_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_w_timesteps = v;
  }
  void set_o_w_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_w_decays = v;
  }

  void set_o_rw_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_rw_values = v;
  }
  void set_o_rw_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_rw_grads = v;
  }
  void set_o_rw_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_rw_velocities = v;
  }
  void set_o_rw_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_rw_m1 = v;
  }
  void set_o_rw_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_rw_m2 = v;
  }
  void set_o_rw_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_rw_timesteps = v;
  }
  void set_o_rw_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_rw_decays = v;
  }

  void set_o_b_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_b_values = v;
  }
  void set_o_b_grads(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_b_grads = v;
  }
  void set_o_b_velocities(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_b_velocities = v;
  }
  void set_o_b_m1(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_b_m1 = v;
  }
  void set_o_b_m2(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_b_m2 = v;
  }
  void set_o_b_timesteps(const std::vector<long long>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_b_timesteps = v;
  }
  void set_o_b_decays(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("LSTMLayer");
    _o_b_decays = v;
  }

private:
  struct BPTTWorkspace
  {
    using AlignedVector = std::vector<double, AlignedAllocator<double, 32>>;
    AlignedVector grad_from_next_all_t;
    AlignedVector d_next_h;
    AlignedVector d_next_c;
    AlignedVector rnn_grad_matrix;
    AlignedVector dx_matrix;
    AlignedVector chunk_df;
    AlignedVector chunk_di;
    AlignedVector chunk_do;
    AlignedVector chunk_dg;
    AlignedVector f_vals;
    AlignedVector i_vals;
    AlignedVector o_vals;
    AlignedVector g_vals;
    AlignedVector c_vals;
    AlignedVector c_prev_vals;
    AlignedVector tanh_c_vals;

    // For Tiling
    AlignedVector temp_Uf_T_df;
    AlignedVector temp_Ui_T_di;
    AlignedVector temp_Uo_T_do;
    AlignedVector temp_Ug_T_dg;

    void resize(size_t n, size_t n_prev, size_t batch_chunk_size, size_t num_time_steps)
    {
      grad_from_next_all_t.assign(batch_chunk_size * num_time_steps * n, 0.0);
      d_next_h.assign(batch_chunk_size * n, 0.0);
      d_next_c.assign(batch_chunk_size * n, 0.0);
      rnn_grad_matrix.assign(batch_chunk_size * num_time_steps * 4 * n, 0.0);
      dx_matrix.assign(batch_chunk_size * num_time_steps * n_prev, 0.0);
      chunk_df.assign(batch_chunk_size * n, 0.0);
      chunk_di.assign(batch_chunk_size * n, 0.0);
      chunk_do.assign(batch_chunk_size * n, 0.0);
      chunk_dg.assign(batch_chunk_size * n, 0.0);
      f_vals.assign(n, 0.0);
      i_vals.assign(n, 0.0);
      o_vals.assign(n, 0.0);
      g_vals.assign(n, 0.0);
      c_vals.assign(n, 0.0);
      c_prev_vals.assign(n, 0.0);
      tanh_c_vals.assign(n, 0.0);
      temp_Uf_T_df.assign(batch_chunk_size * n, 0.0);
      temp_Ui_T_di.assign(batch_chunk_size * n, 0.0);
      temp_Uo_T_do.assign(batch_chunk_size * n, 0.0);
      temp_Ug_T_dg.assign(batch_chunk_size * n, 0.0);
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
    const BPTTWorkspace::AlignedVector& f_rw_values_T,
    const BPTTWorkspace::AlignedVector& i_rw_values_T,
    const BPTTWorkspace::AlignedVector& o_rw_values_T) const;

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

  // SoA for recurrent weights (Cell Candidate g)
  std::vector<double> _rw_values;
  std::vector<double> _rw_grads;
  std::vector<double> _rw_velocities;
  std::vector<double> _rw_m1;
  std::vector<double> _rw_m2;
  std::vector<long long> _rw_timesteps;
  std::vector<double> _rw_decays;

  // --- Forget Gate (f) ---
  std::vector<double> _f_w_values, _f_w_grads, _f_w_velocities, _f_w_m1, _f_w_m2, _f_w_decays;
  std::vector<long long> _f_w_timesteps;
  std::vector<double> _f_rw_values, _f_rw_grads, _f_rw_velocities, _f_rw_m1, _f_rw_m2, _f_rw_decays;
  std::vector<long long> _f_rw_timesteps;
  std::vector<double> _f_b_values, _f_b_grads, _f_b_velocities, _f_b_m1, _f_b_m2, _f_b_decays;
  std::vector<long long> _f_b_timesteps;

  // --- Input Gate (i) ---
  std::vector<double> _i_w_values, _i_w_grads, _i_w_velocities, _i_w_m1, _i_w_m2, _i_w_decays;
  std::vector<long long> _i_w_timesteps;
  std::vector<double> _i_rw_values, _i_rw_grads, _i_rw_velocities, _i_rw_m1, _i_rw_m2, _i_rw_decays;
  std::vector<long long> _i_rw_timesteps;
  std::vector<double> _i_b_values, _i_b_grads, _i_b_velocities, _i_b_m1, _i_b_m2, _i_b_decays;
  std::vector<long long> _i_b_timesteps;

  // --- Output Gate (o) ---
  std::vector<double> _o_w_values, _o_w_grads, _o_w_velocities, _o_w_m1, _o_w_m2, _o_w_decays;
  std::vector<long long> _o_w_timesteps;
  std::vector<double> _o_rw_values, _o_rw_grads, _o_rw_velocities, _o_rw_m1, _o_rw_m2, _o_rw_decays;
  std::vector<long long> _o_rw_timesteps;
  std::vector<double> _o_b_values, _o_b_grads, _o_b_velocities, _o_b_m1, _o_b_m2, _o_b_decays;
  std::vector<long long> _o_b_timesteps;

  // Cached transposed recurrent weights
  BPTTWorkspace::AlignedVector _rw_values_T;
  BPTTWorkspace::AlignedVector _f_rw_values_T;
  BPTTWorkspace::AlignedVector _i_rw_values_T;
  BPTTWorkspace::AlignedVector _o_rw_values_T;

  // Per-thread workspaces for BPTT
  std::vector<std::unique_ptr<BPTTWorkspace>> _thread_workspaces;
};

