#pragma once
#include "layer.h"

#include <array>
#include <vector>


namespace myoddweb::nn
{
// Multi-head causal self-attention block (a small Transformer encoder
// layer): consumes the full T-timestep window of the preceding layer (via
// Layer::calculate_forward_feed's previous_layer.get_rnn_outputs), adds a
// fixed (non-learned) sinusoidal positional encoding, computes Q/K/V
// projections, runs causally-masked scaled dot-product attention per head,
// projects the concatenated heads back to width, adds it as a residual
// (optionally through LayerNorm), runs a position-wise feed-forward
// sub-block, and adds that as a second residual (optionally through a
// second LayerNorm). These two residuals are internal to the block's own
// math and always on - they are unrelated to the base Layer's external
// residual_layer_number/residual_projector mechanism (also supported here,
// forwarded to the base class exactly like FF/Gru/Lstm/Tcn). Output size
// always equals input size (no dimension-changing projection in v1). Always
// strictly causal, with no configuration flag to disable this. Unlike
// AttentionPoolLayer, use_layer_normalisation IS supported here.
class SelfAttentionLayer final : public Layer
{
public:
  // Epsilon used by both internal LayerNorms, matching LSTMLayer's constant.
  static constexpr double LayerNormEpsilon = 1e-5;

  SelfAttentionLayer(
    unsigned layer_index,
    unsigned num_neurons_in_previous_layer,
    unsigned layer_size,
    unsigned number_of_heads,
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

  SelfAttentionLayer(
    unsigned layer_index,
    const Role layer_role,
    const OptimiserType optimiser_type,
    int residual_layer_number,
    unsigned num_neurons,
    unsigned number_of_heads,
    unsigned feed_forward_hidden_size,
    bool use_layer_normalisation,
    const std::vector<Neuron>& neurons,
    const std::vector<double>& b_values,
    const std::vector<double>& b_grads,
    const std::vector<double>& b_velocities,
    const std::vector<double>& b_m1,
    const std::vector<double>& b_m2,
    const std::vector<long long>& b_timesteps,
    const std::vector<double>& b_decays,
    const std::vector<double>& wq_values, const std::vector<double>& wq_grads, const std::vector<double>& wq_velocities, const std::vector<double>& wq_m1, const std::vector<double>& wq_m2, const std::vector<long long>& wq_timesteps, const std::vector<double>& wq_decays,
    const std::vector<double>& bq_values, const std::vector<double>& bq_grads, const std::vector<double>& bq_velocities, const std::vector<double>& bq_m1, const std::vector<double>& bq_m2, const std::vector<long long>& bq_timesteps, const std::vector<double>& bq_decays,
    const std::vector<double>& wk_values, const std::vector<double>& wk_grads, const std::vector<double>& wk_velocities, const std::vector<double>& wk_m1, const std::vector<double>& wk_m2, const std::vector<long long>& wk_timesteps, const std::vector<double>& wk_decays,
    const std::vector<double>& bk_values, const std::vector<double>& bk_grads, const std::vector<double>& bk_velocities, const std::vector<double>& bk_m1, const std::vector<double>& bk_m2, const std::vector<long long>& bk_timesteps, const std::vector<double>& bk_decays,
    const std::vector<double>& wv_values, const std::vector<double>& wv_grads, const std::vector<double>& wv_velocities, const std::vector<double>& wv_m1, const std::vector<double>& wv_m2, const std::vector<long long>& wv_timesteps, const std::vector<double>& wv_decays,
    const std::vector<double>& bv_values, const std::vector<double>& bv_grads, const std::vector<double>& bv_velocities, const std::vector<double>& bv_m1, const std::vector<double>& bv_m2, const std::vector<long long>& bv_timesteps, const std::vector<double>& bv_decays,
    const std::vector<double>& wo_values, const std::vector<double>& wo_grads, const std::vector<double>& wo_velocities, const std::vector<double>& wo_m1, const std::vector<double>& wo_m2, const std::vector<long long>& wo_timesteps, const std::vector<double>& wo_decays,
    const std::vector<double>& bo_values, const std::vector<double>& bo_grads, const std::vector<double>& bo_velocities, const std::vector<double>& bo_m1, const std::vector<double>& bo_m2, const std::vector<long long>& bo_timesteps, const std::vector<double>& bo_decays,
    const std::vector<double>& ff1_w_values, const std::vector<double>& ff1_w_grads, const std::vector<double>& ff1_w_velocities, const std::vector<double>& ff1_w_m1, const std::vector<double>& ff1_w_m2, const std::vector<long long>& ff1_w_timesteps, const std::vector<double>& ff1_w_decays,
    const std::vector<double>& ff1_b_values, const std::vector<double>& ff1_b_grads, const std::vector<double>& ff1_b_velocities, const std::vector<double>& ff1_b_m1, const std::vector<double>& ff1_b_m2, const std::vector<long long>& ff1_b_timesteps, const std::vector<double>& ff1_b_decays,
    const std::vector<double>& ff2_w_values, const std::vector<double>& ff2_w_grads, const std::vector<double>& ff2_w_velocities, const std::vector<double>& ff2_w_m1, const std::vector<double>& ff2_w_m2, const std::vector<long long>& ff2_w_timesteps, const std::vector<double>& ff2_w_decays,
    const std::vector<double>& ff2_b_values, const std::vector<double>& ff2_b_grads, const std::vector<double>& ff2_b_velocities, const std::vector<double>& ff2_b_m1, const std::vector<double>& ff2_b_m2, const std::vector<long long>& ff2_b_timesteps, const std::vector<double>& ff2_b_decays,
    const std::vector<double>& ln1_gain_values, const std::vector<double>& ln1_gain_grads, const std::vector<double>& ln1_gain_velocities, const std::vector<double>& ln1_gain_m1, const std::vector<double>& ln1_gain_m2, const std::vector<long long>& ln1_gain_timesteps, const std::vector<double>& ln1_gain_decays,
    const std::vector<double>& ln1_bias_values, const std::vector<double>& ln1_bias_grads, const std::vector<double>& ln1_bias_velocities, const std::vector<double>& ln1_bias_m1, const std::vector<double>& ln1_bias_m2, const std::vector<long long>& ln1_bias_timesteps, const std::vector<double>& ln1_bias_decays,
    const std::vector<double>& ln2_gain_values, const std::vector<double>& ln2_gain_grads, const std::vector<double>& ln2_gain_velocities, const std::vector<double>& ln2_gain_m1, const std::vector<double>& ln2_gain_m2, const std::vector<long long>& ln2_gain_timesteps, const std::vector<double>& ln2_gain_decays,
    const std::vector<double>& ln2_bias_values, const std::vector<double>& ln2_bias_grads, const std::vector<double>& ln2_bias_velocities, const std::vector<double>& ln2_bias_m1, const std::vector<double>& ln2_bias_m2, const std::vector<long long>& ln2_bias_timesteps, const std::vector<double>& ln2_bias_decays,
    const ResidualProjector* residual_projector,
    int number_of_threads,
    const layer_activation_helper& lah,
    double momentum) noexcept;

  SelfAttentionLayer(const SelfAttentionLayer& src) noexcept;
  SelfAttentionLayer(SelfAttentionLayer&& src) noexcept;
  SelfAttentionLayer& operator=(const SelfAttentionLayer& src) noexcept;
  SelfAttentionLayer& operator=(SelfAttentionLayer&& src) noexcept;
  virtual ~SelfAttentionLayer();

  [[nodiscard]] inline virtual Architecture get_layer_architecture() const override
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return Architecture::SelfAttention;
  }

  [[nodiscard]] inline unsigned get_number_of_heads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _number_of_heads;
  }

  [[nodiscard]] inline unsigned get_feed_forward_hidden_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _feed_forward_hidden_size;
  }

  [[nodiscard]] inline bool get_use_layer_normalisation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _use_layer_normalisation;
  }

  // Full getters for every optimizer-state array of every weight family,
  // required by NeuralNetworkSerializer's save path. Only `values` has a
  // matching public setter (used by tests to inject known weights and by
  // accumulate_swa_average_impl internally) - the other six arrays per
  // family are optimizer/gradient state that NeuralNetworkSerializer's load
  // path restores via the raw-SoA constructor above, exactly like every
  // other layer's deserialize constructor.
    [[nodiscard]] inline const std::vector<double>& get_wq_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wq.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_wq_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wq.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_wq_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wq.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_wq_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wq.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_wq_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wq.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_wq_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wq.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_wq_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wq.decays;
  }
    inline void set_wq_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _wq.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_bq_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bq.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_bq_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bq.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_bq_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bq.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_bq_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bq.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_bq_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bq.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_bq_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bq.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_bq_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bq.decays;
  }
    inline void set_bq_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _bq.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_wk_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wk.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_wk_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wk.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_wk_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wk.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_wk_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wk.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_wk_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wk.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_wk_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wk.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_wk_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wk.decays;
  }
    inline void set_wk_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _wk.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_bk_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bk.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_bk_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bk.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_bk_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bk.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_bk_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bk.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_bk_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bk.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_bk_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bk.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_bk_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bk.decays;
  }
    inline void set_bk_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _bk.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_wv_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wv.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_wv_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wv.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_wv_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wv.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_wv_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wv.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_wv_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wv.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_wv_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wv.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_wv_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wv.decays;
  }
    inline void set_wv_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _wv.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_bv_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bv.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_bv_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bv.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_bv_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bv.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_bv_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bv.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_bv_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bv.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_bv_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bv.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_bv_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bv.decays;
  }
    inline void set_bv_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _bv.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_wo_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wo.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_wo_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wo.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_wo_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wo.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_wo_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wo.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_wo_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wo.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_wo_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wo.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_wo_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _wo.decays;
  }
    inline void set_wo_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _wo.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_bo_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bo.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_bo_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bo.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_bo_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bo.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_bo_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bo.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_bo_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bo.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_bo_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bo.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_bo_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _bo.decays;
  }
    inline void set_bo_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _bo.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_ff1_w_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_w.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_w_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_w.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_w_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_w.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_w_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_w.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_w_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_w.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_ff1_w_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_w.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_w_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_w.decays;
  }
    inline void set_ff1_w_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _ff1_w.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_ff1_b_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_b.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_b_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_b.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_b_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_b.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_b_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_b.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_b_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_b.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_ff1_b_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_b.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff1_b_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff1_b.decays;
  }
    inline void set_ff1_b_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _ff1_b.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_ff2_w_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_w.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_w_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_w.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_w_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_w.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_w_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_w.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_w_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_w.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_ff2_w_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_w.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_w_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_w.decays;
  }
    inline void set_ff2_w_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _ff2_w.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_ff2_b_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_b.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_b_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_b.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_b_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_b.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_b_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_b.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_b_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_b.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_ff2_b_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_b.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_ff2_b_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ff2_b.decays;
  }
    inline void set_ff2_b_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _ff2_b.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_ln1_gain_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_gain.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_gain_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_gain.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_gain_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_gain.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_gain_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_gain.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_gain_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_gain.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_ln1_gain_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_gain.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_gain_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_gain.decays;
  }
    inline void set_ln1_gain_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _ln1_gain.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_ln1_bias_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_bias.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_bias_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_bias.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_bias_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_bias.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_bias_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_bias.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_bias_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_bias.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_ln1_bias_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_bias.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln1_bias_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln1_bias.decays;
  }
    inline void set_ln1_bias_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _ln1_bias.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_ln2_gain_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_gain.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_gain_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_gain.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_gain_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_gain.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_gain_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_gain.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_gain_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_gain.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_ln2_gain_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_gain.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_gain_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_gain.decays;
  }
    inline void set_ln2_gain_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _ln2_gain.values = v;
  }

    [[nodiscard]] inline const std::vector<double>& get_ln2_bias_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_bias.values;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_bias_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_bias.grads;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_bias_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_bias.velocities;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_bias_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_bias.m1;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_bias_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_bias.m2;
  }
    [[nodiscard]] inline const std::vector<long long>& get_ln2_bias_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_bias.timesteps;
  }
    [[nodiscard]] inline const std::vector<double>& get_ln2_bias_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    return _ln2_bias.decays;
  }
    inline void set_ln2_bias_values(const std::vector<double>& v)
  {
    MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
    _ln2_bias.values = v;
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

  void zero_gradients() override;

  Layer* clone() const override;

private:
  struct WeightFamily
  {
    std::vector<double> values, grads, velocities, m1, m2;
    mutable std::vector<long long> timesteps;
    std::vector<double> decays;
  };

  // Every family in declaration order, paired with whether apply_stored_gradients
  // should treat it as bias-like (no weight decay) - mirrors the `is_bias`
  // flag every other layer already passes to apply_update_to_vector.
  struct FamilyRef
  {
    WeightFamily* family;
    bool is_bias;
  };
  [[nodiscard]] std::array<FamilyRef, 16> all_families() noexcept;
  [[nodiscard]] std::array<const WeightFamily*, 16> all_families() const noexcept;

  static void init_family(
    WeightFamily& family,
    size_t num_in,
    size_t num_out,
    double weight_decay,
    const activation& init_activation,
    unsigned block_tag,
    std::optional<uint32_t> seed);

  static void init_bias_family(WeightFamily& family, size_t n, double fill_value);

  // Nullable output pointers for every weight family's gradient accumulator,
  // passed by value into the const self_attention_backward_one below -
  // mirrors AttentionPoolLayer's attention_backward_one nullable-accumulator
  // parameters (scaled from individual params to a struct given the family
  // count), which is how a const method can still accumulate into mutable
  // member storage: the caller (non-const calculate_and_store_gradients)
  // passes plain `double*` values obtained from its own mutable members;
  // calculate_hidden_gradients/_from_output_gradients (which only need
  // `out_dx`) pass a default-constructed (all-nullptr) instance instead.
  struct GradAccumulators
  {
    double* wq = nullptr; double* bq = nullptr;
    double* wk = nullptr; double* bk = nullptr;
    double* wv = nullptr; double* bv = nullptr;
    double* wo = nullptr; double* bo = nullptr;
    double* ff1_w = nullptr; double* ff1_b = nullptr;
    double* ff2_w = nullptr; double* ff2_b = nullptr;
    double* ln1_gain = nullptr; double* ln1_bias = nullptr;
    double* ln2_gain = nullptr; double* ln2_bias = nullptr;
  };

  // Recomputes the full forward pass (positional encoding, Q/K/V, per-head
  // causal attention, output projection, residual+optional-LayerNorm,
  // position-wise FFN, second residual+optional-LayerNorm) for one batch
  // item's T-timestep sequence from `x_seq` (T*d), then backpropagates the
  // already-known local delta `delta` (dL/d(this layer's own output), T*d)
  // through it: writes dL/dx_t for every t into `out_dx` (T*d, fresh write),
  // and accumulates gradients into whichever of `accum`'s pointers are
  // non-null (added, not overwritten, matching this codebase's existing
  // gate-weight-gradient accumulation convention).
  void self_attention_backward_one(
    const double* x_seq,
    size_t T,
    const double* delta,
    double* out_dx,
    const GradAccumulators& accum) const;

  // Shared tail of both calculate_hidden_gradients entry points.
  void finish_hidden_gradients(
    std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
    const std::vector<HiddenStates>& batch_hidden_states,
    size_t batch_size,
    const double* raw_delta_all) const;

  unsigned _number_of_heads;
  unsigned _feed_forward_hidden_size;
  bool _use_layer_normalisation;

  WeightFamily _wq, _bq, _wk, _bk, _wv, _bv, _wo, _bo;
  WeightFamily _ff1_w, _ff1_b, _ff2_w, _ff2_b;
  WeightFamily _ln1_gain, _ln1_bias, _ln2_gain, _ln2_bias;
};

} // namespace myoddweb::nn
