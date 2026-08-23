#include "../libraries/instrumentor.h"
#include "selfattentionlayer.h"
#include "../common/simd_utils.h"
#include "../common/logger.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace myoddweb::nn
{
namespace
{
// y[o] = (has_bias ? b[o] : 0) + sum_i x[i] * W[i*n_out+o]
void linear_forward(const double* x, const double* W, const double* b, double* y, size_t n_in, size_t n_out, bool has_bias)
{
  for (size_t o = 0; o < n_out; ++o)
  {
    y[o] = has_bias ? b[o] : 0.0;
  }
  for (size_t i = 0; i < n_in; ++i)
  {
    const double xi = x[i];
    const double* w_row = W + i * n_out;
    for (size_t o = 0; o < n_out; ++o)
    {
      y[o] += xi * w_row[o];
    }
  }
}

// dx[i] = sum_o dy[o] * W[i*n_out+o]
void linear_backward_input(const double* dy, const double* W, double* dx, size_t n_in, size_t n_out)
{
  for (size_t i = 0; i < n_in; ++i)
  {
    double sum = 0.0;
    const double* w_row = W + i * n_out;
    for (size_t o = 0; o < n_out; ++o)
    {
      sum += dy[o] * w_row[o];
    }
    dx[i] = sum;
  }
}

// dW[i*n_out+o] += x[i]*dy[o]; db[o] += dy[o] (only into non-null accumulators)
void linear_backward_weights(const double* x, const double* dy, double* dW, double* db, size_t n_in, size_t n_out)
{
  if (dW != nullptr)
  {
    for (size_t i = 0; i < n_in; ++i)
    {
      const double xi = x[i];
      double* w_row = dW + i * n_out;
      for (size_t o = 0; o < n_out; ++o)
      {
        w_row[o] += xi * dy[o];
      }
    }
  }
  if (db != nullptr)
  {
    for (size_t o = 0; o < n_out; ++o)
    {
      db[o] += dy[o];
    }
  }
}

// Fixed (non-learned) sinusoidal positional encoding, added in place:
// xp[t][2k] = x[t][2k] + sin(t / 10000^(2k/d)), xp[t][2k+1] = x[t][2k+1] + cos(t / 10000^(2k/d)).
void add_positional_encoding(double* xp, const double* x, size_t T, size_t d)
{
  std::copy(x, x + T * d, xp);
  if (d == 0)
  {
    return;
  }
  for (size_t t = 0; t < T; ++t)
  {
    double* row = xp + t * d;
    for (size_t i = 0; i < d; ++i)
    {
      const size_t k = i / 2;
      const double exponent = (2.0 * static_cast<double>(k)) / static_cast<double>(d);
      const double denom = std::pow(10000.0, exponent);
      const double angle = static_cast<double>(t) / denom;
      row[i] += (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
    }
  }
}
} // namespace

SelfAttentionLayer::SelfAttentionLayer(
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
  std::optional<uint32_t> seed
) :
  Layer(
    layer_index,
    layer_role,
    layer_activation_helper(activation_method, num_neurons_in_previous_layer, layer_size),
    optimiser_type,
    residual_layer_number,
    create_neurons(dropout_rate, layer_size, seed),
    has_bias,
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * layer_size, weight_decay),
    residual_projector,
    number_of_threads,
    momentum,
    seed
  ),
  _number_of_heads(number_of_heads),
  _feed_forward_hidden_size(feed_forward_hidden_size),
  _use_layer_normalisation(use_layer_normalisation)
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  const size_t d = layer_size;
  const size_t d_ff = feed_forward_hidden_size;
  const activation linear_init(activation::method::linear, 0.0);

  init_family(_wq, d, d, weight_decay, linear_init, 0, seed);
  init_family(_wk, d, d, weight_decay, linear_init, 1, seed);
  init_family(_wv, d, d, weight_decay, linear_init, 2, seed);
  init_family(_wo, d, d, weight_decay, linear_init, 3, seed);
  init_family(_ff1_w, d, d_ff, weight_decay, activation_method, 4, seed);
  init_family(_ff2_w, d_ff, d, weight_decay, linear_init, 5, seed);

  init_bias_family(_bq, d, 0.0);
  init_bias_family(_bk, d, 0.0);
  init_bias_family(_bv, d, 0.0);
  init_bias_family(_bo, d, 0.0);
  init_bias_family(_ff1_b, d_ff, 0.0);
  init_bias_family(_ff2_b, d, 0.0);

  init_bias_family(_ln1_gain, d, 1.0);
  init_bias_family(_ln1_bias, d, 0.0);
  init_bias_family(_ln2_gain, d, 1.0);
  init_bias_family(_ln2_bias, d, 0.0);
}

SelfAttentionLayer::SelfAttentionLayer(
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
  double momentum
) noexcept :
  Layer(
    layer_index,
    layer_role,
    optimiser_type,
    residual_layer_number,
    neurons,
    std::vector<double>(static_cast<size_t>(num_neurons) * num_neurons, 0.0),
    std::vector<double>(static_cast<size_t>(num_neurons) * num_neurons, 0.0),
    std::vector<double>(static_cast<size_t>(num_neurons) * num_neurons, 0.0),
    std::vector<double>(static_cast<size_t>(num_neurons) * num_neurons, 0.0),
    std::vector<double>(static_cast<size_t>(num_neurons) * num_neurons, 0.0),
    std::vector<long long>(static_cast<size_t>(num_neurons) * num_neurons, 0),
    std::vector<double>(static_cast<size_t>(num_neurons) * num_neurons, 0.0),
    b_values,
    b_grads,
    b_velocities,
    b_m1,
    b_m2,
    b_timesteps,
    b_decays,
    residual_projector,
    number_of_threads,
    lah,
    momentum
  ),
  _number_of_heads(number_of_heads),
  _feed_forward_hidden_size(feed_forward_hidden_size),
  _use_layer_normalisation(use_layer_normalisation),
  _wq{ wq_values, wq_grads, wq_velocities, wq_m1, wq_m2, wq_timesteps, wq_decays },
  _bq{ bq_values, bq_grads, bq_velocities, bq_m1, bq_m2, bq_timesteps, bq_decays },
  _wk{ wk_values, wk_grads, wk_velocities, wk_m1, wk_m2, wk_timesteps, wk_decays },
  _bk{ bk_values, bk_grads, bk_velocities, bk_m1, bk_m2, bk_timesteps, bk_decays },
  _wv{ wv_values, wv_grads, wv_velocities, wv_m1, wv_m2, wv_timesteps, wv_decays },
  _bv{ bv_values, bv_grads, bv_velocities, bv_m1, bv_m2, bv_timesteps, bv_decays },
  _wo{ wo_values, wo_grads, wo_velocities, wo_m1, wo_m2, wo_timesteps, wo_decays },
  _bo{ bo_values, bo_grads, bo_velocities, bo_m1, bo_m2, bo_timesteps, bo_decays },
  _ff1_w{ ff1_w_values, ff1_w_grads, ff1_w_velocities, ff1_w_m1, ff1_w_m2, ff1_w_timesteps, ff1_w_decays },
  _ff1_b{ ff1_b_values, ff1_b_grads, ff1_b_velocities, ff1_b_m1, ff1_b_m2, ff1_b_timesteps, ff1_b_decays },
  _ff2_w{ ff2_w_values, ff2_w_grads, ff2_w_velocities, ff2_w_m1, ff2_w_m2, ff2_w_timesteps, ff2_w_decays },
  _ff2_b{ ff2_b_values, ff2_b_grads, ff2_b_velocities, ff2_b_m1, ff2_b_m2, ff2_b_timesteps, ff2_b_decays },
  _ln1_gain{ ln1_gain_values, ln1_gain_grads, ln1_gain_velocities, ln1_gain_m1, ln1_gain_m2, ln1_gain_timesteps, ln1_gain_decays },
  _ln1_bias{ ln1_bias_values, ln1_bias_grads, ln1_bias_velocities, ln1_bias_m1, ln1_bias_m2, ln1_bias_timesteps, ln1_bias_decays },
  _ln2_gain{ ln2_gain_values, ln2_gain_grads, ln2_gain_velocities, ln2_gain_m1, ln2_gain_m2, ln2_gain_timesteps, ln2_gain_decays },
  _ln2_bias{ ln2_bias_values, ln2_bias_grads, ln2_bias_velocities, ln2_bias_m1, ln2_bias_m2, ln2_bias_timesteps, ln2_bias_decays }
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
}

SelfAttentionLayer::SelfAttentionLayer(const SelfAttentionLayer& src) noexcept :
  Layer(src),
  _number_of_heads(src._number_of_heads),
  _feed_forward_hidden_size(src._feed_forward_hidden_size),
  _use_layer_normalisation(src._use_layer_normalisation),
  _wq(src._wq), _bq(src._bq), _wk(src._wk), _bk(src._bk),
  _wv(src._wv), _bv(src._bv), _wo(src._wo), _bo(src._bo),
  _ff1_w(src._ff1_w), _ff1_b(src._ff1_b), _ff2_w(src._ff2_w), _ff2_b(src._ff2_b),
  _ln1_gain(src._ln1_gain), _ln1_bias(src._ln1_bias), _ln2_gain(src._ln2_gain), _ln2_bias(src._ln2_bias)
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
}

SelfAttentionLayer::SelfAttentionLayer(SelfAttentionLayer&& src) noexcept :
  Layer(std::move(src)),
  _number_of_heads(src._number_of_heads),
  _feed_forward_hidden_size(src._feed_forward_hidden_size),
  _use_layer_normalisation(src._use_layer_normalisation),
  _wq(std::move(src._wq)), _bq(std::move(src._bq)), _wk(std::move(src._wk)), _bk(std::move(src._bk)),
  _wv(std::move(src._wv)), _bv(std::move(src._bv)), _wo(std::move(src._wo)), _bo(std::move(src._bo)),
  _ff1_w(std::move(src._ff1_w)), _ff1_b(std::move(src._ff1_b)), _ff2_w(std::move(src._ff2_w)), _ff2_b(std::move(src._ff2_b)),
  _ln1_gain(std::move(src._ln1_gain)), _ln1_bias(std::move(src._ln1_bias)), _ln2_gain(std::move(src._ln2_gain)), _ln2_bias(std::move(src._ln2_bias))
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  src._number_of_heads = 0;
  src._feed_forward_hidden_size = 0;
  src._use_layer_normalisation = false;
}

SelfAttentionLayer& SelfAttentionLayer::operator=(const SelfAttentionLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  if (this != &src)
  {
    Layer::operator=(src);
    _number_of_heads = src._number_of_heads;
    _feed_forward_hidden_size = src._feed_forward_hidden_size;
    _use_layer_normalisation = src._use_layer_normalisation;
    _wq = src._wq; _bq = src._bq; _wk = src._wk; _bk = src._bk;
    _wv = src._wv; _bv = src._bv; _wo = src._wo; _bo = src._bo;
    _ff1_w = src._ff1_w; _ff1_b = src._ff1_b; _ff2_w = src._ff2_w; _ff2_b = src._ff2_b;
    _ln1_gain = src._ln1_gain; _ln1_bias = src._ln1_bias; _ln2_gain = src._ln2_gain; _ln2_bias = src._ln2_bias;
  }
  return *this;
}

SelfAttentionLayer& SelfAttentionLayer::operator=(SelfAttentionLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  if (this != &src)
  {
    Layer::operator=(std::move(src));
    _number_of_heads = src._number_of_heads;
    _feed_forward_hidden_size = src._feed_forward_hidden_size;
    _use_layer_normalisation = src._use_layer_normalisation;
    _wq = std::move(src._wq); _bq = std::move(src._bq); _wk = std::move(src._wk); _bk = std::move(src._bk);
    _wv = std::move(src._wv); _bv = std::move(src._bv); _wo = std::move(src._wo); _bo = std::move(src._bo);
    _ff1_w = std::move(src._ff1_w); _ff1_b = std::move(src._ff1_b); _ff2_w = std::move(src._ff2_w); _ff2_b = std::move(src._ff2_b);
    _ln1_gain = std::move(src._ln1_gain); _ln1_bias = std::move(src._ln1_bias); _ln2_gain = std::move(src._ln2_gain); _ln2_bias = std::move(src._ln2_bias);

    src._number_of_heads = 0;
    src._feed_forward_hidden_size = 0;
    src._use_layer_normalisation = false;
  }
  return *this;
}

SelfAttentionLayer::~SelfAttentionLayer()
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
}

Layer* SelfAttentionLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  return new SelfAttentionLayer(*this);
}

void SelfAttentionLayer::init_family(
  WeightFamily& family,
  size_t num_in,
  size_t num_out,
  double weight_decay,
  const activation& init_activation,
  unsigned block_tag,
  std::optional<uint32_t> seed)
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  const size_t n = num_in * num_out;
  family.values.resize(n);
  for (size_t i = 0; i < num_in; ++i)
  {
    for (size_t j = 0; j < num_out; ++j)
    {
      const size_t flat_index = i * num_out + j;
      const auto weight_seed = seed.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(Rng::derive(seed.value(), block_tag, flat_index))) : std::nullopt;
      family.values[flat_index] = init_activation.weight_initialization(static_cast<unsigned>(num_in), static_cast<unsigned>(num_out), weight_seed);
    }
  }
  family.grads.assign(n, 0.0);
  family.velocities.assign(n, 0.0);
  family.m1.assign(n, 0.0);
  family.m2.assign(n, 0.0);
  family.timesteps.assign(n, 0);
  family.decays.assign(n, weight_decay);
}

void SelfAttentionLayer::init_bias_family(WeightFamily& family, size_t n, double fill_value)
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  family.values.assign(n, fill_value);
  family.grads.assign(n, 0.0);
  family.velocities.assign(n, 0.0);
  family.m1.assign(n, 0.0);
  family.m2.assign(n, 0.0);
  family.timesteps.assign(n, 0);
  family.decays.assign(n, 0.0);
}

std::array<SelfAttentionLayer::FamilyRef, 16> SelfAttentionLayer::all_families() noexcept
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  return { {
    { &_wq, false }, { &_bq, true },
    { &_wk, false }, { &_bk, true },
    { &_wv, false }, { &_bv, true },
    { &_wo, false }, { &_bo, true },
    { &_ff1_w, false }, { &_ff1_b, true },
    { &_ff2_w, false }, { &_ff2_b, true },
    { &_ln1_gain, true }, { &_ln1_bias, true },
    { &_ln2_gain, true }, { &_ln2_bias, true }
  } };
}

std::array<const SelfAttentionLayer::WeightFamily*, 16> SelfAttentionLayer::all_families() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  return { {
    &_wq, &_bq, &_wk, &_bk, &_wv, &_bv, &_wo, &_bo,
    &_ff1_w, &_ff1_b, &_ff2_w, &_ff2_b,
    &_ln1_gain, &_ln1_bias, &_ln2_gain, &_ln2_bias
  } };
}

void SelfAttentionLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
  (void)batch_gradients_and_outputs;
  (void)target_outputs_begin;
  (void)batch_hidden_states;
  (void)batch_size;
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  Logger::panic("SelfAttentionLayer: trying to calculate output gradients on a non-output layer!");
}

void SelfAttentionLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  if (batch_size == 0)
  {
    return;
  }

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const size_t d = get_number_neurons();
  const size_t H = _number_of_heads;
  const size_t head_size = (H > 0) ? d / H : 0;
  const size_t d_ff = _feed_forward_hidden_size;
  const double scale = (head_size > 0) ? 1.0 / std::sqrt(static_cast<double>(head_size)) : 0.0;
  const bool use_bias = has_bias();
  const bool have_hs = !batch_hidden_states.empty();
  const activation& ffn_activation = get_activation_helper().get_activation(0);

  std::vector<double> mask(d);
  std::vector<double> scores;

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const size_t T = (d > 0 && !seq.empty()) ? seq.size() / d : 0;

    if (have_hs && batch_hidden_states[b].at(get_layer_index()).size() != T)
    {
      batch_hidden_states[b].assign(get_layer_index(), T, HiddenState(), get_pre_activation_multiplier());
    }

    if (T == 0)
    {
      batch_gradients_and_outputs[b].set_outputs(get_layer_index(), std::vector<double>(d, 0.0));
      batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), std::vector<double>());
      continue;
    }

    std::vector<double> xp(T * d);
    add_positional_encoding(xp.data(), seq.data(), T, d);

    std::vector<double> Q(T * d), K(T * d), V(T * d);
    for (size_t t = 0; t < T; ++t)
    {
      linear_forward(xp.data() + t * d, _wq.values.data(), _bq.values.data(), Q.data() + t * d, d, d, use_bias);
      linear_forward(xp.data() + t * d, _wk.values.data(), _bk.values.data(), K.data() + t * d, d, d, use_bias);
      linear_forward(xp.data() + t * d, _wv.values.data(), _bv.values.data(), V.data() + t * d, d, d, use_bias);
    }

    std::vector<double> ctx(T * d, 0.0);
    for (size_t m = 0; m < H; ++m)
    {
      const size_t off = m * head_size;
      for (size_t t = 0; t < T; ++t)
      {
        scores.assign(t + 1, 0.0);
        const double* q_row = Q.data() + t * d + off;
        for (size_t t2 = 0; t2 <= t; ++t2)
        {
          const double* k_row = K.data() + t2 * d + off;
          double s = 0.0;
          for (size_t a = 0; a < head_size; ++a)
          {
            s += q_row[a] * k_row[a];
          }
          scores[t2] = s * scale;
        }
        simd::softmax_forward(scores.data(), t + 1);
        double* ctx_row = ctx.data() + t * d + off;
        for (size_t t2 = 0; t2 <= t; ++t2)
        {
          const double a_t2 = scores[t2];
          const double* v_row = V.data() + t2 * d + off;
          for (size_t a = 0; a < head_size; ++a)
          {
            ctx_row[a] += a_t2 * v_row[a];
          }
        }
      }
    }

    std::vector<double> ao(T * d);
    for (size_t t = 0; t < T; ++t)
    {
      linear_forward(ctx.data() + t * d, _wo.values.data(), _bo.values.data(), ao.data() + t * d, d, d, use_bias);
    }

    std::vector<double> y1(T * d);
    for (size_t k = 0; k < T * d; ++k)
    {
      y1[k] = xp[k] + ao[k];
    }

    std::vector<double> y1n(T * d);
    std::vector<double> inv_std1(T, 0.0);
    if (_use_layer_normalisation)
    {
      for (size_t t = 0; t < T; ++t)
      {
        simd::layer_norm_forward(y1.data() + t * d, _ln1_gain.values.data(), _ln1_bias.values.data(), y1n.data() + t * d, d, LayerNormEpsilon, inv_std1[t]);
      }
    }
    else
    {
      y1n = y1;
    }

    std::vector<double> ffh_pre(T * d_ff);
    for (size_t t = 0; t < T; ++t)
    {
      linear_forward(y1n.data() + t * d, _ff1_w.values.data(), _ff1_b.values.data(), ffh_pre.data() + t * d_ff, d, d_ff, use_bias);
    }
    std::vector<double> ffh(ffh_pre);
    ffn_activation.activate(ffh.data(), ffh.data() + T * d_ff, is_training);

    std::vector<double> ffo(T * d);
    for (size_t t = 0; t < T; ++t)
    {
      linear_forward(ffh.data() + t * d_ff, _ff2_w.values.data(), _ff2_b.values.data(), ffo.data() + t * d, d_ff, d, use_bias);
    }

    std::vector<double> y2(T * d);
    for (size_t k = 0; k < T * d; ++k)
    {
      y2[k] = y1n[k] + ffo[k];
    }

    if (!batch_residual_output_values.empty() && b < batch_residual_output_values.size() && batch_residual_output_values[b].size() == d)
    {
      simd::add_vectors(batch_residual_output_values[b].data(), y2.data() + (T - 1) * d, d);
    }

    std::vector<double> y2n(T * d);
    std::vector<double> inv_std2(T, 0.0);
    if (_use_layer_normalisation)
    {
      for (size_t t = 0; t < T; ++t)
      {
        simd::layer_norm_forward(y2.data() + t * d, _ln2_gain.values.data(), _ln2_bias.values.data(), y2n.data() + t * d, d, LayerNormEpsilon, inv_std2[t]);
      }
    }
    else
    {
      y2n = y2;
    }

    std::vector<double> out_seq(T * d);
    for (size_t t = 0; t < T; ++t)
    {
      std::fill(mask.begin(), mask.end(), 1.0);
      double* row = out_seq.data() + t * d;
      std::copy(y2n.data() + t * d, y2n.data() + t * d + d, row);
      if (is_training && get_dropout() > 0.0)
      {
        const auto& neurons = get_neurons();
        for (size_t o = 0; o < d; ++o)
        {
          const auto& neuron = neurons[o];
          if (neuron.is_dropout())
          {
            if (neuron.must_randomly_drop(b * T + t))
            {
              row[o] = 0.0;
              mask[o] = 0.0;
            }
            else
            {
              const double sc = 1.0 / (1.0 - neuron.get_dropout_rate());
              row[o] *= sc;
              mask[o] = sc;
            }
          }
        }
      }
      if (have_hs)
      {
        auto& hs_row = batch_hidden_states[b].at(get_layer_index());
        hs_row[t].set_pre_activation_sums(y2n.data() + t * d, d);
        hs_row[t].set_hidden_state_values(row, d);
        hs_row[t].set_cell_state_values(mask.data(), d);
      }
    }

    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), out_seq.data() + (T - 1) * d, d);
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), out_seq.data(), out_seq.size());
  }
}

void SelfAttentionLayer::self_attention_backward_one(
  const double* x_seq,
  size_t T,
  const double* delta,
  double* out_dx,
  const GradAccumulators& accum) const
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  const size_t d = get_number_neurons();
  const size_t H = _number_of_heads;
  const size_t head_size = (H > 0) ? d / H : 0;
  const size_t d_ff = _feed_forward_hidden_size;
  const double scale = (head_size > 0) ? 1.0 / std::sqrt(static_cast<double>(head_size)) : 0.0;
  const bool use_bias = has_bias();
  const activation& ffn_activation = get_activation_helper().get_activation(0);

  if (T == 0 || d == 0)
  {
    return;
  }

  // simd::layer_norm_backward always writes through dgain_accum/dbias_accum
  // (no null-check, unlike this method's own nullable GradAccumulators
  // convention) - when the caller passed nullptr (calculate_hidden_gradients
  // / calculate_hidden_gradients_from_output_gradients only need out_dx, not
  // weight gradients), fall back to scratch buffers so the call is always
  // given valid storage to write into, and simply discard the result.
  std::vector<double> ln1_gain_scratch, ln1_bias_scratch, ln2_gain_scratch, ln2_bias_scratch;
  double* ln1_gain_accum = accum.ln1_gain;
  double* ln1_bias_accum = accum.ln1_bias;
  double* ln2_gain_accum = accum.ln2_gain;
  double* ln2_bias_accum = accum.ln2_bias;
  if (_use_layer_normalisation)
  {
    if (ln1_gain_accum == nullptr) { ln1_gain_scratch.assign(d, 0.0); ln1_gain_accum = ln1_gain_scratch.data(); }
    if (ln1_bias_accum == nullptr) { ln1_bias_scratch.assign(d, 0.0); ln1_bias_accum = ln1_bias_scratch.data(); }
    if (ln2_gain_accum == nullptr) { ln2_gain_scratch.assign(d, 0.0); ln2_gain_accum = ln2_gain_scratch.data(); }
    if (ln2_bias_accum == nullptr) { ln2_bias_scratch.assign(d, 0.0); ln2_bias_accum = ln2_bias_scratch.data(); }
  }

  // --- Recompute forward, caching every intermediate needed for backward ---
  std::vector<double> xp(T * d);
  add_positional_encoding(xp.data(), x_seq, T, d);

  std::vector<double> Q(T * d), K(T * d), V(T * d);
  for (size_t t = 0; t < T; ++t)
  {
    linear_forward(xp.data() + t * d, _wq.values.data(), _bq.values.data(), Q.data() + t * d, d, d, use_bias);
    linear_forward(xp.data() + t * d, _wk.values.data(), _bk.values.data(), K.data() + t * d, d, d, use_bias);
    linear_forward(xp.data() + t * d, _wv.values.data(), _bv.values.data(), V.data() + t * d, d, d, use_bias);
  }

  // alpha[m][t*T+t2] for t2<=t, 0 elsewhere.
  std::vector<double> alpha(H * T * T, 0.0);
  std::vector<double> ctx(T * d, 0.0);
  std::vector<double> scores;
  for (size_t m = 0; m < H; ++m)
  {
    const size_t off = m * head_size;
    double* alpha_head = alpha.data() + m * T * T;
    for (size_t t = 0; t < T; ++t)
    {
      scores.assign(t + 1, 0.0);
      const double* q_row = Q.data() + t * d + off;
      for (size_t t2 = 0; t2 <= t; ++t2)
      {
        const double* k_row = K.data() + t2 * d + off;
        double s = 0.0;
        for (size_t a = 0; a < head_size; ++a)
        {
          s += q_row[a] * k_row[a];
        }
        scores[t2] = s * scale;
      }
      simd::softmax_forward(scores.data(), t + 1);
      std::copy(scores.begin(), scores.end(), alpha_head + t * T);
      double* ctx_row = ctx.data() + t * d + off;
      for (size_t t2 = 0; t2 <= t; ++t2)
      {
        const double a_t2 = scores[t2];
        const double* v_row = V.data() + t2 * d + off;
        for (size_t a = 0; a < head_size; ++a)
        {
          ctx_row[a] += a_t2 * v_row[a];
        }
      }
    }
  }

  std::vector<double> ao(T * d);
  for (size_t t = 0; t < T; ++t)
  {
    linear_forward(ctx.data() + t * d, _wo.values.data(), _bo.values.data(), ao.data() + t * d, d, d, use_bias);
  }

  std::vector<double> y1(T * d);
  for (size_t k = 0; k < T * d; ++k)
  {
    y1[k] = xp[k] + ao[k];
  }

  std::vector<double> y1n(T * d);
  std::vector<double> inv_std1(T, 0.0);
  if (_use_layer_normalisation)
  {
    for (size_t t = 0; t < T; ++t)
    {
      simd::layer_norm_forward(y1.data() + t * d, _ln1_gain.values.data(), _ln1_bias.values.data(), y1n.data() + t * d, d, LayerNormEpsilon, inv_std1[t]);
    }
  }
  else
  {
    y1n = y1;
  }

  std::vector<double> ffh_pre(T * d_ff);
  for (size_t t = 0; t < T; ++t)
  {
    linear_forward(y1n.data() + t * d, _ff1_w.values.data(), _ff1_b.values.data(), ffh_pre.data() + t * d_ff, d, d_ff, use_bias);
  }
  std::vector<double> ffh(ffh_pre);
  ffn_activation.activate(ffh.data(), ffh.data() + T * d_ff, false);

  // --- Backward ---
  std::vector<double> dy2(T * d);
  if (_use_layer_normalisation)
  {
    for (size_t t = 0; t < T; ++t)
    {
      // Recompute y2n[t] (needed by layer_norm_backward to recover a_hat) from y1n[t]+ffo[t].
      std::vector<double> ffo_t(d);
      linear_forward(ffh.data() + t * d_ff, _ff2_w.values.data(), _ff2_b.values.data(), ffo_t.data(), d_ff, d, use_bias);
      std::vector<double> y2_t(d);
      for (size_t o = 0; o < d; ++o)
      {
        y2_t[o] = y1n[t * d + o] + ffo_t[o];
      }
      double inv_std2_t;
      std::vector<double> y2n_t(d);
      simd::layer_norm_forward(y2_t.data(), _ln2_gain.values.data(), _ln2_bias.values.data(), y2n_t.data(), d, LayerNormEpsilon, inv_std2_t);
      simd::layer_norm_backward(delta + t * d, y2n_t.data(), _ln2_gain.values.data(), _ln2_bias.values.data(), inv_std2_t, d, dy2.data() + t * d, ln2_gain_accum, ln2_bias_accum);
    }
  }
  else
  {
    std::copy(delta, delta + T * d, dy2.begin());
  }

  // y2 = y1n + ffo: both branches receive the full dy2.
  std::vector<double> d_ffo(dy2);

  std::vector<double> d_y1n(T * d, 0.0);
  for (size_t t = 0; t < T; ++t)
  {
    std::vector<double> d_ffh(d_ff);
    linear_backward_input(d_ffo.data() + t * d, _ff2_w.values.data(), d_ffh.data(), d_ff, d);
    linear_backward_weights(ffh.data() + t * d_ff, d_ffo.data() + t * d, accum.ff2_w, accum.ff2_b, d_ff, d);

    std::vector<double> deriv(d_ff);
    ffn_activation.activate_derivative(ffh_pre.data() + t * d_ff, ffh_pre.data() + t * d_ff + d_ff, ffh.data() + t * d_ff, deriv.data());
    std::vector<double> d_ffh_pre(d_ff);
    for (size_t k = 0; k < d_ff; ++k)
    {
      d_ffh_pre[k] = d_ffh[k] * deriv[k];
    }

    std::vector<double> d_y1n_from_ffn(d);
    linear_backward_input(d_ffh_pre.data(), _ff1_w.values.data(), d_y1n_from_ffn.data(), d, d_ff);
    linear_backward_weights(y1n.data() + t * d, d_ffh_pre.data(), accum.ff1_w, accum.ff1_b, d, d_ff);

    for (size_t o = 0; o < d; ++o)
    {
      d_y1n[t * d + o] = dy2[t * d + o] + d_y1n_from_ffn[o];
    }
  }

  std::vector<double> dy1(T * d);
  if (_use_layer_normalisation)
  {
    for (size_t t = 0; t < T; ++t)
    {
      double dummy_inv_std;
      std::vector<double> y1n_t(d);
      simd::layer_norm_forward(y1.data() + t * d, _ln1_gain.values.data(), _ln1_bias.values.data(), y1n_t.data(), d, LayerNormEpsilon, dummy_inv_std);
      simd::layer_norm_backward(d_y1n.data() + t * d, y1n_t.data(), _ln1_gain.values.data(), _ln1_bias.values.data(), dummy_inv_std, d, dy1.data() + t * d, ln1_gain_accum, ln1_bias_accum);
    }
  }
  else
  {
    dy1 = d_y1n;
  }

  // y1 = xp + ao: both branches receive the full dy1.
  std::vector<double> d_ao(dy1);
  std::vector<double> d_xp(T * d, 0.0);
  for (size_t k = 0; k < T * d; ++k)
  {
    d_xp[k] = dy1[k];
  }

  std::vector<double> d_ctx(T * d);
  for (size_t t = 0; t < T; ++t)
  {
    linear_backward_input(d_ao.data() + t * d, _wo.values.data(), d_ctx.data() + t * d, d, d);
    linear_backward_weights(ctx.data() + t * d, d_ao.data() + t * d, accum.wo, accum.bo, d, d);
  }

  std::vector<double> d_Q(T * d, 0.0), d_K(T * d, 0.0), d_V(T * d, 0.0);
  std::vector<double> d_alpha, d_scores;
  for (size_t m = 0; m < H; ++m)
  {
    const size_t off = m * head_size;
    const double* alpha_head = alpha.data() + m * T * T;
    for (size_t t = 0; t < T; ++t)
    {
      const double* d_ctx_row = d_ctx.data() + t * d + off;
      d_alpha.assign(t + 1, 0.0);
      for (size_t t2 = 0; t2 <= t; ++t2)
      {
        double* v_row = V.data() + t2 * d + off;
        double dot = 0.0;
        for (size_t a = 0; a < head_size; ++a)
        {
          d_V[t2 * d + off + a] += alpha_head[t * T + t2] * d_ctx_row[a];
          dot += d_ctx_row[a] * v_row[a];
        }
        d_alpha[t2] = dot;
      }
      d_scores.assign(t + 1, 0.0);
      simd::softmax_backward(alpha_head + t * T, d_alpha.data(), d_scores.data(), t + 1);

      double* q_row = Q.data() + t * d + off;
      for (size_t t2 = 0; t2 <= t; ++t2)
      {
        const double ds = d_scores[t2] * scale;
        double* k_row = K.data() + t2 * d + off;
        for (size_t a = 0; a < head_size; ++a)
        {
          d_Q[t * d + off + a] += ds * k_row[a];
          d_K[t2 * d + off + a] += ds * q_row[a];
        }
      }
    }
  }

  for (size_t t = 0; t < T; ++t)
  {
    std::vector<double> d_xp_q(d), d_xp_k(d), d_xp_v(d);
    linear_backward_input(d_Q.data() + t * d, _wq.values.data(), d_xp_q.data(), d, d);
    linear_backward_weights(xp.data() + t * d, d_Q.data() + t * d, accum.wq, accum.bq, d, d);
    linear_backward_input(d_K.data() + t * d, _wk.values.data(), d_xp_k.data(), d, d);
    linear_backward_weights(xp.data() + t * d, d_K.data() + t * d, accum.wk, accum.bk, d, d);
    linear_backward_input(d_V.data() + t * d, _wv.values.data(), d_xp_v.data(), d, d);
    linear_backward_weights(xp.data() + t * d, d_V.data() + t * d, accum.wv, accum.bv, d, d);

    for (size_t o = 0; o < d; ++o)
    {
      // Positional encoding is a constant offset added to x, so d(x)=d(xp).
      out_dx[t * d + o] = d_xp[t * d + o] + d_xp_q[o] + d_xp_k[o] + d_xp_v[o];
    }
  }
}

void SelfAttentionLayer::finish_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  const double* raw_delta_all) const
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  const unsigned this_layer_index = get_layer_index();
  const unsigned prev_layer_index = this_layer_index - 1;
  const size_t d = get_number_neurons();

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& hs_row = batch_hidden_states[b].at(this_layer_index);
    const size_t T = hs_row.size();
    if (T == 0)
    {
      continue;
    }

    std::vector<double> delta(T * d);
    for (size_t t = 0; t < T; ++t)
    {
      const double* raw = raw_delta_all + (b * T + t) * d;
      const double* mask_vals = hs_row[t].get_cell_state_values().data();
      for (size_t o = 0; o < d; ++o)
      {
        delta[t * d + o] = raw[o] * mask_vals[o];
      }
    }

    batch_gradients_and_outputs[b].set_gradients(this_layer_index, delta.data() + (T - 1) * d, d);
    batch_gradients_and_outputs[b].set_rnn_gate_gradients(this_layer_index, delta.data(), delta.size());

    const auto& seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (d == 0 || seq.size() != T * d)
    {
      continue;
    }

    std::vector<double> out_dx(T * d, 0.0);
    self_attention_backward_one(seq.data(), T, delta.data(), out_dx.data(), GradAccumulators{});
    batch_gradients_and_outputs[b].set_rnn_gradients(this_layer_index, out_dx.data(), out_dx.size());
  }
}

void SelfAttentionLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t d = get_number_neurons();
  const size_t N_next = next_layer.get_number_neurons();
  const unsigned next_layer_index = next_layer.get_layer_index();
  const bool use_direct_gradients = batch_next_grad_matrix.empty();

  std::vector<double> raw_delta_all;
  for (size_t b = 0; b < batch_size; ++b)
  {
    const size_t T = batch_hidden_states[b].at(get_layer_index()).size();
    const size_t base = raw_delta_all.size();
    raw_delta_all.resize(base + T * d, 0.0);
    if (T == 0 || N_next == 0)
    {
      continue;
    }

    std::span<const double> src;
    if (use_direct_gradients)
    {
      src = batch_gradients_and_outputs[b].get_gradients(next_layer_index);
    }
    else if (b < batch_next_grad_matrix.size())
    {
      src = std::span<const double>(batch_next_grad_matrix[b].data(), batch_next_grad_matrix[b].size());
    }
    if (src.empty())
    {
      continue;
    }

    const size_t T_avail = (src.size() == N_next) ? T : std::min(T, src.size() / N_next);
    const double* W_next = next_layer.get_w_values().data();
    for (size_t t = 0; t < T_avail; ++t)
    {
      const double* nd = (src.size() == N_next) ? src.data() : (src.data() + t * N_next);
      double* out_row = raw_delta_all.data() + base + t * d;
      simd::gemm_transposed_one_batch(nd, W_next, out_row, d, N_next);
    }
  }

  finish_hidden_gradients(batch_gradients_and_outputs, batch_hidden_states, batch_size, raw_delta_all.data());
}

void SelfAttentionLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t d = get_number_neurons();
  const unsigned this_layer_index = get_layer_index();
  const bool use_direct_gradients = batch_output_gradients.empty();

  std::vector<double> raw_delta_all;
  for (size_t b = 0; b < batch_size; ++b)
  {
    const size_t T = batch_hidden_states[b].at(this_layer_index).size();
    const size_t base = raw_delta_all.size();
    raw_delta_all.resize(base + T * d, 0.0);
    if (T == 0)
    {
      continue;
    }

    std::span<const double> src;
    if (use_direct_gradients)
    {
      const auto& rnn_g = batch_gradients_and_outputs[b].get_rnn_gradients(this_layer_index + 1);
      src = !rnn_g.empty() ? std::span<const double>(rnn_g.data(), rnn_g.size()) : batch_gradients_and_outputs[b].get_gradients(this_layer_index + 1);
    }
    else if (b < batch_output_gradients.size())
    {
      src = std::span<const double>(batch_output_gradients[b].data(), batch_output_gradients[b].size());
    }
    if (src.empty())
    {
      continue;
    }

    double* out_row = raw_delta_all.data() + base;
    if (src.size() == d)
    {
      for (size_t t = 0; t < T; ++t)
      {
        std::copy(src.begin(), src.end(), out_row + t * d);
      }
    }
    else
    {
      const size_t copy_size = std::min(src.size(), T * d);
      std::copy(src.data(), src.data() + copy_size, out_row);
    }
  }

  finish_hidden_gradients(batch_gradients_and_outputs, batch_hidden_states, batch_size, raw_delta_all.data());
}

void SelfAttentionLayer::calculate_and_store_gradients(
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& hidden_states,
  const Layer& previous_layer,
  size_t batch_size,
  int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  (void)hidden_states;
  zero_gradients();
  if (batch_size == 0)
  {
    return;
  }

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const unsigned this_layer_index = get_layer_index();
  const size_t d = get_number_neurons();

  GradAccumulators accum;
  accum.wq = _wq.grads.data(); accum.bq = _bq.grads.data();
  accum.wk = _wk.grads.data(); accum.bk = _bk.grads.data();
  accum.wv = _wv.grads.data(); accum.bv = _bv.grads.data();
  accum.wo = _wo.grads.data(); accum.bo = _bo.grads.data();
  accum.ff1_w = _ff1_w.grads.data(); accum.ff1_b = _ff1_b.grads.data();
  accum.ff2_w = _ff2_w.grads.data(); accum.ff2_b = _ff2_b.grads.data();
  if (_use_layer_normalisation)
  {
    accum.ln1_gain = _ln1_gain.grads.data(); accum.ln1_bias = _ln1_bias.grads.data();
    accum.ln2_gain = _ln2_gain.grads.data(); accum.ln2_bias = _ln2_bias.grads.data();
  }

  size_t contributing = 0;
  std::vector<double> out_dx_scratch;
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& own_delta = batch_gradients_and_outputs[b].get_rnn_gate_gradients(this_layer_index);
    const auto& seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const size_t T = (d > 0 && !own_delta.empty()) ? own_delta.size() / d : 0;
    if (T == 0 || seq.size() != T * d)
    {
      continue;
    }

    out_dx_scratch.assign(T * d, 0.0);
    self_attention_backward_one(seq.data(), T, own_delta.data(), out_dx_scratch.data(), accum);
    ++contributing;
  }

  if (contributing > 0)
  {
    const double inv = 1.0 / static_cast<double>(contributing);
    for (auto& fr : all_families())
    {
      simd::scale_vector(fr.family->grads.data(), inv, fr.family->grads.size());
    }
  }
}

double SelfAttentionLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  double norm_sq = 0.0;
  for (const auto* f : all_families())
  {
    norm_sq += simd::sum_sq(f->grads.data(), f->grads.size());
  }
  return norm_sq;
}

void SelfAttentionLayer::accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count)
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  const auto& other = static_cast<const SelfAttentionLayer&>(snapshot);
  auto mine = all_families();
  auto theirs = other.all_families();
  for (size_t i = 0; i < mine.size(); ++i)
  {
    swa_average_into(mine[i].family->values, theirs[i]->values, existing_swa_count);
  }
}

void SelfAttentionLayer::update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  auto& other = static_cast<SelfAttentionLayer&>(fast_layer);
  auto mine = all_families();
  auto theirs = other.all_families();
  for (size_t i = 0; i < mine.size(); ++i)
  {
    simd::lookahead_step(mine[i].family->values.data(), theirs[i].family->values.data(), alpha, mine[i].family->values.size());
  }
}

void SelfAttentionLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  apply_update_to_vector(_wq.values, _wq.grads, _wq.velocities, _wq.m1, _wq.m2, _wq.timesteps, _wq.decays, learning_rate, clipping_scale, false, _optimiser_type);
  apply_update_to_vector(_wk.values, _wk.grads, _wk.velocities, _wk.m1, _wk.m2, _wk.timesteps, _wk.decays, learning_rate, clipping_scale, false, _optimiser_type);
  apply_update_to_vector(_wv.values, _wv.grads, _wv.velocities, _wv.m1, _wv.m2, _wv.timesteps, _wv.decays, learning_rate, clipping_scale, false, _optimiser_type);
  apply_update_to_vector(_wo.values, _wo.grads, _wo.velocities, _wo.m1, _wo.m2, _wo.timesteps, _wo.decays, learning_rate, clipping_scale, false, _optimiser_type);
  apply_update_to_vector(_ff1_w.values, _ff1_w.grads, _ff1_w.velocities, _ff1_w.m1, _ff1_w.m2, _ff1_w.timesteps, _ff1_w.decays, learning_rate, clipping_scale, false, _optimiser_type);
  apply_update_to_vector(_ff2_w.values, _ff2_w.grads, _ff2_w.velocities, _ff2_w.m1, _ff2_w.m2, _ff2_w.timesteps, _ff2_w.decays, learning_rate, clipping_scale, false, _optimiser_type);

  if (has_bias())
  {
    apply_update_to_vector(_bq.values, _bq.grads, _bq.velocities, _bq.m1, _bq.m2, _bq.timesteps, _bq.decays, learning_rate, clipping_scale, true, _optimiser_type);
    apply_update_to_vector(_bk.values, _bk.grads, _bk.velocities, _bk.m1, _bk.m2, _bk.timesteps, _bk.decays, learning_rate, clipping_scale, true, _optimiser_type);
    apply_update_to_vector(_bv.values, _bv.grads, _bv.velocities, _bv.m1, _bv.m2, _bv.timesteps, _bv.decays, learning_rate, clipping_scale, true, _optimiser_type);
    apply_update_to_vector(_bo.values, _bo.grads, _bo.velocities, _bo.m1, _bo.m2, _bo.timesteps, _bo.decays, learning_rate, clipping_scale, true, _optimiser_type);
    apply_update_to_vector(_ff1_b.values, _ff1_b.grads, _ff1_b.velocities, _ff1_b.m1, _ff1_b.m2, _ff1_b.timesteps, _ff1_b.decays, learning_rate, clipping_scale, true, _optimiser_type);
    apply_update_to_vector(_ff2_b.values, _ff2_b.grads, _ff2_b.velocities, _ff2_b.m1, _ff2_b.m2, _ff2_b.timesteps, _ff2_b.decays, learning_rate, clipping_scale, true, _optimiser_type);
  }

  if (_use_layer_normalisation)
  {
    apply_update_to_vector(_ln1_gain.values, _ln1_gain.grads, _ln1_gain.velocities, _ln1_gain.m1, _ln1_gain.m2, _ln1_gain.timesteps, _ln1_gain.decays, learning_rate, clipping_scale, true, _optimiser_type);
    apply_update_to_vector(_ln1_bias.values, _ln1_bias.grads, _ln1_bias.velocities, _ln1_bias.m1, _ln1_bias.m2, _ln1_bias.timesteps, _ln1_bias.decays, learning_rate, clipping_scale, true, _optimiser_type);
    apply_update_to_vector(_ln2_gain.values, _ln2_gain.grads, _ln2_gain.velocities, _ln2_gain.m1, _ln2_gain.m2, _ln2_gain.timesteps, _ln2_gain.decays, learning_rate, clipping_scale, true, _optimiser_type);
    apply_update_to_vector(_ln2_bias.values, _ln2_bias.grads, _ln2_bias.velocities, _ln2_bias.m1, _ln2_bias.m2, _ln2_bias.timesteps, _ln2_bias.decays, learning_rate, clipping_scale, true, _optimiser_type);
  }

  for (auto& fr : all_families())
  {
    if (!fr.family->grads.empty())
    {
      std::memset(fr.family->grads.data(), 0, fr.family->grads.size() * sizeof(double));
    }
  }
}

void SelfAttentionLayer::zero_gradients()
{
  MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer");
  Layer::zero_gradients();
  for (auto& fr : all_families())
  {
    std::fill(fr.family->grads.begin(), fr.family->grads.end(), 0.0);
  }
}

} // namespace myoddweb::nn
