#include "grnlayer.h"
#include "../common/logger.h"
#include "../common/simd_utils.h"
#include "../libraries/instrumentor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>

namespace myoddweb::nn
{

namespace
{
double stable_sigmoid(double x) noexcept
{
  if (x >= 0.0)
  {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double e = std::exp(x);
  return e / (1.0 + e);
}
} // namespace

void GrnLayer::grn_forward_task::operator()() const
{
  layer->process_forward_range(
    start,
    end,
    *batch_gradients_and_outputs,
    prev_layer_index,
    *batch_residual_output_values,
    *batch_hidden_states,
    is_training,
    *scratch);
}

void GrnLayer::grn_finish_hidden_gradients_task::operator()() const
{
  layer->finish_hidden_gradients_range(
    start,
    end,
    *batch_gradients_and_outputs,
    *batch_hidden_states,
    raw_delta_all,
    *scratch);
}

void GrnLayer::grn_grad_calc_task::operator()() const
{
  layer->accumulate_gradients_range(
    start,
    end,
    *batch_gradients_and_outputs,
    *batch_hidden_states,
    prev_layer_index,
    this_layer_index,
    accum,
    *local_contributing,
    *scratch);
}

void GrnLayer::init_family(
  WeightFamily& family,
  size_t num_in,
  size_t num_out,
  double weight_decay,
  const activation& init_activation,
  unsigned block_tag,
  std::optional<uint32_t> seed)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
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

void GrnLayer::init_bias_family(WeightFamily& family, size_t n, double fill_value)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  family.values.assign(n, fill_value);
  family.grads.assign(n, 0.0);
  family.velocities.assign(n, 0.0);
  family.m1.assign(n, 0.0);
  family.m2.assign(n, 0.0);
  family.timesteps.assign(n, 0);
  family.decays.assign(n, 0.0);
}

std::array<GrnLayer::FamilyRef, 8> GrnLayer::all_families() noexcept
{
  return {{
    { &_w1, false },
    { &_b1, true },
    { &_w2, false },
    { &_b2, true },
    { &_w_skip, false },
    { &_b_skip, true },
    { &_ln_gain, true },
    { &_ln_bias, true }
  }};
}

std::array<const GrnLayer::WeightFamily*, 8> GrnLayer::all_families() const noexcept
{
  return {{
    &_w1,
    &_b1,
    &_w2,
    &_b2,
    &_w_skip,
    &_b_skip,
    &_ln_gain,
    &_ln_bias
  }};
}

GrnLayer::GrnLayer(
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
  std::optional<uint32_t> seed) :
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
    seed),
  _feed_forward_hidden_size(feed_forward_hidden_size),
  _use_layer_normalisation(use_layer_normalisation)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  const size_t d_in = num_neurons_in_previous_layer;
  const size_t d_out = layer_size;
  const size_t d_ff = feed_forward_hidden_size;

  init_family(_w1, d_in, d_ff, weight_decay, activation_method, 0x47524E31, seed);
  init_bias_family(_b1, d_ff, 0.0);

  init_family(_w2, d_ff, 2 * d_out, weight_decay, activation_method, 0x47524E32, seed);
  init_bias_family(_b2, 2 * d_out, 0.0);

  if (d_in != d_out)
  {
    init_family(_w_skip, d_in, d_out, weight_decay, activation(activation::method::linear, 0.0), 0x47524E53, seed);
    init_bias_family(_b_skip, d_out, 0.0);
  }
  else
  {
    _w_skip.values.clear(); _w_skip.grads.clear(); _w_skip.velocities.clear(); _w_skip.m1.clear(); _w_skip.m2.clear(); _w_skip.timesteps.clear(); _w_skip.decays.clear();
    _b_skip.values.clear(); _b_skip.grads.clear(); _b_skip.velocities.clear(); _b_skip.m1.clear(); _b_skip.m2.clear(); _b_skip.timesteps.clear(); _b_skip.decays.clear();
  }

  if (_use_layer_normalisation)
  {
    init_bias_family(_ln_gain, d_out, 1.0);
    init_bias_family(_ln_bias, d_out, 0.0);
  }
  else
  {
    _ln_gain.values.clear(); _ln_gain.grads.clear(); _ln_gain.velocities.clear(); _ln_gain.m1.clear(); _ln_gain.m2.clear(); _ln_gain.timesteps.clear(); _ln_gain.decays.clear();
    _ln_bias.values.clear(); _ln_bias.grads.clear(); _ln_bias.velocities.clear(); _ln_bias.m1.clear(); _ln_bias.m2.clear(); _ln_bias.timesteps.clear(); _ln_bias.decays.clear();
  }
}

GrnLayer::GrnLayer(
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
  double momentum) noexcept :
  Layer(
    layer_index,
    layer_role,
    optimiser_type,
    residual_layer_number,
    neurons,
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * layer_size, 0.0),
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * layer_size, 0.0),
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * layer_size, 0.0),
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * layer_size, 0.0),
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * layer_size, 0.0),
    std::vector<long long>(static_cast<size_t>(num_neurons_in_previous_layer) * layer_size, 0),
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * layer_size, 0.0),
    b1_values,
    b1_grads,
    b1_velocities,
    b1_m1,
    b1_m2,
    b1_timesteps,
    b1_decays,
    residual_projector,
    number_of_threads,
    lah,
    momentum
  ),
  _feed_forward_hidden_size(feed_forward_hidden_size),
  _use_layer_normalisation(use_layer_normalisation)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  _w1.values = w1_values; _w1.grads = w1_grads; _w1.velocities = w1_velocities; _w1.m1 = w1_m1; _w1.m2 = w1_m2; _w1.timesteps = w1_timesteps; _w1.decays = w1_decays;
  _b1.values = b1_values; _b1.grads = b1_grads; _b1.velocities = b1_velocities; _b1.m1 = b1_m1; _b1.m2 = b1_m2; _b1.timesteps = b1_timesteps; _b1.decays = b1_decays;

  _w2.values = w2_values; _w2.grads = w2_grads; _w2.velocities = w2_velocities; _w2.m1 = w2_m1; _w2.m2 = w2_m2; _w2.timesteps = w2_timesteps; _w2.decays = w2_decays;
  _b2.values = b2_values; _b2.grads = b2_grads; _b2.velocities = b2_velocities; _b2.m1 = b2_m1; _b2.m2 = b2_m2; _b2.timesteps = b2_timesteps; _b2.decays = b2_decays;

  _w_skip.values = w_skip_values; _w_skip.grads = w_skip_grads; _w_skip.velocities = w_skip_velocities; _w_skip.m1 = w_skip_m1; _w_skip.m2 = w_skip_m2; _w_skip.timesteps = w_skip_timesteps; _w_skip.decays = w_skip_decays;
  _b_skip.values = b_skip_values; _b_skip.grads = b_skip_grads; _b_skip.velocities = b_skip_velocities; _b_skip.m1 = b_skip_m1; _b_skip.m2 = b_skip_m2; _b_skip.timesteps = b_skip_timesteps; _b_skip.decays = b_skip_decays;

  _ln_gain.values = ln_gain_values; _ln_gain.grads = ln_gain_grads; _ln_gain.velocities = ln_gain_velocities; _ln_gain.m1 = ln_gain_m1; _ln_gain.m2 = ln_gain_m2; _ln_gain.timesteps = ln_gain_timesteps; _ln_gain.decays = ln_gain_decays;
  _ln_bias.values = ln_bias_values; _ln_bias.grads = ln_bias_grads; _ln_bias.velocities = ln_bias_velocities; _ln_bias.m1 = ln_bias_m1; _ln_bias.m2 = ln_bias_m2; _ln_bias.timesteps = ln_bias_timesteps; _ln_bias.decays = ln_bias_decays;
}

GrnLayer::GrnLayer(const GrnLayer& src) noexcept :
  Layer(src),
  _feed_forward_hidden_size(src._feed_forward_hidden_size),
  _use_layer_normalisation(src._use_layer_normalisation),
  _w1(src._w1), _b1(src._b1),
  _w2(src._w2), _b2(src._b2),
  _w_skip(src._w_skip), _b_skip(src._b_skip),
  _ln_gain(src._ln_gain), _ln_bias(src._ln_bias)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
}

GrnLayer::GrnLayer(GrnLayer&& src) noexcept :
  Layer(std::move(src)),
  _feed_forward_hidden_size(src._feed_forward_hidden_size),
  _use_layer_normalisation(src._use_layer_normalisation),
  _w1(std::move(src._w1)), _b1(std::move(src._b1)),
  _w2(std::move(src._w2)), _b2(std::move(src._b2)),
  _w_skip(std::move(src._w_skip)), _b_skip(std::move(src._b_skip)),
  _ln_gain(std::move(src._ln_gain)), _ln_bias(std::move(src._ln_bias))
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  src._feed_forward_hidden_size = 0;
  src._use_layer_normalisation = false;
}

GrnLayer& GrnLayer::operator=(const GrnLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  if (this != &src)
  {
    Layer::operator=(src);
    _feed_forward_hidden_size = src._feed_forward_hidden_size;
    _use_layer_normalisation = src._use_layer_normalisation;
    _w1 = src._w1; _b1 = src._b1;
    _w2 = src._w2; _b2 = src._b2;
    _w_skip = src._w_skip; _b_skip = src._b_skip;
    _ln_gain = src._ln_gain; _ln_bias = src._ln_bias;
  }
  return *this;
}

GrnLayer& GrnLayer::operator=(GrnLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  if (this != &src)
  {
    Layer::operator=(std::move(src));
    _feed_forward_hidden_size = src._feed_forward_hidden_size;
    _use_layer_normalisation = src._use_layer_normalisation;
    _w1 = std::move(src._w1); _b1 = std::move(src._b1);
    _w2 = std::move(src._w2); _b2 = std::move(src._b2);
    _w_skip = std::move(src._w_skip); _b_skip = std::move(src._b_skip);
    _ln_gain = std::move(src._ln_gain); _ln_bias = std::move(src._ln_bias);

    src._feed_forward_hidden_size = 0;
    src._use_layer_normalisation = false;
  }
  return *this;
}

GrnLayer::~GrnLayer()
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
}

Layer* GrnLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  return new GrnLayer(*this);
}

void GrnLayer::process_forward_range(
  size_t b_start,
  size_t b_end,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  unsigned prev_layer_index,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training,
  forward_scratch& scratch) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  const size_t d_in = get_number_input_neurons();
  const size_t d_out = get_number_neurons();
  const size_t d_ff = _feed_forward_hidden_size;
  const bool use_bias = has_bias();
  const bool has_skip_proj = has_skip_projection();
  const bool have_hs = !batch_hidden_states.empty();
  const activation& act = get_activation_helper().get_activation(0);

  for (size_t b = b_start; b < b_end; ++b)
  {
    const auto& rnn_seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const auto std_out = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    const bool is_sequence = !rnn_seq.empty();
    const size_t total_elements = is_sequence ? rnn_seq.size() : std_out.size();
    const size_t T = (d_in > 0 && total_elements >= d_in) ? (total_elements / d_in) : 0;
    const double* src_data = is_sequence ? rnn_seq.data() : std_out.data();

    if (have_hs && batch_hidden_states[b].at(get_layer_index()).size() != T)
    {
      batch_hidden_states[b].assign(get_layer_index(), T, HiddenState(), get_pre_activation_multiplier());
    }

    if (T == 0)
    {
      batch_gradients_and_outputs[b].set_outputs(get_layer_index(), std::vector<double>(d_out, 0.0));
      batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), std::vector<double>());
      continue;
    }

    scratch.z1.resize(d_ff);
    scratch.eta2.resize(d_ff);
    scratch.eta1.resize(2 * d_out);
    scratch.glu.resize(d_out);
    scratch.glu_drop.resize(d_out);
    scratch.mask.assign(d_out, 1.0);
    scratch.x_skip.resize(d_out);
    scratch.y_res.resize(d_out);
    scratch.y_norm.resize(d_out);
    scratch.inv_std.assign(T, 0.0);
    scratch.out_seq.resize(T * d_out);

    for (size_t t = 0; t < T; ++t)
    {
      const double* x_row = src_data + t * d_in;

      // 1. Dense 1: z1 = W1 * x + b1
      if (use_bias)
      {
        std::memcpy(scratch.z1.data(), _b1.values.data(), d_ff * sizeof(double));
      }
      else
      {
        std::memset(scratch.z1.data(), 0, d_ff * sizeof(double));
      }
      for (size_t i = 0; i < d_in; ++i)
      {
        const double xi = x_row[i];
        simd::mul_add(xi, _w1.values.data() + i * d_ff, scratch.z1.data(), d_ff);
      }

      // Activation: eta2 = act(z1)
      for (size_t j = 0; j < d_ff; ++j)
      {
        scratch.eta2[j] = act.activate(scratch.z1[j]);
      }

      // 2. Dense 2: eta1 = W2 * eta2 + b2
      if (use_bias)
      {
        std::memcpy(scratch.eta1.data(), _b2.values.data(), 2 * d_out * sizeof(double));
      }
      else
      {
        std::memset(scratch.eta1.data(), 0, 2 * d_out * sizeof(double));
      }
      for (size_t j = 0; j < d_ff; ++j)
      {
        const double ej = scratch.eta2[j];
        simd::mul_add(ej, _w2.values.data() + j * (2 * d_out), scratch.eta1.data(), 2 * d_out);
      }

      // 3. GLU: eta_val * sigmoid(eta_gate)
      for (size_t k = 0; k < d_out; ++k)
      {
        const double val = scratch.eta1[k];
        const double gate = scratch.eta1[d_out + k];
        const double sig = stable_sigmoid(gate);
        scratch.glu[k] = val * sig;
      }

      // 4. Inverted Dropout
      scratch.mask.assign(d_out, 1.0);
      if (is_training && get_dropout() > 0.0)
      {
        const auto& neurons = get_neurons();
        for (size_t k = 0; k < d_out; ++k)
        {
          const auto& neuron = (k < neurons.size()) ? neurons[k] : neurons[0];
          if (neuron.is_dropout())
          {
            if (neuron.must_randomly_drop(b * T + t))
            {
              scratch.mask[k] = 0.0;
            }
            else
            {
              scratch.mask[k] = 1.0 / (1.0 - neuron.get_dropout_rate());
            }
          }
        }
      }
      for (size_t k = 0; k < d_out; ++k)
      {
        scratch.glu_drop[k] = scratch.glu[k] * scratch.mask[k];
      }

      // 5. Residual skip connection
      if (has_skip_proj)
      {
        if (use_bias)
        {
          std::memcpy(scratch.x_skip.data(), _b_skip.values.data(), d_out * sizeof(double));
        }
        else
        {
          std::memset(scratch.x_skip.data(), 0, d_out * sizeof(double));
        }
        for (size_t i = 0; i < d_in; ++i)
        {
          const double xi = x_row[i];
          simd::mul_add(xi, _w_skip.values.data() + i * d_out, scratch.x_skip.data(), d_out);
        }
      }
      else
      {
        std::memcpy(scratch.x_skip.data(), x_row, d_out * sizeof(double));
      }

      for (size_t k = 0; k < d_out; ++k)
      {
        scratch.y_res[k] = scratch.x_skip[k] + scratch.glu_drop[k];
      }

      // Apply external residual connection if present
      if (!batch_residual_output_values.empty() && b < batch_residual_output_values.size() && batch_residual_output_values[b].size() == d_out)
      {
        simd::add_vectors(batch_residual_output_values[b].data(), scratch.y_res.data(), d_out);
      }

      // 6. Layer Normalisation
      if (_use_layer_normalisation)
      {
        simd::layer_norm_forward(
          scratch.y_res.data(),
          _ln_gain.values.data(),
          _ln_bias.values.data(),
          scratch.y_norm.data(),
          d_out,
          LayerNormEpsilon,
          scratch.inv_std[t]);
      }
      else
      {
        std::memcpy(scratch.y_norm.data(), scratch.y_res.data(), d_out * sizeof(double));
      }

      if (have_hs)
      {
        auto& hs_row = batch_hidden_states[b].at(get_layer_index());
        hs_row[t].set_pre_activation_sums(scratch.y_res.data(), d_out);
        hs_row[t].set_hidden_state_values(scratch.y_norm.data(), d_out);
        hs_row[t].set_cell_state_values(scratch.mask.data(), d_out);
      }

      std::memcpy(scratch.out_seq.data() + t * d_out, scratch.y_norm.data(), d_out * sizeof(double));
    }

    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), std::vector<double>(scratch.out_seq.end() - d_out, scratch.out_seq.end()));
    if (is_sequence)
    {
      batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), scratch.out_seq);
    }
  }
}

void GrnLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  if (batch_size == 0)
  {
    return;
  }

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const auto num_threads = get_number_of_threads();
  const unsigned int active_threads = (num_threads > 1) ? std::min(static_cast<unsigned int>(num_threads), static_cast<unsigned int>(batch_size)) : 1;

  if (active_threads <= 1)
  {
    forward_scratch scratch;
    process_forward_range(
      0,
      batch_size,
      batch_gradients_and_outputs,
      prev_layer_index,
      batch_residual_output_values,
      batch_hidden_states,
      is_training,
      scratch);
    return;
  }

  std::vector<forward_scratch> thread_scratch(active_threads);
  size_t start = 0;
  for (unsigned int t = 0; t < active_threads; ++t)
  {
    size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
    size_t end = start + size;
    if (start < end)
    {
      _task_queue_pool->enqueue(grn_forward_task{
        this,
        start,
        end,
        &batch_gradients_and_outputs,
        prev_layer_index,
        &batch_residual_output_values,
        &batch_hidden_states,
        is_training,
        &thread_scratch[t]
      });
    }
    start = end;
  }
  _task_queue_pool->get();
}

void GrnLayer::grn_backward_one(
  const double* x_seq,
  size_t T,
  const double* delta,
  double* out_dx,
  const std::vector<HiddenState>& hidden_state_row,
  const grad_accumulators& accum,
  backward_scratch& scratch) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  const size_t d_in = get_number_input_neurons();
  const size_t d_out = get_number_neurons();
  const size_t d_ff = _feed_forward_hidden_size;
  const bool use_bias = has_bias();
  const bool has_skip_proj = has_skip_projection();
  const activation& act = get_activation_helper().get_activation(0);

  scratch.z1.resize(d_ff);
  scratch.eta2.resize(d_ff);
  scratch.eta1.resize(2 * d_out);
  scratch.y_norm.resize(d_out);
  scratch.d_y_res.resize(d_out);
  scratch.d_glu.resize(d_out);
  scratch.d_eta1.resize(2 * d_out);
  scratch.d_eta2.resize(d_ff);
  scratch.d_z1.resize(d_ff);
  scratch.d_x_dense.resize(d_in);
  scratch.d_x_skip.resize(d_out);

  for (size_t t = 0; t < T; ++t)
  {
    const double* x_row = x_seq + t * d_in;
    const double* dy_row = delta + t * d_out;

    // --- Recompute forward activations for this timestep ---
    if (use_bias)
    {
      std::memcpy(scratch.z1.data(), _b1.values.data(), d_ff * sizeof(double));
    }
    else
    {
      std::memset(scratch.z1.data(), 0, d_ff * sizeof(double));
    }
    for (size_t i = 0; i < d_in; ++i)
    {
      const double xi = x_row[i];
      simd::mul_add(xi, _w1.values.data() + i * d_ff, scratch.z1.data(), d_ff);
    }

    for (size_t j = 0; j < d_ff; ++j)
    {
      scratch.eta2[j] = act.activate(scratch.z1[j]);
    }

    if (use_bias)
    {
      std::memcpy(scratch.eta1.data(), _b2.values.data(), 2 * d_out * sizeof(double));
    }
    else
    {
      std::memset(scratch.eta1.data(), 0, 2 * d_out * sizeof(double));
    }
    for (size_t j = 0; j < d_ff; ++j)
    {
      const double ej = scratch.eta2[j];
      simd::mul_add(ej, _w2.values.data() + j * (2 * d_out), scratch.eta1.data(), 2 * d_out);
    }

    const double* y_res = hidden_state_row[t].get_pre_activation_sums().data();
    const double* mask = hidden_state_row[t].get_cell_state_values().data();

    double inv_std_val = 0.0;
    if (_use_layer_normalisation)
    {
      simd::layer_norm_forward(
        y_res,
        _ln_gain.values.data(),
        _ln_bias.values.data(),
        scratch.y_norm.data(),
        d_out,
        LayerNormEpsilon,
        inv_std_val);
    }
    else
    {
      std::memcpy(scratch.y_norm.data(), y_res, d_out * sizeof(double));
    }

    // --- Backpropagate delta ---
    // 1. LayerNorm backward
    if (_use_layer_normalisation)
    {
      scratch.d_ln_gain.assign(d_out, 0.0);
      scratch.d_ln_bias.assign(d_out, 0.0);
      simd::layer_norm_backward(
        dy_row,
        scratch.y_norm.data(),
        _ln_gain.values.data(),
        _ln_bias.values.data(),
        inv_std_val,
        d_out,
        scratch.d_y_res.data(),
        scratch.d_ln_gain.data(),
        scratch.d_ln_bias.data());

      if (accum.ln_gain != nullptr)
      {
        simd::add_vectors(scratch.d_ln_gain.data(), accum.ln_gain, d_out);
      }
      if (accum.ln_bias != nullptr)
      {
        simd::add_vectors(scratch.d_ln_bias.data(), accum.ln_bias, d_out);
      }
    }
    else
    {
      std::memcpy(scratch.d_y_res.data(), dy_row, d_out * sizeof(double));
    }

    // 2. Residual skip:
    for (size_t k = 0; k < d_out; ++k)
    {
      scratch.d_glu[k] = scratch.d_y_res[k] * mask[k];
    }
    std::memcpy(scratch.d_x_skip.data(), scratch.d_y_res.data(), d_out * sizeof(double));

    // 3. GLU backward:
    for (size_t k = 0; k < d_out; ++k)
    {
      const double val = scratch.eta1[k];
      const double gate = scratch.eta1[d_out + k];
      const double sig = stable_sigmoid(gate);
      const double d_glu_k = scratch.d_glu[k];

      scratch.d_eta1[k] = d_glu_k * sig;
      scratch.d_eta1[d_out + k] = d_glu_k * val * sig * (1.0 - sig);
    }

    // 4. Dense 2 backward:
    if (use_bias && accum.b2 != nullptr)
    {
      for (size_t k = 0; k < 2 * d_out; ++k)
      {
        accum.b2[k] += scratch.d_eta1[k];
      }
    }
    if (accum.w2 != nullptr)
    {
      for (size_t j = 0; j < d_ff; ++j)
      {
        const double ej = scratch.eta2[j];
        double* w2_row = accum.w2 + j * (2 * d_out);
        simd::mul_add(ej, scratch.d_eta1.data(), w2_row, 2 * d_out);
      }
    }

    // dL/deta2 = W2 * d_eta1
    for (size_t j = 0; j < d_ff; ++j)
    {
      scratch.d_eta2[j] = simd::dot_product(_w2.values.data() + j * (2 * d_out), scratch.d_eta1.data(), 2 * d_out);
    }

    // 5. Activation backward:
    for (size_t j = 0; j < d_ff; ++j)
    {
      scratch.d_z1[j] = scratch.d_eta2[j] * act.activate_derivative(scratch.z1[j]);
    }

    // 6. Dense 1 backward:
    if (use_bias && accum.b1 != nullptr)
    {
      for (size_t j = 0; j < d_ff; ++j)
      {
        accum.b1[j] += scratch.d_z1[j];
      }
    }
    if (accum.w1 != nullptr)
    {
      for (size_t i = 0; i < d_in; ++i)
      {
        const double xi = x_row[i];
        double* w1_row = accum.w1 + i * d_ff;
        simd::mul_add(xi, scratch.d_z1.data(), w1_row, d_ff);
      }
    }

    // dL/dx_dense = W1 * d_z1
    for (size_t i = 0; i < d_in; ++i)
    {
      scratch.d_x_dense[i] = simd::dot_product(_w1.values.data() + i * d_ff, scratch.d_z1.data(), d_ff);
    }

    // 7. Skip backward & combine input gradients:
    if (has_skip_proj)
    {
      if (use_bias && accum.b_skip != nullptr)
      {
        for (size_t k = 0; k < d_out; ++k)
        {
          accum.b_skip[k] += scratch.d_x_skip[k];
        }
      }
      if (accum.w_skip != nullptr)
      {
        for (size_t i = 0; i < d_in; ++i)
        {
          const double xi = x_row[i];
          double* ws_row = accum.w_skip + i * d_out;
          simd::mul_add(xi, scratch.d_x_skip.data(), ws_row, d_out);
        }
      }
      for (size_t i = 0; i < d_in; ++i)
      {
        const double dx_skip_i = simd::dot_product(_w_skip.values.data() + i * d_out, scratch.d_x_skip.data(), d_out);
        out_dx[t * d_in + i] = scratch.d_x_dense[i] + dx_skip_i;
      }
    }
    else
    {
      for (size_t i = 0; i < d_in; ++i)
      {
        out_dx[t * d_in + i] = scratch.d_x_dense[i] + scratch.d_x_skip[i];
      }
    }
  }
}

void GrnLayer::finish_hidden_gradients_range(
  size_t b_start,
  size_t b_end,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states,
  const double* raw_delta_all,
  backward_scratch& scratch) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  (void)raw_delta_all;
  const size_t d_in = get_number_input_neurons();
  const size_t d_out = get_number_neurons();
  const unsigned this_layer_index = get_layer_index();
  const unsigned prev_layer_index = this_layer_index - 1;

  for (size_t b = b_start; b < b_end; ++b)
  {
    const auto& hidden_state_row = batch_hidden_states[b].at(this_layer_index);
    const size_t T = hidden_state_row.size();
    if (T == 0 || d_in == 0)
    {
      continue;
    }

    const auto& own_delta = batch_gradients_and_outputs[b].get_rnn_gate_gradients(this_layer_index);
    if (own_delta.size() != T * d_out)
    {
      continue;
    }

    const auto& rnn_seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const auto std_out = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    const bool is_sequence = !rnn_seq.empty();
    const size_t total_elements = is_sequence ? rnn_seq.size() : std_out.size();
    if (total_elements != T * d_in)
    {
      continue;
    }
    const double* src_data = is_sequence ? rnn_seq.data() : std_out.data();

    scratch.d_x_total.assign(T * d_in, 0.0);
    grn_backward_one(src_data, T, own_delta.data(), scratch.d_x_total.data(), hidden_state_row, grad_accumulators{}, scratch);
    batch_gradients_and_outputs[b].set_rnn_gradients(this_layer_index, scratch.d_x_total.data(), scratch.d_x_total.size());
  }
}

void GrnLayer::finish_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  const double* raw_delta_all) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  if (batch_size == 0)
  {
    return;
  }

  const auto num_threads = get_number_of_threads();
  const unsigned int active_threads = (num_threads > 1) ? std::min(static_cast<unsigned int>(num_threads), static_cast<unsigned int>(batch_size)) : 1;

  if (active_threads <= 1)
  {
    backward_scratch scratch;
    finish_hidden_gradients_range(0, batch_size, batch_gradients_and_outputs, batch_hidden_states, raw_delta_all, scratch);
    return;
  }

  std::vector<backward_scratch> thread_scratch(active_threads);
  size_t start = 0;
  for (unsigned int t = 0; t < active_threads; ++t)
  {
    size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
    size_t end = start + size;
    if (start < end)
    {
      _task_queue_pool->enqueue(grn_finish_hidden_gradients_task{
        this,
        start,
        end,
        &batch_gradients_and_outputs,
        &batch_hidden_states,
        raw_delta_all,
        &thread_scratch[t]
      });
    }
    start = end;
  }
  _task_queue_pool->get();
}

void GrnLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  (void)batch_gradients_and_outputs;
  (void)target_outputs_begin;
  (void)batch_hidden_states;
  (void)batch_size;
  Logger::panic("GrnLayer cannot be used as an output layer directly; use FFOutputLayer or MultiOutputLayer.");
}

void GrnLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  (void)bptt_max_ticks;
  if (batch_size == 0)
  {
    return;
  }

  const size_t d_out = get_number_neurons();
  const unsigned this_layer_index = get_layer_index();
  const unsigned next_layer_index = next_layer.get_layer_index();
  const size_t next_neurons = next_layer.get_number_neurons();

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& hs_span = batch_hidden_states[b].at(this_layer_index);
    const size_t T = hs_span.size();
    if (T == 0)
    {
      continue;
    }

    std::vector<double> deltas(T * d_out, 0.0);
    const bool has_next_rnn = batch_gradients_and_outputs[b].has_rnn_gradients(next_layer_index);
    if (has_next_rnn)
    {
      const auto& next_rnn = batch_gradients_and_outputs[b].get_rnn_gradients(next_layer_index);
      const size_t next_T = (next_neurons > 0) ? next_rnn.size() / next_neurons : 0;
      const size_t common_T = std::min(T, next_T);
      for (size_t t = 0; t < common_T; ++t)
      {
        const double* g_next = next_rnn.data() + t * next_neurons;
        double* d_this = deltas.data() + t * d_out;
        for (size_t j = 0; j < d_out; ++j)
        {
          double sum = 0.0;
          for (size_t k = 0; k < next_neurons; ++k)
          {
            sum += g_next[k] * next_layer.get_weight_value(static_cast<unsigned>(j), static_cast<unsigned>(k));
          }
          d_this[j] = sum;
        }
      }
    }
    else
    {
      std::span<const double> next_grads;
      if (!batch_next_grad_matrix.empty() && b < batch_next_grad_matrix.size())
      {
        next_grads = batch_next_grad_matrix[b];
      }
      else
      {
        next_grads = batch_gradients_and_outputs[b].get_gradients(next_layer_index);
      }

      if (!next_grads.empty())
      {
        double* d_last = deltas.data() + (T - 1) * d_out;
        for (size_t j = 0; j < d_out; ++j)
        {
          double sum = 0.0;
          for (size_t k = 0; k < next_neurons; ++k)
          {
            sum += next_grads[k] * next_layer.get_weight_value(static_cast<unsigned>(j), static_cast<unsigned>(k));
          }
          d_last[j] = sum;
        }
      }
    }

    batch_gradients_and_outputs[b].set_rnn_gate_gradients(this_layer_index, deltas);
    batch_gradients_and_outputs[b].set_gradients(this_layer_index, deltas.data() + (T - 1) * d_out, d_out);
  }

  finish_hidden_gradients(batch_gradients_and_outputs, batch_hidden_states, batch_size, nullptr);
}

void GrnLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  (void)bptt_max_ticks;
  if (batch_size == 0)
  {
    return;
  }

  const size_t d_out = get_number_neurons();
  const unsigned this_layer_index = get_layer_index();

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& hs_span = batch_hidden_states[b].at(this_layer_index);
    const size_t T = hs_span.size();
    if (T == 0)
    {
      continue;
    }

    std::span<const double> og;
    if (!batch_output_gradients.empty())
    {
      if (b < batch_output_gradients.size())
      {
        og = std::span<const double>(batch_output_gradients[b].data(), batch_output_gradients[b].size());
      }
    }
    else
    {
      const auto& next_rnn = batch_gradients_and_outputs[b].get_rnn_gradients(this_layer_index + 1);
      og = !next_rnn.empty() ? std::span<const double>(next_rnn.data(), next_rnn.size()) : batch_gradients_and_outputs[b].get_gradients(this_layer_index + 1);
    }

    if (og.size() == T * d_out)
    {
      batch_gradients_and_outputs[b].set_rnn_gate_gradients(this_layer_index, og.data(), og.size());
      batch_gradients_and_outputs[b].set_gradients(this_layer_index, og.data() + (T - 1) * d_out, d_out);
    }
    else if (og.size() == d_out)
    {
      std::vector<double> seq(T * d_out, 0.0);
      std::memcpy(seq.data() + (T - 1) * d_out, og.data(), d_out * sizeof(double));
      batch_gradients_and_outputs[b].set_rnn_gate_gradients(this_layer_index, seq);
      batch_gradients_and_outputs[b].set_gradients(this_layer_index, og.data(), d_out);
    }
  }

  finish_hidden_gradients(batch_gradients_and_outputs, batch_hidden_states, batch_size, nullptr);
}

void GrnLayer::accumulate_gradients_range(
  size_t b_start,
  size_t b_end,
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states,
  unsigned prev_layer_index,
  unsigned this_layer_index,
  const grad_accumulators& accum,
  size_t& local_contributing,
  backward_scratch& scratch) const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  const size_t d_in = get_number_input_neurons();
  const size_t d_out = get_number_neurons();

  for (size_t b = b_start; b < b_end; ++b)
  {
    const auto& rnn_seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const auto std_out = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    const bool is_sequence = !rnn_seq.empty();
    const size_t total_elements = is_sequence ? rnn_seq.size() : std_out.size();
    const size_t T = (d_in > 0 && total_elements >= d_in) ? (total_elements / d_in) : 0;
    const double* src_data = is_sequence ? rnn_seq.data() : std_out.data();

    if (T == 0)
    {
      continue;
    }

    const auto& hidden_state_row = batch_hidden_states[b].at(this_layer_index);
    if (hidden_state_row.size() != T)
    {
      continue;
    }

    const auto& rnn_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(this_layer_index);
    const auto std_grads = batch_gradients_and_outputs[b].get_gradients(this_layer_index);

    scratch.delta.resize(T * d_out);
    if (!rnn_grads.empty() && rnn_grads.size() == T * d_out)
    {
      std::memcpy(scratch.delta.data(), rnn_grads.data(), T * d_out * sizeof(double));
    }
    else if (!std_grads.empty() && std_grads.size() == d_out)
    {
      std::memset(scratch.delta.data(), 0, T * d_out * sizeof(double));
      std::memcpy(scratch.delta.data() + (T - 1) * d_out, std_grads.data(), d_out * sizeof(double));
    }
    else
    {
      continue;
    }

    scratch.d_x_total.resize(T * d_in);
    grn_backward_one(src_data, T, scratch.delta.data(), scratch.d_x_total.data(), hidden_state_row, accum, scratch);
    ++local_contributing;
  }
}

void GrnLayer::calculate_and_store_gradients(
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& hidden_states,
  const Layer& previous_layer,
  size_t batch_size,
  int bptt_max_ticks)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  (void)bptt_max_ticks;
  if (batch_size == 0)
  {
    return;
  }

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const unsigned this_layer_index = get_layer_index();
  const auto num_threads = get_number_of_threads();
  const unsigned int active_threads = (num_threads > 1) ? std::min(static_cast<unsigned int>(num_threads), static_cast<unsigned int>(batch_size)) : 1;

  if (active_threads <= 1)
  {
    backward_scratch scratch;
    grad_accumulators accum{
      _w1.grads.data(), _b1.grads.data(),
      _w2.grads.data(), _b2.grads.data(),
      _w_skip.grads.data(), _b_skip.grads.data(),
      _ln_gain.grads.data(), _ln_bias.grads.data()
    };
    size_t contributing = 0;
    accumulate_gradients_range(0, batch_size, batch_gradients_and_outputs, hidden_states, prev_layer_index, this_layer_index, accum, contributing, scratch);

    if (contributing > 0)
    {
      const double scale = 1.0 / static_cast<double>(contributing);
      for (auto& fr : all_families())
      {
        if (!fr.family->grads.empty())
        {
          simd::mul_scalar(fr.family->grads.data(), scale, fr.family->grads.data(), fr.family->grads.size());
        }
      }
    }
    return;
  }

  std::vector<backward_scratch> thread_scratch(active_threads);
  std::vector<thread_grad_accumulators> thread_accums(active_threads);

  for (size_t i = 0; i < active_threads; ++i)
  {
    thread_accums[i].w1.assign(_w1.grads.size(), 0.0);
    thread_accums[i].b1.assign(_b1.grads.size(), 0.0);
    thread_accums[i].w2.assign(_w2.grads.size(), 0.0);
    thread_accums[i].b2.assign(_b2.grads.size(), 0.0);
    thread_accums[i].w_skip.assign(_w_skip.grads.size(), 0.0);
    thread_accums[i].b_skip.assign(_b_skip.grads.size(), 0.0);
    thread_accums[i].ln_gain.assign(_ln_gain.grads.size(), 0.0);
    thread_accums[i].ln_bias.assign(_ln_bias.grads.size(), 0.0);
  }

  size_t start = 0;
  for (unsigned int t = 0; t < active_threads; ++t)
  {
    size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
    size_t end = start + size;
    if (start < end)
    {
      grad_accumulators accum{
        thread_accums[t].w1.data(), thread_accums[t].b1.data(),
        thread_accums[t].w2.data(), thread_accums[t].b2.data(),
        thread_accums[t].w_skip.data(), thread_accums[t].b_skip.data(),
        thread_accums[t].ln_gain.data(), thread_accums[t].ln_bias.data()
      };

      _task_queue_pool->enqueue(grn_grad_calc_task{
        this,
        start,
        end,
        &batch_gradients_and_outputs,
        &hidden_states,
        prev_layer_index,
        this_layer_index,
        accum,
        &thread_accums[t].contributing,
        &thread_scratch[t]
      });
    }
    start = end;
  }
  _task_queue_pool->get();

  size_t total_contributing = 0;
  for (size_t i = 0; i < active_threads; ++i)
  {
    total_contributing += thread_accums[i].contributing;
    if (!_w1.grads.empty())
    {
      simd::add_vectors(thread_accums[i].w1.data(), _w1.grads.data(), _w1.grads.size());
    }
    if (!_b1.grads.empty())
    {
      simd::add_vectors(thread_accums[i].b1.data(), _b1.grads.data(), _b1.grads.size());
    }
    if (!_w2.grads.empty())
    {
      simd::add_vectors(thread_accums[i].w2.data(), _w2.grads.data(), _w2.grads.size());
    }
    if (!_b2.grads.empty())
    {
      simd::add_vectors(thread_accums[i].b2.data(), _b2.grads.data(), _b2.grads.size());
    }
    if (!_w_skip.grads.empty())
    {
      simd::add_vectors(thread_accums[i].w_skip.data(), _w_skip.grads.data(), _w_skip.grads.size());
    }
    if (!_b_skip.grads.empty())
    {
      simd::add_vectors(thread_accums[i].b_skip.data(), _b_skip.grads.data(), _b_skip.grads.size());
    }
    if (!_ln_gain.grads.empty())
    {
      simd::add_vectors(thread_accums[i].ln_gain.data(), _ln_gain.grads.data(), _ln_gain.grads.size());
    }
    if (!_ln_bias.grads.empty())
    {
      simd::add_vectors(thread_accums[i].ln_bias.data(), _ln_bias.grads.data(), _ln_bias.grads.size());
    }
  }

  if (total_contributing > 0)
  {
    const double scale = 1.0 / static_cast<double>(total_contributing);
    for (auto& fr : all_families())
    {
      if (!fr.family->grads.empty())
      {
        simd::mul_scalar(fr.family->grads.data(), scale, fr.family->grads.data(), fr.family->grads.size());
      }
    }
  }
}

double GrnLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  double total = 0.0;
  for (const auto* fam : all_families())
  {
    if (!fam->grads.empty())
    {
      total += simd::dot_product(fam->grads.data(), fam->grads.data(), fam->grads.size());
    }
  }
  return total;
}

void GrnLayer::accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  const auto* grn_snapshot = dynamic_cast<const GrnLayer*>(&snapshot);
  if (grn_snapshot == nullptr)
  {
    Logger::panic("Cannot SWA-average a non-GrnLayer snapshot into a GrnLayer!");
    return;
  }

  const double weight = 1.0 / static_cast<double>(existing_swa_count + 1);
  auto my_fams = all_families();
  auto snap_fams = grn_snapshot->all_families();

  for (size_t i = 0; i < my_fams.size(); ++i)
  {
    auto* dst = my_fams[i].family;
    const auto* src = snap_fams[i];
    for (size_t k = 0; k < dst->values.size(); ++k)
    {
      dst->values[k] += weight * (src->values[k] - dst->values[k]);
    }
  }
}

void GrnLayer::update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  auto* fast_grn = dynamic_cast<GrnLayer*>(&fast_layer);
  if (fast_grn == nullptr)
  {
    Logger::panic("Cannot update Lookahead slow weights: fast layer is not a GrnLayer!");
    return;
  }

  auto slow_fams = all_families();
  auto fast_fams = fast_grn->all_families();

  for (size_t i = 0; i < slow_fams.size(); ++i)
  {
    auto* slow = slow_fams[i].family;
    auto* fast = fast_fams[i].family;
    for (size_t k = 0; k < slow->values.size(); ++k)
    {
      slow->values[k] += alpha * (fast->values[k] - slow->values[k]);
      fast->values[k] = slow->values[k];
    }
  }
}

void GrnLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  for (auto& fr : all_families())
  {
    if (!fr.family->values.empty())
    {
      apply_update_to_vector(
        fr.family->values,
        fr.family->grads,
        fr.family->velocities,
        fr.family->m1,
        fr.family->m2,
        fr.family->timesteps,
        fr.family->decays,
        learning_rate,
        clipping_scale,
        fr.is_bias,
        _optimiser_type);
    }
  }

  zero_gradients();
}

void GrnLayer::zero_gradients()
{
  MYODDWEB_PROFILE_FUNCTION("GrnLayer");
  Layer::zero_gradients();
  for (auto& fr : all_families())
  {
    if (!fr.family->grads.empty())
    {
      std::fill(fr.family->grads.begin(), fr.family->grads.end(), 0.0);
    }
  }
}

} // namespace myoddweb::nn
