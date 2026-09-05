#include "../libraries/instrumentor.h"
#include "attentionpoollayer.h"
#include "../common/simd_utils.h"
#include "../common/logger.h"
#include <cmath>
#include <cstring>

namespace myoddweb::nn
{
AttentionPoolLayer::AttentionPoolLayer(
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
  std::optional<uint32_t> seed
) :
  Layer(
    layer_index,
    layer_role,
    layer_activation_helper(activation_method, num_neurons_in_previous_layer, num_neurons_in_previous_layer),
    optimiser_type,
    -1,
    create_neurons(dropout_rate, num_neurons_in_previous_layer, seed),
    has_bias,
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * num_neurons_in_previous_layer, 0.0),
    nullptr,
    number_of_threads,
    momentum,
    seed
  ),
  _attention_hidden_size(attention_hidden_size)
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  const size_t N = num_neurons_in_previous_layer;
  const size_t d_a = attention_hidden_size;
  const size_t num_wa = N * d_a;

  const activation scoring_activation(activation::method::tanh, 0.0);

  _wa_values.resize(num_wa);
  for (size_t i = 0; i < N; ++i)
  {
    for (size_t j = 0; j < d_a; ++j)
    {
      const size_t flat_index = i * d_a + j;
      const auto weight_seed = seed.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(Rng::derive(seed.value(), 0, flat_index))) : std::nullopt;
      _wa_values[flat_index] = scoring_activation.weight_initialization(static_cast<unsigned>(N), static_cast<unsigned>(d_a), weight_seed);
    }
  }
  _wa_grads.assign(num_wa, 0.0);
  _wa_velocities.assign(num_wa, 0.0);
  _wa_m1.assign(num_wa, 0.0);
  _wa_m2.assign(num_wa, 0.0);
  _wa_timesteps.assign(num_wa, 0);
  _wa_decays.assign(num_wa, weight_decay);

  _ba_values.assign(d_a, 0.0);
  _ba_grads.assign(d_a, 0.0);
  _ba_velocities.assign(d_a, 0.0);
  _ba_m1.assign(d_a, 0.0);
  _ba_m2.assign(d_a, 0.0);
  _ba_timesteps.assign(d_a, 0);
  _ba_decays.assign(d_a, 0.0);

  _v_values.resize(d_a);
  for (size_t j = 0; j < d_a; ++j)
  {
    const auto weight_seed = seed.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(Rng::derive(seed.value(), 1, j))) : std::nullopt;
    _v_values[j] = scoring_activation.weight_initialization(static_cast<unsigned>(d_a), 1u, weight_seed);
  }
  _v_grads.assign(d_a, 0.0);
  _v_velocities.assign(d_a, 0.0);
  _v_m1.assign(d_a, 0.0);
  _v_m2.assign(d_a, 0.0);
  _v_timesteps.assign(d_a, 0);
  _v_decays.assign(d_a, weight_decay);
}

AttentionPoolLayer::AttentionPoolLayer(
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
  double momentum
) noexcept :
  Layer(
    layer_index,
    layer_role,
    optimiser_type,
    -1,
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
    nullptr,
    number_of_threads,
    lah,
    momentum
  ),
  _attention_hidden_size(attention_hidden_size),
  _wa_values(wa_values),
  _wa_grads(wa_grads),
  _wa_velocities(wa_velocities),
  _wa_m1(wa_m1),
  _wa_m2(wa_m2),
  _wa_timesteps(wa_timesteps),
  _wa_decays(wa_decays),
  _ba_values(ba_values),
  _ba_grads(ba_grads),
  _ba_velocities(ba_velocities),
  _ba_m1(ba_m1),
  _ba_m2(ba_m2),
  _ba_timesteps(ba_timesteps),
  _ba_decays(ba_decays),
  _v_values(v_values),
  _v_grads(v_grads),
  _v_velocities(v_velocities),
  _v_m1(v_m1),
  _v_m2(v_m2),
  _v_timesteps(v_timesteps),
  _v_decays(v_decays)
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
}

AttentionPoolLayer::AttentionPoolLayer(const AttentionPoolLayer& src) noexcept :
  Layer(src),
  _attention_hidden_size(src._attention_hidden_size),
  _wa_values(src._wa_values),
  _wa_grads(src._wa_grads),
  _wa_velocities(src._wa_velocities),
  _wa_m1(src._wa_m1),
  _wa_m2(src._wa_m2),
  _wa_timesteps(src._wa_timesteps),
  _wa_decays(src._wa_decays),
  _ba_values(src._ba_values),
  _ba_grads(src._ba_grads),
  _ba_velocities(src._ba_velocities),
  _ba_m1(src._ba_m1),
  _ba_m2(src._ba_m2),
  _ba_timesteps(src._ba_timesteps),
  _ba_decays(src._ba_decays),
  _v_values(src._v_values),
  _v_grads(src._v_grads),
  _v_velocities(src._v_velocities),
  _v_m1(src._v_m1),
  _v_m2(src._v_m2),
  _v_timesteps(src._v_timesteps),
  _v_decays(src._v_decays)
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
}

AttentionPoolLayer::AttentionPoolLayer(AttentionPoolLayer&& src) noexcept :
  Layer(std::move(src)),
  _attention_hidden_size(src._attention_hidden_size),
  _wa_values(std::move(src._wa_values)),
  _wa_grads(std::move(src._wa_grads)),
  _wa_velocities(std::move(src._wa_velocities)),
  _wa_m1(std::move(src._wa_m1)),
  _wa_m2(std::move(src._wa_m2)),
  _wa_timesteps(std::move(src._wa_timesteps)),
  _wa_decays(std::move(src._wa_decays)),
  _ba_values(std::move(src._ba_values)),
  _ba_grads(std::move(src._ba_grads)),
  _ba_velocities(std::move(src._ba_velocities)),
  _ba_m1(std::move(src._ba_m1)),
  _ba_m2(std::move(src._ba_m2)),
  _ba_timesteps(std::move(src._ba_timesteps)),
  _ba_decays(std::move(src._ba_decays)),
  _v_values(std::move(src._v_values)),
  _v_grads(std::move(src._v_grads)),
  _v_velocities(std::move(src._v_velocities)),
  _v_m1(std::move(src._v_m1)),
  _v_m2(std::move(src._v_m2)),
  _v_timesteps(std::move(src._v_timesteps)),
  _v_decays(std::move(src._v_decays))
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  src._attention_hidden_size = 0;
}

AttentionPoolLayer& AttentionPoolLayer::operator=(const AttentionPoolLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  if (this != &src)
  {
    Layer::operator=(src);
    _attention_hidden_size = src._attention_hidden_size;
    _wa_values = src._wa_values;
    _wa_grads = src._wa_grads;
    _wa_velocities = src._wa_velocities;
    _wa_m1 = src._wa_m1;
    _wa_m2 = src._wa_m2;
    _wa_timesteps = src._wa_timesteps;
    _wa_decays = src._wa_decays;
    _ba_values = src._ba_values;
    _ba_grads = src._ba_grads;
    _ba_velocities = src._ba_velocities;
    _ba_m1 = src._ba_m1;
    _ba_m2 = src._ba_m2;
    _ba_timesteps = src._ba_timesteps;
    _ba_decays = src._ba_decays;
    _v_values = src._v_values;
    _v_grads = src._v_grads;
    _v_velocities = src._v_velocities;
    _v_m1 = src._v_m1;
    _v_m2 = src._v_m2;
    _v_timesteps = src._v_timesteps;
    _v_decays = src._v_decays;
  }
  return *this;
}

AttentionPoolLayer& AttentionPoolLayer::operator=(AttentionPoolLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  if (this != &src)
  {
    Layer::operator=(std::move(src));
    _attention_hidden_size = src._attention_hidden_size;
    _wa_values = std::move(src._wa_values);
    _wa_grads = std::move(src._wa_grads);
    _wa_velocities = std::move(src._wa_velocities);
    _wa_m1 = std::move(src._wa_m1);
    _wa_m2 = std::move(src._wa_m2);
    _wa_timesteps = std::move(src._wa_timesteps);
    _wa_decays = std::move(src._wa_decays);
    _ba_values = std::move(src._ba_values);
    _ba_grads = std::move(src._ba_grads);
    _ba_velocities = std::move(src._ba_velocities);
    _ba_m1 = std::move(src._ba_m1);
    _ba_m2 = std::move(src._ba_m2);
    _ba_timesteps = std::move(src._ba_timesteps);
    _ba_decays = std::move(src._ba_decays);
    _v_values = std::move(src._v_values);
    _v_grads = std::move(src._v_grads);
    _v_velocities = std::move(src._v_velocities);
    _v_m1 = std::move(src._v_m1);
    _v_m2 = std::move(src._v_m2);
    _v_timesteps = std::move(src._v_timesteps);
    _v_decays = std::move(src._v_decays);

    src._attention_hidden_size = 0;
  }
  return *this;
}

AttentionPoolLayer::~AttentionPoolLayer()
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
}

Layer* AttentionPoolLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  return new AttentionPoolLayer(*this);
}

void AttentionPoolLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
  (void)batch_gradients_and_outputs;
  (void)target_outputs_begin;
  (void)batch_hidden_states;
  (void)batch_size;
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  Logger::panic("AttentionPoolLayer: trying to calculate output gradients on a non-output layer!");
}

void AttentionPoolLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  (void)batch_residual_output_values;
  if (batch_size == 0)
  {
    return;
  }

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const size_t N = get_number_neurons();
  const size_t d_a = _attention_hidden_size;
  const bool have_hs = !batch_hidden_states.empty();
  const bool use_bias = has_bias();

  std::vector<double> context(N);
  std::vector<double> output(N);
  std::vector<double> mask(N);
  std::vector<double> scores;

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const size_t T = (N > 0 && !seq.empty()) ? seq.size() / N : 0;

    std::fill(context.begin(), context.end(), 0.0);

    if (T > 0)
    {
      scores.assign(T, 0.0);
      for (size_t t = 0; t < T; ++t)
      {
        const double* h_t = seq.data() + t * N;
        double score = 0.0;
        for (size_t j = 0; j < d_a; ++j)
        {
          double e = use_bias ? _ba_values[j] : 0.0;
          const double* wa_row = _wa_values.data() + j;
          for (size_t i = 0; i < N; ++i)
          {
            e += h_t[i] * wa_row[i * d_a];
          }
          const double u = std::tanh(e);
          score += _v_values[j] * u;
        }
        scores[t] = score;
      }
      simd::softmax_forward(scores.data(), T);
      for (size_t t = 0; t < T; ++t)
      {
        const double* h_t = seq.data() + t * N;
        const double a_t = scores[t];
        for (size_t i = 0; i < N; ++i)
        {
          context[i] += a_t * h_t[i];
        }
      }
    }

    output = context;
    for (const auto& r : get_activation_helper().ranges())
    {
      r.activation_method.activate(output.data() + r.start, output.data() + r.end, is_training);
    }

    std::fill(mask.begin(), mask.end(), 1.0);
    if (is_training && get_dropout() > 0.0)
    {
      const auto& neurons = get_neurons();
      for (size_t j = 0; j < N; ++j)
      {
        const auto& neuron = neurons[j];
        if (neuron.is_dropout())
        {
          if (neuron.must_randomly_drop(b))
          {
            output[j] = 0.0;
            mask[j] = 0.0;
          }
          else
          {
            const double scale = 1.0 / (1.0 - neuron.get_dropout_rate());
            output[j] *= scale;
            mask[j] = scale;
          }
        }
      }
    }

    if (have_hs)
    {
      if (batch_hidden_states[b].at(get_layer_index()).size() != 1)
      {
        batch_hidden_states[b].assign(get_layer_index(), 1, HiddenState(), get_pre_activation_multiplier());
      }
      auto& hs_row = batch_hidden_states[b].at(get_layer_index());
      hs_row[0].set_pre_activation_sums(context.data(), N);
      hs_row[0].set_hidden_state_values(output.data(), N);
      hs_row[0].set_cell_state_values(mask.data(), N);
    }

    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), output.data(), N);
  }
}

void AttentionPoolLayer::attention_backward_one(
  const double* h_seq,
  size_t T,
  const double* delta,
  double* out_dh,
  double* wa_grad_accum,
  double* ba_grad_accum,
  double* v_grad_accum) const
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  const size_t N = get_number_neurons();
  const size_t d_a = _attention_hidden_size;
  if (T == 0 || N == 0)
  {
    return;
  }
  const bool use_bias = has_bias();

  std::vector<double> scores(T);
  std::vector<double> u_flat(T * d_a);

  for (size_t t = 0; t < T; ++t)
  {
    const double* h_t = h_seq + t * N;
    double score = 0.0;
    for (size_t j = 0; j < d_a; ++j)
    {
      double e = use_bias ? _ba_values[j] : 0.0;
      for (size_t i = 0; i < N; ++i)
      {
        e += h_t[i] * _wa_values[i * d_a + j];
      }
      const double u = std::tanh(e);
      u_flat[t * d_a + j] = u;
      score += _v_values[j] * u;
    }
    scores[t] = score;
  }

  std::vector<double> alpha(scores);
  simd::softmax_forward(alpha.data(), T);

  std::vector<double> dalpha(T);
  for (size_t t = 0; t < T; ++t)
  {
    const double* h_t = h_seq + t * N;
    double dot = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
      dot += delta[i] * h_t[i];
    }
    dalpha[t] = dot;
  }

  std::vector<double> dscore(T);
  simd::softmax_backward(alpha.data(), dalpha.data(), dscore.data(), T);

  for (size_t t = 0; t < T; ++t)
  {
    const double* h_t = h_seq + t * N;
    double* dh_t = out_dh + t * N;
    const double a_t = alpha[t];

    for (size_t i = 0; i < N; ++i)
    {
      dh_t[i] = a_t * delta[i];
    }

    for (size_t j = 0; j < d_a; ++j)
    {
      const double u = u_flat[t * d_a + j];
      const double de = dscore[t] * _v_values[j] * (1.0 - u * u);

      if (v_grad_accum != nullptr)
      {
        v_grad_accum[j] += dscore[t] * u;
      }
      if (ba_grad_accum != nullptr)
      {
        ba_grad_accum[j] += de;
      }
      for (size_t i = 0; i < N; ++i)
      {
        if (wa_grad_accum != nullptr)
        {
          wa_grad_accum[i * d_a + j] += h_t[i] * de;
        }
        dh_t[i] += _wa_values[i * d_a + j] * de;
      }
    }
  }
}

void AttentionPoolLayer::finish_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  const double* raw_delta_all) const
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  const size_t N = get_number_neurons();
  const unsigned this_layer_index = get_layer_index();
  const unsigned prev_layer_index = this_layer_index - 1;

  std::vector<double> delta(N);
  std::vector<double> deriv(N);
  std::vector<double> dh_scratch;

  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* raw = raw_delta_all + b * N;

    // Undo dropout mask and this layer's own output activation derivative.
    const auto& hs_row = batch_hidden_states[b].at(this_layer_index);
    const double* pre_act = hs_row[0].get_pre_activation_sums().data();
    const double* mask_vals = hs_row[0].get_cell_state_values().data();
    const double* y_vals = hs_row[0].get_hidden_state_values().data();

    for (const auto& r : get_activation_helper().ranges())
    {
      r.activation_method.activate_derivative(pre_act + r.start, pre_act + r.end, y_vals + r.start, deriv.data() + r.start);
    }
    for (size_t i = 0; i < N; ++i)
    {
      delta[i] = raw[i] * deriv[i] * mask_vals[i];
    }

    batch_gradients_and_outputs[b].set_gradients(this_layer_index, delta.data(), N);

    const auto& seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const size_t T = (N > 0 && !seq.empty()) ? seq.size() / N : 0;
    if (T > 0)
    {
      dh_scratch.assign(T * N, 0.0);
      attention_backward_one(seq.data(), T, delta.data(), dh_scratch.data(), nullptr, nullptr, nullptr);
      batch_gradients_and_outputs[b].set_rnn_gradients(this_layer_index, dh_scratch.data(), T * N);
    }
  }
}

void AttentionPoolLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t N = get_number_neurons();
  const size_t N_next = next_layer.get_number_neurons();
  const unsigned next_layer_index = next_layer.get_layer_index();
  const bool use_direct_gradients = batch_next_grad_matrix.empty();

  std::vector<double> raw_delta_all(batch_size * N, 0.0);
  std::vector<double> next_grad(N_next);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::span<const double> src;
    if (use_direct_gradients)
    {
      src = batch_gradients_and_outputs[b].get_gradients(next_layer_index);
    }
    else if (b < batch_next_grad_matrix.size())
    {
      src = std::span<const double>(batch_next_grad_matrix[b].data(), batch_next_grad_matrix[b].size());
    }

    std::fill(next_grad.begin(), next_grad.end(), 0.0);
    if (!src.empty())
    {
      const size_t copy_size = std::min(src.size(), N_next);
      std::copy(src.data(), src.data() + copy_size, next_grad.begin());
    }

    // Chain-rule through next_layer's dense weight matrix: raw_delta[i] = sum_j next_grad[j] * W_next[i,j]
    double* raw = raw_delta_all.data() + b * N;
    simd::gemm_transposed_one_batch(next_grad.data(), next_layer.get_w_values().data(), raw, N, N_next);
  }

  finish_hidden_gradients(batch_gradients_and_outputs, batch_hidden_states, batch_size, raw_delta_all.data());
}

void AttentionPoolLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  const size_t N = get_number_neurons();
  if (N == 0 || batch_size == 0)
  {
    return;
  }

  const unsigned this_layer_index = get_layer_index();
  const bool use_direct_gradients = batch_output_gradients.empty();

  std::vector<double> raw_delta_all(batch_size * N, 0.0);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::span<const double> src;
    if (use_direct_gradients)
    {
      const auto& rnn_g = batch_gradients_and_outputs[b].get_rnn_gradients(this_layer_index + 1);
      src = !rnn_g.empty() ? rnn_g : batch_gradients_and_outputs[b].get_gradients(this_layer_index + 1);
    }
    else if (b < batch_output_gradients.size())
    {
      src = std::span<const double>(batch_output_gradients[b].data(), batch_output_gradients[b].size());
    }
    if (src.empty())
    {
      continue;
    }
    double* raw = raw_delta_all.data() + b * N;
    const size_t copy_size = std::min(src.size(), N);
    std::copy(src.data(), src.data() + copy_size, raw);
  }

  finish_hidden_gradients(batch_gradients_and_outputs, batch_hidden_states, batch_size, raw_delta_all.data());
}

void AttentionPoolLayer::calculate_and_store_gradients(
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& hidden_states,
  const Layer& previous_layer,
  size_t batch_size,
  int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  (void)hidden_states;
  zero_gradients();
  if (batch_size == 0)
  {
    return;
  }

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const unsigned this_layer_index = get_layer_index();
  const size_t N = get_number_neurons();

  std::vector<double> dh_scratch;
  size_t contributing = 0;

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto delta = batch_gradients_and_outputs[b].get_gradients(this_layer_index);
    const auto& seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const size_t T = (N > 0 && !seq.empty()) ? seq.size() / N : 0;
    if (delta.size() != N || T == 0)
    {
      continue;
    }

    dh_scratch.assign(T * N, 0.0);
    attention_backward_one(seq.data(), T, delta.data(), dh_scratch.data(), _wa_grads.data(), _ba_grads.data(), _v_grads.data());
    ++contributing;
  }

  if (contributing > 0)
  {
    const double inv = 1.0 / static_cast<double>(contributing);
    simd::scale_vector(_wa_grads.data(), inv, _wa_grads.size());
    simd::scale_vector(_ba_grads.data(), inv, _ba_grads.size());
    simd::scale_vector(_v_grads.data(), inv, _v_grads.size());
  }
}

double AttentionPoolLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  double norm_sq = simd::sum_sq(_wa_grads.data(), _wa_grads.size());
  norm_sq += simd::sum_sq(_ba_grads.data(), _ba_grads.size());
  norm_sq += simd::sum_sq(_v_grads.data(), _v_grads.size());
  return norm_sq;
}

void AttentionPoolLayer::accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count)
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  const auto& other = static_cast<const AttentionPoolLayer&>(snapshot);
  swa_average_into(_wa_values, other._wa_values, existing_swa_count);
  swa_average_into(_ba_values, other._ba_values, existing_swa_count);
  swa_average_into(_v_values, other._v_values, existing_swa_count);
}

void AttentionPoolLayer::update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  auto& other = static_cast<AttentionPoolLayer&>(fast_layer);
  simd::lookahead_step(_wa_values.data(), other._wa_values.data(), alpha, _wa_values.size());
  if (has_bias())
  {
    simd::lookahead_step(_ba_values.data(), other._ba_values.data(), alpha, _ba_values.size());
  }
  simd::lookahead_step(_v_values.data(), other._v_values.data(), alpha, _v_values.size());
}

void AttentionPoolLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  apply_update_to_vector(_wa_values, _wa_grads, _wa_velocities, _wa_m1, _wa_m2, _wa_timesteps, _wa_decays, learning_rate, clipping_scale, false, _optimiser_type);
  if (has_bias())
  {
    apply_update_to_vector(_ba_values, _ba_grads, _ba_velocities, _ba_m1, _ba_m2, _ba_timesteps, _ba_decays, learning_rate, clipping_scale, true, _optimiser_type);
  }
  apply_update_to_vector(_v_values, _v_grads, _v_velocities, _v_m1, _v_m2, _v_timesteps, _v_decays, learning_rate, clipping_scale, false, _optimiser_type);

  if (!_wa_grads.empty()) std::memset(_wa_grads.data(), 0, _wa_grads.size() * sizeof(double));
  if (!_ba_grads.empty()) std::memset(_ba_grads.data(), 0, _ba_grads.size() * sizeof(double));
  if (!_v_grads.empty()) std::memset(_v_grads.data(), 0, _v_grads.size() * sizeof(double));
}

void AttentionPoolLayer::zero_gradients()
{
  MYODDWEB_PROFILE_FUNCTION("AttentionPoolLayer");
  Layer::zero_gradients();
  std::fill(_wa_grads.begin(), _wa_grads.end(), 0.0);
  std::fill(_ba_grads.begin(), _ba_grads.end(), 0.0);
  std::fill(_v_grads.begin(), _v_grads.end(), 0.0);
}

} // namespace myoddweb::nn
