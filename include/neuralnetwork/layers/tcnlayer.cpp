#include "../libraries/instrumentor.h"
#include "tcnlayer.h"
#include "../common/simd_utils.h"
#include "../common/logger.h"
#include <algorithm>
#include <cstring>

namespace myoddweb::nn
{
TcnLayer::TcnLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer,
  unsigned layer_size,
  unsigned kernel_size,
  unsigned dilation,
  double weight_decay,
  const Role layer_role,
  const activation& activation_method,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias,
  double momentum,
  std::optional<uint32_t> seed
) :
  Layer(
    layer_index,
    layer_role,
    layer_activation_helper(activation_method, kernel_size * num_neurons_in_previous_layer, layer_size),
    optimiser_type,
    residual_layer_number,
    create_neurons(dropout_rate, layer_size, seed),
    has_bias,
    std::vector<double>(static_cast<size_t>(kernel_size) * num_neurons_in_previous_layer * layer_size, weight_decay),
    residual_projector,
    number_of_threads,
    momentum,
    seed
  ),
  _kernel_size(kernel_size),
  _dilation(dilation)
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
}

TcnLayer::TcnLayer(
  unsigned layer_index,
  const Role layer_role,
  const OptimiserType optimiser_type,
  int residual_layer_number,
  unsigned kernel_size,
  unsigned dilation,
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
    w_values,
    w_grads,
    w_velocities,
    w_m1,
    w_m2,
    w_timesteps,
    w_decays,
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
  _kernel_size(kernel_size),
  _dilation(dilation)
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
}

TcnLayer::TcnLayer(const TcnLayer& src) noexcept :
  Layer(src),
  _kernel_size(src._kernel_size),
  _dilation(src._dilation)
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
}

TcnLayer::TcnLayer(TcnLayer&& src) noexcept :
  Layer(std::move(src)),
  _kernel_size(src._kernel_size),
  _dilation(src._dilation)
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  src._kernel_size = 0;
  src._dilation = 0;
}

TcnLayer& TcnLayer::operator=(const TcnLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  if (this != &src)
  {
    Layer::operator=(src);
    _kernel_size = src._kernel_size;
    _dilation = src._dilation;
  }
  return *this;
}

TcnLayer& TcnLayer::operator=(TcnLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  if (this != &src)
  {
    Layer::operator=(std::move(src));
    _kernel_size = src._kernel_size;
    _dilation = src._dilation;
    src._kernel_size = 0;
    src._dilation = 0;
  }
  return *this;
}

TcnLayer::~TcnLayer()
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
}

Layer* TcnLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  return new TcnLayer(*this);
}

void TcnLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
  (void)batch_gradients_and_outputs;
  (void)target_outputs_begin;
  (void)batch_hidden_states;
  (void)batch_size;
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  Logger::panic("TcnLayer: trying to calculate output gradients on a non-output layer!");
}

void TcnLayer::process_forward_range(
  size_t b_start,
  size_t b_end,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  unsigned prev_layer_index,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  const size_t N_out = get_number_neurons();
  const size_t N_gathered = get_number_input_neurons();
  const size_t K = _kernel_size;
  const size_t D = _dilation;
  const size_t N_in = K > 0 ? N_gathered / K : 0;
  const bool use_bias = has_bias();
  const bool have_hs = !batch_hidden_states.empty();

  std::vector<double> gathered(N_gathered);
  std::vector<double> pre_act(N_out);
  std::vector<double> output(N_out);
  std::vector<double> mask(N_out);

  for (size_t b = b_start; b < b_end; ++b)
  {
    const auto& seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const size_t T = (N_in > 0 && !seq.empty()) ? seq.size() / N_in : 0;

    if (have_hs && batch_hidden_states[b].at(get_layer_index()).size() != T)
    {
      batch_hidden_states[b].assign(get_layer_index(), T, HiddenState(), get_pre_activation_multiplier());
    }

    std::vector<double> out_seq(T * N_out);

    for (size_t t = 0; t < T; ++t)
    {
      std::fill(gathered.begin(), gathered.end(), 0.0);
      for (size_t j = 0; j < K; ++j)
      {
        if (t >= j * D)
        {
          const size_t s = t - j * D;
          const double* x_s = seq.data() + s * N_in;
          std::copy(x_s, x_s + N_in, gathered.begin() + j * N_in);
        }
      }

      for (size_t o = 0; o < N_out; ++o)
      {
        double sum = use_bias ? _b_values[o] : 0.0;
        for (size_t g = 0; g < N_gathered; ++g)
        {
          sum += gathered[g] * _w_values[g * N_out + o];
        }
        pre_act[o] = sum;
      }

      if (!batch_residual_output_values.empty() && b < batch_residual_output_values.size() &&
        batch_residual_output_values[b].size() == N_out && t == T - 1)
      {
        simd::add_vectors(batch_residual_output_values[b].data(), pre_act.data(), N_out);
      }

      if (have_hs)
      {
        auto& hs_row = batch_hidden_states[b].at(get_layer_index());
        hs_row[t].set_pre_activation_sums(pre_act.data(), N_out);
      }

      output = pre_act;
      for (const auto& r : get_activation_helper().ranges())
      {
        r.activation_method.activate(output.data() + r.start, output.data() + r.end, is_training);
      }

      std::fill(mask.begin(), mask.end(), 1.0);
      if (is_training && get_dropout() > 0.0)
      {
        const auto& neurons = get_neurons();
        for (size_t o = 0; o < N_out; ++o)
        {
          const auto& neuron = neurons[o];
          if (neuron.is_dropout())
          {
            if (neuron.must_randomly_drop(b * T + t))
            {
              output[o] = 0.0;
              mask[o] = 0.0;
            }
            else
            {
              const double scale = 1.0 / (1.0 - neuron.get_dropout_rate());
              output[o] *= scale;
              mask[o] = scale;
            }
          }
        }
      }

      if (have_hs)
      {
        auto& hs_row = batch_hidden_states[b].at(get_layer_index());
        hs_row[t].set_hidden_state_values(output.data(), N_out);
        hs_row[t].set_cell_state_values(mask.data(), N_out);
      }

      std::copy(output.begin(), output.end(), out_seq.begin() + t * N_out);
    }

    if (T > 0)
    {
      batch_gradients_and_outputs[b].set_outputs(get_layer_index(), out_seq.data() + (T - 1) * N_out, N_out);
    }
    else
    {
      batch_gradients_and_outputs[b].set_outputs(get_layer_index(), std::vector<double>(N_out, 0.0));
    }
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), out_seq.data(), out_seq.size());
  }
}

void TcnLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  if (batch_size == 0)
  {
    return;
  }

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const auto num_threads = get_number_of_threads();
  const unsigned int active_threads = (num_threads > 1) ? std::min(static_cast<unsigned int>(num_threads), static_cast<unsigned int>(batch_size)) : 1;

  if (active_threads <= 1)
  {
    process_forward_range(0, batch_size, batch_gradients_and_outputs, prev_layer_index, batch_residual_output_values, batch_hidden_states, is_training);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, &batch_gradients_and_outputs, prev_layer_index, &batch_residual_output_values, &batch_hidden_states, is_training, this]()
        {
          process_forward_range(start, end, batch_gradients_and_outputs, prev_layer_index, batch_residual_output_values, batch_hidden_states, is_training);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void TcnLayer::finish_hidden_gradients_range(
  size_t b_start,
  size_t b_end,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states,
  const std::vector<std::vector<double>>& raw_delta_all) const
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  const unsigned this_layer_index = get_layer_index();
  const size_t N_out = get_number_neurons();
  const size_t N_gathered = get_number_input_neurons();
  const size_t K = _kernel_size;
  const size_t D = _dilation;
  const size_t N_in = K > 0 ? N_gathered / K : 0;

  std::vector<double> deriv(N_out);
  std::vector<double> d_gathered(N_gathered);

  for (size_t b = b_start; b < b_end; ++b)
  {
    const auto& hs_row = batch_hidden_states[b].at(this_layer_index);
    const size_t T = hs_row.size();
    if (T == 0 || raw_delta_all[b].size() != T * N_out)
    {
      continue;
    }

    std::vector<double> d_pre_act(T * N_out);
    for (size_t t = 0; t < T; ++t)
    {
      const double* raw = raw_delta_all[b].data() + t * N_out;
      const double* pre_act = hs_row[t].get_pre_activation_sums().data();
      const double* y_vals = hs_row[t].get_hidden_state_values().data();
      const double* mask_vals = hs_row[t].get_cell_state_values().data();

      for (const auto& r : get_activation_helper().ranges())
      {
        r.activation_method.activate_derivative(pre_act + r.start, pre_act + r.end, y_vals + r.start, deriv.data() + r.start);
      }
      for (size_t o = 0; o < N_out; ++o)
      {
        d_pre_act[t * N_out + o] = raw[o] * deriv[o] * mask_vals[o];
      }
    }

    batch_gradients_and_outputs[b].set_gradients(this_layer_index, d_pre_act.data() + (T - 1) * N_out, N_out);
    batch_gradients_and_outputs[b].set_rnn_gate_gradients(this_layer_index, d_pre_act.data(), d_pre_act.size());

    std::vector<double> d_prev(T * N_in, 0.0);
    for (size_t t = 0; t < T; ++t)
    {
      const double* delta_t = d_pre_act.data() + t * N_out;
      for (size_t g = 0; g < N_gathered; ++g)
      {
        double sum = 0.0;
        const double* w_row = _w_values.data() + g * N_out;
        for (size_t o = 0; o < N_out; ++o)
        {
          sum += delta_t[o] * w_row[o];
        }
        d_gathered[g] = sum;
      }
      for (size_t j = 0; j < K; ++j)
      {
        if (t >= j * D)
        {
          const size_t s = t - j * D;
          simd::add_vectors(d_gathered.data() + j * N_in, d_prev.data() + s * N_in, N_in);
        }
      }
    }
    batch_gradients_and_outputs[b].set_rnn_gradients(this_layer_index, d_prev.data(), d_prev.size());
  }
}

void TcnLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t N_out = get_number_neurons();
  const size_t N_next = next_layer.get_number_neurons();
  const unsigned next_layer_index = next_layer.get_layer_index();
  const bool use_direct_gradients = batch_next_grad_matrix.empty();

  std::vector<std::vector<double>> raw_delta_all(batch_size);
  for (size_t b = 0; b < batch_size; ++b)
  {
    const size_t T = batch_hidden_states[b].at(get_layer_index()).size();
    auto& raw = raw_delta_all[b];
    raw.assign(T * N_out, 0.0);
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
      double* out_row = raw.data() + t * N_out;
      simd::gemm_transposed_one_batch(nd, W_next, out_row, N_out, N_next);
    }
  }

  const auto num_threads = get_number_of_threads();
  const unsigned int active_threads = (num_threads > 1) ? std::min(static_cast<unsigned int>(num_threads), static_cast<unsigned int>(batch_size)) : 1;
  if (active_threads <= 1)
  {
    finish_hidden_gradients_range(0, batch_size, batch_gradients_and_outputs, batch_hidden_states, raw_delta_all);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, &batch_gradients_and_outputs, &batch_hidden_states, &raw_delta_all, this]()
        {
          finish_hidden_gradients_range(start, end, batch_gradients_and_outputs, batch_hidden_states, raw_delta_all);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void TcnLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t N_out = get_number_neurons();
  const unsigned this_layer_index = get_layer_index();
  const bool use_direct_gradients = batch_output_gradients.empty();

  std::vector<std::vector<double>> raw_delta_all(batch_size);
  for (size_t b = 0; b < batch_size; ++b)
  {
    const size_t T = batch_hidden_states[b].at(this_layer_index).size();
    auto& raw = raw_delta_all[b];
    raw.assign(T * N_out, 0.0);
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

    if (src.size() == N_out)
    {
      for (size_t t = 0; t < T; ++t)
      {
        std::copy(src.begin(), src.end(), raw.begin() + t * N_out);
      }
    }
    else
    {
      const size_t copy_size = std::min(src.size(), T * N_out);
      std::copy(src.data(), src.data() + copy_size, raw.data());
    }
  }

  const auto num_threads = get_number_of_threads();
  const unsigned int active_threads = (num_threads > 1) ? std::min(static_cast<unsigned int>(num_threads), static_cast<unsigned int>(batch_size)) : 1;
  if (active_threads <= 1)
  {
    finish_hidden_gradients_range(0, batch_size, batch_gradients_and_outputs, batch_hidden_states, raw_delta_all);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, &batch_gradients_and_outputs, &batch_hidden_states, &raw_delta_all, this]()
        {
          finish_hidden_gradients_range(start, end, batch_gradients_and_outputs, batch_hidden_states, raw_delta_all);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void TcnLayer::accumulate_gradients_range(
  size_t b_start,
  size_t b_end,
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  unsigned prev_layer_index,
  std::vector<double>& local_w_grads,
  std::vector<double>& local_b_grads,
  size_t& local_contributing) const
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  const unsigned this_layer_index = get_layer_index();
  const size_t N_out = get_number_neurons();
  const size_t N_gathered = get_number_input_neurons();
  const size_t K = _kernel_size;
  const size_t D = _dilation;
  const size_t N_in = K > 0 ? N_gathered / K : 0;
  const bool use_bias = has_bias();

  std::vector<double> gathered(N_gathered);
  local_contributing = 0;

  for (size_t b = b_start; b < b_end; ++b)
  {
    const auto& own_delta = batch_gradients_and_outputs[b].get_rnn_gate_gradients(this_layer_index);
    const auto& seq = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const size_t T = (N_in > 0 && !seq.empty()) ? seq.size() / N_in : 0;
    if (T == 0 || own_delta.size() != T * N_out)
    {
      continue;
    }

    for (size_t t = 0; t < T; ++t)
    {
      std::fill(gathered.begin(), gathered.end(), 0.0);
      for (size_t j = 0; j < K; ++j)
      {
        if (t >= j * D)
        {
          const size_t s = t - j * D;
          const double* x_s = seq.data() + s * N_in;
          std::copy(x_s, x_s + N_in, gathered.begin() + j * N_in);
        }
      }

      const double* delta_t = own_delta.data() + t * N_out;
      for (size_t g = 0; g < N_gathered; ++g)
      {
        double* w_row = &local_w_grads[g * N_out];
        simd::mul_add(gathered[g], delta_t, w_row, N_out);
      }
      if (use_bias)
      {
        simd::add_vectors(delta_t, local_b_grads.data(), N_out);
      }
    }
    ++local_contributing;
  }
}

void TcnLayer::calculate_and_store_gradients(
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& hidden_states,
  const Layer& previous_layer,
  size_t batch_size,
  int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  (void)hidden_states;
  zero_gradients();
  if (batch_size == 0)
  {
    return;
  }

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const bool use_bias = has_bias();

  size_t contributing = 0;
  accumulate_gradients_range(0, batch_size, batch_gradients_and_outputs, prev_layer_index, _w_grads, _b_grads, contributing);

  if (contributing > 0)
  {
    const double inv = 1.0 / static_cast<double>(contributing);
    simd::scale_vector(_w_grads.data(), inv, _w_grads.size());
    if (use_bias)
    {
      simd::scale_vector(_b_grads.data(), inv, _b_grads.size());
    }
  }
}

double TcnLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  double norm_sq = simd::sum_sq(_w_grads.data(), _w_grads.size());
  if (has_bias())
  {
    norm_sq += simd::sum_sq(_b_grads.data(), _b_grads.size());
  }
  return norm_sq;
}

void TcnLayer::accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count)
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  const auto& other = static_cast<const TcnLayer&>(snapshot);
  swa_average_into(_w_values, other._w_values, existing_swa_count);
  swa_average_into(_b_values, other._b_values, existing_swa_count);
}

void TcnLayer::update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  auto& other = static_cast<TcnLayer&>(fast_layer);
  simd::lookahead_step(_w_values.data(), other._w_values.data(), alpha, _w_values.size());
  if (has_bias())
  {
    simd::lookahead_step(_b_values.data(), other._b_values.data(), alpha, _b_values.size());
  }
}

void TcnLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("TcnLayer");
  apply_update_to_vector(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, learning_rate, clipping_scale, false, _optimiser_type);
  if (has_bias())
  {
    apply_update_to_vector(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, learning_rate, clipping_scale, true, _optimiser_type);
  }
  if (!_w_grads.empty())
  {
    std::memset(_w_grads.data(), 0, _w_grads.size() * sizeof(double));
  }
  if (has_bias() && !_b_grads.empty())
  {
    std::memset(_b_grads.data(), 0, _b_grads.size() * sizeof(double));
  }
}

} // namespace myoddweb::nn
