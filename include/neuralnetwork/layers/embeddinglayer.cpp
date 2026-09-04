#include "../libraries/instrumentor.h"
#include "embeddinglayer.h"
#include "fflayer.h"
#include "../common/simd_utils.h"
#include "../common/logger.h"
#include "../common/tempbuffer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace myoddweb::nn
{
EmbeddingLayer::EmbeddingLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer,
  unsigned vocabulary_size,
  unsigned embedding_dimension,
  double weight_decay,
  const Role layer_role,
  const activation& activation_method,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  double momentum,
  std::optional<uint32_t> seed) :
  Layer(
    layer_index,
    layer_role,
    optimiser_type,
    residual_layer_number,
    create_neurons(dropout_rate, num_neurons_in_previous_layer * embedding_dimension, seed),
    create_embedding_weights(vocabulary_size, embedding_dimension, activation_method, seed),
    std::vector<double>(static_cast<size_t>(vocabulary_size) * embedding_dimension, 0.0),
    std::vector<double>(static_cast<size_t>(vocabulary_size) * embedding_dimension, 0.0),
    std::vector<double>(static_cast<size_t>(vocabulary_size) * embedding_dimension, 0.0),
    std::vector<double>(static_cast<size_t>(vocabulary_size) * embedding_dimension, 0.0),
    std::vector<long long>(static_cast<size_t>(vocabulary_size) * embedding_dimension, 0),
    std::vector<double>(static_cast<size_t>(vocabulary_size) * embedding_dimension, weight_decay),
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    residual_projector,
    number_of_threads,
    layer_activation_helper(activation_method, num_neurons_in_previous_layer, num_neurons_in_previous_layer * embedding_dimension),
    momentum),
  _vocabulary_size(vocabulary_size),
  _embedding_dimension(embedding_dimension)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  if (_vocabulary_size == 0)
  {
    throw std::invalid_argument("EmbeddingLayer: vocabulary_size must be greater than 0");
  }
  if (_embedding_dimension == 0)
  {
    throw std::invalid_argument("EmbeddingLayer: embedding_dimension must be greater than 0");
  }
}

EmbeddingLayer::EmbeddingLayer(
  unsigned layer_index,
  const Role layer_role,
  const OptimiserType optimiser_type,
  int residual_layer_number,
  unsigned vocabulary_size,
  unsigned embedding_dimension,
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
  const ResidualProjector* residual_projector,
  int number_of_threads,
  const layer_activation_helper& lah,
  double momentum) :
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
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    residual_projector,
    number_of_threads,
    lah,
    momentum),
  _vocabulary_size(vocabulary_size),
  _embedding_dimension(embedding_dimension)
{
  (void)number_input_neurons;
  (void)number_output_neurons;
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  if (_vocabulary_size == 0)
  {
    throw std::invalid_argument("EmbeddingLayer: vocabulary_size must be greater than 0");
  }
  if (_embedding_dimension == 0)
  {
    throw std::invalid_argument("EmbeddingLayer: embedding_dimension must be greater than 0");
  }
}

EmbeddingLayer::EmbeddingLayer(const EmbeddingLayer& src) noexcept :
  Layer(src),
  _vocabulary_size(src._vocabulary_size),
  _embedding_dimension(src._embedding_dimension)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
}

EmbeddingLayer::EmbeddingLayer(EmbeddingLayer&& src) noexcept :
  Layer(std::move(src)),
  _vocabulary_size(src._vocabulary_size),
  _embedding_dimension(src._embedding_dimension),
  _thread_w_grads(std::move(src._thread_w_grads))
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
}

EmbeddingLayer& EmbeddingLayer::operator=(const EmbeddingLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  if (this != &src)
  {
    Layer::operator=(src);
    _vocabulary_size = src._vocabulary_size;
    _embedding_dimension = src._embedding_dimension;
  }
  return *this;
}

EmbeddingLayer& EmbeddingLayer::operator=(EmbeddingLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  if (this != &src)
  {
    Layer::operator=(std::move(src));
    _vocabulary_size = src._vocabulary_size;
    _embedding_dimension = src._embedding_dimension;
    _thread_w_grads = std::move(src._thread_w_grads);
  }
  return *this;
}

EmbeddingLayer::~EmbeddingLayer()
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
}

Layer* EmbeddingLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  return new EmbeddingLayer(*this);
}

void EmbeddingLayer::zero_gradients()
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
}

void EmbeddingLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  if (batch_size == 0)
  {
    return;
  }

  const auto N_prev = get_number_input_neurons();
  const auto N_this = get_number_neurons();
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  // 1. Determine sequence length
  size_t num_time_steps = 1;
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty() && N_prev > 0)
    {
      num_time_steps = rnn_in.size() / N_prev;
      break;
    }
  }

  TempBuffer<double, 0> mask_buf(N_this);
  TempBuffer<double, 1> pre_act_buf(N_this);
  TempBuffer<double, 2> output_row_seq_buf(num_time_steps * N_this);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::fill(mask_buf.vec().begin(), mask_buf.vec().begin() + N_this, 1.0);

    if (!batch_hidden_states.empty())
    {
      if (batch_hidden_states[b].at(get_layer_index()).size() != num_time_steps)
      {
        batch_hidden_states[b].assign(get_layer_index(), num_time_steps, {}, get_pre_activation_multiplier());
      }
    }

    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    const bool is_rnn_in = !rnn_in.empty();
    const double* src_input_base = is_rnn_in ? rnn_in.data() : std_in.data();

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const double* in_step = is_rnn_in ? &src_input_base[t * N_prev] : src_input_base;
      double* current_pre_act = pre_act_buf.data();
      double* current_output_row = output_row_seq_buf.data() + t * N_this;

      // Lookup embedding vectors for each categorical input
      for (size_t k = 0; k < N_prev; ++k)
      {
        const double x_val = in_step[k];
        int cat_idx = std::clamp(static_cast<int>(std::round(x_val)), 0, static_cast<int>(_vocabulary_size - 1));
        const double* w_row = &_w_values[static_cast<size_t>(cat_idx) * _embedding_dimension];
        std::copy(w_row, w_row + _embedding_dimension, current_pre_act + k * _embedding_dimension);
      }

      // Add residual if present
      if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
      {
        if (num_time_steps == 1 || t == num_time_steps - 1)
        {
          simd::add_vectors(batch_residual_output_values[b].data(), current_pre_act, N_this);
        }
      }

      if (!batch_hidden_states.empty())
      {
        auto& layer_states_ref = batch_hidden_states[b].at(get_layer_index());
        layer_states_ref[t].set_pre_activation_sums(current_pre_act, N_this);
      }

      // Activations and dropout
      for (const auto& r : _layer_activation_helper.ranges())
      {
        r.activation_method.activate(current_pre_act + r.start, current_pre_act + r.end, is_training);
        if (is_training && get_dropout() > 0.0)
        {
          const auto& neurons = get_neurons();
          for (size_t j = r.start; j < r.end; ++j)
          {
            const auto& neuron = neurons[j];
            double output_val = current_pre_act[j];
            if (neuron.is_dropout())
            {
              if (neuron.must_randomly_drop(b * num_time_steps + t))
              {
                output_val = 0.0;
                mask_buf.data()[j] = 0.0;
              }
              else
              {
                double scale = 1.0 / (1.0 - neuron.get_dropout_rate());
                output_val *= scale;
                mask_buf.data()[j] = scale;
              }
            }
            current_output_row[j] = output_val;
          }
        }
        else
        {
          std::copy(current_pre_act + r.start, current_pre_act + r.end, current_output_row + r.start);
        }
      }

      if (!batch_hidden_states.empty())
      {
        auto& layer_states_ref = batch_hidden_states[b].at(get_layer_index());
        layer_states_ref[t].set_cell_state_values(mask_buf.data(), N_this);
        layer_states_ref[t].set_hidden_state_values(current_output_row, N_this);
      }
    }

    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), output_row_seq_buf.data() + (num_time_steps * N_this) - N_this, N_this);
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), output_row_seq_buf.data(), output_row_seq_buf.size());
  }
}

void EmbeddingLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
  (void)batch_gradients_and_outputs;
  (void)target_outputs_begin;
  (void)batch_hidden_states;
  (void)batch_size;
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  Logger::panic("EmbeddingLayer: Trying to calculate output gradient with a non output layer!");
}

void EmbeddingLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  const auto N_this = get_number_neurons();
  const auto N_next = next_layer.get_number_neurons();
  if (batch_size == 0)
  {
    return;
  }

  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0)
  {
    return;
  }

  TempBuffer<double, 3> flattened_next_grads_buffer(batch_size * num_time_steps * N_next, true);
  const bool use_direct_gradients = batch_next_grad_matrix.empty();
  for (size_t b = 0; b < batch_size; ++b)
  {
    std::span<const double> next_grads;
    if (use_direct_gradients)
    {
      next_grads = batch_gradients_and_outputs[b].get_gradients(next_layer.get_layer_index());
    }
    else
    {
      if (b < batch_next_grad_matrix.size())
      {
        next_grads = batch_next_grad_matrix[b];
      }
    }
    if (next_grads.empty())
    {
      continue;
    }

    double* dest_base = flattened_next_grads_buffer.data() + b * num_time_steps * N_next;
    const double* src_ptr = next_grads.data();
    if (next_grads.size() == N_next)
    {
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        std::copy(src_ptr, src_ptr + N_next, dest_base + t * N_next);
      }
    }
    else if (next_grads.size() == num_time_steps * N_next)
    {
      std::copy(src_ptr, src_ptr + num_time_steps * N_next, dest_base);
    }
    else
    {
      const size_t copy_size = std::min(next_grads.size(), num_time_steps * N_next);
      std::copy(src_ptr, src_ptr + copy_size, dest_base);
    }
  }

  const size_t effective_batch_size = batch_size * num_time_steps;
  TempBuffer<double, 4> flattened_this_grads_buffer(effective_batch_size * N_this, true);

  const FFLayer* ff_next = next_layer.is_ff_layer() ? static_cast<const FFLayer*>(&next_layer) : nullptr;
  const double* W_next_T = (ff_next != nullptr && !ff_next->get_w_values_T().empty()) ? ff_next->get_w_values_T().data() : nullptr;
  const double* W_next = (W_next_T == nullptr) ? next_layer.get_w_values().data() : nullptr;

  if (W_next_T != nullptr)
  {
    for (size_t eb = 0; eb < effective_batch_size; ++eb)
    {
      const double* g_next_row = flattened_next_grads_buffer.data() + eb * N_next;
      double* g_this_row = flattened_this_grads_buffer.data() + eb * N_this;
      for (size_t i = 0; i < N_this; ++i)
      {
        double sum = 0.0;
        for (size_t j = 0; j < N_next; ++j)
        {
          sum += g_next_row[j] * W_next_T[j * N_this + i];
        }
        g_this_row[i] = sum;
      }
    }
  }
  else if (W_next != nullptr)
  {
    for (size_t eb = 0; eb < effective_batch_size; ++eb)
    {
      const double* g_next_row = flattened_next_grads_buffer.data() + eb * N_next;
      double* g_this_row = flattened_this_grads_buffer.data() + eb * N_this;
      simd::gemm_transposed_one_batch(g_next_row, W_next, g_this_row, N_this, N_next);
    }
  }

  run_post_gemm_backward(0, batch_size, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer.vec());
}

void EmbeddingLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  const auto N_this = get_number_neurons();
  if (batch_size == 0)
  {
    return;
  }

  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0)
  {
    return;
  }

  const size_t effective_batch_size = batch_size * num_time_steps;
  TempBuffer<double, 4> flattened_this_grads_buffer(effective_batch_size * N_this, true);
  const bool use_direct_gradients = batch_output_gradients.empty();

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::span<const double> next_grads;
    if (use_direct_gradients)
    {
      const auto& rnn_g = batch_gradients_and_outputs[b].get_rnn_gradients(get_layer_index() + 1);
      if (!rnn_g.empty())
      {
        next_grads = rnn_g;
      }
      else
      {
        next_grads = batch_gradients_and_outputs[b].get_gradients(get_layer_index() + 1);
      }
    }
    else
    {
      if (b < batch_output_gradients.size())
      {
        next_grads = batch_output_gradients[b];
      }
    }
    if (next_grads.empty())
    {
      continue;
    }

    if (next_grads.size() == N_this)
    {
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        std::copy(next_grads.begin(), next_grads.end(), flattened_this_grads_buffer.vec().begin() + (b * num_time_steps + t) * N_this);
      }
    }
    else if (next_grads.size() == num_time_steps * N_this)
    {
      std::copy(next_grads.begin(), next_grads.end(), flattened_this_grads_buffer.vec().begin() + b * num_time_steps * N_this);
    }
    else
    {
      const size_t copy_size = std::min(next_grads.size(), num_time_steps * N_this);
      std::copy(next_grads.begin(), next_grads.begin() + copy_size, flattened_this_grads_buffer.vec().begin() + b * num_time_steps * N_this);
    }
  }

  run_post_gemm_backward(0, batch_size, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer.vec());
}

void EmbeddingLayer::run_post_gemm_backward(
  size_t start,
  size_t end,
  size_t N_this,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states,
  const std::vector<double>& flattened_this_grads_buffer) const
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  TempBuffer<double, 5> deriv_buf(N_this);
  TempBuffer<double, 6> rnn_grads_row(0);

  for (size_t b = start; b < end; ++b)
  {
    const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
    const size_t num_time_steps = layer_states.size();
    if (num_time_steps == 0)
    {
      continue;
    }

    rnn_grads_row.assign(num_time_steps * N_this, 0.0);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const double* g_this_row = &flattened_this_grads_buffer[(b * num_time_steps + t) * N_this];
      const auto& current_hidden_state = layer_states[t];
      const double* pre_act = current_hidden_state.get_pre_activation_sums().data();
      const double* mask_vals = current_hidden_state.get_cell_state_values().data();
      const double* y_vals = current_hidden_state.get_hidden_state_values().data();

      for (const auto& r : _layer_activation_helper.ranges())
      {
        r.activation_method.activate_derivative(
          pre_act + r.start,
          pre_act + r.end,
          y_vals + r.start,
          deriv_buf.data() + r.start
        );
        simd::mul_three_vectors(
          g_this_row + r.start,
          deriv_buf.data() + r.start,
          mask_vals + r.start,
          rnn_grads_row.data() + t * N_this + r.start,
          r.end - r.start
        );
      }
    }

    if (!rnn_grads_row.empty())
    {
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), rnn_grads_row.data() + rnn_grads_row.size() - N_this, N_this);
    }
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), rnn_grads_row.data(), rnn_grads_row.size());
  }
}

void EmbeddingLayer::calculate_and_store_gradients(
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& hidden_states,
  const Layer& previous_layer,
  size_t batch_size,
  int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  if (batch_size == 0)
  {
    return;
  }

  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();
  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const unsigned this_layer_index = get_layer_index();

  const size_t num_time_steps = hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0)
  {
    return;
  }

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * num_inputs * _embedding_dimension) / 100000))) : 1;

  if (active_threads == 1)
  {
    std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
    calculate_and_store_gradients_chunk(0, batch_size, batch_gradients_and_outputs, prev_layer_index, this_layer_index, num_inputs, num_outputs, num_time_steps, _w_grads);
  }
  else
  {
    if (_thread_w_grads.size() < active_threads)
    {
      _thread_w_grads.resize(active_threads, std::vector<double>(_w_grads.size(), 0.0));
    }
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      std::fill(_thread_w_grads[t].begin(), _thread_w_grads[t].end(), 0.0);
    }

    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, &batch_gradients_and_outputs, prev_layer_index, this_layer_index, num_inputs, num_outputs, num_time_steps, t, this]()
        {
          calculate_and_store_gradients_chunk(start, end, batch_gradients_and_outputs, prev_layer_index, this_layer_index, num_inputs, num_outputs, num_time_steps, _thread_w_grads[t]);
        });
      }
      start = end;
    }
    _task_queue_pool->get();

    std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      simd::add_vectors(_thread_w_grads[t].data(), _w_grads.data(), _w_grads.size());
    }
  }

  const double scale = 1.0 / static_cast<double>(batch_size);
  simd::scale_vector(_w_grads.data(), scale, _w_grads.size());
}

void EmbeddingLayer::calculate_and_store_gradients_chunk(
  size_t start,
  size_t end,
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  unsigned prev_layer_index,
  unsigned this_layer_index,
  unsigned num_inputs,
  unsigned num_outputs,
  size_t num_time_steps,
  std::vector<double>& w_grads_out) const
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  if (start >= end)
  {
    return;
  }

  const bool is_rnn_in = !batch_gradients_and_outputs[start].get_rnn_outputs(prev_layer_index).empty();
  const bool is_rnn_grad = batch_gradients_and_outputs[start].has_rnn_gradients(this_layer_index);

  for (size_t b = start; b < end; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const auto& std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    const double* x_base = is_rnn_in ? rnn_in.data() : std_in.data();

    const auto& rnn_grad = batch_gradients_and_outputs[b].get_rnn_gradients(this_layer_index);
    const auto& std_grad = batch_gradients_and_outputs[b].get_gradients(this_layer_index);
    const double* g_base = is_rnn_grad ? rnn_grad.data() : std_grad.data();

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const double* x_t = is_rnn_in ? &x_base[t * num_inputs] : x_base;
      const double* g_t = is_rnn_grad ? &g_base[t * num_outputs] : g_base;

      for (size_t k = 0; k < num_inputs; ++k)
      {
        const double x_val = x_t[k];
        int cat_idx = std::clamp(static_cast<int>(std::round(x_val)), 0, static_cast<int>(_vocabulary_size - 1));
        double* w_grad_row = &w_grads_out[static_cast<size_t>(cat_idx) * _embedding_dimension];
        const double* g_slice = &g_t[k * _embedding_dimension];
        simd::add_vectors(g_slice, w_grad_row, _embedding_dimension);
      }
    }
  }
}

double EmbeddingLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  return simd::sum_sq(_w_grads.data(), _w_grads.size());
}

void EmbeddingLayer::accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  const auto& other = static_cast<const EmbeddingLayer&>(snapshot);
  swa_average_into(_w_values, other._w_values, existing_swa_count);
}

void EmbeddingLayer::update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  auto& other = static_cast<EmbeddingLayer&>(fast_layer);
  simd::lookahead_step(_w_values.data(), other._w_values.data(), alpha, _w_values.size());
}

void EmbeddingLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  apply_update_to_vector(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, learning_rate, clipping_scale, false, _optimiser_type);
  if (!_w_grads.empty())
  {
    std::memset(_w_grads.data(), 0, _w_grads.size() * sizeof(double));
  }
}

std::vector<double> EmbeddingLayer::create_embedding_weights(
  unsigned vocabulary_size,
  unsigned embedding_dimension,
  const activation& activation_method,
  std::optional<uint32_t> seed)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayer");
  const size_t num_weights = static_cast<size_t>(vocabulary_size) * embedding_dimension;
  std::vector<double> w_values(num_weights);
  layer_activation_helper helper(activation_method, vocabulary_size, embedding_dimension);
  for (size_t i = 0; i < vocabulary_size; ++i)
  {
    for (size_t j = 0; j < embedding_dimension; ++j)
    {
      const size_t flat_index = i * embedding_dimension + j;
      const auto weight_seed = seed.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(Rng::derive(seed.value(), flat_index))) : std::nullopt;
      w_values[flat_index] = helper.weight_initialization(static_cast<unsigned>(j), weight_seed);
    }
  }
  return w_values;
}

} // namespace myoddweb::nn
