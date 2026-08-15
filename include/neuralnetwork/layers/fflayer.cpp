#include "../libraries/instrumentor.h"
#include "fflayer.h"
#include "../common/simd_utils.h"
#include "../common/logger.h"
#include "../common/tempbuffer.h"
#include <cstring>
#include <numeric>


namespace myoddweb::nn
{
FFLayer::FFLayer(
  unsigned layer_index,
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
  double momentum
) :
  FFLayer(
    layer_index,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer)* num_neurons_in_this_layer, weight_decay),
    layer_role,
    activation_method,
    optimiser_type,
    residual_layer_number,
    dropout_rate,
    residual_projector,
    number_of_threads,
    has_bias,
    momentum
  )
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(
  unsigned layer_index,
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
  double momentum
) :
  FFLayer(
    layer_index,
    weight_decays,
    layer_role,
    layer_activation_helper(activation_method, num_neurons_in_previous_layer, num_neurons_in_this_layer),
    optimiser_type,
    residual_layer_number,
    dropout_rate,
    residual_projector,
    number_of_threads,
    has_bias,
    momentum
  )
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(
  unsigned layer_index,
  const std::vector<double>& weight_decays,
  const Role layer_role,
  const layer_activation_helper& lah,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias,
  double momentum
) :
  Layer(
    layer_index,
    layer_role,
    lah,
    optimiser_type,
    residual_layer_number,
    create_neurons(dropout_rate, lah.get_number_output_neurons()),
    has_bias,
    weight_decays,
    residual_projector,
    number_of_threads,
    momentum
  )
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  cache_recurrent_weights();
}

FFLayer::FFLayer(const FFLayer& src) noexcept :
  Layer(src),
  _w_values_T(src._w_values_T)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(
  unsigned layer_index,
  const Role layer_role,
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
  const ResidualProjector* residual_projector,
  int number_of_threads,
  const layer_activation_helper& lah,
  double momentum
) noexcept :
  Layer
  (
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
  )
{
  (void)number_input_neurons;
  (void)number_output_neurons;
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  cache_recurrent_weights();
}

FFLayer::FFLayer(FFLayer&& src) noexcept :
  Layer(std::move(src)),
  _w_values_T(std::move(src._w_values_T))
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer& FFLayer::operator=(const FFLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (this != &src)
  {
    Layer::operator=(src);
    _w_values_T = src._w_values_T;
  }
  return *this;
}

FFLayer& FFLayer::operator=(FFLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (this != &src)
  {
    Layer::operator=(std::move(src));
    _w_values_T = std::move(src._w_values_T);
  }
  return *this;
}

FFLayer::~FFLayer()
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

void FFLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (batch_size == 0) return;

  const auto N_prev = get_number_input_neurons();
  const auto N_this = get_number_neurons();
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  // 1. Determine sequence length and flatten inputs
  size_t num_time_steps = 1;
  for (size_t b = 0; b < batch_size; ++b)
  {
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      if (!rnn_in.empty())
      {
          num_time_steps = rnn_in.size() / N_prev;
          break;
      }
  }

  const size_t effective_batch_size = batch_size * num_time_steps;
  TempBuffer<double, 0> batch_inputs_buffer(effective_batch_size * N_prev);
  
  double* in_buf_ptr = batch_inputs_buffer.data();
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    double* dest_base = in_buf_ptr + b * num_time_steps * N_prev;
    if (!rnn_in.empty())
    {
      const size_t copy_size = std::min(rnn_in.size(), num_time_steps * N_prev);
      std::copy(rnn_in.data(), rnn_in.data() + copy_size, dest_base);
    }
    else
    {
      const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      const size_t copy_size = std::min<size_t>(std_in.size(), N_prev);
      if (num_time_steps == 1)
      {
        std::copy(std_in.data(), std_in.data() + copy_size, dest_base);
      }
      else
      {
        for (size_t t = 0; t < num_time_steps; ++t)
        {
          std::copy(std_in.data(), std_in.data() + copy_size, dest_base + t * N_prev);
        }
      }
    }
  }

  TempBuffer<double, 1> batch_pre_activation_sums_buffer(effective_batch_size * N_this, true);

  // 2. Initialize with bias values
  if (has_bias())
  {
    const auto& biases = get_b_values();
    const size_t copy_size = std::min<size_t>(biases.size(), N_this);
    const double* b_ptr = biases.data();
    double* dest_base = batch_pre_activation_sums_buffer.data();
    for (size_t eb = 0; eb < effective_batch_size; ++eb)
    {
      double* dest = dest_base + eb * N_this;
      std::copy(b_ptr, b_ptr + copy_size, dest);
      if (copy_size < N_this)
      {
        std::fill(dest + copy_size, dest + N_this, 0.0);
      }
    }
  }

  // 3. Batched Matrix-Matrix multiplication (GEMM)
  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_gemm_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((effective_batch_size * N_prev * N_this) / 100000))) : 1;
  const bool use_gemm_mt = is_training && (active_gemm_threads > 1);
  if (!use_gemm_mt)
  {
    run_gemm(0, effective_batch_size, N_prev, N_this, batch_inputs_buffer.vec(), batch_pre_activation_sums_buffer.vec());
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_gemm_threads; ++t)
    {
      size_t size = (effective_batch_size / active_gemm_threads) + (t < (effective_batch_size % active_gemm_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, N_prev, N_this, &batch_inputs_buffer, &batch_pre_activation_sums_buffer, this]()
        {
          run_gemm(start, end, N_prev, N_this, batch_inputs_buffer.vec(), batch_pre_activation_sums_buffer.vec());
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 4. Residuals, Activation and Dropout
  const unsigned int active_post_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * N_this) / 50000))) : 1;
  const bool use_post_mt = is_training && (active_post_threads > 1);
  if (!use_post_mt)
  {
    run_post_gemm(0, batch_size, num_time_steps, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, batch_inputs_buffer.vec(), batch_pre_activation_sums_buffer.vec(), is_training);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_post_threads; ++t)
    {
      size_t size = (batch_size / active_post_threads) + (t < (batch_size % active_post_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, num_time_steps, N_this, &batch_gradients_and_outputs, &batch_residual_output_values, &batch_hidden_states, &batch_inputs_buffer, &batch_pre_activation_sums_buffer, is_training, this]()
        {
          run_post_gemm(start, end, num_time_steps, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, batch_inputs_buffer.vec(), batch_pre_activation_sums_buffer.vec(), is_training);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void FFLayer::run_gemm(
  size_t b_start,
  size_t b_end,
  size_t N_prev,
  size_t N_this,
  const std::vector<double>& batch_inputs_buffer,
  std::vector<double>& batch_pre_activation_sums_buffer) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const double* W = get_w_values().data();
  size_t b = b_start;
  for (; b + 3 < b_end; b += 4)
  {
    const double* x0 = &batch_inputs_buffer[b * N_prev];
    const double* x1 = &batch_inputs_buffer[(b + 1) * N_prev];
    const double* x2 = &batch_inputs_buffer[(b + 2) * N_prev];
    const double* x3 = &batch_inputs_buffer[(b + 3) * N_prev];

    double* y0 = &batch_pre_activation_sums_buffer[b * N_this];
    double* y1 = &batch_pre_activation_sums_buffer[(b + 1) * N_this];
    double* y2 = &batch_pre_activation_sums_buffer[(b + 2) * N_this];
    double* y3 = &batch_pre_activation_sums_buffer[(b + 3) * N_this];

    simd::gemm_four_batches(
      x0, x1, x2, x3,
      W,
      y0, y1, y2, y3,
      N_prev, N_this
    );
  }

  // Cleanup loops
  for (; b + 1 < b_end; b += 2)
  {
    const double* x0 = &batch_inputs_buffer[b * N_prev];
    const double* x1 = &batch_inputs_buffer[(b + 1) * N_prev];

    double* y0 = &batch_pre_activation_sums_buffer[b * N_this];
    double* y1 = &batch_pre_activation_sums_buffer[(b + 1) * N_this];

    simd::gemm_two_batches(
      x0, x1,
      W,
      y0, y1,
      N_prev, N_this
    );
  }

  for (; b < b_end; ++b)
  {
    const double* x_row = &batch_inputs_buffer[b * N_prev];
    double* y_row = &batch_pre_activation_sums_buffer[b * N_this];

    simd::gemm_one_batch(
      x_row,
      W,
      y_row,
      N_prev, N_this
    );
  }
}

void FFLayer::run_post_gemm(
  size_t start,
  size_t end,
  size_t num_time_steps,
  size_t N_this,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  const std::vector<double>& /*batch_inputs_buffer*/,
  std::vector<double>& batch_pre_activation_sums_buffer,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  // Reuse thread-local buffers via TempBuffer to avoid repeated heap allocations.
  TempBuffer<double, 7> mask_buf(N_this);
  TempBuffer<double, 8> output_row_seq_buf(num_time_steps * N_this);

  for (size_t b = start; b < end; b++)
  {
    std::fill(mask_buf.vec().begin(), mask_buf.vec().begin() + N_this, 1.0);
    if (!batch_hidden_states.empty())
    {
      if (batch_hidden_states[b].at(get_layer_index()).size() != num_time_steps)
      {
        batch_hidden_states[b].assign(get_layer_index(), num_time_steps, {}, get_pre_activation_multiplier());
      }
    }

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      double* current_pre_act = &batch_pre_activation_sums_buffer[(b * num_time_steps + t) * N_this];
      double* current_output_row = output_row_seq_buf.data() + t * N_this;

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

      for (const auto& r : _layer_activation_helper.ranges())
      {
        r.activation_method.activate(current_pre_act + r.start, current_pre_act + r.end, is_training);
        if (is_training && get_dropout() > 0.0)
        {
          const auto& neurons = get_neurons();
          for (size_t j = r.start; j < r.end; j++)
          {
            const auto& neuron = neurons[j];
            double output = current_pre_act[j];
            if (neuron.is_dropout())
            {
              if (neuron.must_randomly_drop())
              {
                output = 0.0;
                mask_buf.data()[j] = 0.0;
              }
              else
              {
                double scale = 1.0 / (1.0 - neuron.get_dropout_rate());
                output *= scale;
                mask_buf.data()[j] = scale;
              }
            }
            current_output_row[j] = output;
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

    double* dest_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    std::copy(output_row_seq_buf.data() + (num_time_steps * N_this) - N_this, output_row_seq_buf.data() + (num_time_steps * N_this), dest_ptr);
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), output_row_seq_buf.data(), output_row_seq_buf.size());
  }
}

void FFLayer::calculate_output_gradients(std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, std::vector<std::vector<double>>::const_iterator target_outputs_begin, const std::vector<HiddenStates>& batch_hidden_states, size_t batch_size) const
{
  (void)batch_gradients_and_outputs;
  (void)target_outputs_begin;
  (void)batch_hidden_states;
  (void)batch_size;
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  Logger::panic("FFLayer: Trying to calculate output gradient with a non output layer!");
}

void FFLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const auto N_this = get_number_neurons();
  const auto N_next = next_layer.get_number_neurons();
  if (batch_size == 0) return;

  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0) return;

  TempBuffer<double, 2> flattened_next_grads_buffer(batch_size * num_time_steps * N_next, true);
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
      // Broadcast single gradient to all time steps
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
  TempBuffer<double, 3> flattened_this_grads_buffer(effective_batch_size * N_this, true);

  const FFLayer* ff_next = next_layer.is_ff_layer() ? static_cast<const FFLayer*>(&next_layer) : nullptr;
  const double* W_next_T = (ff_next != nullptr && !ff_next->get_w_values_T().empty()) ? ff_next->get_w_values_T().data() : nullptr;
  const double* W_next = (W_next_T == nullptr) ? next_layer.get_w_values().data() : nullptr;

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_gemm_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((effective_batch_size * N_next * N_this) / 100000))) : 1;
  const bool use_gemm_mt = (active_gemm_threads > 1);
  const unsigned int active_post_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * N_this) / 50000))) : 1;
  const bool use_post_mt = (active_post_threads > 1);

  if (!use_gemm_mt && !use_post_mt)
  {
    if (W_next_T != nullptr)
    {
      run_gemm_backward_fast(0, effective_batch_size, N_next, N_this, W_next_T, flattened_next_grads_buffer.vec(), flattened_this_grads_buffer.vec());
    }
    else
    {
      run_gemm_backward(0, effective_batch_size, N_next, N_this, W_next, flattened_next_grads_buffer.vec(), flattened_this_grads_buffer.vec());
    }
    run_post_gemm_backward(0, batch_size, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer.vec());
  }
  else
  {
    if (!use_gemm_mt)
    {
      if (W_next_T != nullptr)
      {
        run_gemm_backward_fast(0, effective_batch_size, N_next, N_this, W_next_T, flattened_next_grads_buffer.vec(), flattened_this_grads_buffer.vec());
      }
      else
      {
        run_gemm_backward(0, effective_batch_size, N_next, N_this, W_next, flattened_next_grads_buffer.vec(), flattened_this_grads_buffer.vec());
      }
    }
    else
    {
      size_t start = 0;
      for (unsigned int t = 0; t < active_gemm_threads; ++t)
      {
        size_t size = (effective_batch_size / active_gemm_threads) + (t < (effective_batch_size % active_gemm_threads) ? 1 : 0);
        size_t end = start + size;
        if (start < end)
        {
          if (W_next_T != nullptr)
          {
            _task_queue_pool->enqueue([start, end, N_next, N_this, W_next_T, &flattened_next_grads_buffer, &flattened_this_grads_buffer, this]()
            {
              run_gemm_backward_fast(start, end, N_next, N_this, W_next_T, flattened_next_grads_buffer.vec(), flattened_this_grads_buffer.vec());
            });
          }
          else
          {
            _task_queue_pool->enqueue([start, end, N_next, N_this, W_next, &flattened_next_grads_buffer, &flattened_this_grads_buffer, this]()
            {
              run_gemm_backward(start, end, N_next, N_this, W_next, flattened_next_grads_buffer.vec(), flattened_this_grads_buffer.vec());
            });
          }
        }
        start = end;
      }
      _task_queue_pool->get();
    }

    if (!use_post_mt)
    {
      run_post_gemm_backward(0, batch_size, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer.vec());
    }
    else
    {
      size_t start = 0;
      for (unsigned int t = 0; t < active_post_threads; ++t)
      {
        size_t size = (batch_size / active_post_threads) + (t < (batch_size % active_post_threads) ? 1 : 0);
        size_t end = start + size;
        if (start < end) _task_queue_pool->enqueue([start, end, N_this, &batch_gradients_and_outputs, &batch_hidden_states, &flattened_this_grads_buffer, this]() { run_post_gemm_backward(start, end, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer.vec()); });
        start = end;
      }
      _task_queue_pool->get();
    }
  }
}

void FFLayer::calculate_hidden_gradients_from_output_gradients(std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, const std::vector<std::vector<double>>& batch_output_gradients, const std::vector<HiddenStates>& batch_hidden_states, size_t batch_size, int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const auto N_this = get_number_neurons();
  if (batch_size == 0) return;

  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0) return;

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

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * N_this) / 50000))) : 1;
  const bool use_multithreading = (active_threads > 1);
  if (!use_multithreading)
  {
    run_post_gemm_backward(0, batch_size, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer.vec());
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
        _task_queue_pool->enqueue([start, end, N_this, &batch_gradients_and_outputs, &batch_hidden_states, &flattened_this_grads_buffer, this]() 
          { 
            run_post_gemm_backward(start, end, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer.vec()); 
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

Layer* FFLayer::clone() const { MYODDWEB_PROFILE_FUNCTION("FFLayer"); return new FFLayer(*this); }

void FFLayer::calculate_and_store_gradients(const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, const std::vector<HiddenStates>& hidden_states, const Layer& previous_layer, size_t batch_size, int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
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
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * num_inputs * num_outputs) / 100000))) : 1;

  if (active_threads == 1)
  {
    std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
    if (has_bias())
    {
      std::fill(_b_grads.begin(), _b_grads.end(), 0.0);
    }
    calculate_and_store_gradients_chunk(0, batch_size, batch_gradients_and_outputs, prev_layer_index, this_layer_index, num_inputs, num_outputs, num_time_steps, _w_grads, _b_grads);
  }
  else
  {
    if (_thread_w_grads.size() < active_threads)
    {
      _thread_w_grads.resize(active_threads);
    }
    if (_thread_b_grads.size() < active_threads)
    {
      _thread_b_grads.resize(active_threads);
    }

    for (unsigned int t = 0; t < active_threads; ++t)
    {
      _thread_w_grads[t].resize(_w_grads.size());
      std::fill(_thread_w_grads[t].begin(), _thread_w_grads[t].end(), 0.0);
      _thread_b_grads[t].resize(has_bias() ? num_outputs : 0);
      std::fill(_thread_b_grads[t].begin(), _thread_b_grads[t].end(), 0.0);
    }

    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        auto& thread_w_grads_t = _thread_w_grads[t];
        auto& thread_b_grads_t = _thread_b_grads[t];
        _task_queue_pool->enqueue(
          [
            this,
            start,
            end,
            &batch_gradients_and_outputs,
            prev_layer_index,
            this_layer_index,
            num_inputs,
            num_outputs,
            num_time_steps,
            &thread_w_grads_t,
            &thread_b_grads_t
          ]()
        {
          calculate_and_store_gradients_chunk(start, end, batch_gradients_and_outputs, prev_layer_index, this_layer_index, num_inputs, num_outputs, num_time_steps, thread_w_grads_t, thread_b_grads_t);
        });
      }
      start = end;
    }
    _task_queue_pool->get();

    // Merge results
    std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
    if (has_bias())
    {
      std::fill(_b_grads.begin(), _b_grads.end(), 0.0);
    }

    for (unsigned int t = 0; t < active_threads; ++t)
    {
      simd::add_vectors(_thread_w_grads[t].data(), _w_grads.data(), _w_grads.size());
      if (has_bias())
      {
        simd::add_vectors(_thread_b_grads[t].data(), _b_grads.data(), _b_grads.size());
      }
    }
  }

  const double inv_batch = 1.0 / static_cast<double>(batch_size);
  simd::scale_vector(_w_grads.data(), inv_batch, _w_grads.size());
  if (has_bias())
  {
    simd::scale_vector(_b_grads.data(), inv_batch, _b_grads.size());
  }
}

double FFLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  double norm_sq = simd::sum_sq(_w_grads.data(), _w_grads.size());
  if (has_bias())
  {
    norm_sq += simd::sum_sq(_b_grads.data(), _b_grads.size());
  }
  return norm_sq;
}

void FFLayer::accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const auto& other = static_cast<const FFLayer&>(snapshot);
  swa_average_into(_w_values, other._w_values, existing_swa_count);
  swa_average_into(_b_values, other._b_values, existing_swa_count);
}

void FFLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  apply_update_to_vector(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, learning_rate, clipping_scale, false, _optimiser_type);
  if (has_bias()) apply_update_to_vector(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, learning_rate, clipping_scale, true, _optimiser_type);
  if (!_w_grads.empty()) std::memset(_w_grads.data(), 0, _w_grads.size() * sizeof(double));
  if (has_bias() && !_b_grads.empty()) std::memset(_b_grads.data(), 0, _b_grads.size() * sizeof(double));
  cache_recurrent_weights();
}

void FFLayer::run_gemm_backward(
  size_t b_start,
  size_t b_end,
  size_t N_next,
  size_t N_this,
  const double* W_next,
  const std::vector<double>& flattened_next_grads_buffer,
  std::vector<double>& flattened_this_grads_buffer) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  size_t b = b_start;
  for (; b + 3 < b_end; b += 4)
  {
    const double* x0 = &flattened_next_grads_buffer[b * N_next];
    const double* x1 = &flattened_next_grads_buffer[(b + 1) * N_next];
    const double* x2 = &flattened_next_grads_buffer[(b + 2) * N_next];
    const double* x3 = &flattened_next_grads_buffer[(b + 3) * N_next];

    double* y0 = &flattened_this_grads_buffer[b * N_this];
    double* y1 = &flattened_this_grads_buffer[(b + 1) * N_this];
    double* y2 = &flattened_this_grads_buffer[(b + 2) * N_this];
    double* y3 = &flattened_this_grads_buffer[(b + 3) * N_this];

    simd::gemm_transposed_four_batches(
      x0, x1, x2, x3,
      W_next,
      y0, y1, y2, y3,
      N_this, N_next
    );
  }

  for (; b + 1 < b_end; b += 2)
  {
    const double* x0 = &flattened_next_grads_buffer[b * N_next];
    const double* x1 = &flattened_next_grads_buffer[(b + 1) * N_next];

    double* y0 = &flattened_this_grads_buffer[b * N_this];
    double* y1 = &flattened_this_grads_buffer[(b + 1) * N_this];

    simd::gemm_transposed_two_batches(
      x0, x1,
      W_next,
      y0, y1,
      N_this, N_next
    );
  }

  for (; b < b_end; ++b)
  {
    simd::gemm_transposed_one_batch(
      &flattened_next_grads_buffer[b * N_next],
      W_next,
      &flattened_this_grads_buffer[b * N_this],
      N_this, N_next
    );
  }
}

void FFLayer::run_post_gemm_backward(
  size_t start,
  size_t end,
  size_t N_this,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states,
  const std::vector<double>& flattened_this_grads_buffer) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  
  TempBuffer<double, 5> deriv_buf(N_this);
  TempBuffer<double, 6> rnn_grads_row(0);
  for (size_t b = start; b < end; b++)
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

void FFLayer::calculate_and_store_gradients_chunk(
  size_t start,
  size_t end,
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  unsigned prev_layer_index,
  unsigned this_layer_index,
  unsigned num_inputs,
  unsigned num_outputs,
  size_t num_time_steps,
  std::vector<double>& local_w_grads,
  std::vector<double>& local_b_grads) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (start >= end)
  {
    return;
  }

  const bool is_rnn_in = !batch_gradients_and_outputs[start].get_rnn_outputs(prev_layer_index).empty();
  const bool is_rnn_grad = batch_gradients_and_outputs[start].has_rnn_gradients(this_layer_index);

  if (has_bias())
  {
    for (size_t b = start; b < end; ++b)
    {
      const auto& rnn_grad = batch_gradients_and_outputs[b].get_rnn_gradients(this_layer_index);
      const auto& std_grad = batch_gradients_and_outputs[b].get_gradients(this_layer_index);
      const double* g_base = is_rnn_grad ? rnn_grad.data() : std_grad.data();
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        const double* g_t = is_rnn_grad ? &g_base[t * num_outputs] : g_base;
        simd::add_vectors(g_t, local_b_grads.data(), num_outputs);
      }
    }
  }

  // Weight gradients (outer loop over input neuron i to eliminate L1 cache thrashing)
  for (size_t i = 0; i < num_inputs; ++i)
  {
    double* w_grad_row = &local_w_grads[i * num_outputs];
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
        const double x_val = x_t[i];
        simd::mul_add(x_val, g_t, w_grad_row, num_outputs);
      }
    }
  }
}


void FFLayer::cache_recurrent_weights()
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t n_prev = get_number_input_neurons();
  const size_t n_this = get_number_neurons();
  const auto& w_vals = get_w_values();
  if (w_vals.empty() || n_prev == 0 || n_this == 0)
  {
    return;
  }
  _w_values_T.resize(n_this * n_prev);
  simd::transpose(w_vals.data(), _w_values_T.data(), n_prev, n_this);
}

void FFLayer::run_gemm_backward_fast(
  size_t b_start,
  size_t b_end,
  size_t N_next,
  size_t N_this,
  const double* W_next_T,
  const std::vector<double>& flattened_next_grads_buffer,
  std::vector<double>& flattened_this_grads_buffer) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  size_t b = b_start;
  for (; b + 3 < b_end; b += 4)
  {
    const double* x0 = &flattened_next_grads_buffer[b * N_next];
    const double* x1 = &flattened_next_grads_buffer[(b + 1) * N_next];
    const double* x2 = &flattened_next_grads_buffer[(b + 2) * N_next];
    const double* x3 = &flattened_next_grads_buffer[(b + 3) * N_next];

    double* y0 = &flattened_this_grads_buffer[b * N_this];
    double* y1 = &flattened_this_grads_buffer[(b + 1) * N_this];
    double* y2 = &flattened_this_grads_buffer[(b + 2) * N_this];
    double* y3 = &flattened_this_grads_buffer[(b + 3) * N_this];

    simd::gemm_four_batches(
      x0, x1, x2, x3,
      W_next_T,
      y0, y1, y2, y3,
      N_next, N_this
    );
  }

  for (; b + 1 < b_end; b += 2)
  {
    const double* x0 = &flattened_next_grads_buffer[b * N_next];
    const double* x1 = &flattened_next_grads_buffer[(b + 1) * N_next];

    double* y0 = &flattened_this_grads_buffer[b * N_this];
    double* y1 = &flattened_this_grads_buffer[(b + 1) * N_this];

    simd::gemm_two_batches(
      x0, x1,
      W_next_T,
      y0, y1,
      N_next, N_this
    );
  }

  for (; b < b_end; ++b)
  {
    const double* x_row = &flattened_next_grads_buffer[b * N_next];
    double* y_row = &flattened_this_grads_buffer[b * N_this];

    simd::gemm_one_batch(
      x_row,
      W_next_T,
      y_row,
      N_next, N_this
    );
  }
}

} // namespace myoddweb::nn
