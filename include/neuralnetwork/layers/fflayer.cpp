#include "../libraries/instrumentor.h"
#include "fflayer.h"
#include "../common/simd_utils.h"
#include "../common/logger.h"
#include "../common/tempbuffer.h"
#include <array>
#include <cstring>
#include <numeric>


namespace myoddweb::nn
{
namespace
{
struct FfForwardTask
{
  const FFLayer* layer;
  size_t start;
  size_t end;
  size_t num_time_steps;
  size_t n_prev;
  size_t n_this;
  std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
  const std::vector<std::vector<double>>* batch_residual_output_values;
  std::vector<HiddenStates>* batch_hidden_states;
  const double* in_buf_ptr;
  double* pre_act_ptr;
  bool is_training;

  void operator()() const
  {
    layer->run_forward_chunk(
      start,
      end,
      num_time_steps,
      n_prev,
      n_this,
      *batch_gradients_and_outputs,
      *batch_residual_output_values,
      *batch_hidden_states,
      in_buf_ptr,
      pre_act_ptr,
      is_training
    );
  }
};

struct FfBackwardTask
{
  const FFLayer* layer;
  size_t start;
  size_t end;
  size_t num_time_steps;
  size_t n_next;
  size_t n_this;
  const double* w_next;
  const double* w_next_t;
  const double* next_grads_ptr;
  double* this_grads_ptr;
  std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
  const std::vector<HiddenStates>* batch_hidden_states;

  void operator()() const
  {
    layer->run_backward_chunk(
      start,
      end,
      num_time_steps,
      n_next,
      n_this,
      w_next,
      w_next_t,
      next_grads_ptr,
      this_grads_ptr,
      *batch_gradients_and_outputs,
      *batch_hidden_states
    );
  }
};

struct FfPostGemmBackwardTask
{
  const FFLayer* layer;
  size_t start;
  size_t end;
  size_t n_this;
  std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
  const std::vector<HiddenStates>* batch_hidden_states;
  const double* this_grads_ptr;

  void operator()() const
  {
    layer->run_post_gemm_backward(
      start,
      end,
      n_this,
      *batch_gradients_and_outputs,
      *batch_hidden_states,
      this_grads_ptr
    );
  }
};

struct FfGradsTask
{
  const FFLayer* layer;
  size_t start;
  size_t end;
  const std::vector<GradientsAndOutputs>* batch_gradients_and_outputs;
  unsigned prev_layer_index;
  unsigned this_layer_index;
  unsigned num_inputs;
  unsigned num_outputs;
  size_t num_time_steps;
  std::span<double> local_w_grads;
  std::span<double> local_b_grads;

  void operator()() const
  {
    layer->calculate_and_store_gradients_chunk(
      start,
      end,
      *batch_gradients_and_outputs,
      prev_layer_index,
      this_layer_index,
      num_inputs,
      num_outputs,
      num_time_steps,
      local_w_grads,
      local_b_grads
    );
  }
};

static std::span<const double> get_output_grads_span(
  const GradientsAndOutputs& go,
  unsigned next_layer_idx,
  unsigned this_layer_idx)
{
  const auto& rnn_next = go.get_rnn_gradients(next_layer_idx);
  if (!rnn_next.empty())
  {
    return rnn_next;
  }
  if (next_layer_idx < go.number_layers())
  {
    const auto std_next = go.get_gradients(next_layer_idx);
    if (!std_next.empty())
    {
      return std_next;
    }
  }
  const auto& rnn_this = go.get_rnn_gradients(this_layer_idx);
  if (!rnn_this.empty())
  {
    return rnn_this;
  }
  return go.get_gradients(this_layer_idx);
}
} // namespace

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
  double momentum,
  std::optional<uint32_t> seed
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
    momentum,
    seed
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
  double momentum,
  std::optional<uint32_t> seed
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
    momentum,
    seed
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
  double momentum,
  std::optional<uint32_t> seed
) :
  Layer(
    layer_index,
    layer_role,
    lah,
    optimiser_type,
    residual_layer_number,
    create_neurons(dropout_rate, lah.get_number_output_neurons(), seed),
    has_bias,
    weight_decays,
    residual_projector,
    number_of_threads,
    momentum,
    seed
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
  if (batch_size == 0)
  {
    return;
  }

  const auto N_prev = get_number_input_neurons();
  const auto N_this = get_number_neurons();
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  // 1. Determine sequence length and flatten inputs
  size_t num_time_steps = 1;
  if (N_prev > 0)
  {
    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      if (!rnn_in.empty())
      {
        num_time_steps = rnn_in.size() / N_prev;
        break;
      }
    }
  }

  if (!batch_hidden_states.empty())
  {
    const size_t hs_steps = batch_hidden_states[0].at(get_layer_index()).size();
    if (hs_steps > num_time_steps)
    {
      num_time_steps = hs_steps;
    }
  }

  const size_t effective_batch_size = batch_size * num_time_steps;
  const double* in_buf_ptr = nullptr;

  const bool can_bypass_input = (batch_size == 1) &&
    ((!batch_gradients_and_outputs[0].get_rnn_outputs(prev_layer_index).empty() &&
      batch_gradients_and_outputs[0].get_rnn_outputs(prev_layer_index).size() == num_time_steps * N_prev) ||
     (num_time_steps == 1 && batch_gradients_and_outputs[0].get_outputs(prev_layer_index).size() == N_prev));

  TempBuffer<double, 0> batch_inputs_buffer(can_bypass_input ? 0 : effective_batch_size * N_prev);

  if (can_bypass_input)
  {
    const auto& rnn_in = batch_gradients_and_outputs[0].get_rnn_outputs(prev_layer_index);
    in_buf_ptr = !rnn_in.empty() ? rnn_in.data() : batch_gradients_and_outputs[0].get_outputs(prev_layer_index).data();
  }
  else
  {
    double* dest_ptr = batch_inputs_buffer.data();
    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      double* dest_base = dest_ptr + b * num_time_steps * N_prev;
      if (!rnn_in.empty())
      {
        if (rnn_in.size() == N_prev && num_time_steps > 1)
        {
          for (size_t t = 0; t < num_time_steps; ++t)
          {
            std::memcpy(dest_base + t * N_prev, rnn_in.data(), N_prev * sizeof(double));
          }
        }
        else
        {
          const size_t copy_size = std::min(rnn_in.size(), num_time_steps * N_prev);
          if (copy_size > 0 && rnn_in.data() != nullptr)
          {
            std::memcpy(dest_base, rnn_in.data(), copy_size * sizeof(double));
          }
          if (copy_size < num_time_steps * N_prev)
          {
            std::memset(dest_base + copy_size, 0, (num_time_steps * N_prev - copy_size) * sizeof(double));
          }
        }
      }
      else
      {
        const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
        const size_t copy_size = std::min<size_t>(std_in.size(), N_prev);
        for (size_t t = 0; t < num_time_steps; ++t)
        {
          double* t_dest = dest_base + t * N_prev;
          if (copy_size > 0 && std_in.data() != nullptr)
          {
            std::memcpy(t_dest, std_in.data(), copy_size * sizeof(double));
          }
          if (copy_size < N_prev)
          {
            std::memset(t_dest + copy_size, 0, (N_prev - copy_size) * sizeof(double));
          }
        }
      }
    }
    in_buf_ptr = batch_inputs_buffer.data();
  }

  TempBuffer<double, 1> batch_pre_act_buffer(effective_batch_size * N_this);
  double* pre_act_ptr = batch_pre_act_buffer.data();

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1 && batch_size > 1)
    ? std::max(1U, std::min({ max_layer_threads, static_cast<unsigned int>(batch_size), static_cast<unsigned int>((effective_batch_size * N_prev * N_this) / 50000) }))
    : 1U;

  if (active_threads <= 1)
  {
    run_forward_chunk(0, batch_size, num_time_steps, N_prev, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, in_buf_ptr, pre_act_ptr, is_training);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      const size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      const size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue(FfForwardTask{
          this,
          start,
          end,
          num_time_steps,
          N_prev,
          N_this,
          &batch_gradients_and_outputs,
          &batch_residual_output_values,
          &batch_hidden_states,
          in_buf_ptr,
          pre_act_ptr,
          is_training
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void FFLayer::run_forward_chunk(
  size_t start,
  size_t end,
  size_t num_time_steps,
  size_t N_prev,
  size_t N_this,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  const double* in_buf_ptr,
  double* pre_act_ptr,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t b_step_start = start * num_time_steps;
  const size_t b_step_end = end * num_time_steps;

  if (has_bias())
  {
    const auto& biases = get_b_values();
    const size_t copy_size = std::min<size_t>(biases.size(), N_this);
    const double* b_ptr = biases.data();
    for (size_t eb = b_step_start; eb < b_step_end; ++eb)
    {
      double* dest = pre_act_ptr + eb * N_this;
      std::copy_n(b_ptr, copy_size, dest);
      if (copy_size < N_this)
      {
        std::fill_n(dest + copy_size, N_this - copy_size, 0.0);
      }
    }
  }
  else
  {
    std::fill_n(pre_act_ptr + b_step_start * N_this, (b_step_end - b_step_start) * N_this, 0.0);
  }

  run_gemm(b_step_start, b_step_end, N_prev, N_this, in_buf_ptr, pre_act_ptr);

  run_post_gemm(start, end, num_time_steps, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, in_buf_ptr, pre_act_ptr, is_training);
}

void FFLayer::run_gemm(
  size_t b_start,
  size_t b_end,
  size_t N_prev,
  size_t N_this,
  const double* batch_inputs,
  double* batch_pre_activation_sums) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const double* W = get_w_values().data();
  size_t b = b_start;
  for (; b + 3 < b_end; b += 4)
  {
    const double* x0 = &batch_inputs[b * N_prev];
    const double* x1 = &batch_inputs[(b + 1) * N_prev];
    const double* x2 = &batch_inputs[(b + 2) * N_prev];
    const double* x3 = &batch_inputs[(b + 3) * N_prev];

    double* y0 = &batch_pre_activation_sums[b * N_this];
    double* y1 = &batch_pre_activation_sums[(b + 1) * N_this];
    double* y2 = &batch_pre_activation_sums[(b + 2) * N_this];
    double* y3 = &batch_pre_activation_sums[(b + 3) * N_this];

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
    const double* x0 = &batch_inputs[b * N_prev];
    const double* x1 = &batch_inputs[(b + 1) * N_prev];

    double* y0 = &batch_pre_activation_sums[b * N_this];
    double* y1 = &batch_pre_activation_sums[(b + 1) * N_this];

    simd::gemm_two_batches(
      x0, x1,
      W,
      y0, y1,
      N_prev, N_this
    );
  }

  for (; b < b_end; ++b)
  {
    const double* x_row = &batch_inputs[b * N_prev];
    double* y_row = &batch_pre_activation_sums[b * N_this];

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
  const double* /*batch_inputs*/,
  double* batch_pre_activation_sums,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const bool has_dropout = (is_training && get_dropout() > 0.0);
  const double layer_dropout = get_dropout();
  const double default_dropout_scale = (layer_dropout > 0.0 && layer_dropout < 1.0) ? (1.0 / (1.0 - layer_dropout)) : 1.0;
  const auto& neurons = get_neurons();

  std::array<double, 64> mask_stack;
  TempBuffer<double, 2> mask_heap((has_dropout && N_this > 64) ? N_this : 0);
  double* mask_ptr = has_dropout ? ((N_this <= 64) ? mask_stack.data() : mask_heap.data()) : nullptr;

  std::array<double, 128> output_row_stack;
  const size_t seq_elem_count = has_dropout ? num_time_steps * N_this : 0;
  TempBuffer<double, 3> output_row_heap(seq_elem_count > 128 ? seq_elem_count : 0);
  double* output_row_ptr = has_dropout ? ((seq_elem_count <= 128) ? output_row_stack.data() : output_row_heap.data()) : nullptr;

  for (size_t b = start; b < end; b++)
  {
    if (!batch_hidden_states.empty())
    {
      if (batch_hidden_states[b].at(get_layer_index()).size() != num_time_steps)
      {
        batch_hidden_states[b].assign(get_layer_index(), num_time_steps, {}, get_pre_activation_multiplier());
      }
    }

    double* b_pre_act_base = &batch_pre_activation_sums[b * num_time_steps * N_this];

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      if (has_dropout)
      {
        std::fill_n(mask_ptr, N_this, 1.0);
      }

      double* current_pre_act = b_pre_act_base + t * N_this;

      if (!batch_residual_output_values.empty())
      {
        const auto& res_vec = batch_residual_output_values[b];
        if (res_vec.size() == num_time_steps * N_this)
        {
          simd::add_vectors(res_vec.data() + t * N_this, current_pre_act, N_this);
        }
        else if (res_vec.size() == N_this && (num_time_steps == 1 || t == num_time_steps - 1))
        {
          simd::add_vectors(res_vec.data(), current_pre_act, N_this);
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
        if (has_dropout)
        {
          double* current_output_row = output_row_ptr + t * N_this;
          for (size_t j = r.start; j < r.end; j++)
          {
            const auto& neuron = neurons[j];
            double output = current_pre_act[j];
            if (neuron.is_dropout())
            {
              if (neuron.must_randomly_drop(b * num_time_steps + t))
              {
                output = 0.0;
                mask_ptr[j] = 0.0;
              }
              else
              {
                const double rate = neuron.get_dropout_rate();
                const double scale = (rate == layer_dropout) ? default_dropout_scale : (1.0 / (1.0 - rate));
                output *= scale;
                mask_ptr[j] = scale;
              }
            }
            current_output_row[j] = output;
          }
        }
      }

      if (!batch_hidden_states.empty())
      {
        auto& layer_states_ref = batch_hidden_states[b].at(get_layer_index());
        if (has_dropout)
        {
          layer_states_ref[t].set_cell_state_values(mask_ptr, N_this);
          layer_states_ref[t].set_hidden_state_values(output_row_ptr + t * N_this, N_this);
        }
        else
        {
          layer_states_ref[t].set_hidden_state_values(current_pre_act, N_this);
        }
      }
    }

    const double* seq_ptr = has_dropout ? output_row_ptr : b_pre_act_base;
    double* dest_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    std::memcpy(dest_ptr, seq_ptr + (num_time_steps - 1) * N_this, N_this * sizeof(double));
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), seq_ptr, num_time_steps * N_this);
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
  if (batch_size == 0 || batch_hidden_states.empty())
  {
    return;
  }
  const auto this_layer_index = get_layer_index();
  if (this_layer_index >= batch_hidden_states[0].size())
  {
    return;
  }
  const size_t num_time_steps = batch_hidden_states[0].at(this_layer_index).size();
  if (num_time_steps == 0)
  {
    return;
  }
  const auto N_next = next_layer.get_number_neurons();

  const bool use_direct_gradients = batch_next_grad_matrix.empty();
  const double* next_grads_ptr = nullptr;

  const bool can_bypass_next_grads = (batch_size == 1) &&
    ((use_direct_gradients && (batch_gradients_and_outputs[0].has_rnn_gradients(next_layer.get_layer_index())
        ? batch_gradients_and_outputs[0].get_rnn_gradients(next_layer.get_layer_index()).size() == num_time_steps * N_next
        : batch_gradients_and_outputs[0].get_gradients(next_layer.get_layer_index()).size() == num_time_steps * N_next)) ||
     (!use_direct_gradients && !batch_next_grad_matrix.empty() && batch_next_grad_matrix[0].size() == num_time_steps * N_next));

  TempBuffer<double, 4> flattened_next_grads_buffer(can_bypass_next_grads ? 0 : batch_size * num_time_steps * N_next);

  if (can_bypass_next_grads)
  {
    if (use_direct_gradients)
    {
      const auto& rnn_g = batch_gradients_and_outputs[0].get_rnn_gradients(next_layer.get_layer_index());
      next_grads_ptr = !rnn_g.empty()
        ? rnn_g.data()
        : batch_gradients_and_outputs[0].get_gradients(next_layer.get_layer_index()).data();
    }
    else
    {
      next_grads_ptr = batch_next_grad_matrix[0].data();
    }
  }
  else
  {
    double* dest_ptr = flattened_next_grads_buffer.data();
    for (size_t b = 0; b < batch_size; ++b)
    {
      std::span<const double> next_grads;
      if (use_direct_gradients)
      {
        const auto& rnn_g = batch_gradients_and_outputs[b].get_rnn_gradients(next_layer.get_layer_index());
        if (!rnn_g.empty())
        {
          next_grads = rnn_g;
        }
        else
        {
          next_grads = batch_gradients_and_outputs[b].get_gradients(next_layer.get_layer_index());
        }
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
        std::memset(dest_ptr + b * num_time_steps * N_next, 0, num_time_steps * N_next * sizeof(double));
        continue;
      }

      double* dest_base = dest_ptr + b * num_time_steps * N_next;
      const double* src_ptr = next_grads.data();
      if (next_grads.size() == N_next)
      {
        // Broadcast single gradient to all time steps
        for (size_t t = 0; t < num_time_steps; ++t)
        {
          std::memcpy(dest_base + t * N_next, src_ptr, N_next * sizeof(double));
        }
      }
      else if (next_grads.size() == num_time_steps * N_next)
      {
        std::memcpy(dest_base, src_ptr, num_time_steps * N_next * sizeof(double));
      }
      else
      {
        const size_t copy_size = std::min(next_grads.size(), num_time_steps * N_next);
        std::memcpy(dest_base, src_ptr, copy_size * sizeof(double));
        if (copy_size < num_time_steps * N_next)
        {
          std::memset(dest_base + copy_size, 0, (num_time_steps * N_next - copy_size) * sizeof(double));
        }
      }
    }
    next_grads_ptr = flattened_next_grads_buffer.data();
  }

  const size_t effective_batch_size = batch_size * num_time_steps;
  TempBuffer<double, 5> flattened_this_grads_buffer(effective_batch_size * N_this, false);
  double* this_grads_ptr = flattened_this_grads_buffer.data();

  const FFLayer* ff_next = next_layer.is_ff_layer() ? static_cast<const FFLayer*>(&next_layer) : nullptr;
  const double* W_next_T = (ff_next != nullptr && !ff_next->get_w_values_T().empty()) ? ff_next->get_w_values_T().data() : nullptr;
  const double* W_next = (W_next_T == nullptr) ? next_layer.get_w_values().data() : nullptr;

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1 && batch_size > 1)
    ? std::max(1U, std::min({ max_layer_threads, static_cast<unsigned int>(batch_size), static_cast<unsigned int>((effective_batch_size * N_next * N_this) / 50000) }))
    : 1U;

  if (active_threads <= 1)
  {
    run_backward_chunk(0, batch_size, num_time_steps, N_next, N_this, W_next, W_next_T, next_grads_ptr, this_grads_ptr, batch_gradients_and_outputs, batch_hidden_states);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      const size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      const size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue(FfBackwardTask{
          this,
          start,
          end,
          num_time_steps,
          N_next,
          N_this,
          W_next,
          W_next_T,
          next_grads_ptr,
          this_grads_ptr,
          &batch_gradients_and_outputs,
          &batch_hidden_states
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void FFLayer::run_backward_chunk(
  size_t start,
  size_t end,
  size_t num_time_steps,
  size_t N_next,
  size_t N_this,
  const double* W_next,
  const double* W_next_T,
  const double* next_grads_ptr,
  double* this_grads_ptr,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& batch_hidden_states) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t b_step_start = start * num_time_steps;
  const size_t b_step_end = end * num_time_steps;

  std::memset(this_grads_ptr + b_step_start * N_this, 0, (b_step_end - b_step_start) * N_this * sizeof(double));

  if (W_next_T != nullptr)
  {
    run_gemm_backward_fast(b_step_start, b_step_end, N_next, N_this, W_next_T, next_grads_ptr, this_grads_ptr);
  }
  else
  {
    run_gemm_backward(b_step_start, b_step_end, N_next, N_this, W_next, next_grads_ptr, this_grads_ptr);
  }

  run_post_gemm_backward(start, end, N_this, batch_gradients_and_outputs, batch_hidden_states, this_grads_ptr);
}

void FFLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const auto N_this = get_number_neurons();
  if (batch_size == 0 || batch_hidden_states.empty())
  {
    return;
  }

  const auto this_layer_index = get_layer_index();
  if (this_layer_index >= batch_hidden_states[0].size())
  {
    return;
  }
  const size_t num_time_steps = batch_hidden_states[0].at(this_layer_index).size();
  if (num_time_steps == 0)
  {
    return;
  }

  const size_t effective_batch_size = batch_size * num_time_steps;
  const bool use_direct_gradients = batch_output_gradients.empty();
  const double* this_grads_ptr = nullptr;

  const unsigned this_layer_idx = get_layer_index();
  const unsigned next_layer_idx = this_layer_idx + 1;

  const std::span<const double> first_grads = use_direct_gradients
    ? get_output_grads_span(batch_gradients_and_outputs[0], next_layer_idx, this_layer_idx)
    : (!batch_output_gradients.empty() ? std::span<const double>(batch_output_gradients[0]) : std::span<const double>{});

  const bool can_bypass = (batch_size == 1) && (first_grads.size() == num_time_steps * N_this);

  TempBuffer<double, 8> output_this_grads_buffer(can_bypass ? 0 : effective_batch_size * N_this);

  if (can_bypass)
  {
    this_grads_ptr = first_grads.data();
  }
  else
  {
    double* dest_ptr = output_this_grads_buffer.data();
    for (size_t b = 0; b < batch_size; ++b)
    {
      const std::span<const double> next_grads = use_direct_gradients
        ? get_output_grads_span(batch_gradients_and_outputs[b], next_layer_idx, this_layer_idx)
        : (b < batch_output_gradients.size() ? std::span<const double>(batch_output_gradients[b]) : std::span<const double>{});

      double* dest_base = dest_ptr + b * num_time_steps * N_this;
      if (next_grads.empty() || next_grads.data() == nullptr)
      {
        std::memset(dest_base, 0, num_time_steps * N_this * sizeof(double));
        continue;
      }

      if (next_grads.size() == N_this)
      {
        for (size_t t = 0; t < num_time_steps; ++t)
        {
          std::memcpy(dest_base + t * N_this, next_grads.data(), N_this * sizeof(double));
        }
      }
      else if (next_grads.size() == num_time_steps * N_this)
      {
        std::memcpy(dest_base, next_grads.data(), num_time_steps * N_this * sizeof(double));
      }
      else
      {
        const size_t copy_size = std::min(next_grads.size(), num_time_steps * N_this);
        std::memcpy(dest_base, next_grads.data(), copy_size * sizeof(double));
        if (copy_size < num_time_steps * N_this)
        {
          std::memset(dest_base + copy_size, 0, (num_time_steps * N_this - copy_size) * sizeof(double));
        }
      }
    }
    this_grads_ptr = output_this_grads_buffer.data();
  }

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1 && batch_size > 1)
    ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((effective_batch_size * N_this) / 50000)))
    : 1U;
  const bool use_multithreading = (active_threads > 1);

  if (!use_multithreading)
  {
    run_post_gemm_backward(0, batch_size, N_this, batch_gradients_and_outputs, batch_hidden_states, this_grads_ptr);
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
        _task_queue_pool->enqueue(FfPostGemmBackwardTask{
          this,
          start,
          end,
          N_this,
          &batch_gradients_and_outputs,
          &batch_hidden_states,
          this_grads_ptr
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
  if (batch_size == 0 || hidden_states.empty())
  {
    return;
  }
  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();
  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const unsigned this_layer_index = get_layer_index();
  if (this_layer_index >= hidden_states[0].size())
  {
    return;
  }
  const size_t num_time_steps = hidden_states[0].at(this_layer_index).size();
  if (num_time_steps == 0)
  {
    return;
  }

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1 && batch_size > 1)
    ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * num_inputs * num_outputs) / 100000)))
    : 1U;

  if (active_threads == 1)
  {
    if (!_w_grads.empty())
    {
      std::memset(_w_grads.data(), 0, _w_grads.size() * sizeof(double));
    }
    if (has_bias() && !_b_grads.empty())
    {
      std::memset(_b_grads.data(), 0, _b_grads.size() * sizeof(double));
    }
    calculate_and_store_gradients_chunk(0, batch_size, batch_gradients_and_outputs, prev_layer_index, this_layer_index, num_inputs, num_outputs, num_time_steps, _w_grads, _b_grads);
  }
  else
  {
    const unsigned int aux_threads = active_threads - 1;
    if (_thread_grad_accumulators.size() < aux_threads)
    {
      _thread_grad_accumulators.resize(aux_threads);
    }

    for (unsigned int t = 0; t < aux_threads; ++t)
    {
      _thread_grad_accumulators[t].w_grads.resize_and_zero(_w_grads.size());
      if (has_bias())
      {
        _thread_grad_accumulators[t].b_grads.resize_and_zero(num_outputs);
      }
    }

    if (!_w_grads.empty())
    {
      std::memset(_w_grads.data(), 0, _w_grads.size() * sizeof(double));
    }
    if (has_bias() && !_b_grads.empty())
    {
      std::memset(_b_grads.data(), 0, _b_grads.size() * sizeof(double));
    }

    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      const size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      const size_t end = start + size;
      if (start < end)
      {
        std::span<double> target_w = (t == 0) ? std::span<double>(_w_grads) : std::span<double>(_thread_grad_accumulators[t - 1].w_grads);
        std::span<double> target_b = (t == 0) ? std::span<double>(_b_grads) : std::span<double>(_thread_grad_accumulators[t - 1].b_grads);

        _task_queue_pool->enqueue(FfGradsTask{
          this,
          start,
          end,
          &batch_gradients_and_outputs,
          prev_layer_index,
          this_layer_index,
          num_inputs,
          num_outputs,
          num_time_steps,
          target_w,
          target_b
        });
      }
      start = end;
    }
    _task_queue_pool->get();

    // Merge auxiliary accumulators directly into _w_grads and _b_grads
    unsigned int t = 0;
    for (; t + 3 < aux_threads; t += 4)
    {
      simd::accumulate_four_vectors(
        _thread_grad_accumulators[t].w_grads.data(),
        _thread_grad_accumulators[t + 1].w_grads.data(),
        _thread_grad_accumulators[t + 2].w_grads.data(),
        _thread_grad_accumulators[t + 3].w_grads.data(),
        _w_grads.data(),
        _w_grads.size()
      );
      if (has_bias())
      {
        simd::accumulate_four_vectors(
          _thread_grad_accumulators[t].b_grads.data(),
          _thread_grad_accumulators[t + 1].b_grads.data(),
          _thread_grad_accumulators[t + 2].b_grads.data(),
          _thread_grad_accumulators[t + 3].b_grads.data(),
          _b_grads.data(),
          _b_grads.size()
        );
      }
    }
    for (; t + 1 < aux_threads; t += 2)
    {
      simd::accumulate_two_vectors(
        _thread_grad_accumulators[t].w_grads.data(),
        _thread_grad_accumulators[t + 1].w_grads.data(),
        _w_grads.data(),
        _w_grads.size()
      );
      if (has_bias())
      {
        simd::accumulate_two_vectors(
          _thread_grad_accumulators[t].b_grads.data(),
          _thread_grad_accumulators[t + 1].b_grads.data(),
          _b_grads.data(),
          _b_grads.size()
        );
      }
    }
    for (; t < aux_threads; ++t)
    {
      simd::add_vectors(_thread_grad_accumulators[t].w_grads.data(), _w_grads.data(), _w_grads.size());
      if (has_bias())
      {
        simd::add_vectors(_thread_grad_accumulators[t].b_grads.data(), _b_grads.data(), _b_grads.size());
      }
    }
  }

  if (batch_size > 1)
  {
    const double inv_batch = 1.0 / static_cast<double>(batch_size);
    simd::scale_vector(_w_grads.data(), inv_batch, _w_grads.size());
    if (has_bias())
    {
      simd::scale_vector(_b_grads.data(), inv_batch, _b_grads.size());
    }
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
  cache_recurrent_weights();
}

void FFLayer::update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  auto& other = static_cast<FFLayer&>(fast_layer);
  simd::lookahead_step(_w_values.data(), other._w_values.data(), alpha, _w_values.size());
  if (has_bias())
  {
    simd::lookahead_step(_b_values.data(), other._b_values.data(), alpha, _b_values.size());
  }
  cache_recurrent_weights();
  other.cache_recurrent_weights();
}

void FFLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
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
  cache_recurrent_weights();
}

void FFLayer::run_gemm_backward(
  size_t b_start,
  size_t b_end,
  size_t N_next,
  size_t N_this,
  const double* W_next,
  const double* flattened_next_grads,
  double* flattened_this_grads) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  size_t b = b_start;
  for (; b + 3 < b_end; b += 4)
  {
    const double* x0 = &flattened_next_grads[b * N_next];
    const double* x1 = &flattened_next_grads[(b + 1) * N_next];
    const double* x2 = &flattened_next_grads[(b + 2) * N_next];
    const double* x3 = &flattened_next_grads[(b + 3) * N_next];

    double* y0 = &flattened_this_grads[b * N_this];
    double* y1 = &flattened_this_grads[(b + 1) * N_this];
    double* y2 = &flattened_this_grads[(b + 2) * N_this];
    double* y3 = &flattened_this_grads[(b + 3) * N_this];

    simd::gemm_transposed_four_batches(
      x0, x1, x2, x3,
      W_next,
      y0, y1, y2, y3,
      N_this, N_next
    );
  }

  for (; b + 1 < b_end; b += 2)
  {
    const double* x0 = &flattened_next_grads[b * N_next];
    const double* x1 = &flattened_next_grads[(b + 1) * N_next];

    double* y0 = &flattened_this_grads[b * N_this];
    double* y1 = &flattened_this_grads[(b + 1) * N_this];

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
      &flattened_next_grads[b * N_next],
      W_next,
      &flattened_this_grads[b * N_this],
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
  const double* flattened_this_grads) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  
  const size_t initial_time_steps = (start < batch_hidden_states.size() && !batch_hidden_states[start].at(get_layer_index()).empty())
    ? batch_hidden_states[start].at(get_layer_index()).size()
    : 1;

  std::array<double, 64> deriv_stack;
  TempBuffer<double, 6> deriv_heap(N_this > 64 ? N_this : 0);
  double* deriv_ptr = (N_this <= 64) ? deriv_stack.data() : deriv_heap.data();

  TempBuffer<double, 7> rnn_grads_row(initial_time_steps * N_this);

  const bool has_dropout = (get_dropout() > 0.0);

  for (size_t b = start; b < end; b++)
  {
    const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
    const size_t num_time_steps = layer_states.size();
    if (num_time_steps == 0)
    {
      continue;
    }

    if (rnn_grads_row.size() < num_time_steps * N_this)
    {
      rnn_grads_row.assign(num_time_steps * N_this, 0.0);
    }
    double* rnn_grads_ptr = rnn_grads_row.data();

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const double* g_this_row = &flattened_this_grads[(b * num_time_steps + t) * N_this];
      const auto& current_hidden_state = layer_states[t];
      const double* pre_act = current_hidden_state.get_pre_activation_sums().data();
      const double* y_vals = has_dropout ? nullptr : current_hidden_state.get_hidden_state_values().data();

      for (const auto& r : _layer_activation_helper.ranges())
      {
        const size_t range_size = r.end - r.start;
        double* out_grad_dest = rnn_grads_ptr + t * N_this + r.start;
        const double* g_src = g_this_row + r.start;

        if (r.activation_method.get_method() == activation::method::linear)
        {
          if (has_dropout)
          {
            const double* mask_vals = current_hidden_state.get_cell_state_values().data();
            simd::mul_vectors(
              g_src,
              mask_vals + r.start,
              out_grad_dest,
              range_size
            );
          }
          else
          {
            std::memcpy(out_grad_dest, g_src, range_size * sizeof(double));
          }
        }
        else
        {
          r.activation_method.activate_derivative(
            pre_act + r.start,
            pre_act + r.end,
            y_vals ? (y_vals + r.start) : nullptr,
            deriv_ptr + r.start
          );

          if (has_dropout)
          {
            const double* mask_vals = current_hidden_state.get_cell_state_values().data();
            simd::mul_three_vectors(
              g_src,
              deriv_ptr + r.start,
              mask_vals + r.start,
              out_grad_dest,
              range_size
            );
          }
          else
          {
            simd::mul_vectors(
              g_src,
              deriv_ptr + r.start,
              out_grad_dest,
              range_size
            );
          }
        }
      }
    }

    batch_gradients_and_outputs[b].set_gradients(get_layer_index(), rnn_grads_ptr + (num_time_steps - 1) * N_this, N_this);
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), rnn_grads_ptr, num_time_steps * N_this);
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
  std::span<double> local_w_grads,
  std::span<double> local_b_grads) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (start >= end)
  {
    return;
  }

  struct batch_item_info
  {
    const double* x_base;
    const double* g_base;
    size_t x_stride;
    size_t g_stride;
  };

  const size_t chunk_size = end - start;
  std::array<batch_item_info, 64> batch_items_stack;
  std::vector<batch_item_info> batch_items_heap;
  batch_item_info* batch_items_ptr = batch_items_stack.data();
  if (chunk_size > 64)
  {
    batch_items_heap.resize(chunk_size);
    batch_items_ptr = batch_items_heap.data();
  }
  for (size_t b = start; b < end; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    const auto& std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    const bool item_rnn_in = (!rnn_in.empty() && rnn_in.size() >= num_time_steps * num_inputs);
    const double* x_ptr = !rnn_in.empty() ? rnn_in.data() : std_in.data();
    const size_t x_stride = item_rnn_in ? num_inputs : ((!rnn_in.empty() || std_in.size() < num_time_steps * num_inputs) ? 0 : num_inputs);

    const auto& rnn_grad = batch_gradients_and_outputs[b].get_rnn_gradients(this_layer_index);
    const auto& std_grad = batch_gradients_and_outputs[b].get_gradients(this_layer_index);
    const bool item_rnn_grad = (batch_gradients_and_outputs[b].has_rnn_gradients(this_layer_index) && rnn_grad.size() >= num_time_steps * num_outputs);
    const double* g_ptr = (!rnn_grad.empty()) ? rnn_grad.data() : std_grad.data();
    const size_t g_stride = item_rnn_grad ? num_outputs : ((!rnn_grad.empty() || std_grad.size() < num_time_steps * num_outputs) ? 0 : num_outputs);

    batch_items_ptr[b - start] = {
      x_ptr,
      g_ptr,
      x_stride,
      g_stride
    };
  }

  const batch_item_info* const items_ptr = batch_items_ptr;

  if (has_bias() && !local_b_grads.empty())
  {
    double* b_grad_data = local_b_grads.data();
    if (num_time_steps == 1)
    {
      std::array<const double*, 64> valid_g_stack;
      std::vector<const double*> valid_g_heap;
      const double** valid_g_ptr = valid_g_stack.data();
      if (chunk_size > 64)
      {
        valid_g_heap.resize(chunk_size);
        valid_g_ptr = valid_g_heap.data();
      }
      size_t valid_g_count = 0;
      for (size_t k = 0; k < chunk_size; ++k)
      {
        if (items_ptr[k].g_base != nullptr)
        {
          valid_g_ptr[valid_g_count++] = items_ptr[k].g_base;
        }
      }

      size_t k = 0;
      for (; k + 3 < valid_g_count; k += 4)
      {
        simd::accumulate_four_vectors(
          valid_g_ptr[k],
          valid_g_ptr[k + 1],
          valid_g_ptr[k + 2],
          valid_g_ptr[k + 3],
          b_grad_data,
          num_outputs
        );
      }
      for (; k + 1 < valid_g_count; k += 2)
      {
        simd::accumulate_two_vectors(
          valid_g_ptr[k],
          valid_g_ptr[k + 1],
          b_grad_data,
          num_outputs
        );
      }
      for (; k < valid_g_count; ++k)
      {
        simd::add_vectors(valid_g_ptr[k], b_grad_data, num_outputs);
      }
    }
    else
    {
      for (size_t k = 0; k < chunk_size; ++k)
      {
        const auto& item = items_ptr[k];
        if (item.g_base == nullptr)
        {
          continue;
        }
        if (item.g_stride == 0)
        {
          simd::mul_add(static_cast<double>(num_time_steps), item.g_base, b_grad_data, num_outputs);
        }
        else
        {
          size_t t = 0;
          for (; t + 3 < num_time_steps; t += 4)
          {
            simd::accumulate_four_vectors(
              item.g_base + t * item.g_stride,
              item.g_base + (t + 1) * item.g_stride,
              item.g_base + (t + 2) * item.g_stride,
              item.g_base + (t + 3) * item.g_stride,
              b_grad_data,
              num_outputs
            );
          }
          for (; t + 1 < num_time_steps; t += 2)
          {
            simd::accumulate_two_vectors(
              item.g_base + t * item.g_stride,
              item.g_base + (t + 1) * item.g_stride,
              b_grad_data,
              num_outputs
            );
          }
          for (; t < num_time_steps; ++t)
          {
            simd::add_vectors(item.g_base + t * item.g_stride, b_grad_data, num_outputs);
          }
        }
      }
    }
  }

  // Weight gradients
  if (num_time_steps == 1)
  {
    // Fast path for non-recurrent / single-step training
    size_t i = 0;
    for (; i + 3 < num_inputs; i += 4)
    {
      double* w0 = &local_w_grads[i * num_outputs];
      double* w1 = &local_w_grads[(i + 1) * num_outputs];
      double* w2 = &local_w_grads[(i + 2) * num_outputs];
      double* w3 = &local_w_grads[(i + 3) * num_outputs];

      for (size_t k = 0; k < chunk_size; ++k)
      {
        const auto& item = items_ptr[k];
        if (item.x_base == nullptr || item.g_base == nullptr)
        {
          continue;
        }
        const double x0 = item.x_base[i];
        const double x1 = item.x_base[i + 1];
        const double x2 = item.x_base[i + 2];
        const double x3 = item.x_base[i + 3];
        if (x0 == 0.0 && x1 == 0.0 && x2 == 0.0 && x3 == 0.0)
        {
          continue;
        }
        simd::mul_add_four_scalars(x0, x1, x2, x3, item.g_base, w0, w1, w2, w3, num_outputs);
      }
    }

    for (; i + 1 < num_inputs; i += 2)
    {
      double* w0 = &local_w_grads[i * num_outputs];
      double* w1 = &local_w_grads[(i + 1) * num_outputs];

      for (size_t k = 0; k < chunk_size; ++k)
      {
        const auto& item = items_ptr[k];
        if (item.x_base == nullptr || item.g_base == nullptr)
        {
          continue;
        }
        const double x0 = item.x_base[i];
        const double x1 = item.x_base[i + 1];
        if (x0 == 0.0 && x1 == 0.0)
        {
          continue;
        }
        simd::mul_add_two_scalars(x0, x1, item.g_base, w0, w1, num_outputs);
      }
    }

    for (; i < num_inputs; ++i)
    {
      double* w_grad_row = &local_w_grads[i * num_outputs];
      for (size_t k = 0; k < chunk_size; ++k)
      {
        const auto& item = items_ptr[k];
        if (item.x_base == nullptr || item.g_base == nullptr)
        {
          continue;
        }
        const double x_val = item.x_base[i];
        if (x_val == 0.0)
        {
          continue;
        }
        simd::mul_add(x_val, item.g_base, w_grad_row, num_outputs);
      }
    }
  }
  else
  {
    // Check if any items have static inputs (x_stride == 0) with dynamic gradients (g_stride > 0)
    std::vector<double> summed_g_buffer;
    std::vector<const double*> static_g_ptrs(chunk_size, nullptr);
    size_t dynamic_g_static_x_count = 0;
    for (size_t k = 0; k < chunk_size; ++k)
    {
      if (items_ptr[k].x_base != nullptr && items_ptr[k].g_base != nullptr &&
          items_ptr[k].x_stride == 0 && items_ptr[k].g_stride > 0)
      {
        ++dynamic_g_static_x_count;
      }
    }
    if (dynamic_g_static_x_count > 0)
    {
      summed_g_buffer.resize(dynamic_g_static_x_count * num_outputs, 0.0);
      size_t cur_buf_idx = 0;
      for (size_t k = 0; k < chunk_size; ++k)
      {
        const auto& item = items_ptr[k];
        if (item.x_base != nullptr && item.g_base != nullptr &&
            item.x_stride == 0 && item.g_stride > 0)
        {
          double* sum_ptr = &summed_g_buffer[cur_buf_idx * num_outputs];
          ++cur_buf_idx;
          size_t t = 0;
          for (; t + 3 < num_time_steps; t += 4)
          {
            simd::accumulate_four_vectors(
              item.g_base + t * item.g_stride,
              item.g_base + (t + 1) * item.g_stride,
              item.g_base + (t + 2) * item.g_stride,
              item.g_base + (t + 3) * item.g_stride,
              sum_ptr,
              num_outputs
            );
          }
          for (; t + 1 < num_time_steps; t += 2)
          {
            simd::accumulate_two_vectors(
              item.g_base + t * item.g_stride,
              item.g_base + (t + 1) * item.g_stride,
              sum_ptr,
              num_outputs
            );
          }
          for (; t < num_time_steps; ++t)
          {
            simd::add_vectors(item.g_base + t * item.g_stride, sum_ptr, num_outputs);
          }
          static_g_ptrs[k] = sum_ptr;
        }
      }
    }

    // General sequence loop with static stride optimization
    size_t i = 0;
    for (; i + 3 < num_inputs; i += 4)
    {
      double* w0 = &local_w_grads[i * num_outputs];
      double* w1 = &local_w_grads[(i + 1) * num_outputs];
      double* w2 = &local_w_grads[(i + 2) * num_outputs];
      double* w3 = &local_w_grads[(i + 3) * num_outputs];

      for (size_t k = 0; k < chunk_size; ++k)
      {
        const auto& item = items_ptr[k];
        if (item.x_base == nullptr || item.g_base == nullptr)
        {
          continue;
        }
        const double* x_ptr = item.x_base;
        const double* g_ptr = item.g_base;
        const size_t x_stride = item.x_stride;
        const size_t g_stride = item.g_stride;

        if (x_stride == 0)
        {
          const bool is_dynamic_g = (g_stride > 0);
          const double scale = is_dynamic_g ? 1.0 : static_cast<double>(num_time_steps);
          const double* eff_g = is_dynamic_g ? static_g_ptrs[k] : g_ptr;
          const double x0 = x_ptr[i] * scale;
          const double x1 = x_ptr[i + 1] * scale;
          const double x2 = x_ptr[i + 2] * scale;
          const double x3 = x_ptr[i + 3] * scale;
          if (x0 == 0.0 && x1 == 0.0 && x2 == 0.0 && x3 == 0.0)
          {
            continue;
          }
          simd::mul_add_four_scalars(x0, x1, x2, x3, eff_g, w0, w1, w2, w3, num_outputs);
        }
        else
        {
          for (size_t t = 0; t < num_time_steps; ++t)
          {
            const double* x_t = x_ptr + t * x_stride;
            const double* g_t = g_ptr + t * g_stride;
            const double x0 = x_t[i];
            const double x1 = x_t[i + 1];
            const double x2 = x_t[i + 2];
            const double x3 = x_t[i + 3];
            if (x0 == 0.0 && x1 == 0.0 && x2 == 0.0 && x3 == 0.0)
            {
              continue;
            }
            simd::mul_add_four_scalars(x0, x1, x2, x3, g_t, w0, w1, w2, w3, num_outputs);
          }
        }
      }
    }

    for (; i + 1 < num_inputs; i += 2)
    {
      double* w0 = &local_w_grads[i * num_outputs];
      double* w1 = &local_w_grads[(i + 1) * num_outputs];

      for (size_t k = 0; k < chunk_size; ++k)
      {
        const auto& item = items_ptr[k];
        if (item.x_base == nullptr || item.g_base == nullptr)
        {
          continue;
        }
        const double* x_ptr = item.x_base;
        const double* g_ptr = item.g_base;
        const size_t x_stride = item.x_stride;
        const size_t g_stride = item.g_stride;

        if (x_stride == 0)
        {
          const bool is_dynamic_g = (g_stride > 0);
          const double scale = is_dynamic_g ? 1.0 : static_cast<double>(num_time_steps);
          const double* eff_g = is_dynamic_g ? static_g_ptrs[k] : g_ptr;
          const double x0 = x_ptr[i] * scale;
          const double x1 = x_ptr[i + 1] * scale;
          if (x0 == 0.0 && x1 == 0.0)
          {
            continue;
          }
          simd::mul_add_two_scalars(x0, x1, eff_g, w0, w1, num_outputs);
        }
        else
        {
          for (size_t t = 0; t < num_time_steps; ++t)
          {
            const double* x_t = x_ptr + t * x_stride;
            const double* g_t = g_ptr + t * g_stride;
            const double x0 = x_t[i];
            const double x1 = x_t[i + 1];
            if (x0 == 0.0 && x1 == 0.0)
            {
              continue;
            }
            simd::mul_add_two_scalars(x0, x1, g_t, w0, w1, num_outputs);
          }
        }
      }
    }

    for (; i < num_inputs; ++i)
    {
      double* w_grad_row = &local_w_grads[i * num_outputs];
      for (size_t k = 0; k < chunk_size; ++k)
      {
        const auto& item = items_ptr[k];
        if (item.x_base == nullptr || item.g_base == nullptr)
        {
          continue;
        }
        const double* x_ptr = item.x_base;
        const double* g_ptr = item.g_base;
        const size_t x_stride = item.x_stride;
        const size_t g_stride = item.g_stride;

        if (x_stride == 0)
        {
          const bool is_dynamic_g = (g_stride > 0);
          const double scale = is_dynamic_g ? 1.0 : static_cast<double>(num_time_steps);
          const double* eff_g = is_dynamic_g ? static_g_ptrs[k] : g_ptr;
          const double x_val = x_ptr[i] * scale;
          if (x_val == 0.0)
          {
            continue;
          }
          simd::mul_add(x_val, eff_g, w_grad_row, num_outputs);
        }
        else
        {
          for (size_t t = 0; t < num_time_steps; ++t)
          {
            const double* x_t = x_ptr + t * x_stride;
            const double* g_t = g_ptr + t * g_stride;
            const double x_val = x_t[i];
            if (x_val == 0.0)
            {
              continue;
            }
            simd::mul_add(x_val, g_t, w_grad_row, num_outputs);
          }
        }
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
  const double* flattened_next_grads,
  double* flattened_this_grads) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  size_t b = b_start;
  for (; b + 3 < b_end; b += 4)
  {
    const double* x0 = &flattened_next_grads[b * N_next];
    const double* x1 = &flattened_next_grads[(b + 1) * N_next];
    const double* x2 = &flattened_next_grads[(b + 2) * N_next];
    const double* x3 = &flattened_next_grads[(b + 3) * N_next];

    double* y0 = &flattened_this_grads[b * N_this];
    double* y1 = &flattened_this_grads[(b + 1) * N_this];
    double* y2 = &flattened_this_grads[(b + 2) * N_this];
    double* y3 = &flattened_this_grads[(b + 3) * N_this];

    simd::gemm_four_batches(
      x0, x1, x2, x3,
      W_next_T,
      y0, y1, y2, y3,
      N_next, N_this
    );
  }

  for (; b + 1 < b_end; b += 2)
  {
    const double* x0 = &flattened_next_grads[b * N_next];
    const double* x1 = &flattened_next_grads[(b + 1) * N_next];

    double* y0 = &flattened_this_grads[b * N_this];
    double* y1 = &flattened_this_grads[(b + 1) * N_this];

    simd::gemm_two_batches(
      x0, x1,
      W_next_T,
      y0, y1,
      N_next, N_this
    );
  }

  for (; b < b_end; ++b)
  {
    const double* x_row = &flattened_next_grads[b * N_next];
    double* y_row = &flattened_this_grads[b * N_this];

    simd::gemm_one_batch(
      x_row,
      W_next_T,
      y_row,
      N_next, N_this
    );
  }
}

} // namespace myoddweb::nn
