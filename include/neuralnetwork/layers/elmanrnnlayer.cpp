#include "../libraries/instrumentor.h"
#include "elmanrnnlayer.h"
#include "fflayer.h"
#include "../common/logger.h"
#include "../common/simd_utils.h"
#include <algorithm>
#include <cmath>


namespace myoddweb::nn
{
ElmanRNNLayer::ElmanRNNLayer(
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
  ElmanRNNLayer(
    layer_index,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * num_neurons_in_this_layer, weight_decay),
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
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  allocate_workspace();
}

ElmanRNNLayer::ElmanRNNLayer(
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
  Layer(
    layer_index,
    layer_role,
    activation_method,
    optimiser_type,
    residual_layer_number,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    create_neurons(dropout_rate, num_neurons_in_this_layer),
    has_bias,
    weight_decays,
    residual_projector,
    number_of_threads,
    momentum
  )
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  initialize_recurrent_weights(weight_decays.empty() ? 0.0 : weight_decays[0]);
  cache_recurrent_weights();
  allocate_workspace();
}

ElmanRNNLayer::ElmanRNNLayer(const ElmanRNNLayer& src) noexcept :
  Layer(src),
  _rw_values(src._rw_values),
  _rw_grads(src._rw_grads),
  _rw_velocities(src._rw_velocities),
  _rw_m1(src._rw_m1),
  _rw_m2(src._rw_m2),
  _rw_timesteps(src._rw_timesteps),
  _rw_decays(src._rw_decays)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  cache_recurrent_weights();
  allocate_workspace();
}

ElmanRNNLayer::ElmanRNNLayer(ElmanRNNLayer&& src) noexcept :
  Layer(std::move(src)),
  _rw_values(std::move(src._rw_values)),
  _rw_grads(std::move(src._rw_grads)),
  _rw_velocities(std::move(src._rw_velocities)),
  _rw_m1(std::move(src._rw_m1)),
  _rw_m2(std::move(src._rw_m2)),
  _rw_timesteps(std::move(src._rw_timesteps)),
  _rw_decays(std::move(src._rw_decays)),
  _rw_values_T(std::move(src._rw_values_T)),
  _thread_workspaces(std::move(src._thread_workspaces))
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
}

ElmanRNNLayer::ElmanRNNLayer(
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
  double momentum) noexcept :
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
  ),
  _rw_values(rw_values),
  _rw_grads(rw_grads),
  _rw_velocities(rw_velocities),
  _rw_m1(rw_m1),
  _rw_m2(rw_m2),
  _rw_timesteps(rw_timesteps),
  _rw_decays(rw_decays)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  cache_recurrent_weights();
  allocate_workspace();
}

ElmanRNNLayer& ElmanRNNLayer::operator=(const ElmanRNNLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (this != &src)
  {
    Layer::operator=(src);
    _rw_values = src._rw_values;
    _rw_grads = src._rw_grads;
    _rw_velocities = src._rw_velocities;
    _rw_m1 = src._rw_m1;
    _rw_m2 = src._rw_m2;
    _rw_timesteps = src._rw_timesteps;
    _rw_decays = src._rw_decays;
    allocate_workspace();
    cache_recurrent_weights();
  }
  return *this;
}

ElmanRNNLayer& ElmanRNNLayer::operator=(ElmanRNNLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (this != &src)
  {
    Layer::operator=(std::move(src));
    _rw_values = std::move(src._rw_values);
    _rw_grads = std::move(src._rw_grads);
    _rw_velocities = std::move(src._rw_velocities);
    _rw_m1 = std::move(src._rw_m1);
    _rw_m2 = std::move(src._rw_m2);
    _rw_timesteps = std::move(src._rw_timesteps);
    _rw_decays = std::move(src._rw_decays);
    _rw_values_T = std::move(src._rw_values_T);
    _thread_workspaces = std::move(src._thread_workspaces);
  }
  return *this;
}

ElmanRNNLayer::~ElmanRNNLayer()
{
}

void ElmanRNNLayer::initialize_recurrent_weights(double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const auto num_neurons = get_number_output_neurons();
  const size_t num_weights = static_cast<size_t>(num_neurons) * num_neurons;

  _rw_values.resize(num_weights);
  for (size_t i = 0; i < num_neurons; ++i)
  {
    for (unsigned int o = 0; o < num_neurons; ++o)
    {
      _rw_values[i * num_neurons + o] = get_activation().weight_initialization(num_neurons, num_neurons);
    }
  }

  _rw_grads.assign(num_weights, 0.0);
  _rw_velocities.assign(num_weights, 0.0);
  _rw_m1.assign(num_weights, 0.0);
  _rw_m2.assign(num_weights, 0.0);
  _rw_timesteps.assign(num_weights, 0);
  _rw_decays.assign(num_weights, weight_decay);
  cache_recurrent_weights();
}

void ElmanRNNLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  // 1. Determine sequence length and flatten inputs
  size_t num_time_steps = 0;
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty())
    {
      num_time_steps = rnn_in.size() / N_prev;
      break;
    }
    const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    if (std_in.size() == N_prev)
    {
      num_time_steps = 1;
      break;
    }
  }
  if (num_time_steps == 0)
  {
    return;
  }

  std::vector<double> flattened_batch_inputs(batch_size * num_time_steps * N_prev);
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty())
    {
      std::copy(rnn_in.begin(), rnn_in.end(), flattened_batch_inputs.begin() + b * num_time_steps * N_prev);
    }
    else
    {
      const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        std::copy(std_in.begin(), std_in.end(), flattened_batch_inputs.begin() + (b * num_time_steps + t) * N_prev);
      }
    }
  }

  // 2. Pre-calculate Input-to-Hidden (W * x_t) for all ticks
  std::vector<double> batch_pre_act(batch_size * num_time_steps * N_this, 0.0);

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  const bool use_multithreading = (num_threads > 1) && (batch_size >= num_threads * 16);
  if (!use_multithreading)
  {
    pre_calculate_gates(0, batch_size, N_this, N_prev, num_time_steps, flattened_batch_inputs, batch_pre_act);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, N_this, N_prev, num_time_steps, &flattened_batch_inputs, &batch_pre_act, this]()
          {
            pre_calculate_gates(start, end, N_this, N_prev, num_time_steps, flattened_batch_inputs, batch_pre_act);
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 3. Sequential Recurrent Pass and Activations
  std::vector<double> batch_output_sequences(batch_size * num_time_steps * N_this);

  auto recurrent_pass = [&](size_t b_start, size_t b_end)
  {
    std::vector<double> current_h(N_this, 0.0);
    std::vector<double> mask(N_this, 1.0);
    for (size_t b = b_start; b < b_end; ++b)
    {
      std::fill(current_h.begin(), current_h.end(), 0.0);
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        double* pre_t = &batch_pre_act[(b * num_time_steps + t) * N_this];

        // Recurrent-to-Hidden (U * h_{t-1})
        simd::gemv_add(_rw_values_T.data(), current_h.data(), pre_t, N_this, N_this);

        if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
        {
          for (size_t j = 0; j < N_this; ++j)
          {
            pre_t[j] += batch_residual_output_values[b][j];
          }
        }

        auto& state = batch_hidden_states[b].at(get_layer_index())[t];
        state.set_pre_activation_sums(pre_t, N_this);

        get_activation().activate(pre_t, pre_t + N_this, is_training);

        std::fill(mask.begin(), mask.end(), 1.0);
        if (is_training)
        {
          const auto& neurons = get_neurons();
          for (size_t j = 0; j < N_this; ++j)
          {
            double out = pre_t[j];
            const auto& neuron = neurons[j];
            if (neuron.is_dropout())
            {
              if (neuron.must_randomly_drop())
              {
                out = 0.0;
                mask[j] = 0.0;
              }
              else
              {
                const double scale = 1.0 / (1.0 - neuron.get_dropout_rate());
                out *= scale;
                mask[j] = scale;
              }
            }
            current_h[j] = out;
            batch_output_sequences[(b * num_time_steps + t) * N_this + j] = out;
          }
        }
        else
        {
          for (size_t j = 0; j < N_this; ++j)
          {
            double out = pre_t[j];
            current_h[j] = out;
            batch_output_sequences[(b * num_time_steps + t) * N_this + j] = out;
          }
        }
        state.set_cell_state_values(mask.data(), N_this);
        state.set_hidden_state_values(current_h.data(), N_this);
      }
    }
  };

  if (!use_multithreading)
  {
    recurrent_pass(0, batch_size);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([&recurrent_pass, start, end]()
        {
          recurrent_pass(start, end);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* seq_ptr = &batch_output_sequences[b * num_time_steps * N_this];
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), seq_ptr, num_time_steps * N_this);
    const double* last_ptr = seq_ptr + (num_time_steps - 1) * N_this;
    double* dest_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    std::copy(last_ptr, last_ptr + N_this, dest_ptr);
  }
}

void ElmanRNNLayer::pre_calculate_gates(
  const size_t& b_start, 
  const size_t& b_end,
  const size_t N_this,
  const size_t N_prev,
  const size_t num_time_steps,
  const std::vector<double>& flattened_batch_inputs,
  std::vector<double>& batch_pre_act
) const
{
  for (size_t b = b_start; b < b_end; ++b)
  {
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const double* x_t = &flattened_batch_inputs[(b * num_time_steps + t) * N_prev];
      double* pre_t = &batch_pre_act[(b * num_time_steps + t) * N_this];

      if (has_bias())
      {
        std::copy(get_b_values().begin(), get_b_values().end(), pre_t);
      }

      for (size_t i = 0; i < N_prev; ++i)
      {
        const double xi = x_t[i];
        if (xi == 0.0)
        {
          continue;
        }
        const double* w_row = &get_w_values()[i * N_this];
        simd::mul_add(xi, w_row, pre_t, N_this);
      }
    }
  }
}

void ElmanRNNLayer::calculate_output_gradients(std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, std::vector<std::vector<double>>::const_iterator target_outputs_begin, const std::vector<HiddenStates>& batch_hidden_states, size_t batch_size) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_this = get_number_neurons();
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& states = batch_hidden_states[b].at(get_layer_index());
    const size_t T = states.size();
    std::vector<double> deltas(T * N_this);
    const std::vector<double>& targets = *(target_outputs_begin + b);
    for (size_t t = 0; t < T; ++t)
    {
      const auto& given = states[t].get_hidden_state_values();
      for (size_t j = 0; j < N_this; ++j)
      {
        size_t idx = t * N_this + j;
        if (idx < targets.size())
        {
          deltas[idx] = given[j] - targets[idx];
        }
        else
        {
          deltas[idx] = 0.0;
        }
      }
    }
    double* dest_ptr = batch_gradients_and_outputs[b].get_gradients_raw(get_layer_index());
    std::copy(deltas.end() - N_this, deltas.end(), dest_ptr);
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), std::move(deltas));
  }
}

void ElmanRNNLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (batch_size == 0)
  {
    return;
  }
  const size_t N_this = get_number_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0 || N_this == 0)
  {
    return;
  }

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  const bool use_multithreading = (num_threads > 1) && (batch_size >= num_threads * 16);

  if (!use_multithreading)
  {
    auto& workspace = get_workspace(0);
    calculate_bptt_batch_chunk(0, batch_size, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, workspace, _rw_values_T);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, t, &batch_gradients_and_outputs, &next_layer, &batch_next_grad_matrix, &batch_hidden_states, bptt_max_ticks, this]()
        {
          auto& workspace = get_workspace(t);
          calculate_bptt_batch_chunk(start, end, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, workspace, _rw_values_T);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void ElmanRNNLayer::calculate_hidden_gradients_from_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_output_gradients,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (batch_size == 0)
  {
    return;
  }
  const size_t N_this = get_number_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0 || N_this == 0)
  {
    return;
  }

  // Use local FFLayer proxy
  FFLayer proxy(0, static_cast<unsigned>(N_this), static_cast<unsigned>(N_this), 0.0, Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, false, 0.0);
  std::vector<double> id(static_cast<size_t>(N_this) * N_this, 0.0);
  for (unsigned i = 0; i < N_this; ++i) id[i * N_this + i] = 1.0;
  proxy.set_w_values(id);

  calculate_hidden_gradients(batch_gradients_and_outputs, proxy, batch_output_gradients, batch_hidden_states, batch_size, bptt_max_ticks);
}

void ElmanRNNLayer::calculate_bptt_batch_chunk(
  size_t start,
  size_t end,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  int bptt_max_ticks,
  BPTTWorkspace& workspace,
  const BPTTWorkspace::AlignedVector& /*rw_values_T*/) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  workspace.resize(N_this, N_prev, end - start, num_time_steps);

  const size_t N_next = next_layer.get_number_neurons();
  const bool next_is_seq = (batch_next_grad_matrix[0].size() == num_time_steps * N_next);

  const double* next_w_data = next_layer.get_w_values().data();
  for (size_t b = start; b < end; ++b)
  {
    const size_t b_idx = b - start;
    const double* next_grads_base = batch_next_grad_matrix[b].data();
    double* dest_base = &workspace.grad_from_next_all_t[b_idx * num_time_steps * N_this];

    for (int t = t_start; t >= t_end; --t)
    {
      if (!next_is_seq && t < t_start)
      {
        continue;
      }
      const double* g_next_t = next_is_seq ? &next_grads_base[t * N_next] : next_grads_base;
      double* dest_t = &dest_base[t * N_this];
      simd::gemv_add(next_w_data, g_next_t, dest_t, N_this, N_next);
    }
  }

  for (int t = t_start; t >= t_end; --t)
  {
    for (size_t b = start; b < end; ++b)
    {
      const size_t b_idx = b - start;
      const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
      const auto& state = layer_states[t];

      double* dh_next = &workspace.d_next_h[b_idx * N_this];
      const double* upstream_grads = &workspace.grad_from_next_all_t[(b_idx * num_time_steps + t) * N_this];
      double* g_this_tick = &workspace.rnn_grad_matrix[(b_idx * num_time_steps + t) * N_this];

      const auto mask = state.get_cell_state_values();
      const double* pre_act = state.get_pre_activation_sums().data();
      double* deriv_buf = &workspace.deriv_buf[b_idx * N_this];

      get_activation().activate_derivative(pre_act, pre_act + N_this, nullptr, deriv_buf);

      for (size_t j = 0; j < N_this; ++j)
      {
        double dh = std::clamp(upstream_grads[j] + dh_next[j], -50.0, 50.0);
        g_this_tick[j] = dh * deriv_buf[j] * mask[j];
      }

      // Calculate dX_t
      double* dx_t = &workspace.dx_matrix[(b_idx * num_time_steps + t) * N_prev];
      std::fill(dx_t, dx_t + N_prev, 0.0);
      simd::gemv_add(get_w_values().data(), g_this_tick, dx_t, N_prev, N_this);

      std::fill(dh_next, dh_next + N_this, 0.0);
      simd::gemv_add(_rw_values.data(), g_this_tick, dh_next, N_this, N_this);
    }
  }

  for (size_t b = start; b < end; ++b)
  {
    const size_t b_idx = b - start;
    const double* dX_src = &workspace.dx_matrix[b_idx * num_time_steps * N_prev];
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), dX_src, num_time_steps * N_prev);

    const double* gates_src = &workspace.rnn_grad_matrix[b_idx * num_time_steps * N_this];
    batch_gradients_and_outputs[b].set_rnn_gate_gradients(get_layer_index(), gates_src, num_time_steps * N_this);
  }
}

double ElmanRNNLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  double norm_sq = simd::sum_sq(_w_grads.data(), _w_grads.size()) +
                   simd::sum_sq(_rw_grads.data(), _rw_grads.size());
  if (has_bias())
  {
    norm_sq += simd::sum_sq(_b_grads.data(), _b_grads.size());
  }
  return norm_sq;
}

void ElmanRNNLayer::calculate_and_store_gradients(const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, const std::vector<HiddenStates>& hidden_states, const Layer& previous_layer, size_t batch_size, int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (batch_size == 0)
  {
    return;
  }

  const size_t N_this = get_number_neurons();
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t T = hidden_states[0].at(get_layer_index()).size();
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  _thread_w_grads.resize(num_threads);
  _thread_rw_grads.resize(num_threads);
  _thread_b_grads.resize(num_threads);

  for (unsigned int t = 0; t < num_threads; ++t)
  {
    _thread_w_grads[t].resize(_w_grads.size());
    std::fill(_thread_w_grads[t].begin(), _thread_w_grads[t].end(), 0.0);
    _thread_rw_grads[t].resize(_rw_grads.size());
    std::fill(_thread_rw_grads[t].begin(), _thread_rw_grads[t].end(), 0.0);
    _thread_b_grads[t].resize(has_bias() ? N_this : 0);
    std::fill(_thread_b_grads[t].begin(), _thread_b_grads[t].end(), 0.0);
  }

  auto run_chunk = [&](size_t start, size_t end, size_t thread_idx)
  {
    auto& local_w_grads = _thread_w_grads[thread_idx];
    auto& local_rw_grads = _thread_rw_grads[thread_idx];
    auto& local_b_grads = _thread_b_grads[thread_idx];

    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty())
      {
        continue;
      }

      const auto& layer_states = hidden_states[b].at(get_layer_index());
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      const auto& std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      const double* x_base = !rnn_in.empty() ? rnn_in.data() : std_in.data();
      const size_t x_seq_len = !rnn_in.empty() ? rnn_in.size() / N_prev : 1;

      for (size_t t = 0; t < T; ++t)
      {
        const double* g_t = &packed_grads[t * N_this];
        const double* x_t = (x_seq_len == T) ? &x_base[t * N_prev] : x_base;
        const double* h_prev = (t > 0) ? layer_states[t - 1].get_hidden_state_values().data() : nullptr;

        for (size_t j = 0; j < N_this; ++j)
        {
          const auto gj = g_t[j];
          if (std::abs(gj) < 1e-15)
          {
            continue;
          }

          if (has_bias())
          {
            local_b_grads[j] += gj;
          }
          for (size_t k = 0; k < N_prev; ++k)
          {
            local_w_grads[k * N_this + j] += gj * x_t[k];
          }
          if (h_prev)
          {
            for (size_t k = 0; k < N_this; ++k)
            {
              local_rw_grads[k * N_this + j] += gj * h_prev[k];
            }
          }
        }
      }
    }
  };

  const bool use_multithreading = (num_threads > 1) && (batch_size >= num_threads * 16);
  if (!use_multithreading)
  {
    run_chunk(0, batch_size, 0);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, t, &run_chunk]() { run_chunk(start, end, t); });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // Merge results
  zero_gradients();
  for (unsigned int t = 0; t < num_threads; ++t)
  {
    for (size_t i = 0; i < _w_grads.size(); ++i)
    {
      _w_grads[i] += _thread_w_grads[t][i];
    }
    for (size_t i = 0; i < _rw_grads.size(); ++i)
    {
      _rw_grads[i] += _thread_rw_grads[t][i];
    }
    if (has_bias())
    {
      for (size_t i = 0; i < _b_grads.size(); ++i)
      {
        _b_grads[i] += _thread_b_grads[t][i];
      }
    }
  }

  const double inv_batch = 1.0 / static_cast<double>(batch_size);
  for (double& x : _w_grads)
  {
    x *= inv_batch;
  }
  for (double& x : _rw_grads)
  {
    x *= inv_batch;
  }
  if (has_bias())
  {
    for (double& x : _b_grads)
    {
      x *= inv_batch;
    }
  }
}

void ElmanRNNLayer::zero_gradients()
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  Layer::zero_gradients();
  std::fill(_rw_grads.begin(), _rw_grads.end(), 0.0);
}

void ElmanRNNLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  apply_update_to_vector(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, learning_rate, clipping_scale, false, get_optimiser_type());
  apply_update_to_vector(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, learning_rate, clipping_scale, false, get_optimiser_type());
  if (has_bias())
  {
    apply_update_to_vector(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, learning_rate, clipping_scale, true, get_optimiser_type());
  }
  cache_recurrent_weights();
  zero_gradients();
}

void ElmanRNNLayer::allocate_workspace()
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (_task_queue_pool == nullptr)
  {
    return;
  }
  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  allocate_workspace(num_threads);
}

void ElmanRNNLayer::allocate_workspace(unsigned int num_threads)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (_thread_workspaces.size() <= num_threads)
  {
    _thread_workspaces.resize(num_threads);
  }
  for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx)
  {
    if (!_thread_workspaces[thread_idx])
    {
      _thread_workspaces[thread_idx] = std::make_unique<BPTTWorkspace>();
    }
  }
}

ElmanRNNLayer::BPTTWorkspace& ElmanRNNLayer::get_workspace(size_t thread_idx) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
#if VALIDATE_DATA == 1
  if (thread_idx >= _thread_workspaces.size())
  {
    Logger::panic("Trying to get a workspace thread ", thread_idx, " past the workspaces size!");
  }
#endif
  return *_thread_workspaces[thread_idx];
}

double ElmanRNNLayer::get_recurrent_weight_value(unsigned f, unsigned t) const
{
  return _rw_values[f * get_number_neurons() + t];
}

Layer* ElmanRNNLayer::clone() const
{
  return new ElmanRNNLayer(*this);
}

void ElmanRNNLayer::cache_recurrent_weights()
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t n = get_number_neurons();
  if (n == 0) return;
  _rw_values_T.resize(n * n);
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      _rw_values_T[j * n + i] = _rw_values[i * n + j];
    }
  }
}

} // namespace myoddweb::nn
