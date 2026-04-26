#include "./libraries/instrumentor.h"
#include "elmanrnnlayer.h"
#include "fflayer.h"
#include "logger.h"
#include <algorithm>
#include <cmath>

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
}

ElmanRNNLayer::ElmanRNNLayer(ElmanRNNLayer&& src) noexcept :
  Layer(std::move(src)),
  _rw_values(std::move(src._rw_values)),
  _rw_grads(std::move(src._rw_grads)),
  _rw_velocities(std::move(src._rw_velocities)),
  _rw_m1(std::move(src._rw_m1)),
  _rw_m2(std::move(src._rw_m2)),
  _rw_timesteps(std::move(src._rw_timesteps)),
  _rw_decays(std::move(src._rw_decays))
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

  std::vector<double> flattened_inputs(batch_size * num_time_steps * N_prev);
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty())
    {
      std::copy(rnn_in.begin(), rnn_in.end(), flattened_inputs.begin() + b * num_time_steps * N_prev);
    }
    else
    {
      const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        std::copy(std_in.begin(), std_in.end(), flattened_inputs.begin() + (b * num_time_steps + t) * N_prev);
      }
    }
  }

  // 2. Pre-calculate Input-to-Hidden (W * x_t) for all ticks
  std::vector<double> batch_pre_act(batch_size * num_time_steps * N_this, 0.0);

  auto precalc_gates = [&](size_t b_start, size_t b_end)
  {
    for (size_t b = b_start; b < b_end; ++b)
    {
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        const double* x_t = &flattened_inputs[(b * num_time_steps + t) * N_prev];
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
          for (size_t j = 0; j < N_this; ++j)
          {
            pre_t[j] += xi * w_row[j];
          }
        }
      }
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    precalc_gates(0, batch_size);
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
        _task_queue_pool->enqueue([&precalc_gates, start, end]()
        {
          precalc_gates(start, end);
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
    for (size_t b = b_start; b < b_end; ++b)
    {
      std::fill(current_h.begin(), current_h.end(), 0.0);
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        double* pre_t = &batch_pre_act[(b * num_time_steps + t) * N_this];

        // Recurrent-to-Hidden (U * h_{t-1})
        for (size_t i = 0; i < N_this; ++i)
        {
          const double hi = current_h[i];
          if (hi == 0.0)
          {
            continue;
          }
          const double* u_row = &_rw_values[i * N_this];
          for (size_t j = 0; j < N_this; ++j)
          {
            pre_t[j] += hi * u_row[j];
          }
        }

        if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
        {
          for (size_t j = 0; j < N_this; ++j)
          {
            pre_t[j] += batch_residual_output_values[b][j];
          }
        }

        auto& state = batch_hidden_states[b].at(get_layer_index())[t];
        state.set_pre_activation_sums(std::vector<double>(pre_t, pre_t + N_this));

        get_activation().activate(pre_t, pre_t + N_this, is_training);

        for (size_t j = 0; j < N_this; ++j)
        {
          double out = pre_t[j];
          if (is_training && get_neuron((unsigned)j).is_dropout())
          {
            const auto& neuron = get_neuron((unsigned)j);
            if (neuron.must_randomly_drop())
            {
              out = 0.0;
            }
            else
            {
              out /= (1.0 - neuron.get_dropout_rate());
            }
          }
          current_h[j] = out;
          batch_output_sequences[(b * num_time_steps + t) * N_this + j] = out;
        }
        state.set_hidden_state_values(current_h);
      }
    }
  };

  if (num_threads <= 1)
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
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), std::vector<double>(seq_ptr, seq_ptr + num_time_steps * N_this));
    const double* last_ptr = seq_ptr + (num_time_steps - 1) * N_this;
    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), std::vector<double>(last_ptr, last_ptr + N_this));
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
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), deltas);
    std::vector<double> last_step_deltas(N_this);
    std::copy(deltas.end() - N_this, deltas.end(), last_step_deltas.begin());
    batch_gradients_and_outputs[b].set_gradients(get_layer_index(), last_step_deltas);
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

  if (num_threads <= 1)
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
          auto& workspace = const_cast<ElmanRNNLayer*>(this)->get_workspace(t);
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

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  
  auto run_chunk = [this, &batch_gradients_and_outputs, &batch_output_gradients, &batch_hidden_states, bptt_max_ticks, N_this, num_time_steps](size_t start, size_t end, size_t t_id)
  {
    auto& workspace = const_cast<ElmanRNNLayer*>(this)->get_workspace(t_id);
    workspace.resize(N_this, end - start, num_time_steps);

    for (size_t b = start; b < end; ++b)
    {
      const size_t b_idx = b - start;
      const auto& next_grads = batch_output_gradients[b];
      double* dest_base = &workspace.grad_from_next_all_t[b_idx * num_time_steps * N_this];
      if (next_grads.size() == num_time_steps * N_this)
      {
        std::copy(next_grads.begin(), next_grads.end(), dest_base);
      }
      else
      {
        if (next_grads.size() != N_this)
        {
          Logger::panic("ElmanRNNLayer #", get_layer_index(), " output gradient size mismatch! Expected ", N_this, " but got ", next_grads.size());
        }
        std::copy(next_grads.begin(), next_grads.end(), dest_base + (num_time_steps - 1) * N_this);
      }
    }

    const int t_start = static_cast<int>(num_time_steps) - 1;
    int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

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

        for (size_t j = 0; j < N_this; ++j)
        {
          double dh = upstream_grads[j] + dh_next[j];
          double deriv = get_activation().activate_derivative(state.get_pre_activation_sum_at_neuron((unsigned)j));
          g_this_tick[j] = dh * deriv;
        }

        std::fill(dh_next, dh_next + N_this, 0.0);
        if (t > 0)
        {
          const double* rw = _rw_values_T.data();
          for (size_t i = 0; i < N_this; ++i)
          {
            const double g_val = g_this_tick[i];
            const double* rw_row = &rw[i * N_this];
            for (size_t j = 0; j < N_this; ++j)
            {
              dh_next[j] += g_val * rw_row[j];
            }
          }
        }
      }
    }

    const size_t grad_size = num_time_steps * N_this;
    for (size_t b = start; b < end; ++b)
    {
      const size_t b_idx = b - start;
      const double* src = &workspace.rnn_grad_matrix[b_idx * grad_size];
      batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), std::vector<double>(src, src + grad_size));

      std::vector<double> last_step_deltas(N_this);
      std::copy(src + grad_size - N_this, src + grad_size, last_step_deltas.begin());
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), last_step_deltas);
    }
  };

  if (num_threads <= 1)
  {
    run_chunk(0, batch_size, 0);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t_id = 0; t_id < num_threads; ++t_id)
    {
      size_t size = (batch_size / num_threads) + (t_id < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([start, end, t_id, &run_chunk]()
        {
          run_chunk(start, end, t_id);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
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
  const BPTTWorkspace::AlignedVector& rw_values_T) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  workspace.resize(N_this, end - start, num_time_steps);

  const size_t N_next = next_layer.get_number_neurons();
  const bool next_is_seq = (batch_next_grad_matrix[0].size() == num_time_steps * N_next);

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
      for (size_t j = 0; j < N_this; ++j)
      {
        const double* next_w_row = next_layer.get_w_values().data() + j * N_next;
        __m256d sum_vec = _mm256_setzero_pd();
        size_t k = 0;
        for (; k + 3 < N_next; k += 4)
        {
          sum_vec = _mm256_fmadd_pd(_mm256_loadu_pd(&g_next_t[k]), _mm256_loadu_pd(&next_w_row[k]), sum_vec);
        }
        double buffer[4];
        _mm256_storeu_pd(buffer, sum_vec);
        double sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
        for (; k < N_next; ++k)
        {
          sum += g_next_t[k] * next_w_row[k];
        }
        dest_t[j] += sum;
      }
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

      for (size_t j = 0; j < N_this; ++j)
      {
        double dh = upstream_grads[j] + dh_next[j];
        double deriv = get_activation().activate_derivative(state.get_pre_activation_sum_at_neuron((unsigned)j));
        g_this_tick[j] = dh * deriv;
      }

      std::fill(dh_next, dh_next + N_this, 0.0);
      for (size_t j = 0; j < N_this; ++j)
      {
        const double gj = g_this_tick[j];
        if (gj == 0.0)
        {
          continue;
        }
        for (size_t k = 0; k < N_this; ++k)
        {
          dh_next[k] += gj * rw_values_T[k * N_this + j];
        }
      }
    }
  }

  for (size_t b = start; b < end; ++b)
  {
    const size_t b_idx = b - start;
    const double* src = &workspace.rnn_grad_matrix[b_idx * num_time_steps * N_this];
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), std::vector<double>(src, src + num_time_steps * N_this));
  }
}

void ElmanRNNLayer::calculate_and_store_gradients(const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, const std::vector<HiddenStates>& hidden_states, const Layer& previous_layer, size_t batch_size, int bptt_max_ticks)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t T = hidden_states[0].at(get_layer_index()).size();
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gradients(get_layer_index());
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
        if (has_bias())
        {
          _b_grads[j] += g_t[j];
        }
        for (size_t k = 0; k < N_prev; ++k)
        {
          _w_grads[k * N_this + j] += g_t[j] * x_t[k];
        }
        if (h_prev)
        {
          for (size_t k = 0; k < N_this; ++k)
          {
            _rw_grads[k * N_this + j] += g_t[j] * h_prev[k];
          }
        }
      }
    }
  }

  const double inv_batch = 1.0 / static_cast<double>(batch_size * T);
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

double ElmanRNNLayer::get_gradient_norm_sq() const
{
  auto ssq = [](const std::vector<double>& v)
  {
    double s = 0; for (double x : v)
    {
      s += x * x;
    }
    return s;
  };
  return ssq(_w_grads) + ssq(_rw_grads) + (has_bias() ? ssq(_b_grads) : 0.0);
}

void ElmanRNNLayer::zero_gradients()
{
  std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
  std::fill(_rw_grads.begin(), _rw_grads.end(), 0.0);
  if (has_bias())
  {
    std::fill(_b_grads.begin(), _b_grads.end(), 0.0);
  }
}

void ElmanRNNLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  apply_update_to_vector(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, learning_rate, clipping_scale, false, get_optimiser_type());
  apply_update_to_vector(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, learning_rate, clipping_scale, false, get_optimiser_type());
  if (has_bias())
  {
    apply_update_to_vector(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, learning_rate, clipping_scale, true, get_optimiser_type());
  }
}

void ElmanRNNLayer::cache_recurrent_weights()
{
  const size_t n = get_number_neurons();
  if (n == 0)
  {
    return;
  }
  _rw_values_T.resize(n * n);
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < n; ++j)
    {
      _rw_values_T[j * n + i] = _rw_values[i * n + j];
    }
  }
}

ElmanRNNLayer::BPTTWorkspace& ElmanRNNLayer::get_workspace(size_t thread_idx) const
{
  std::unique_lock<std::shared_mutex> lock(_workspace_mutex);
  if (_thread_workspaces.size() <= thread_idx)
  {
    _thread_workspaces.resize(thread_idx + 1);
  }
  if (!_thread_workspaces[thread_idx])
  {
    _thread_workspaces[thread_idx] = std::make_unique<BPTTWorkspace>();
  }
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
