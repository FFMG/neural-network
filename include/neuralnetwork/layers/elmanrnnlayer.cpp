#include "../libraries/instrumentor.h"
#include "elmanrnnlayer.h"
#include "fflayer.h"
#include "../common/logger.h"
#include "../common/simd_utils.h"
#include "../common/tempbuffer.h"
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
  double momentum,
  std::optional<uint32_t> seed
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
    momentum,
    seed
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
  double momentum,
  std::optional<uint32_t> seed
) :
  Layer(
    layer_index,
    layer_role,
    activation_method,
    optimiser_type,
    residual_layer_number,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    create_neurons(dropout_rate, num_neurons_in_this_layer, seed),
    has_bias,
    weight_decays,
    residual_projector,
    number_of_threads,
    momentum,
    seed
  )
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  initialize_recurrent_weights(weight_decays.empty() ? 0.0 : weight_decays[0], seed);
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
  _identity_proxy = nullptr;
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
  _w_values_T(std::move(src._w_values_T)),
  _thread_workspaces(std::move(src._thread_workspaces)),
  _thread_grad_accumulators(std::move(src._thread_grad_accumulators))
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  _identity_proxy = src._identity_proxy;
  src._identity_proxy = nullptr;
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
    delete _identity_proxy;
    _identity_proxy = nullptr;
    Layer::operator=(src);
    _rw_values = src._rw_values;
    _rw_grads = src._rw_grads;
    _rw_velocities = src._rw_velocities;
    _rw_m1 = src._rw_m1;
    _rw_m2 = src._rw_m2;
    _rw_timesteps = src._rw_timesteps;
    _rw_decays = src._rw_decays;
    _thread_grad_accumulators.clear();
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
    delete _identity_proxy;
    _identity_proxy = src._identity_proxy;
    src._identity_proxy = nullptr;
    Layer::operator=(std::move(src));
    _rw_values = std::move(src._rw_values);
    _rw_grads = std::move(src._rw_grads);
    _rw_velocities = std::move(src._rw_velocities);
    _rw_m1 = std::move(src._rw_m1);
    _rw_m2 = std::move(src._rw_m2);
    _rw_timesteps = std::move(src._rw_timesteps);
    _rw_decays = std::move(src._rw_decays);
    _rw_values_T = std::move(src._rw_values_T);
    _w_values_T = std::move(src._w_values_T);
    _thread_workspaces = std::move(src._thread_workspaces);
    _thread_grad_accumulators = std::move(src._thread_grad_accumulators);
  }
  return *this;
}

ElmanRNNLayer::~ElmanRNNLayer()
{
  delete _identity_proxy;
}

void ElmanRNNLayer::initialize_recurrent_weights(double weight_decay, std::optional<uint32_t> seed)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const auto num_neurons = get_number_output_neurons();
  const size_t num_weights = static_cast<size_t>(num_neurons) * num_neurons;

  _rw_values.resize(num_weights);
  for (size_t i = 0; i < num_neurons; ++i)
  {
    for (unsigned int o = 0; o < num_neurons; ++o)
    {
      const size_t flat_index = i * num_neurons + o;
      const auto weight_seed = seed.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(Rng::derive(seed.value(), 1, flat_index))) : std::nullopt;
      _rw_values[flat_index] = get_activation().weight_initialization(num_neurons, num_neurons, weight_seed);
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

  const double* flattened_inputs_ptr = nullptr;
  if (batch_size == 1)
  {
    const auto& rnn_in = batch_gradients_and_outputs[0].get_rnn_outputs(prev_layer_index);
    if (rnn_in.size() == num_time_steps * N_prev)
    {
      flattened_inputs_ptr = rnn_in.data();
    }
  }

  TempBuffer<double, 10> flattened_batch_inputs(flattened_inputs_ptr == nullptr ? batch_size * num_time_steps * N_prev : 0);

  if (flattened_inputs_ptr == nullptr)
  {
    double* dst = flattened_batch_inputs.data();
    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      if (!rnn_in.empty())
      {
        std::copy_n(rnn_in.begin(), num_time_steps * N_prev, dst + b * num_time_steps * N_prev);
      }
      else
      {
        const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
        for (size_t t = 0; t < num_time_steps; ++t)
        {
          std::copy_n(std_in.begin(), N_prev, dst + (b * num_time_steps + t) * N_prev);
        }
      }
    }
    flattened_inputs_ptr = flattened_batch_inputs.data();
  }

  // 2. Pre-calculate Input-to-Hidden (W * x_t) for all ticks
  TempBuffer<double, 11> batch_pre_act(batch_size * num_time_steps * N_this);
  double* batch_pre_act_ptr = batch_pre_act.data();

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * N_prev * N_this) / 100000))) : 1;
  const bool use_multithreading = (active_threads > 1);

  if (!use_multithreading)
  {
    pre_calculate_gates(0, batch_size, N_this, N_prev, num_time_steps, flattened_inputs_ptr, batch_pre_act_ptr);
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
        _task_queue_pool->enqueue([start, end, N_this, N_prev, num_time_steps, flattened_inputs_ptr, batch_pre_act_ptr, this]()
        {
          pre_calculate_gates(start, end, N_this, N_prev, num_time_steps, flattened_inputs_ptr, batch_pre_act_ptr);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 3. Sequential Recurrent Pass and Activations
  TempBuffer<double, 12> batch_output_seq(batch_size * num_time_steps * N_this);
  double* batch_output_seq_ptr = batch_output_seq.data();

  auto recurrent_pass = [this, N_this, num_time_steps, is_training, batch_pre_act_ptr, batch_output_seq_ptr,
    &batch_residual_output_values, &batch_hidden_states](size_t b_start, size_t b_end)
  {
    const size_t chunk_size = b_end - b_start;
    TempBuffer<double, 13> current_h(chunk_size * N_this, true);

    const bool has_dropout = (is_training && get_dropout() > 0.0);
    TempBuffer<double, 14> mask((has_dropout || !batch_hidden_states.empty()) ? chunk_size * N_this : 0);
    if (!has_dropout && !mask.empty())
    {
      std::fill_n(mask.data(), chunk_size * N_this, 1.0);
    }

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      // 1. Recurrent GEMM across batches (only for t > 0 because at t = 0, h_{-1} = 0)
      if (t > 0)
      {
        size_t b_idx = 0;
        for (; b_idx + 3 < chunk_size; b_idx += 4)
        {
          simd::gemm_four_batches(
            &current_h.data()[b_idx * N_this],
            &current_h.data()[(b_idx + 1) * N_this],
            &current_h.data()[(b_idx + 2) * N_this],
            &current_h.data()[(b_idx + 3) * N_this],
            _rw_values.data(),
            &batch_pre_act_ptr[((b_start + b_idx) * num_time_steps + t) * N_this],
            &batch_pre_act_ptr[((b_start + b_idx + 1) * num_time_steps + t) * N_this],
            &batch_pre_act_ptr[((b_start + b_idx + 2) * num_time_steps + t) * N_this],
            &batch_pre_act_ptr[((b_start + b_idx + 3) * num_time_steps + t) * N_this],
            N_this, N_this
          );
        }
        for (; b_idx + 1 < chunk_size; b_idx += 2)
        {
          simd::gemm_two_batches(
            &current_h.data()[b_idx * N_this],
            &current_h.data()[(b_idx + 1) * N_this],
            _rw_values.data(),
            &batch_pre_act_ptr[((b_start + b_idx) * num_time_steps + t) * N_this],
            &batch_pre_act_ptr[((b_start + b_idx + 1) * num_time_steps + t) * N_this],
            N_this, N_this
          );
        }
        for (; b_idx < chunk_size; ++b_idx)
        {
          simd::gemm_one_batch(
            &current_h.data()[b_idx * N_this],
            _rw_values.data(),
            &batch_pre_act_ptr[((b_start + b_idx) * num_time_steps + t) * N_this],
            N_this, N_this
          );
        }
      }

      // 2. Residuals, activation, dropout, and recording
      for (size_t b_idx = 0; b_idx < chunk_size; ++b_idx)
      {
        const size_t b = b_start + b_idx;
        double* pre_t = &batch_pre_act_ptr[(b * num_time_steps + t) * N_this];
        double* h_out = &current_h.data()[b_idx * N_this];
        double* seq_dest = &batch_output_seq_ptr[(b * num_time_steps + t) * N_this];

        if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
        {
          simd::add_vectors(batch_residual_output_values[b].data(), pre_t, N_this);
        }

        if (!batch_hidden_states.empty())
        {
          auto& state = batch_hidden_states[b].at(get_layer_index())[t];
          state.set_pre_activation_sums(pre_t, N_this);
        }

        get_activation().activate(pre_t, pre_t + N_this, is_training);

        if (has_dropout)
        {
          double* mask_ptr = &mask.data()[b_idx * N_this];
          std::fill_n(mask_ptr, N_this, 1.0);
          const auto& neurons = get_neurons();
          for (size_t j = 0; j < N_this; ++j)
          {
            double out = pre_t[j];
            const auto& neuron = neurons[j];
            if (neuron.is_dropout())
            {
              if (neuron.must_randomly_drop(b * num_time_steps + t))
              {
                out = 0.0;
                mask_ptr[j] = 0.0;
              }
              else
              {
                const double scale = 1.0 / (1.0 - neuron.get_dropout_rate());
                out *= scale;
                mask_ptr[j] = scale;
              }
            }
            h_out[j] = out;
            seq_dest[j] = out;
          }
          if (!batch_hidden_states.empty())
          {
            auto& state = batch_hidden_states[b].at(get_layer_index())[t];
            state.set_cell_state_values(mask_ptr, N_this);
            state.set_hidden_state_values(h_out, N_this);
          }
        }
        else
        {
          std::copy_n(pre_t, N_this, h_out);
          std::copy_n(pre_t, N_this, seq_dest);
          if (!batch_hidden_states.empty())
          {
            auto& state = batch_hidden_states[b].at(get_layer_index())[t];
            state.set_cell_state_values(&mask.data()[b_idx * N_this], N_this);
            state.set_hidden_state_values(h_out, N_this);
          }
        }
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
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
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
    const double* seq_ptr = &batch_output_seq_ptr[b * num_time_steps * N_this];
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), seq_ptr, num_time_steps * N_this);
    const double* last_ptr = seq_ptr + (num_time_steps - 1) * N_this;
    double* dest_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    std::copy_n(last_ptr, N_this, dest_ptr);
  }
}

void ElmanRNNLayer::pre_calculate_gates(
  size_t b_start, 
  size_t b_end,
  size_t N_this,
  size_t N_prev,
  size_t num_time_steps,
  const double* flattened_batch_inputs,
  double* batch_pre_act
) const
{
  const double* W = get_w_values().data();
  const size_t step_start = b_start * num_time_steps;
  const size_t step_end = b_end * num_time_steps;

  if (has_bias())
  {
    const auto& biases = get_b_values();
    const size_t copy_size = std::min<size_t>(biases.size(), N_this);
    for (size_t step = step_start; step < step_end; ++step)
    {
      double* dest = &batch_pre_act[step * N_this];
      std::copy_n(biases.begin(), copy_size, dest);
      if (copy_size < N_this)
      {
        std::fill_n(dest + copy_size, N_this - copy_size, 0.0);
      }
    }
  }
  else
  {
    std::fill_n(&batch_pre_act[step_start * N_this], (step_end - step_start) * N_this, 0.0);
  }

  size_t step = step_start;
  for (; step + 3 < step_end; step += 4)
  {
    const double* x0 = &flattened_batch_inputs[step * N_prev];
    const double* x1 = &flattened_batch_inputs[(step + 1) * N_prev];
    const double* x2 = &flattened_batch_inputs[(step + 2) * N_prev];
    const double* x3 = &flattened_batch_inputs[(step + 3) * N_prev];

    double* y0 = &batch_pre_act[step * N_this];
    double* y1 = &batch_pre_act[(step + 1) * N_this];
    double* y2 = &batch_pre_act[(step + 2) * N_this];
    double* y3 = &batch_pre_act[(step + 3) * N_this];

    simd::gemm_four_batches(
      x0, x1, x2, x3,
      W,
      y0, y1, y2, y3,
      N_prev, N_this
    );
  }

  for (; step + 1 < step_end; step += 2)
  {
    const double* x0 = &flattened_batch_inputs[step * N_prev];
    const double* x1 = &flattened_batch_inputs[(step + 1) * N_prev];

    double* y0 = &batch_pre_act[step * N_this];
    double* y1 = &batch_pre_act[(step + 1) * N_this];

    simd::gemm_two_batches(
      x0, x1,
      W,
      y0, y1,
      N_prev, N_this
    );
  }

  for (; step < step_end; ++step)
  {
    const double* x_row = &flattened_batch_inputs[step * N_prev];
    double* y_row = &batch_pre_act[step * N_this];

    simd::gemm_one_batch(
      x_row,
      W,
      y_row,
      N_prev, N_this
    );
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
  pre_calculate_gates(b_start, b_end, N_this, N_prev, num_time_steps, flattened_batch_inputs.data(), batch_pre_act.data());
}

void ElmanRNNLayer::calculate_output_gradients(std::vector<GradientsAndOutputs>& batch_gradients_and_outputs, std::vector<std::vector<double>>::const_iterator target_outputs_begin, const std::vector<HiddenStates>& batch_hidden_states, size_t batch_size) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_this = get_number_neurons();
  TempBuffer<double, 15> deltas_buf(0);
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& states = batch_hidden_states[b].at(get_layer_index());
    const size_t T = states.size();
    if (deltas_buf.size() < T * N_this)
    {
      deltas_buf.assign(T * N_this, 0.0);
    }
    double* deltas = deltas_buf.data();
    const std::vector<double>& targets = *(target_outputs_begin + b);
    if (targets.size() == T * N_this)
    {
      for (size_t t = 0; t < T; ++t)
      {
        const auto& given = states[t].get_hidden_state_values();
        simd::sub_vectors(given.data(), &targets[t * N_this], deltas + t * N_this, N_this);
      }
    }
    else
    {
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
    }
    double* dest_ptr = batch_gradients_and_outputs[b].get_gradients_raw(get_layer_index());
    std::copy_n(deltas + (T - 1) * N_this, N_this, dest_ptr);
    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), deltas, T * N_this);
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

  const auto num_threads = get_number_of_threads();
  const size_t N_next = next_layer.get_number_neurons();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * num_time_steps * N_this * (N_next + N_this)) / 100000))) : 1;
  const bool use_multithreading = (active_threads > 1);

  if (!use_multithreading)
  {
    auto& workspace = get_workspace(0);
    calculate_bptt_batch_chunk(0, batch_size, batch_gradients_and_outputs, next_layer, batch_next_grad_matrix, batch_hidden_states, bptt_max_ticks, workspace, _rw_values_T);
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

  if (_identity_proxy == nullptr)
  {
    _identity_proxy = new FFLayer(0, static_cast<unsigned>(N_this), static_cast<unsigned>(N_this), 0.0, Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, false, 0.0, std::nullopt);
    std::vector<double> id(static_cast<size_t>(N_this) * N_this, 0.0);
    for (unsigned i = 0; i < N_this; ++i)
    {
      id[i * N_this + i] = 1.0;
    }
    _identity_proxy->set_w_values(id);
  }

  calculate_hidden_gradients(batch_gradients_and_outputs, *_identity_proxy, batch_output_gradients, batch_hidden_states, batch_size, bptt_max_ticks);
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
  const AlignedVector& /*rw_values_T*/) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_prev = get_number_input_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  workspace.resize(N_this, N_prev, end - start, num_time_steps);

  const size_t N_next = next_layer.get_number_neurons();
  const bool use_direct_gradients = batch_next_grad_matrix.empty();
  const bool is_identity = (_identity_proxy != nullptr && &next_layer == _identity_proxy);
  const unsigned target_layer_idx = (use_direct_gradients && is_identity) ? (get_layer_index() + 1) : next_layer.get_layer_index();

  bool next_is_seq = false;
  if (use_direct_gradients)
  {
    if (end > start)
    {
      const auto& rnn_g = batch_gradients_and_outputs[start].get_rnn_gradients(target_layer_idx);
      if (!rnn_g.empty())
      {
        next_is_seq = (rnn_g.size() == num_time_steps * N_next);
      }
      else
      {
        const auto next_grads = batch_gradients_and_outputs[start].get_gradients(target_layer_idx);
        next_is_seq = (next_grads.size() == num_time_steps * N_next);
      }
    }
  }
  else
  {
    if (!batch_next_grad_matrix.empty())
    {
      next_is_seq = (batch_next_grad_matrix[0].size() == num_time_steps * N_next);
    }
  }

  const double* next_w_data = next_layer.get_w_values().data();
  for (size_t b = start; b < end; ++b)
  {
    const size_t b_idx = b - start;
    const double* next_grads_base = nullptr;
    if (use_direct_gradients)
    {
      const auto& rnn_g = batch_gradients_and_outputs[b].get_rnn_gradients(target_layer_idx);
      if (!rnn_g.empty())
      {
        next_grads_base = rnn_g.data();
      }
      else
      {
        const auto std_g = batch_gradients_and_outputs[b].get_gradients(target_layer_idx);
        next_grads_base = std_g.data();
      }
    }
    else
    {
      if (b < batch_next_grad_matrix.size())
      {
        next_grads_base = batch_next_grad_matrix[b].data();
      }
    }

    if (next_grads_base == nullptr)
    {
      continue;
    }

    double* dest_base = &workspace.grad_from_next_all_t[b_idx * num_time_steps * N_this];

    if (next_is_seq)
    {
      if (is_identity)
      {
        std::copy_n(next_grads_base, num_time_steps * N_this, dest_base);
      }
      else
      {
        int t = t_start;
        for (; t - 3 >= t_end; t -= 4)
        {
          const double* g0 = &next_grads_base[t * N_next];
          const double* g1 = &next_grads_base[(t - 1) * N_next];
          const double* g2 = &next_grads_base[(t - 2) * N_next];
          const double* g3 = &next_grads_base[(t - 3) * N_next];

          double* d0 = &dest_base[t * N_this];
          double* d1 = &dest_base[(t - 1) * N_this];
          double* d2 = &dest_base[(t - 2) * N_this];
          double* d3 = &dest_base[(t - 3) * N_this];

          simd::gemm_transposed_four_batches(g0, g1, g2, g3, next_w_data, d0, d1, d2, d3, N_this, N_next);
        }
        for (; t - 1 >= t_end; t -= 2)
        {
          const double* g0 = &next_grads_base[t * N_next];
          const double* g1 = &next_grads_base[(t - 1) * N_next];

          double* d0 = &dest_base[t * N_this];
          double* d1 = &dest_base[(t - 1) * N_this];

          simd::gemm_transposed_two_batches(g0, g1, next_w_data, d0, d1, N_this, N_next);
        }
        for (; t >= t_end; --t)
        {
          const double* g0 = &next_grads_base[t * N_next];
          double* d0 = &dest_base[t * N_this];

          simd::gemm_transposed_one_batch(g0, next_w_data, d0, N_this, N_next);
        }
      }
    }
    else
    {
      if (t_start >= t_end)
      {
        if (is_identity)
        {
          std::copy_n(next_grads_base, N_this, &dest_base[t_start * N_this]);
        }
        else
        {
          const double* g_next_t = next_grads_base;
          double* dest_t = &dest_base[t_start * N_this];
          simd::gemv_add(next_w_data, g_next_t, dest_t, N_this, N_next);
        }
      }
    }
  }

  const size_t batch_size_chunk = end - start;
  const size_t total_elements = batch_size_chunk * N_this;

  for (int t = t_start; t >= t_end; --t)
  {
    // 1. Gather raw inputs contiguously
    for (size_t b_idx = 0; b_idx < batch_size_chunk; ++b_idx)
    {
      size_t b = start + b_idx;
      const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
      const auto& state = layer_states[t];

      std::copy(state.get_pre_activation_sums().begin(), state.get_pre_activation_sums().end(), &workspace.pre_act_buf[b_idx * N_this]);
      std::copy(state.get_hidden_state_values().begin(), state.get_hidden_state_values().end(), &workspace.hidden_val_buf[b_idx * N_this]);
    }

    // 2. Contiguous activation derivative
    const bool has_dropout = (get_dropout() > 0.0);
    const double* y_vals = has_dropout ? nullptr : workspace.hidden_val_buf.data();
    get_activation().activate_derivative(workspace.pre_act_buf.data(), workspace.pre_act_buf.data() + total_elements, y_vals, workspace.deriv_buf.data());

    // 3. Batch loop for gate steps
    for (size_t b_idx = 0; b_idx < batch_size_chunk; ++b_idx)
    {
      size_t b = start + b_idx;
      const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
      const auto& state = layer_states[t];

      double* dh_next = &workspace.d_next_h[b_idx * N_this];
      const double* upstream_grads = &workspace.grad_from_next_all_t[(b_idx * num_time_steps + t) * N_this];
      double* g_this_tick = &workspace.rnn_grad_matrix[(b_idx * num_time_steps + t) * N_this];

      const auto mask = state.get_cell_state_values();
      double* deriv_buf = &workspace.deriv_buf[b_idx * N_this];

      if (t == t_start)
      {
        std::fill(dh_next, dh_next + N_this, 0.0);
      }

      // Calculate gate gradients using SIMD helper
      simd::elman_bptt_gate_step(upstream_grads, dh_next, deriv_buf, mask.data(), g_this_tick, N_this);

      std::fill(dh_next, dh_next + N_this, 0.0);
    }

    // 4. Contiguous batched GEMM operations outside the batch loop
    {
      size_t b = 0;
      for (; b + 3 < batch_size_chunk; b += 4)
      {
        const double* g0 = &workspace.rnn_grad_matrix[(b * num_time_steps + t) * N_this];
        const double* g1 = &workspace.rnn_grad_matrix[((b + 1) * num_time_steps + t) * N_this];
        const double* g2 = &workspace.rnn_grad_matrix[((b + 2) * num_time_steps + t) * N_this];
        const double* g3 = &workspace.rnn_grad_matrix[((b + 3) * num_time_steps + t) * N_this];

        double* y0_rw = workspace.d_next_h.data() + b * N_this;
        double* y1_rw = workspace.d_next_h.data() + (b + 1) * N_this;
        double* y2_rw = workspace.d_next_h.data() + (b + 2) * N_this;
        double* y3_rw = workspace.d_next_h.data() + (b + 3) * N_this;

        double* y0_w = &workspace.dx_matrix[(b * num_time_steps + t) * N_prev];
        double* y1_w = &workspace.dx_matrix[((b + 1) * num_time_steps + t) * N_prev];
        double* y2_w = &workspace.dx_matrix[((b + 2) * num_time_steps + t) * N_prev];
        double* y3_w = &workspace.dx_matrix[((b + 3) * num_time_steps + t) * N_prev];

        simd::gemm_four_batches(g0, g1, g2, g3, _rw_values_T.data(), y0_rw, y1_rw, y2_rw, y3_rw, N_this, N_this);
        simd::gemm_four_batches(g0, g1, g2, g3, _w_values_T.data(), y0_w, y1_w, y2_w, y3_w, N_this, N_prev);
      }
      for (; b + 1 < batch_size_chunk; b += 2)
      {
        const double* g0 = &workspace.rnn_grad_matrix[(b * num_time_steps + t) * N_this];
        const double* g1 = &workspace.rnn_grad_matrix[((b + 1) * num_time_steps + t) * N_this];

        double* y0_rw = workspace.d_next_h.data() + b * N_this;
        double* y1_rw = workspace.d_next_h.data() + (b + 1) * N_this;

        double* y0_w = &workspace.dx_matrix[(b * num_time_steps + t) * N_prev];
        double* y1_w = &workspace.dx_matrix[((b + 1) * num_time_steps + t) * N_prev];

        simd::gemm_two_batches(g0, g1, _rw_values_T.data(), y0_rw, y1_rw, N_this, N_this);
        simd::gemm_two_batches(g0, g1, _w_values_T.data(), y0_w, y1_w, N_this, N_prev);
      }
      for (; b < batch_size_chunk; ++b)
      {
        const double* g = &workspace.rnn_grad_matrix[(b * num_time_steps + t) * N_this];
        double* y_rw = workspace.d_next_h.data() + b * N_this;
        double* y_w = &workspace.dx_matrix[(b * num_time_steps + t) * N_prev];

        simd::gemm_one_batch(g, _rw_values_T.data(), y_rw, N_this, N_this);
        simd::gemm_one_batch(g, _w_values_T.data(), y_w, N_this, N_prev);
      }
    }
  }

  for (size_t b = start; b < end; ++b)
  {
    const size_t b_idx = b - start;
    const double* dX_src = &workspace.dx_matrix[b_idx * num_time_steps * N_prev];
    if (num_time_steps > 0 && N_prev > 0 && get_layer_index() > 0)
    {
      batch_gradients_and_outputs[b].set_gradients(get_layer_index() - 1, dX_src + (num_time_steps - 1) * N_prev, N_prev);
    }
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

void ElmanRNNLayer::accumulate_swa_average_impl(const Layer& snapshot, size_t existing_swa_count)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const auto& other = static_cast<const ElmanRNNLayer&>(snapshot);
  swa_average_into(_w_values, other._w_values, existing_swa_count);
  swa_average_into(_rw_values, other._rw_values, existing_swa_count);
  swa_average_into(_b_values, other._b_values, existing_swa_count);
}

void ElmanRNNLayer::update_lookahead_slow_weights_impl(Layer& fast_layer, double alpha)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  auto& other = static_cast<ElmanRNNLayer&>(fast_layer);
  simd::lookahead_step(_w_values.data(), other._w_values.data(), alpha, _w_values.size());
  simd::lookahead_step(_rw_values.data(), other._rw_values.data(), alpha, _rw_values.size());
  if (has_bias())
  {
    simd::lookahead_step(_b_values.data(), other._b_values.data(), alpha, _b_values.size());
  }
  other.cache_recurrent_weights();
}

namespace
{
struct ElmanGradCalcTask
{
  const ElmanRNNLayer& layer;
  size_t start;
  size_t end;
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs;
  const std::vector<HiddenStates>& hidden_states;
  unsigned prev_layer_index;
  size_t N_prev;
  size_t N_this;
  size_t T;
  std::span<double> local_w_grads;
  std::span<double> local_rw_grads;
  std::span<double> local_b_grads;

  void operator()() const
  {
    MYODDWEB_PROFILE_FUNCTION("ElmanGradCalcTask");
    layer.calculate_and_store_gradients_chunk(
      start,
      end,
      batch_gradients_and_outputs,
      hidden_states,
      prev_layer_index,
      N_prev,
      N_this,
      T,
      local_w_grads,
      local_rw_grads,
      local_b_grads
    );
  }
};
}

void ElmanRNNLayer::calculate_and_store_gradients_chunk(
  size_t start,
  size_t end,
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& hidden_states,
  unsigned prev_layer_index,
  size_t N_prev,
  size_t N_this,
  size_t T,
  std::vector<double>& local_w_grads,
  std::vector<double>& local_rw_grads,
  std::vector<double>& local_b_grads) const
{
  calculate_and_store_gradients_chunk(
    start, end,
    batch_gradients_and_outputs, hidden_states,
    prev_layer_index, N_prev, N_this, T,
    std::span<double>(local_w_grads),
    std::span<double>(local_rw_grads),
    std::span<double>(local_b_grads)
  );
}

void ElmanRNNLayer::calculate_and_store_gradients_chunk(
  size_t start,
  size_t end,
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& hidden_states,
  unsigned prev_layer_index,
  size_t N_prev,
  size_t N_this,
  size_t T,
  std::span<double> local_w_grads,
  std::span<double> local_rw_grads,
  std::span<double> local_b_grads) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  if (start >= end)
  {
    return;
  }

  if (has_bias() && !local_b_grads.empty())
  {
    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty())
      {
        continue;
      }
      for (size_t t = 0; t < T; ++t)
      {
        const double* g_t = &packed_grads[t * N_this];
        simd::add_vectors(g_t, local_b_grads.data(), N_this);
      }
    }
  }

  // 1. Input weight gradients: 4-wide and 2-wide unrolled outer product
  size_t k = 0;
  for (; k + 3 < N_prev; k += 4)
  {
    double* row0 = &local_w_grads[k * N_this];
    double* row1 = &local_w_grads[(k + 1) * N_this];
    double* row2 = &local_w_grads[(k + 2) * N_this];
    double* row3 = &local_w_grads[(k + 3) * N_this];

    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty())
      {
        continue;
      }
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      const auto& std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      const double* x_base = !rnn_in.empty() ? rnn_in.data() : std_in.data();
      const size_t x_seq_len = !rnn_in.empty() ? rnn_in.size() / N_prev : 1;

      for (size_t t = 0; t < T; ++t)
      {
        const double* g_t = &packed_grads[t * N_this];
        const double* x_t = (x_seq_len == T) ? &x_base[t * N_prev] : x_base;
        simd::mul_add_four_scalars(x_t[k], x_t[k + 1], x_t[k + 2], x_t[k + 3], g_t, row0, row1, row2, row3, N_this);
      }
    }
  }
  for (; k + 1 < N_prev; k += 2)
  {
    double* row0 = &local_w_grads[k * N_this];
    double* row1 = &local_w_grads[(k + 1) * N_this];

    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty())
      {
        continue;
      }
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      const auto& std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      const double* x_base = !rnn_in.empty() ? rnn_in.data() : std_in.data();
      const size_t x_seq_len = !rnn_in.empty() ? rnn_in.size() / N_prev : 1;

      for (size_t t = 0; t < T; ++t)
      {
        const double* g_t = &packed_grads[t * N_this];
        const double* x_t = (x_seq_len == T) ? &x_base[t * N_prev] : x_base;
        simd::mul_add_two_scalars(x_t[k], x_t[k + 1], g_t, row0, row1, N_this);
      }
    }
  }
  for (; k < N_prev; ++k)
  {
    double* row0 = &local_w_grads[k * N_this];
    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty())
      {
        continue;
      }
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      const auto& std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
      const double* x_base = !rnn_in.empty() ? rnn_in.data() : std_in.data();
      const size_t x_seq_len = !rnn_in.empty() ? rnn_in.size() / N_prev : 1;

      for (size_t t = 0; t < T; ++t)
      {
        const double* g_t = &packed_grads[t * N_this];
        const double* x_t = (x_seq_len == T) ? &x_base[t * N_prev] : x_base;
        simd::mul_add(x_t[k], g_t, row0, N_this);
      }
    }
  }

  // 2. Recurrent weight gradients: 4-wide and 2-wide unrolled outer product
  size_t rk = 0;
  for (; rk + 3 < N_this; rk += 4)
  {
    double* row0 = &local_rw_grads[rk * N_this];
    double* row1 = &local_rw_grads[(rk + 1) * N_this];
    double* row2 = &local_rw_grads[(rk + 2) * N_this];
    double* row3 = &local_rw_grads[(rk + 3) * N_this];

    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty())
      {
        continue;
      }
      const auto& layer_states = hidden_states[b].at(get_layer_index());
      for (size_t t = 1; t < T; ++t)
      {
        const double* g_t = &packed_grads[t * N_this];
        const double* h_prev = layer_states[t - 1].get_hidden_state_values().data();
        simd::mul_add_four_scalars(h_prev[rk], h_prev[rk + 1], h_prev[rk + 2], h_prev[rk + 3], g_t, row0, row1, row2, row3, N_this);
      }
    }
  }
  for (; rk + 1 < N_this; rk += 2)
  {
    double* row0 = &local_rw_grads[rk * N_this];
    double* row1 = &local_rw_grads[(rk + 1) * N_this];

    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty())
      {
        continue;
      }
      const auto& layer_states = hidden_states[b].at(get_layer_index());
      for (size_t t = 1; t < T; ++t)
      {
        const double* g_t = &packed_grads[t * N_this];
        const double* h_prev = layer_states[t - 1].get_hidden_state_values().data();
        simd::mul_add_two_scalars(h_prev[rk], h_prev[rk + 1], g_t, row0, row1, N_this);
      }
    }
  }
  for (; rk < N_this; ++rk)
  {
    double* row0 = &local_rw_grads[rk * N_this];
    for (size_t b = start; b < end; ++b)
    {
      const auto& packed_grads = batch_gradients_and_outputs[b].get_rnn_gate_gradients(get_layer_index());
      if (packed_grads.empty())
      {
        continue;
      }
      const auto& layer_states = hidden_states[b].at(get_layer_index());
      for (size_t t = 1; t < T; ++t)
      {
        const double* g_t = &packed_grads[t * N_this];
        const double* h_prev = layer_states[t - 1].get_hidden_state_values().data();
        simd::mul_add(h_prev[rk], g_t, row0, N_this);
      }
    }
  }
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

  const auto num_threads = get_number_of_threads();
  const unsigned int max_layer_threads = std::min(num_threads, 4U);
  const unsigned int active_threads = (num_threads > 1) ? std::max(1U, std::min(max_layer_threads, static_cast<unsigned int>((batch_size * T * N_this * (N_prev + N_this)) / 100000))) : 1;

  const bool use_multithreading = (active_threads > 1);
  if (!use_multithreading)
  {
    zero_gradients();
    calculate_and_store_gradients_chunk(0, batch_size, batch_gradients_and_outputs, hidden_states, prev_layer_index, N_prev, N_this, T, _w_grads, _rw_grads, _b_grads);
  }
  else
  {
    if (_thread_grad_accumulators.size() < active_threads)
    {
      _thread_grad_accumulators.resize(active_threads);
    }

    for (unsigned int t = 0; t < active_threads; ++t)
    {
      _thread_grad_accumulators[t].w_grads.resize(_w_grads.size());
      std::fill_n(_thread_grad_accumulators[t].w_grads.data(), _w_grads.size(), 0.0);
      _thread_grad_accumulators[t].rw_grads.resize(_rw_grads.size());
      std::fill_n(_thread_grad_accumulators[t].rw_grads.data(), _rw_grads.size(), 0.0);
      _thread_grad_accumulators[t].b_grads.resize(has_bias() ? N_this : 0);
      std::fill_n(_thread_grad_accumulators[t].b_grads.data(), has_bias() ? N_this : 0, 0.0);
    }

    size_t start = 0;
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      size_t size = (batch_size / active_threads) + (t < (batch_size % active_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue(ElmanGradCalcTask{
          *this,
          start,
          end,
          batch_gradients_and_outputs,
          hidden_states,
          prev_layer_index,
          N_prev,
          N_this,
          T,
          _thread_grad_accumulators[t].w_grads,
          _thread_grad_accumulators[t].rw_grads,
          _thread_grad_accumulators[t].b_grads
        });
      }
      start = end;
    }
    _task_queue_pool->get();

    // Merge results
    zero_gradients();
    for (unsigned int t = 0; t < active_threads; ++t)
    {
      simd::add_vectors(_thread_grad_accumulators[t].w_grads.data(), _w_grads.data(), _w_grads.size());
      simd::add_vectors(_thread_grad_accumulators[t].rw_grads.data(), _rw_grads.data(), _rw_grads.size());
      if (has_bias())
      {
        simd::add_vectors(_thread_grad_accumulators[t].b_grads.data(), _b_grads.data(), _b_grads.size());
      }
    }
  }

  const double inv_batch = 1.0 / static_cast<double>(batch_size);
  simd::scale_vector(_w_grads.data(), inv_batch, _w_grads.size());
  simd::scale_vector(_rw_grads.data(), inv_batch, _rw_grads.size());
  if (has_bias())
  {
    simd::scale_vector(_b_grads.data(), inv_batch, _b_grads.size());
  }
}

void ElmanRNNLayer::zero_gradients()
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  Layer::zero_gradients();
  std::fill(_rw_grads.begin(), _rw_grads.end(), 0.0);
}

void ElmanRNNLayer::set_w_values(const std::vector<double>& v)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  Layer::set_w_values(v);
  cache_recurrent_weights();
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

void ElmanRNNLayer::set_number_of_threads(int number_of_threads)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  Layer::set_number_of_threads(number_of_threads);
  allocate_workspace();
}

void ElmanRNNLayer::allocate_workspace()
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const auto num_threads = std::max(1U, get_number_of_threads());
  allocate_workspace(num_threads);
}

void ElmanRNNLayer::allocate_workspace(unsigned int num_threads)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const unsigned count = std::max(1U, num_threads);
  if (_thread_workspaces.size() < count)
  {
    _thread_workspaces.resize(count);
  }
  for (size_t thread_idx = 0; thread_idx < count; ++thread_idx)
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
  if (thread_idx >= _thread_workspaces.size() || !_thread_workspaces[thread_idx])
  {
    const_cast<ElmanRNNLayer*>(this)->allocate_workspace(static_cast<unsigned int>(thread_idx + 1));
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

void ElmanRNNLayer::cache_recurrent_weights()
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t n = get_number_neurons();
  const size_t n_prev = get_number_input_neurons();
  if (n == 0)
  {
    return;
  }
  _rw_values_T.resize(n * n);
  simd::transpose(_rw_values.data(), _rw_values_T.data(), n, n);

  if (n_prev > 0)
  {
    _w_values_T.resize(n * n_prev);
    const auto& w_vals = get_w_values();
    simd::transpose(w_vals.data(), _w_values_T.data(), n_prev, n);
  }
}

} // namespace myoddweb::nn
