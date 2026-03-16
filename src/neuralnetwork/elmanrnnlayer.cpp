#include "./libraries/instrumentor.h"
#include "elmanrnnlayer.h"
#include "logger.h"

constexpr bool _has_bias_neuron = true;

ElmanRNNLayer::ElmanRNNLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer, 
  unsigned num_neurons_in_this_layer, 
  double weight_decay,
  LayerType layer_type, 
  const activation& activation_method,
  const OptimiserType& optimiser_type, 
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads
  ) noexcept :
  Layer(
    layer_index, 
    layer_type, 
    activation_method, 
    optimiser_type, 
    residual_layer_number,
    num_neurons_in_previous_layer, 
    num_neurons_in_this_layer,
    create_neurons(dropout_rate, num_neurons_in_this_layer),
    _has_bias_neuron,
    weight_decay,
    residual_projector,
    number_of_threads
  )
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  initialize_recurrent_weights(weight_decay);
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
  const LayerType layer_type,
  const activation activation,
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
  const std::vector<double>& rw_values,
  const std::vector<double>& rw_grads,
  const std::vector<double>& rw_velocities,
  const std::vector<double>& rw_m1,
  const std::vector<double>& rw_m2,
  const std::vector<long long>& rw_timesteps,
  const std::vector<double>& rw_decays,
  const ResidualProjector* residual_projector,
  int number_of_threads
) noexcept :
  Layer(
    layer_index,
    layer_type,
    activation,
    optimiser_type,
    residual_layer_number,
    number_input_neurons,
    number_output_neurons,
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
    number_of_threads
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
  if(this != &src)
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
  if(this != &src)
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

  auto initial_weights = get_activation().weight_initialization(num_neurons, num_neurons);
  _rw_values.resize(num_weights);
  for (size_t i = 0; i < num_neurons; ++i) {
    for (size_t o = 0; o < num_neurons; ++o) {
        _rw_values[i * num_neurons + o] = initial_weights[o];
    }
  }

  _rw_grads.assign(num_weights, 0.0);
  _rw_velocities.assign(num_weights, 0.0);
  _rw_m1.assign(num_weights, 0.0);
  _rw_m2.assign(num_weights, 0.0);
  _rw_timesteps.assign(num_weights, 0);
  _rw_decays.assign(num_weights, weight_decay);
}

bool ElmanRNNLayer::has_bias() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _has_bias_neuron;
}

void ElmanRNNLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  if (batch_size == 0) return;

  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();

  // 1. Determine time steps and flatten inputs [BatchSize x T x N_prev]
  std::vector<double> sample_inputs = batch_gradients_and_outputs[0].get_rnn_outputs(previous_layer.get_layer_index());
  if (sample_inputs.empty())
  {
      sample_inputs = batch_gradients_and_outputs[0].get_outputs(previous_layer.get_layer_index());
  }
  const size_t num_time_steps = N_prev > 0 ? sample_inputs.size() / N_prev : 0;
  if (num_time_steps == 0) return;

  _flattened_batch_inputs_buffer.resize(batch_size * num_time_steps * N_prev);
  for (size_t b = 0; b < batch_size; ++b)
  {
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(previous_layer.get_layer_index());
      const auto& std_in = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
      const double* src = !rnn_in.empty() ? rnn_in.data() : std_in.data();
      const size_t src_size = !rnn_in.empty() ? rnn_in.size() : std_in.size();

      if (src_size == num_time_steps * N_prev)
      {
          std::copy(src, src + src_size, _flattened_batch_inputs_buffer.begin() + b * num_time_steps * N_prev);
      }
      else if (src_size == N_prev)
      {
          // Broadcast single input across all time steps
          for (size_t t = 0; t < num_time_steps; ++t)
          {
              std::copy(src, src + N_prev, _flattened_batch_inputs_buffer.begin() + (b * num_time_steps + t) * N_prev);
          }
      }
  }

  // 2. Output buffer
  _batch_output_sequences_buffer.assign(batch_size * num_time_steps * N_this, 0.0);

  // 3. Process time steps sequentially (due to recurrent dependency)
  auto run_forward_pass = [&](size_t start, size_t end)
  {
    std::vector<double> pre_activation_sums(N_this);
    std::vector<double> current_hidden_state_values(N_this);
    const double* W = get_w_values().data();
    const double* U = _rw_values.data();

    for (size_t b = start; b < end; ++b)
    {
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        // a. Initialize with bias
        if (has_bias())
        {
          const double* B = get_b_values().data();
          std::copy(B, B + N_this, pre_activation_sums.begin());
        }
        else
        {
          std::fill(pre_activation_sums.begin(), pre_activation_sums.end(), 0.0);
        }

        // b. Input-to-Hidden (W * x_t) - Tiled
        const double* x_t = &_flattened_batch_inputs_buffer[(b * num_time_steps + t) * N_prev];
        constexpr size_t BLOCK_SIZE = 64;
        for (size_t i0 = 0; i0 < N_prev; i0 += BLOCK_SIZE)
        {
            size_t i_limit = std::min(i0 + BLOCK_SIZE, N_prev);
            for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
            {
                size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);
                for (size_t i = i0; i < i_limit; ++i)
                {
                    const double val = x_t[i];
                    if (val == 0.0) continue;
                    const double* w_row = &W[i * N_this];
                    for (size_t j = j0; j < j_limit; ++j) pre_activation_sums[j] += val * w_row[j];
                }
            }
        }

        // c. Hidden-to-Hidden (U * h_{t-1}) - Tiled
        if (t > 0)
        {
          const auto& prev_h_vec = batch_hidden_states[b].at(get_layer_index())[t - 1].get_hidden_state_values();
          const double* h_prev = prev_h_vec.data();
          for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
          {
              size_t i_limit = std::min(i0 + BLOCK_SIZE, N_this);
              for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
              {
                  size_t j_limit = std::min(j0 + BLOCK_SIZE, N_this);
                  for (size_t i = i0; i < i_limit; ++i)
                  {
                      const double val = h_prev[i];
                      if (val == 0.0) continue;
                      const double* u_row = &U[i * N_this];
                      for (size_t j = j0; j < j_limit; ++j) pre_activation_sums[j] += val * u_row[j];
                  }
              }
          }
        }

        // d. Residuals
        if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
        {
            for (size_t j = 0; j < N_this; ++j) pre_activation_sums[j] += batch_residual_output_values[b][j];
        }

        // e. Activation and Store
        get_activation().activate(pre_activation_sums.data(), pre_activation_sums.data() + N_this);

        for (size_t j = 0; j < N_this; ++j)
        {
          const auto& neuron = get_neuron((unsigned)j);
          double output = pre_activation_sums[j];
          if (is_training && neuron.is_dropout())
          {
            if (neuron.must_randomly_drop()) output = 0.0;
            else output /= (1.0 - neuron.get_dropout_rate());
          }
          current_hidden_state_values[j] = output;
          _batch_output_sequences_buffer[(b * num_time_steps + t) * N_this + j] = output;
        }

        batch_hidden_states[b].at(get_layer_index())[t].set_pre_activation_sums(pre_activation_sums);
        batch_hidden_states[b].at(get_layer_index())[t].set_hidden_state_values(current_hidden_state_values);
      }
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_forward_pass(0, batch_size);
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
        _task_queue_pool->enqueue([=]()
          {
            run_forward_pass(start, end);
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* seq_ptr = &_batch_output_sequences_buffer[b * num_time_steps * N_this];
    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), std::vector<double>(seq_ptr, seq_ptr + num_time_steps * N_this));
    
    const double* last_ptr = &_batch_output_sequences_buffer[(b * num_time_steps + num_time_steps - 1) * N_this];
    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), std::vector<double>(last_ptr, last_ptr + N_this));
  }
}

void ElmanRNNLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_total = get_number_neurons();

  // --- Neuron-Parallel Implementation (Small Batch) ---
  auto run_output_gradients_neuron_parallel = [&](size_t start, size_t end)
  {
    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& given_outputs = batch_gradients_and_outputs[b].get_outputs(get_layer_index());
      const auto& target_outputs = *(target_outputs_begin + b);
      double* gradients_ptr = batch_gradients_and_outputs[b].get_gradients_raw(get_layer_index());
      
      const auto& last_hs = batch_hidden_states[b].at(get_layer_index()).back();

      // Determine correct output index (last time step)
      size_t output_offset = 0;
      if (given_outputs.size() >= N_total && N_total > 0)
      {
        const size_t num_time_steps = given_outputs.size() / N_total;
        output_offset = (num_time_steps - 1) * N_total;
      }
      
      const auto is_not_using_activation_derivative = Layer::is_not_using_activation_derivative(error_calculation_type);

      std::vector<double> deltas(N_total, 0.0);
      std::vector<double> target_slice(target_outputs.begin(), target_outputs.begin() + std::min(target_outputs.size(), (size_t)N_total));
      if (target_slice.size() < N_total) target_slice.resize(N_total, 0.0);
      std::vector<double> given_slice(given_outputs.begin() + output_offset, given_outputs.begin() + output_offset + N_total);

      calculate_error_deltas(deltas, target_slice, given_slice, error_calculation_type);

      for (size_t j = start; j < end; ++j)
      {
        if (is_not_using_activation_derivative)
        {
          gradients_ptr[j] = deltas[j];
        }
        else
        {
          const double deriv = get_activation().activate_derivative(last_hs.get_pre_activation_sum_at_neuron((unsigned)j));
          gradients_ptr[j] = deltas[j] * deriv;
        }
      }
    }
  };

  // --- Batch-Parallel Implementation (Large Batch) ---
  auto run_output_gradients_batch_parallel = [&](size_t start, size_t end)
  {
    std::vector<double> gradients(N_total, 0.0);
    std::vector<double> deltas(N_total, 0.0);

    for (size_t b = start; b < end; b++)
    {
      const auto& given_outputs = batch_gradients_and_outputs[b].get_outputs(get_layer_index());
      const auto& target_outputs = *(target_outputs_begin + b);
      
      std::fill(gradients.begin(), gradients.end(), 0.0);

      if (given_outputs.size() >= N_total && N_total > 0)
      {
        const size_t num_time_steps = given_outputs.size() / N_total;
        const auto& last_hs = batch_hidden_states[b].at(get_layer_index()).back();
        const auto is_not_using_activation_derivative = Layer::is_not_using_activation_derivative(error_calculation_type);

        std::vector<double> target_slice(target_outputs.begin(), target_outputs.begin() + std::min(target_outputs.size(), (size_t)N_total));
        if (target_slice.size() < N_total) target_slice.resize(N_total, 0.0);

        if (given_outputs.size() == N_total)
        {
          calculate_error_deltas(deltas, target_slice, given_outputs, error_calculation_type);
        }
        else
        {
          // Sequence case
          const size_t output_offset = (num_time_steps - 1) * N_total;
          std::vector<double> given_slice(given_outputs.begin() + output_offset, given_outputs.begin() + output_offset + N_total);
          calculate_error_deltas(deltas, target_slice, given_slice, error_calculation_type);
        }

        if (is_not_using_activation_derivative)
        {
          for (unsigned j = 0; j < N_total; ++j)
          {
            gradients[j] = deltas[j];
          }
        }
        else
        {
          for (unsigned j = 0; j < N_total; ++j)
          {
            double deriv = get_activation().activate_derivative(last_hs.get_pre_activation_sum_at_neuron(j));
            gradients[j] = deltas[j] * deriv;
          }
        }
      }
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), gradients);
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads > 1 && batch_size < num_threads && N_total >= 64)
  {
    // Small batch but many neurons: Use neuron parallelism
    // Pre-allocate gradients
    for (size_t b = 0; b < batch_size; ++b) 
    {
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), std::vector<double>(N_total, 0.0));
    }
       
    size_t chunk_size = (N_total + num_threads - 1) / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = std::min(start + chunk_size, N_total);
      if (start < end)
      {
        _task_queue_pool->enqueue([=]() { run_output_gradients_neuron_parallel(start, end); });
      }
    }
    _task_queue_pool->get();
  }
  else if (num_threads <= 1)
  {
    run_output_gradients_batch_parallel(0, batch_size);
  }
  else
  {
    // Batch parallelism
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end)
      {
        _task_queue_pool->enqueue([=]() {
          run_output_gradients_batch_parallel(start, end);
        });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void ElmanRNNLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  if (batch_size == 0) return;

  const size_t N_this = get_number_neurons();
  const size_t N_next = next_layer.get_number_output_neurons();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();
  if (num_time_steps == 0 || N_this == 0) return;

  const int t_start = static_cast<int>(num_time_steps) - 1;
  int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  // 1. Flatten next-layer gradients [BatchSize x T x N_next]
  std::vector<double> flattened_next_grads(batch_size * num_time_steps * N_next, 0.0);
  for (size_t b = 0; b < batch_size; ++b)
  {
      const auto& next_grad_vec = batch_next_grad_matrix[b];
      if (next_grad_vec.size() == N_next * num_time_steps)
      {
          std::copy(next_grad_vec.begin(), next_grad_vec.end(), flattened_next_grads.begin() + b * num_time_steps * N_next);
      }
      else if (next_grad_vec.size() == N_next)
      {
          // Only last step has gradient (typical sequence-to-one)
          std::copy(next_grad_vec.begin(), next_grad_vec.end(), flattened_next_grads.begin() + (b * num_time_steps + t_start) * N_next);
      }
  }

  auto run_hidden_gradients = [&](size_t start, size_t end)
  {
    const size_t chunk_count = end - start;
    std::vector<double> chunk_rnn_grads(chunk_count * num_time_steps * N_this, 0.0);
    std::vector<double> d_next_h(chunk_count * N_this, 0.0);
    const double* W_next = next_layer.get_w_values().data();
    const double* U = _rw_values.data(); // Standard recurrent weights

    for (int t = t_start; t >= t_end; --t)
    {
      for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
      {
        size_t b = start + b_idx;
        const auto& current_h_step = batch_hidden_states[b].at(get_layer_index())[t];
        const double* g_next = &flattened_next_grads[(b * num_time_steps + t) * N_next];
        double* g_this_step = &chunk_rnn_grads[(b_idx * num_time_steps + t) * N_this];
        double* dh_accum = &d_next_h[b_idx * N_this];

        for (size_t i = 0; i < N_this; ++i)
        {
          // dh = sum(g_next * W_next[i,:]) + d_next_h[i]
          double dh = dh_accum[i];
          const double* w_next_row = &W_next[i * N_next];
          for (size_t k = 0; k < N_next; ++k) dh += g_next[k] * w_next_row[k];

          const double deriv = get_activation().activate_derivative(current_h_step.get_pre_activation_sum_at_neuron((unsigned)i));
          g_this_step[i] = dh * deriv;
        }
      }

      // Propagate backward through recurrent weights U for step t-1
      if (t > t_end)
      {
        std::vector<double> prev_dh_accum(chunk_count * N_this, 0.0);
        for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
        {
          const double* g_this_step = &chunk_rnn_grads[(b_idx * num_time_steps + t) * N_this];
          double* dh_prev = &prev_dh_accum[b_idx * N_this];
          for (size_t i = 0; i < N_this; ++i)
          {
            const double* u_row = &U[i * N_this];
            double sum = 0.0;
            for (size_t j = 0; j < N_this; ++j) sum += g_this_step[j] * u_row[j];
            dh_prev[i] = sum;
          }
        }
        d_next_h = std::move(prev_dh_accum);
      }
    }

    // Store results
    for (size_t b_idx = 0; b_idx < chunk_count; ++b_idx)
    {
      size_t b = start + b_idx;
      const double* chunk_ptr = &chunk_rnn_grads[b_idx * num_time_steps * N_this];
      std::vector<double> rnn_grad_vec(chunk_ptr, chunk_ptr + num_time_steps * N_this);
      batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), rnn_grad_vec);

      // Sum over time for standard gradient
      std::vector<double> grad_sum(N_this, 0.0);
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        for (size_t i = 0; i < N_this; ++i) grad_sum[i] += rnn_grad_vec[t * N_this + i];
      }
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), grad_sum);
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_hidden_gradients(0, batch_size);
  }
  else
  {
    size_t start = 0;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t size = (batch_size / num_threads) + (t < (batch_size % num_threads) ? 1 : 0);
      size_t end = start + size;
      if (start < end) _task_queue_pool->enqueue([=]() { run_hidden_gradients(start, end); });
      start = end;
    }
    _task_queue_pool->get();
  }
}



double ElmanRNNLayer::get_recurrent_weight_value(unsigned from_neuron, unsigned to_neuron) const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return _rw_values[from_neuron * get_number_neurons() + to_neuron];
}

Layer* ElmanRNNLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  return new ElmanRNNLayer(*this);
}

void ElmanRNNLayer::calculate_and_store_gradients(
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& hidden_states,
  const Layer& previous_layer,
  int bptt_max_ticks)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  if (batch_size == 0) return;

  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();
  const unsigned num_time_steps = (unsigned)hidden_states[0].at(get_layer_index()).size();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  const int t_end = (bptt_max_ticks > 0) ? std::max(0, t_start - bptt_max_ticks + 1) : 0;

  // 1. Reset gradients
  std::fill(this->_w_grads.begin(), this->_w_grads.end(), 0.0);
  std::fill(_rw_grads.begin(), _rw_grads.end(), 0.0);
  if (has_bias()) std::fill(this->_b_grads.begin(), this->_b_grads.end(), 0.0);

  // 2. Flatten batch inputs and rnn gradients for efficiency
  _flattened_inputs_buffer.resize(batch_size * num_time_steps * num_inputs);
  std::fill(_flattened_inputs_buffer.begin(), _flattened_inputs_buffer.end(), 0.0);
  _flattened_rnn_grads_buffer.resize(batch_size * num_time_steps * num_outputs);
  std::fill(_flattened_rnn_grads_buffer.begin(), _flattened_rnn_grads_buffer.end(), 0.0);
  _flattened_prev_h_buffer.resize(batch_size * num_time_steps * num_outputs);
  std::fill(_flattened_prev_h_buffer.begin(), _flattened_prev_h_buffer.end(), 0.0);

  for (size_t b = 0; b < batch_size; ++b)
  {
      const auto& rnn_grads = batch_gradients_and_outputs[b].get_rnn_gradients(get_layer_index());
      if (rnn_grads.size() == num_time_steps * num_outputs)
      {
          std::copy(rnn_grads.begin(), rnn_grads.end(), _flattened_rnn_grads_buffer.begin() + b * num_time_steps * num_outputs);
      }

      const auto& prev_rnn_out = batch_gradients_and_outputs[b].get_rnn_outputs(previous_layer.get_layer_index());
      const auto& prev_std_out = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
      const double* src_in = !prev_rnn_out.empty() ? prev_rnn_out.data() : prev_std_out.data();
      const size_t src_in_size = !prev_rnn_out.empty() ? prev_rnn_out.size() : prev_std_out.size();

      if (src_in_size == num_time_steps * num_inputs)
      {
          std::copy(src_in, src_in + src_in_size, _flattened_inputs_buffer.begin() + b * num_time_steps * num_inputs);
      }
      else if (src_in_size == num_inputs)
      {
          for (size_t t = 0; t < num_time_steps; ++t)
              std::copy(src_in, src_in + num_inputs, _flattened_inputs_buffer.begin() + (b * num_time_steps + t) * num_inputs);
      }

      for (size_t t = 1; t < num_time_steps; ++t)
      {
          const auto& h_prev = hidden_states[b].at(get_layer_index())[t - 1].get_hidden_state_values();
          std::copy(h_prev.begin(), h_prev.end(), _flattened_prev_h_buffer.begin() + (b * num_time_steps + t) * num_outputs);
      }
  }

  // 3. Batched Outer Product for W_grads (Input-to-Hidden)
  constexpr size_t BLOCK_SIZE = 64;
  for (size_t i0 = 0; i0 < num_inputs; i0 += BLOCK_SIZE)
  {
      size_t i_limit = std::min(i0 + BLOCK_SIZE, (size_t)num_inputs);
      for (size_t j0 = 0; j0 < num_outputs; j0 += BLOCK_SIZE)
      {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, (size_t)num_outputs);
          for (size_t b = 0; b < batch_size; ++b)
          {
              for (int t = t_start; t >= t_end; --t)
              {
                  const double* x_row = &_flattened_inputs_buffer[(b * num_time_steps + t) * num_inputs];
                  const double* g_row = &_flattened_rnn_grads_buffer[(b * num_time_steps + t) * num_outputs];
                  for (size_t i = i0; i < i_limit; ++i)
                  {
                      const double x_val = x_row[i];
                      if (x_val == 0.0) continue;
                      double* w_grad_row = &this->_w_grads[i * num_outputs];
                      for (size_t j = j0; j < j_limit; ++j) w_grad_row[j] += x_val * g_row[j];
                  }
              }
          }
      }
  }

  // 4. Batched Outer Product for RW_grads (Recurrent)
  for (size_t i0 = 0; i0 < num_outputs; i0 += BLOCK_SIZE)
  {
      size_t i_limit = std::min(i0 + BLOCK_SIZE, (size_t)num_outputs);
      for (size_t j0 = 0; j0 < num_outputs; j0 += BLOCK_SIZE)
      {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, (size_t)num_outputs);
          for (size_t b = 0; b < batch_size; ++b)
          {
              for (int t = t_start; t >= std::max(1, t_end); --t)
              {
                  const double* h_prev_row = &_flattened_prev_h_buffer[(b * num_time_steps + t) * num_outputs];
                  const double* g_row = &_flattened_rnn_grads_buffer[(b * num_time_steps + t) * num_outputs];
                  for (size_t i = i0; i < i_limit; ++i)
                  {
                      const double h_val = h_prev_row[i];
                      if (h_val == 0.0) continue;
                      double* rw_grad_row = &_rw_grads[i * num_outputs];
                      for (size_t j = j0; j < j_limit; ++j) rw_grad_row[j] += h_val * g_row[j];
                  }
              }
          }
      }
  }

  // 5. Bias Gradients
  if (has_bias())
  {
      for (size_t b = 0; b < batch_size; ++b)
      {
          for (int t = t_start; t >= t_end; --t)
          {
              const double* g_row = &_flattened_rnn_grads_buffer[(b * num_time_steps + t) * num_outputs];
              for (unsigned j = 0; j < num_outputs; ++j) this->_b_grads[j] += g_row[j];
          }
      }
  }

  // 6. Normalization
  const int active_ticks = t_start - t_end + 1;
  const double denom = static_cast<double>(batch_size) * active_ticks;
  const double inv_denom = 1.0 / (denom > 0 ? denom : 1.0);
  for (double& grad : this->_w_grads) grad *= inv_denom;
  if (has_bias()) for (double& grad : this->_b_grads) grad *= inv_denom;

  const double denom_rec = static_cast<double>(batch_size) * (active_ticks > 1 ? active_ticks - 1 : 1.0);
  const double inv_denom_rec = 1.0 / denom_rec;
  for (double& grad : _rw_grads) grad *= inv_denom_rec;
}

double ElmanRNNLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  double norm_sq = 0.0;
  for (const double grad : this->_w_grads)
  {
    norm_sq += grad * grad;
  }
  for (const double grad : _rw_grads)
  {
    norm_sq += grad * grad;
  }
  if (has_bias())
  {
      for (const double grad : this->_b_grads) norm_sq += grad * grad;
  }
  return norm_sq;
}

void ElmanRNNLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("ElmanRNNLayer");
  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();

  for (unsigned j = 0; j < num_outputs; ++j)
  {
    // Apply input-to-hidden weights
    for (unsigned i = 0; i < num_inputs; ++i)
    {
      unsigned weight_index = i * num_outputs + j;
      apply_weight_gradient(this->_w_grads[weight_index], learning_rate, false, weight_index, clipping_scale);
    }

    // Apply bias weights
    if (has_bias())
    {
      apply_weight_gradient(this->_b_grads[j], learning_rate, true, j, clipping_scale);
    }
        
    // Apply recurrent weights
    for (unsigned k = 0; k < num_outputs; ++k)
    {
      const unsigned idx = k * num_outputs + j;
      apply_update_to_weight(_rw_values, _rw_grads, _rw_velocities, _rw_m1, _rw_m2, _rw_timesteps, _rw_decays, idx, _rw_grads[idx], learning_rate, clipping_scale);
    }
  }
}