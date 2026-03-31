#include "./libraries/instrumentor.h"
#include "fflayer.h"
#include "logger.h"
#include <numeric>

constexpr bool _has_bias_neuron = true;

FFLayer::FFLayer(
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
) :
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
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(const FFLayer& src) noexcept :
  Layer(src)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(
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
  number_of_threads)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(FFLayer&& src) noexcept :
  Layer(std::move(src))
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer& FFLayer::operator=(const FFLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if(this != &src)
  {
  Layer::operator=(src);
  }
  return *this;
}

FFLayer& FFLayer::operator=(FFLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if(this != &src)
  {
  Layer::operator=(std::move(src));
  }
  return *this;
}

FFLayer::~FFLayer()
{
}

bool FFLayer::has_bias() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _has_bias_neuron;
}

void FFLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const auto N_prev = get_number_input_neurons();
  const auto N_this = get_number_neurons();

  if (batch_size == 0) return;

  // 1. Flatten inputs for the whole batch into a contiguous matrix [BatchSize x N_prev]
  _batch_inputs_buffer.resize(batch_size * N_prev);
  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* src = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index() - 1);
    std::copy(src, src + N_prev, _batch_inputs_buffer.begin() + b * N_prev);
  }

  _batch_pre_activation_sums_buffer.assign(batch_size * N_this, 0.0);

  // 2. Initialize with bias values
  if (has_bias())
  {
  for (size_t b = 0; b < batch_size; b++)
  {
    double* dest = &_batch_pre_activation_sums_buffer[b * N_this];
    for (size_t j = 0; j < N_this; j++)
    {
    dest[j] = get_bias_value((unsigned)j);
    }
  }
  }

  // 3. Batched Matrix-Matrix multiplication (GEMM)
  // Y = X * W where X is [BatchSize x N_prev] and W is [N_prev x N_this]
  auto run_gemm = [&](size_t b_start, size_t b_end)
  {
    constexpr size_t BLOCK_SIZE = 64;
    const double* W = get_w_values().data();

    for (size_t i0 = 0; i0 < N_prev; i0 += BLOCK_SIZE)
    {
      size_t i_limit = std::min(i0 + BLOCK_SIZE, (size_t)N_prev);
      for (size_t b0 = b_start; b0 < b_end; b0 += BLOCK_SIZE)
      {
        size_t b_limit = std::min(b0 + BLOCK_SIZE, b_end);
        for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
        {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, (size_t)N_this);

          for (size_t b = b0; b < b_limit; ++b)
          {
            for (size_t i = i0; i < i_limit; ++i)
            {
              const double x_val = _batch_inputs_buffer[b * N_prev + i];
              if (x_val == 0.0) continue;

              double* y_row = &_batch_pre_activation_sums_buffer[b * N_this];
              const double* w_row = &W[i * N_this];

              for (size_t j = j0; j < j_limit; ++j)
              {
                y_row[j] += x_val * w_row[j];
              }
            }
          }
        }
      }
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_gemm(0, batch_size);
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
      _task_queue_pool->enqueue([=]() { run_gemm(start, end); });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 4. Residuals, Activation and Dropout
  auto run_post_gemm = [&](size_t start, size_t end)
  {
    std::vector<double> output_row(N_this);
    std::vector<double> temp_pre_activations;
    if (!batch_hidden_states.empty())
    {
      temp_pre_activations.resize(N_this);
    }

    for (size_t b = start; b < end; b++)
    {
      double* current_pre_act = &_batch_pre_activation_sums_buffer[b * N_this];

      // Residuals
      if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
      {
        for (size_t j = 0; j < N_this; j++)
        {
          current_pre_act[j] += batch_residual_output_values[b][j];
        }
      }

      // Activation
      get_activation().activate(current_pre_act, current_pre_act + N_this);

      const auto output_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
      for (size_t j = 0; j < N_this; j++)
      {
        const auto& neuron = get_neuron((unsigned)j);
        double output = current_pre_act[j];

        if (is_training && neuron.is_dropout())
        {
          if (neuron.must_randomly_drop())
          {
            output = 0.0;
          }
          else
          {
            output /= (1.0 - neuron.get_dropout_rate());
          }
        }
        output_row[j] = output;
        output_ptr[j] = output;
      }

      if (!batch_hidden_states.empty())
      {
        for (size_t j = 0; j < N_this; ++j)
        {
          temp_pre_activations[j] = _batch_pre_activation_sums_buffer[b * N_this + j];
        }
        batch_hidden_states[b].at(get_layer_index())[0].set_pre_activation_sums(temp_pre_activations);
        batch_hidden_states[b].at(get_layer_index())[0].set_hidden_state_values(output_row);
      }
    }
  };

  if (num_threads <= 1)
  {
    run_post_gemm(0, batch_size);
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
        _task_queue_pool->enqueue([=]() { run_post_gemm(start, end); });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void FFLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  const OutputLayerDetails& output_layer_detail) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  Logger::panic("FFLayer: Trying to calculate output gradient with a non output layer!");
}

void FFLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const auto N_this = get_number_neurons();
  const auto N_next = next_layer.get_number_neurons();

  if (batch_size == 0) return;

  // 1. Flatten next-layer gradients for the whole batch [BatchSize x N_next]
  _flattened_next_grads_buffer.resize(batch_size * N_next);
  for (size_t b = 0; b < batch_size; ++b)
  {
    std::copy(batch_next_grad_matrix[b].begin(), batch_next_grad_matrix[b].end(), _flattened_next_grads_buffer.begin() + b * N_next);
  }

  // 2. Transposed Matrix-Matrix multiplication (G_this = G_next * W_next^T)
  // G_this is [BatchSize x N_this], G_next is [BatchSize x N_next], W_next is [N_this x N_next]
  _flattened_this_grads_buffer.assign(batch_size * N_this, 0.0);

  auto run_gemm_backward = [&](size_t b_start, size_t b_end)
  {
    constexpr size_t BLOCK_SIZE = 64;
    // W_next is stored as [N_this x N_next]
    const double* W_next = next_layer.get_w_values().data();

    for (size_t j0 = 0; j0 < N_next; j0 += BLOCK_SIZE)
    {
      size_t j_limit = std::min(j0 + BLOCK_SIZE, (size_t)N_next);
      for (size_t b0 = b_start; b0 < b_end; b0 += BLOCK_SIZE)
      {
        size_t b_limit = std::min(b0 + BLOCK_SIZE, b_end);
        for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, (size_t)N_this);
        for (size_t b = b0; b < b_limit; ++b)
        {
            const double* g_next_row = &_flattened_next_grads_buffer[b * N_next];
            double* g_this_row = &_flattened_this_grads_buffer[b * N_this];

            for (size_t i = i0; i < i_limit; ++i)
            {
              const double* w_next_row = &W_next[i * N_next];
              double sum = 0.0;
              for (size_t j = j0; j < j_limit; ++j)
              {
                sum += g_next_row[j] * w_next_row[j];
              }
              g_this_row[i] += sum;
            }
          }
        }
      }
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_gemm_backward(0, batch_size);
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
        _task_queue_pool->enqueue([=]() { run_gemm_backward(start, end); });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 3. Apply activation derivative and store results
  auto run_post_gemm_backward = [&](size_t start, size_t end)
  {
    for (size_t b = start; b < end; b++)
    {
      const auto& current_hidden_state = batch_hidden_states[b].at(get_layer_index())[0];
      double* grad_ptr = batch_gradients_and_outputs[b].get_gradients_raw(get_layer_index());
      const double* g_this_row = &_flattened_this_grads_buffer[b * N_this];

      for (size_t i = 0; i < N_this; i++)
      {
        double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron((unsigned)i));
        grad_ptr[i] = g_this_row[i] * deriv;
      }
    }
  };

  if (num_threads <= 1)
  {
    run_post_gemm_backward(0, batch_size);
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
        _task_queue_pool->enqueue([=]() { run_post_gemm_backward(start, end); });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

Layer* FFLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return new FFLayer(*this);
}

void FFLayer::calculate_and_store_gradients(
  const std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<HiddenStates>& /*hidden_states*/,
  const Layer& previous_layer,
  int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  if (batch_size == 0)
  {
  return;
  }

  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();

  // 1. Flatten inputs and gradients for the whole batch
  _batch_inputs_buffer.resize(batch_size * num_inputs);
  _batch_grads_buffer.resize(batch_size * num_outputs);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* src_in = batch_gradients_and_outputs[b].get_outputs_raw(previous_layer.get_layer_index());
    std::copy(src_in, src_in + num_inputs, _batch_inputs_buffer.begin() + b * num_inputs);

    const double* src_grad = batch_gradients_and_outputs[b].get_gradients_raw(get_layer_index());
    std::copy(src_grad, src_grad + num_outputs, _batch_grads_buffer.begin() + b * num_outputs);
  }

  // 2. Reset gradients
  std::fill(this->_w_grads.begin(), this->_w_grads.end(), 0.0);
  if(has_bias())
  {
  std::fill(this->_b_grads.begin(), this->_b_grads.end(), 0.0);
  }

  // 3. Batched Weight Gradient Calculation (W_grad = X^T * G)
  constexpr size_t BLOCK_SIZE = 64;
  for (size_t i0 = 0; i0 < num_inputs; i0 += BLOCK_SIZE)
  {
    size_t i_limit = std::min(i0 + BLOCK_SIZE, (size_t)num_inputs);
    for (size_t j0 = 0; j0 < num_outputs; j0 += BLOCK_SIZE)
    {
      size_t j_limit = std::min(j0 + BLOCK_SIZE, (size_t)num_outputs);
      for (size_t b0 = 0; b0 < batch_size; b0 += BLOCK_SIZE)
      {
        size_t b_limit = std::min(b0 + BLOCK_SIZE, batch_size);

        for (size_t i = i0; i < i_limit; ++i)
        {
          for (size_t b = b0; b < b_limit; ++b)
          {
            const double x_val = _batch_inputs_buffer[b * num_inputs + i];
            if (x_val == 0.0) continue;

            const double* g_row = &_batch_grads_buffer[b * num_outputs];
            double* w_grad_row = &this->_w_grads[i * num_outputs];

            for (size_t j = j0; j < j_limit; ++j)
            {
              w_grad_row[j] += x_val * g_row[j];
            }
          }
        }
      }
    }
  }

  // 4. Bias Gradients (Sum of batch gradients)
  if (has_bias())
  {
    for (size_t b = 0; b < batch_size; ++b)
    {
      const double* g_row = &_batch_grads_buffer[b * num_outputs];
      for (unsigned j = 0; j < num_outputs; ++j)
      {
        this->_b_grads[j] += g_row[j];
      }
    }
  }

  // 5. Average gradients over batch
  const double inv_batch_size = 1.0 / static_cast<double>(batch_size);
  for (double& grad : this->_w_grads) grad *= inv_batch_size;
  if (has_bias())
  {
  for (double& grad : this->_b_grads)
  {
    grad *= inv_batch_size;
  }
  }
}

double FFLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  double norm_sq = 0.0;
  for (const double grad : this->_w_grads) norm_sq += grad * grad;
  if (has_bias())
  {
  for (const double grad : this->_b_grads)
  {
    norm_sq += grad * grad;
  }
  }
  return norm_sq;
}

void FFLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();

  for (unsigned j = 0; j < num_outputs; ++j)
  {
  for (unsigned i = 0; i < num_inputs; ++i)
  {
    unsigned weight_index = i * num_outputs + j;
    apply_weight_gradient(this->_w_grads[weight_index], learning_rate, false, weight_index, clipping_scale);
  }

  if (has_bias())
  {
    apply_weight_gradient(this->_b_grads[j], learning_rate, true, j, clipping_scale);
  }
  }
}