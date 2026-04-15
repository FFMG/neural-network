#include "./libraries/instrumentor.h"
#include "fflayer.h"
#include "logger.h"
#include <numeric>

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
  int number_of_threads,
  bool has_bias
) :
  FFLayer(
  layer_index,
  num_neurons_in_previous_layer,
  num_neurons_in_this_layer,
  std::vector<double>(static_cast<size_t>(num_neurons_in_previous_layer) * num_neurons_in_this_layer, weight_decay),
  layer_type,
  activation_method,
  optimiser_type,
  residual_layer_number,
  dropout_rate,
  residual_projector,
  number_of_threads,
  has_bias
  )
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  const std::vector<double>& weight_decays,
  LayerType layer_type,
  const activation& activation_method,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector,
  int number_of_threads,
  bool has_bias
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
  has_bias,
  weight_decays,
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
  const activation& activation_method,
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
  activation_method,
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
  const auto N_prev = get_number_input_neurons();
  const auto N_this = get_number_neurons();

  if (batch_size == 0)
  {
    return;
  }

  // 1. Flatten inputs for the whole batch
  std::vector<double> batch_inputs_buffer(batch_size * N_prev);
  const unsigned prev_layer_index = previous_layer.get_layer_index();
  for (size_t b = 0; b < batch_size; ++b)
  {
    const std::vector<double> src_vec = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    if (src_vec.size() != N_prev)
    {
      Logger::panic("FFLayer #", get_layer_index(), " input size mismatch! Expected ", N_prev, " but got ", src_vec.size(), " from layer #", prev_layer_index, " at batch sample ", b);
    }
    std::copy(src_vec.begin(), src_vec.end(), batch_inputs_buffer.begin() + b * N_prev);
  }

  std::vector<double> batch_pre_activation_sums_buffer(batch_size * N_this, 0.0);
  
  // 2. Initialize with bias values
  if (has_bias())
  {
    for (size_t b = 0; b < batch_size; b++)
    {
      double* dest = &batch_pre_activation_sums_buffer[b * N_this];
      for (size_t j = 0; j < N_this; j++)
      {
        dest[j] = get_bias_value((unsigned)j);
      }
    }
  }

  // 3. Batched Matrix-Matrix multiplication (GEMM)
  // Y = X * W where X is [BatchSize x N_prev] and W is [N_prev x N_this]
  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_gemm(0, batch_size, N_prev, N_this, batch_inputs_buffer, batch_pre_activation_sums_buffer);
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
        _task_queue_pool->enqueue([start, end, N_prev, N_this, &batch_inputs_buffer, &batch_pre_activation_sums_buffer, this]()
          { 
            run_gemm(start, end, N_prev, N_this, batch_inputs_buffer, batch_pre_activation_sums_buffer); 
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 4. Residuals, Activation and Dropout
  if (num_threads <= 1)
  {
    run_post_gemm(0, batch_size, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, batch_inputs_buffer, batch_pre_activation_sums_buffer, is_training);
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
        _task_queue_pool->enqueue([start, end, N_this, &batch_gradients_and_outputs, &batch_residual_output_values, &batch_hidden_states, &batch_inputs_buffer, &batch_pre_activation_sums_buffer, is_training, this]()
          { 
            run_post_gemm(start, end, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, batch_inputs_buffer, batch_pre_activation_sums_buffer, is_training); 
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
          // Weight range debug (only for first batch and first thread)
          if (b == 0 && i0 == 0 && b_start == 0) 
          {
            double min_w = W[0], max_w = W[0];
            for (size_t k = 0; k < N_prev * N_this; ++k) 
            {
              min_w = std::min(min_w, W[k]);
              max_w = std::max(max_w, W[k]);
            }
            Logger::trace([=]
              {
                return Logger::factory("DEBUG: [GEMM] Weight Range: [", min_w, ", ", max_w, "]");
              });
          }

          for (size_t i = i0; i < i_limit; ++i)
          {
            const double x_val = batch_inputs_buffer[b * N_prev + i];
            if (x_val == 0.0) continue;

            double* y_row = &batch_pre_activation_sums_buffer[b * N_this];
            const double* w_row = &W[i * N_this];

            for (size_t j = j0; j < j_limit; ++j)
            {
              y_row[j] += x_val * w_row[j];
            }
          }
          if (b < 2 && i0 == 0) // Log first two batch samples' state
          {
            Logger::trace([=]
              {
                return Logger::factory("DEBUG: [GEMM] b=", b, " pre_act[0]=", batch_pre_activation_sums_buffer[b * N_this]);
              });
          }
        }
      }
    }
  }
}

void FFLayer::run_post_gemm(
  size_t start,
  size_t end,
  size_t N_this,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  const std::vector<double>& /*batch_inputs_buffer*/,
  std::vector<double>& batch_pre_activation_sums_buffer,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  std::vector<double> output_row(N_this);
  std::vector<double> temp_pre_activations;
  if (!batch_hidden_states.empty())
  {
    temp_pre_activations.resize(N_this);
  }

  for (size_t b = start; b < end; b++)
  {
    double* current_pre_act = &batch_pre_activation_sums_buffer[b * N_this];

    // Residuals
    if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
    {
      for (size_t j = 0; j < N_this; j++)
      {
        current_pre_act[j] += batch_residual_output_values[b][j];
      }
    }

    // Activation
    const auto output_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    for (size_t j = 0; j < N_this; j++)
    {
      const auto& neuron = get_neuron((unsigned)j);
      double output = get_activation().activate(current_pre_act[j]);

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
        temp_pre_activations[j] = batch_pre_activation_sums_buffer[b * N_this + j];
      }
      batch_hidden_states[b].at(get_layer_index())[0].set_pre_activation_sums(temp_pre_activations);
      batch_hidden_states[b].at(get_layer_index())[0].set_hidden_state_values(output_row);
    }
  }
}

void FFLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
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

  // 1. Flatten next-layer gradients for the whole batch [BatchSize x N_next]
  std::vector<double> flattened_next_grads_buffer(batch_size * N_next);
  for (size_t b = 0; b < batch_size; ++b)
  {
    std::copy(batch_next_grad_matrix[b].begin(), batch_next_grad_matrix[b].end(), flattened_next_grads_buffer.begin() + b * N_next);
  }

  // 2. Transposed Matrix-Matrix multiplication (G_this = G_next * W_next^T)
  // G_this is [BatchSize x N_this], G_next is [BatchSize x N_next], W_next is [N_this x N_next]
  std::vector<double> flattened_this_grads_buffer(batch_size * N_this, 0.0);

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
            const double* g_next_row = &flattened_next_grads_buffer[b * N_next];
            double* g_this_row = &flattened_this_grads_buffer[b * N_this];

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
        _task_queue_pool->enqueue([start, end, &run_gemm_backward]() {
          run_gemm_backward(start, end); 
          }
        );
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
      const double* g_this_row = &flattened_this_grads_buffer[b * N_this];

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
        _task_queue_pool->enqueue([start, end, &run_post_gemm_backward]()
          { 
            run_post_gemm_backward(start, end); 
          });
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
  size_t batch_size,
  int /*bptt_max_ticks*/)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (batch_size == 0)
  {
    return;
  }

  const unsigned num_outputs = get_number_neurons();
  const unsigned num_inputs = get_number_input_neurons();

  // 1. Flatten inputs and gradients for the whole batch
  std::vector<double> batch_inputs_buffer(batch_size * num_inputs);
  std::vector<double> batch_grads_buffer(batch_size * num_outputs);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const std::vector<double> src_in_vec = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
    std::copy(src_in_vec.begin(), src_in_vec.end(), batch_inputs_buffer.begin() + b * num_inputs);

    const std::vector<double> src_grad_vec = batch_gradients_and_outputs[b].get_gradients(get_layer_index());
    std::copy(src_grad_vec.begin(), src_grad_vec.end(), batch_grads_buffer.begin() + b * num_outputs);
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
            const double x_val = batch_inputs_buffer[b * num_inputs + i];
            if (x_val == 0.0) continue;

            const double* g_row = &batch_grads_buffer[b * num_outputs];
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
      const double* g_row = &batch_grads_buffer[b * num_outputs];
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

  double update_sum = 0.0;
  for (unsigned j = 0; j < num_outputs; ++j)
  {
    for (unsigned i = 0; i < num_inputs; ++i)
    {
      unsigned weight_index = i * num_outputs + j;
      double w_before = this->_w_values[weight_index];
      apply_weight_gradient(this->_w_grads[weight_index], learning_rate, false, weight_index, clipping_scale, _optimiser_type);
      double w_after = this->_w_values[weight_index];
      update_sum += std::abs(w_after - w_before);
    }

    if (has_bias())
    {
      double b_before = this->_b_values[j];
      apply_weight_gradient(this->_b_grads[j], learning_rate, true, j, clipping_scale, _optimiser_type);
      double b_after = this->_b_values[j];
      update_sum += std::abs(b_after - b_before);
    }
  }
}
