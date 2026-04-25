#include "./libraries/instrumentor.h"
#include "fflayer.h"
#include "logger.h"
#include <numeric>

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
}

FFLayer::FFLayer(const FFLayer& src) noexcept :
  Layer(src)
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
  if (this != &src)
  {
    Layer::operator=(src);
  }
  return *this;
}

FFLayer& FFLayer::operator=(FFLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (this != &src)
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
  if (batch_size == 0)
  {
    return;
  }

  const auto N_prev = get_number_input_neurons();
  const auto N_this = get_number_neurons();

  // 1. Flatten inputs for the whole batch
  std::vector<double> batch_inputs_buffer(batch_size * N_prev);
  const unsigned prev_layer_index = previous_layer.get_layer_index();
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto src_span = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
    if (src_span.size() != N_prev)
    {
      Logger::panic("FFLayer #", get_layer_index(), " input size mismatch! Expected ", N_prev, " but got ", src_span.size(), " from layer #", prev_layer_index, " at batch sample ", b);
    }
    std::copy(src_span.begin(), src_span.end(), batch_inputs_buffer.begin() + b * N_prev);
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
  const double* W = get_w_values().data();

  for (size_t b = b_start; b < b_end; ++b)
  {
    const double* x_row = &batch_inputs_buffer[b * N_prev];
    double* y_row = &batch_pre_activation_sums_buffer[b * N_this];

    for (size_t i = 0; i < N_prev; ++i)
    {
      const double x_val = x_row[i];
      if (x_val == 0.0)
      {
        continue;
      }

      const double* w_row = &W[i * N_this];
      for (size_t j = 0; j < N_this; ++j)
      {
        y_row[j] += x_val * w_row[j];
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

    // 1. Store pre-activation sums BEFORE activation (as activation is in-place)
    if (!batch_hidden_states.empty())
    {
      std::copy(current_pre_act, current_pre_act + N_this, temp_pre_activations.begin());
      batch_hidden_states[b].at(get_layer_index())[0].set_pre_activation_sums(temp_pre_activations);
    }

    // 2. Activation and Dropout
    const auto output_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    for (const auto& r : _layer_activation_helper.ranges())
    {
      // 1. Batch activation for the range (modifies current_pre_act in-place)
      r.activation_method.activate(current_pre_act + r.start, current_pre_act + r.end);

      // 2. Apply dropout and store
      for (size_t j = r.start; j < r.end; j++)
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
    }

    if (!batch_hidden_states.empty())
    {
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

  if (batch_size == 0)
  {
    return;
  }

  // 1. Flatten next-layer gradients for the whole batch [BatchSize x N_next]
  std::vector<double> flattened_next_grads_buffer(batch_size * N_next);
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& next_grads = batch_next_grad_matrix[b];
    if (next_grads.size() != N_next)
    {
      Logger::panic("FFLayer #", get_layer_index(), " next gradient size mismatch! Expected ", N_next, " but got ", next_grads.size(), " at batch sample ", b);
    }
    std::copy(next_grads.begin(), next_grads.end(), flattened_next_grads_buffer.begin() + b * N_next);
  }

  // 2. Transposed Matrix-Matrix multiplication (G_this = G_next * W_next^T)
  // G_this is [BatchSize x N_this], G_next is [BatchSize x N_next], W_next is [N_this x N_next]
  std::vector<double> flattened_this_grads_buffer(batch_size* N_this, 0.0);
  const double* W_next = next_layer.get_w_values().data();

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_gemm_backward(0, batch_size, N_next, N_this, W_next, flattened_next_grads_buffer, flattened_this_grads_buffer);
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
        _task_queue_pool->enqueue([start, end, N_next, N_this, W_next, &flattened_next_grads_buffer, &flattened_this_grads_buffer, this]()
        {
          run_gemm_backward(start, end, N_next, N_this, W_next, flattened_next_grads_buffer, flattened_this_grads_buffer);
        }
        );
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 3. Apply activation derivative and store results
  if (num_threads <= 1)
  {
    run_post_gemm_backward(0, batch_size, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer);
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
        _task_queue_pool->enqueue([start, end, N_this, &batch_gradients_and_outputs, &batch_hidden_states, &flattened_this_grads_buffer, this]()
        {
          run_post_gemm_backward(start, end, N_this, batch_gradients_and_outputs, batch_hidden_states, flattened_this_grads_buffer);
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
  std::vector<double> batch_grads_buffer(batch_size* num_outputs);

  const unsigned prev_layer_index = previous_layer.get_layer_index();
  const unsigned this_layer_index = get_layer_index();

  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* src_in = batch_gradients_and_outputs[b].get_outputs_raw(prev_layer_index);
    std::copy(src_in, src_in + num_inputs, batch_inputs_buffer.begin() + b * num_inputs);

    const double* src_grad = batch_gradients_and_outputs[b].get_gradients_raw(this_layer_index);
    std::copy(src_grad, src_grad + num_outputs, batch_grads_buffer.begin() + b * num_outputs);
  }

  // 2. Reset gradients
  std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
  if (has_bias())
  {
    std::fill(_b_grads.begin(), _b_grads.end(), 0.0);
  }

  // 3. Batched Weight Gradient Calculation (W_grad = X^T * G)
  for (size_t b = 0; b < batch_size; ++b)
  {
    const double* x_row = &batch_inputs_buffer[b * num_inputs];
    const double* g_row = &batch_grads_buffer[b * num_outputs];

    for (size_t i = 0; i < num_inputs; ++i)
    {
      const double x_val = x_row[i];
      if (x_val == 0.0)
      {
        continue;
      }

      double* w_grad_row = &_w_grads[i * num_outputs];
      for (size_t j = 0; j < num_outputs; ++j)
      {
        w_grad_row[j] += x_val * g_row[j];
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
        _b_grads[j] += g_row[j];
      }
    }
  }

  // 5. Average gradients over batch
  const double inv_batch_size = 1.0 / static_cast<double>(batch_size);
  for (double& grad : _w_grads)
  {
    grad *= inv_batch_size;
  }
  if (has_bias())
  {
    for (double& grad : _b_grads)
    {
      grad *= inv_batch_size;
    }
  }
}

double FFLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  double norm_sq = 0.0;
  for (const double grad : _w_grads)
  {
    norm_sq += grad * grad;
  }
  if (has_bias())
  {
    for (const double grad : _b_grads)
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
  for (unsigned neuron_number = 0; neuron_number < num_outputs; ++neuron_number)
  {
    for (unsigned i = 0; i < num_inputs; ++i)
    {
      unsigned weight_index = i * num_outputs + neuron_number;
      double w_before = _w_values[weight_index];
      apply_weight_gradient(_w_grads[weight_index], learning_rate, false, weight_index, clipping_scale, _optimiser_type, neuron_number);
      double w_after = _w_values[weight_index];
      update_sum += std::abs(w_after - w_before);
    }

    if (has_bias())
    {
      double b_before = _b_values[neuron_number];
      apply_weight_gradient(_b_grads[neuron_number], learning_rate, true, neuron_number, clipping_scale, _optimiser_type, neuron_number);
      double b_after = _b_values[neuron_number];
      update_sum += std::abs(b_after - b_before);
    }
  }

  // Clear gradients
  std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
  std::fill(_b_grads.begin(), _b_grads.end(), 0.0);
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
  constexpr size_t BLOCK_SIZE = 64;
  for (size_t j0 = 0; j0 < N_next; j0 += BLOCK_SIZE)
  {
    size_t j_limit = std::min(j0 + BLOCK_SIZE, N_next);
    for (size_t b0 = b_start; b0 < b_end; b0 += BLOCK_SIZE)
    {
      size_t b_limit = std::min(b0 + BLOCK_SIZE, b_end);
      for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, N_this);
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
  for (size_t b = start; b < end; b++)
  {
    const auto& current_hidden_state = batch_hidden_states[b].at(get_layer_index())[0];
    double* grad_ptr = batch_gradients_and_outputs[b].get_gradients_raw(get_layer_index());
    const double* g_this_row = &flattened_this_grads_buffer[b * N_this];

    for (const auto& r : _layer_activation_helper.ranges())
    {
      for (size_t i = r.start; i < r.end; i++)
      {
        double deriv = r.activation_method.activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron((unsigned)i));
        grad_ptr[i] = g_this_row[i] * deriv;
      }
    }
  }
}
