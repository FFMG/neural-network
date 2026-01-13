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

  std::vector<double> batch_pre_activation_sums(batch_size * N_this, 0.0);

  // Initialize with bias values and Matrix-Matrix multiplication
  auto run_forward_pass = [&](size_t start, size_t end) 
  {
    std::vector<double> output_row(N_this);
    std::vector<double> temp_pre_activations;
    if (!batch_hidden_states.empty())
    {
      temp_pre_activations.resize(N_this);
    }

    // Initialize with bias values
    if (has_bias())
    {
      for (size_t b = start; b < end; b++)
      {
        for (size_t j = 0; j < N_this; j++)
        {
          batch_pre_activation_sums[b * N_this + j] = get_bias_value((unsigned)j);
        }
      }
    }

    // Multiply and accumulate weights and inputs
    // Tiled matrix multiplication for cache locality
    constexpr size_t BLOCK_SIZE = 64; 

    for (size_t i0 = 0; i0 < N_prev; i0 += BLOCK_SIZE)
    {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, (size_t)N_prev);
        for (size_t b0 = start; b0 < end; b0 += BLOCK_SIZE)
        {
            size_t b_limit = std::min(b0 + BLOCK_SIZE, end);
            for (size_t j0 = 0; j0 < N_this; j0 += BLOCK_SIZE)
            {
                size_t j_limit = std::min(j0 + BLOCK_SIZE, (size_t)N_this);

                for (size_t i = i0; i < i_limit; ++i)
                {
                    for (size_t b = b0; b < b_limit; ++b)
                    {
                        const double input_val = batch_gradients_and_outputs[b].get_output(get_layer_index() - 1, (unsigned)i);
                        // Accessing raw pointer here would be faster if we exposed it for read as well, 
                        // but get_output is inline and checks bounds, optimizing access inside inner loop is key.
                        // Optimization: hoist get_output or use raw pointer if available.
                        // Since we optimized GradientsAndOutputs, let's assume we can get raw ptrs if we refactor input access.
                        // For now, keeping get_output but blocking helps.

                        if (input_val == 0.0) continue;
                        
                        double* sum_ptr = &batch_pre_activation_sums[b * N_this];
                        for (size_t j = j0; j < j_limit; ++j)
                        {
                             sum_ptr[j] += input_val * get_weight_value((unsigned)i, (unsigned)j);
                        }
                    }
                }
            }
        }
    }

    // Residuals
    if (!batch_residual_output_values.empty())
    {
      for (size_t b = start; b < end; b++)
      {
        if (batch_residual_output_values[b].size() == N_this)
        {
          for (size_t j = 0; j < N_this; j++)
          {
            batch_pre_activation_sums[b * N_this + j] += batch_residual_output_values[b][j];
          }
        }
      }
    }

    // Activation
    for (size_t b = start; b < end; b++)
    {
      const auto output_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
      
      for (size_t j = 0; j < N_this; j++)
      {
        const auto& neuron = get_neuron((unsigned)j);
        double output = get_activation().activate(batch_pre_activation_sums[b * N_this + j]);

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
        output_ptr[j] = output; // Write directly to raw output buffer
      }
      
      if(!batch_hidden_states.empty())
      {
        for (size_t j = 0; j < N_this; ++j) 
        {
          temp_pre_activations[j] = batch_pre_activation_sums[b * N_this + j];
        }
        batch_hidden_states[b].at(get_layer_index())[0].set_pre_activation_sums(temp_pre_activations);
        batch_hidden_states[b].at(get_layer_index())[0].set_hidden_state_values(output_row);
      }
      // batch_gradients_and_outputs[b].set_outputs(get_layer_index(), output_row); // Removed: Writing to raw ptr instead
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (batch_size < (num_threads * 2))
  {
    run_forward_pass(0, batch_size);
  }
  else
  {
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      _task_queue_pool->enqueue([=]()
        {
          run_forward_pass(start, end);
        });
    }
    _task_queue_pool->get();
  }
}

void FFLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t batch_size = batch_gradients_and_outputs.size();
  const size_t N_total = get_number_neurons();

  auto run_output_gradients = [&](size_t start, size_t end)
  {
    std::vector<double> gradients(N_total, 0.0);
    std::vector<double> deltas(N_total, 0.0);

    for (size_t b = start; b < end; b++)
    {
      const auto& given_outputs = batch_gradients_and_outputs[b].get_outputs(get_layer_index());
      const auto& target_outputs = *(target_outputs_begin + b);

      calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type);

      if (error_calculation_type == ErrorCalculation::type::bce_loss && get_activation().get_method() == activation::method::sigmoid)
      {
        for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
        {
          gradients[neuron_index] = deltas[neuron_index];
        }
      }
      else
      {
        const auto& current_hidden_state = batch_hidden_states[b].at(get_layer_index())[0];
        for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
        {
          double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(neuron_index));
          gradients[neuron_index] = deltas[neuron_index] * deriv;
        }
      }
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), gradients);
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (batch_size < (num_threads * 2))
  {
    run_output_gradients(0, batch_size);
  }
  else
  {
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      _task_queue_pool->enqueue([=]()
        {
          run_output_gradients(start, end);
        });
    }
    _task_queue_pool->get();
  }
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

  // --- Neuron-Parallel Implementation (Small Batch) ---
  auto run_hidden_gradients_neuron_parallel = [&](size_t i_start, size_t i_end)
  {
    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& next_grad_matrix = batch_next_grad_matrix[b];
      const auto& current_hidden_state = batch_hidden_states[b].at(get_layer_index())[0];
      double* grad_ptr = batch_gradients_and_outputs[b].get_gradients_raw(get_layer_index());

      // G_this[i] = sum(G_next[j] * W_next[i, j]) * activation_derivative(pre_act[i])
      for (size_t i = i_start; i < i_end; ++i)
      {
        double sum = 0.0;
        for (size_t j = 0; j < N_next; ++j)
        {
          sum += next_grad_matrix[j] * next_layer.get_weight_value(static_cast<unsigned>(i), static_cast<unsigned>(j));
        }
         
        double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron((unsigned)i));
        grad_ptr[i] = sum * deriv;
      }
    }
  };

  // --- Batch-Parallel Implementation (Large Batch) ---
  auto run_hidden_gradients_batch_parallel = [&](size_t start, size_t end)
  {
    std::vector<double> grad_matrix(N_this, 0.0);
    constexpr size_t BLOCK_SIZE = 64;

    for (size_t b = start; b < end; b++)
    {
      std::fill(grad_matrix.begin(), grad_matrix.end(), 0.0);
      const auto& next_grad_matrix = batch_next_grad_matrix[b];
      const auto& current_hidden_state = batch_hidden_states[b].at(get_layer_index())[0];

      // G_this = (G_next * W_next^T)
      // Tiled multiplication
      for (size_t i0 = 0; i0 < N_this; i0 += BLOCK_SIZE)
      {
        size_t i_limit = std::min(i0 + BLOCK_SIZE, (size_t)N_this);
        for (size_t j0 = 0; j0 < N_next; j0 += BLOCK_SIZE)
        {
          size_t j_limit = std::min(j0 + BLOCK_SIZE, (size_t)N_next);
          for (size_t i = i0; i < i_limit; ++i)
          {
            double partial_sum = 0.0;
            for (size_t j = j0; j < j_limit; ++j)
            {
              partial_sum += next_grad_matrix[j] * next_layer.get_weight_value(static_cast<unsigned>(i), static_cast<unsigned>(j));
            }
            grad_matrix[i] += partial_sum;
          }
        }
      }

      for (unsigned i = 0; i < N_this; i++)
      {
        double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(i));
        grad_matrix[i] *= deriv;
      }
      batch_gradients_and_outputs[b].set_gradients(get_layer_index(), grad_matrix);
    }
  };

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (batch_size < num_threads)
  {
    // Small batch: Use neuron parallelism if beneficial
    if (N_this >= 64 && num_threads > 1)
    {
      // Pre-allocate gradients for all batches
      for (size_t b = 0; b < batch_size; ++b) 
      {
        batch_gradients_and_outputs[b].set_gradients(get_layer_index(), std::vector<double>(N_this, 0.0));
      }

      size_t chunk_size = (N_this + num_threads - 1) / num_threads;
      for (unsigned int t = 0; t < num_threads; ++t)
      {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, (size_t)N_this);
        if (start < end)
        {
          _task_queue_pool->enqueue([=]() { run_hidden_gradients_neuron_parallel(start, end); });
        }
      }
      _task_queue_pool->get();
    }
    else
    {
      run_hidden_gradients_batch_parallel(0, batch_size);
    }
  }
  else
  {
    size_t chunk_size = batch_size / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t)
    {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? batch_size : start + chunk_size;
      _task_queue_pool->enqueue([=]()
        {
          run_hidden_gradients_batch_parallel(start, end);
        });
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
  const std::vector<HiddenStates>& hidden_states,
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

  // Reset gradients
  std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
  if(has_bias())
  {
    std::fill(_b_grads.begin(), _b_grads.end(), 0.0);
  }

  for (unsigned b = 0; b < batch_size; ++b)
  {
    const auto& layer_outputs = batch_gradients_and_outputs[b].get_outputs(previous_layer.get_layer_index());
    const auto& layer_grads = batch_gradients_and_outputs[b].get_gradients(get_layer_index());

    for (unsigned i = 0; i < num_inputs; ++i)
    {
      const double input_val = layer_outputs[i];
      for (unsigned j = 0; j < num_outputs; ++j)
      {
        _w_grads[i * num_outputs + j] += layer_grads[j] * input_val;
      }
    }

    if (has_bias())
    {
      for (unsigned j = 0; j < num_outputs; ++j)
      {
        _b_grads[j] += layer_grads[j];
      }
    }
  }

  // Average gradients over batch
  for (double& grad : _w_grads) grad /= static_cast<double>(batch_size);
  if (has_bias())
  {
    for (double& grad : _b_grads)
    {
      grad /= static_cast<double>(batch_size);
    }
  }
}

double FFLayer::get_gradient_norm_sq() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  double norm_sq = 0.0;
  for (const double grad : _w_grads) norm_sq += grad * grad;
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

  for (unsigned j = 0; j < num_outputs; ++j)
  {
    for (unsigned i = 0; i < num_inputs; ++i)
    {
      unsigned weight_index = i * num_outputs + j;
      apply_weight_gradient(_w_grads[weight_index], learning_rate, false, weight_index, clipping_scale);
    }

    if (has_bias())
    {
      apply_weight_gradient(_b_grads[j], learning_rate, true, j, clipping_scale);
    }
  }
}