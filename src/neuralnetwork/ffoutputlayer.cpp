#include "./libraries/instrumentor.h"
#include "ffoutputlayer.h"
#include "logger.h"
#include <numeric>

constexpr bool _has_bias_neuron = true;

FFOutputLayer::FFOutputLayer(
  unsigned layer_index,
  const std::vector<OutputLayerDetails>& output_layer_details,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  double weight_decay,
  const OptimiserType& optimiser_type,
  int number_of_threads
) :
  FFLayer(
    layer_index,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    weight_decay,
    Layer::LayerType::Output,
    output_layer_details.front().get_activation(),
    optimiser_type,
    -1,       //  no residual layer
    0.0,      //  no dropout for output layer
    nullptr,  //  no residual projector
    number_of_threads),
  _output_layer_details(output_layer_details)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  create_activation_per_neuron(output_layer_details);
  create_using_activation_derivatives_per_neuron(output_layer_details);
}

FFOutputLayer::FFOutputLayer(const FFOutputLayer& src) noexcept :
  FFLayer(src),
  _output_layer_details(src._output_layer_details),
  _activations(src._activations),
  _is_not_using_activation_derivatives(src._is_not_using_activation_derivatives)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

FFOutputLayer::FFOutputLayer(
  unsigned layer_index,
  const std::vector<OutputLayerDetails>& output_layer_details,
  const OptimiserType optimiser_type,
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
  int number_of_threads
) noexcept : 
  FFLayer(
    layer_index,
    Layer::LayerType::Output,
    output_layer_details.front().get_activation(),
    optimiser_type,
    -1,       //  no residual layer
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
    nullptr,  //  no residual projector
    number_of_threads),
    _output_layer_details(output_layer_details)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  create_activation_per_neuron(output_layer_details);
  create_using_activation_derivatives_per_neuron(output_layer_details);
}

FFOutputLayer::FFOutputLayer(FFOutputLayer&& src) noexcept :
  FFLayer(std::move(src)),
  _output_layer_details(std::move(src._output_layer_details)),
  _activations(std::move(src._activations)),
  _is_not_using_activation_derivatives(std::move(src._is_not_using_activation_derivatives))
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

FFOutputLayer& FFOutputLayer::operator=(const FFOutputLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  if(this != &src)
  {
    FFLayer::operator=(src);
    _output_layer_details = src._output_layer_details;
    _activations = src._activations;
    _is_not_using_activation_derivatives = src._is_not_using_activation_derivatives;
  }
  return *this;
}

FFOutputLayer& FFOutputLayer::operator=(FFOutputLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  if(this != &src)
  {
    FFLayer::operator=(std::move(src));
    _output_layer_details = std::move(src._output_layer_details);
    _activations = std::move(src._activations);
    _is_not_using_activation_derivatives = std::move(src._is_not_using_activation_derivatives);
  }
  return *this;
}

FFOutputLayer::~FFOutputLayer()
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

Layer* FFOutputLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  return new FFOutputLayer(*this);
}

void FFOutputLayer::calculate_hidden_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<std::vector<double>>& batch_next_grad_matrix,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  int bptt_max_ticks) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  Logger::panic("The output layer cannot do hidden layer calculations!");
}

void FFOutputLayer::calculate_output_gradients(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  const size_t N_total = get_number_neurons();

  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_output_gradients(
      0, 
      batch_size, 
      batch_gradients_and_outputs, 
      target_outputs_begin, 
      batch_hidden_states,
      N_total);
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
        _task_queue_pool->enqueue([=, &batch_gradients_and_outputs]()
          {
            run_output_gradients(
              start, 
              end, 
              batch_gradients_and_outputs, 
              target_outputs_begin, 
              batch_hidden_states,
              N_total);
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void FFOutputLayer::run_output_gradients(
  size_t start,
  size_t end,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  std::vector<std::vector<double>>::const_iterator target_outputs_begin,
  const std::vector<HiddenStates>& batch_hidden_states,
  size_t num_neurons) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  std::vector<double> gradients(num_neurons, 0.0);
  std::vector<double> deltas(num_neurons, 0.0);

  for (size_t b = start; b < end; b++)
  {
    const auto& given_outputs = batch_gradients_and_outputs[b].get_outputs(get_layer_index());
    const auto& target_outputs = *(target_outputs_begin + b);

    calculate_error_deltas(deltas, target_outputs, given_outputs);

    for (unsigned neuron_index = 0; neuron_index < num_neurons; ++neuron_index)
    {
      if (get_is_not_using_activation_derivatives(neuron_index))
      {
        gradients[neuron_index] = deltas[neuron_index];
      }
      else
      {
        const auto& current_hidden_state = batch_hidden_states[b].at(get_layer_index())[0];
        double deriv = get_activation(neuron_index).activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(neuron_index));
        gradients[neuron_index] = deltas[neuron_index] * deriv;
      }
    }
    batch_gradients_and_outputs[b].set_gradients(get_layer_index(), gradients);
  }
}

void FFOutputLayer::calculate_forward_feed(
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  size_t batch_size,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  const auto N_prev = get_number_input_neurons();
  const auto N_this = get_number_neurons();

  if (batch_size == 0)
  {
    return;
  }

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
  const auto& num_threads = _task_queue_pool->get_number_of_threads();
  if (num_threads <= 1)
  {
    run_gemm(0, batch_size, N_prev, N_this);
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
        _task_queue_pool->enqueue([=]() { run_gemm(start, end, N_prev, N_this); });
      }
      start = end;
    }
    _task_queue_pool->get();
  }

  // 4. Residuals, Activation and Dropout
  if (num_threads <= 1)
  {
    run_post_gemm(0, batch_size, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, is_training);
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
        _task_queue_pool->enqueue([=, &batch_gradients_and_outputs, &batch_residual_output_values, &batch_hidden_states]() 
          { 
            run_post_gemm(start, end, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, is_training); 
          });
      }
      start = end;
    }
    _task_queue_pool->get();
  }
}

void FFOutputLayer::run_post_gemm(
  size_t start,
  size_t end,
  size_t N_this,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
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

    // Activation and Dropout
    const auto output_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    for (size_t j = 0; j < N_this; j++)
    {
      const auto& neuron = get_neuron((unsigned)j);
      
      // Use scalar activation
      double output = get_activation(static_cast<unsigned>(j)).activate(current_pre_act[j]);
      
      // Store the activated value back in the buffer if needed later for hidden states
      current_pre_act[j] = output;

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
}

void FFOutputLayer::create_activation_per_neuron(const std::vector<OutputLayerDetails>& output_layer_details)
{
  for (const auto& output_layer_detail : output_layer_details)
  {
    for (size_t i = 0; i < output_layer_detail.get_size(); ++i)
    {
      _activations.push_back(output_layer_detail.get_activation());
    }
  }
}

void FFOutputLayer::create_using_activation_derivatives_per_neuron(const std::vector<OutputLayerDetails>& output_layer_details)
{
  for (const auto& output_layer_detail : output_layer_details)
  {
    for (size_t i = 0; i < output_layer_detail.get_size(); ++i)
    {
      const auto& error_calculation_type = output_layer_detail.get_output_error_calculation_type();
      const auto is_not_using_activation_derivative = Layer::is_not_using_activation_derivative(output_layer_detail.get_activation().get_method(), error_calculation_type);
      _is_not_using_activation_derivatives.push_back(is_not_using_activation_derivative);
    }
  }
}

void FFOutputLayer::calculate_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  unsigned start_neuron = 0;
  unsigned end_neuron = 0;
  for (const auto& output_layer_detail : _output_layer_details)
  {
    const auto error_calculation_type = output_layer_detail.get_output_error_calculation_type();
    const auto evaluation_config = output_layer_detail.get_error_evaluation_config();
    end_neuron = start_neuron + output_layer_detail.get_size() - 1;
    Layer::calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type, evaluation_config, start_neuron, end_neuron);
    start_neuron = end_neuron + 1;
  }
}
