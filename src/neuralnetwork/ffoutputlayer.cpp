#include "./libraries/instrumentor.h"
#include "ffoutputlayer.h"
#include "logger.h"
#include "neuralnetworkhelpermetrics.h"

FFOutputLayer::FFOutputLayer(
  unsigned layer_index,
  const std::vector<OutputLayerDetails>& output_layer_details,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  const OptimiserType& optimiser_type,
  int number_of_threads,
  bool has_bias
) :
  FFLayer(
    layer_index,
    num_neurons_in_previous_layer,
    num_neurons_in_this_layer,
    create_weight_decays(num_neurons_in_previous_layer, num_neurons_in_this_layer, output_layer_details),
    Layer::LayerType::Output,
    output_layer_details.front().get_activation(),
    optimiser_type,
    -1,       //  no residual layer
    0.0,      //  no dropout for output layer
    nullptr,  //  no residual projector
    number_of_threads,
    has_bias),
  OutputLayer(output_layer_details)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

std::vector<double> FFOutputLayer::create_weight_decays(
  unsigned num_inputs,
  unsigned num_neurons_in_this_layer,
  const std::vector<OutputLayerDetails>& output_layer_details)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  unsigned total_neurons = 0;
  for (const auto& detail : output_layer_details)
  {
    total_neurons += detail.get_size();
  }

  if (total_neurons != num_neurons_in_this_layer)
  {
    Logger::panic("The total number of neurons in output_layer_details does not match num_neurons_in_this_layer.");
  }

  const size_t num_weights = static_cast<size_t>(num_inputs) * num_neurons_in_this_layer;
  std::vector<double> weight_decays(num_weights);

  unsigned current_output_neuron = 0;
  for (const auto& detail : output_layer_details)
  {
    const double decay = detail.get_weight_decay();
    const unsigned section_size = detail.get_size();
    for (unsigned s = 0; s < section_size; ++s)
    {
      const unsigned j = current_output_neuron + s;
      for (unsigned i = 0; i < num_inputs; ++i)
      {
        weight_decays[i * num_neurons_in_this_layer + j] = decay;
      }
    }
    current_output_neuron += section_size;
  }

  return weight_decays;
}

FFOutputLayer::FFOutputLayer(const FFOutputLayer& src) noexcept :
  FFLayer(src),
  OutputLayer(src)
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
    OutputLayer(output_layer_details)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

FFOutputLayer::FFOutputLayer(FFOutputLayer&& src) noexcept :
  FFLayer(std::move(src)),
  OutputLayer(std::move(src))
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

FFOutputLayer& FFOutputLayer::operator=(const FFOutputLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  if(this != &src)
  {
    FFLayer::operator=(src);
    OutputLayer::operator=(src);
  }
  return *this;
}

FFOutputLayer& FFOutputLayer::operator=(FFOutputLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  if(this != &src)
  {
    FFLayer::operator=(std::move(src));
    OutputLayer::operator=(std::move(src));
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
        _task_queue_pool->enqueue([start, end, &batch_gradients_and_outputs, target_outputs_begin, &batch_hidden_states, N_total, this]()
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
  const size_t start,
  const size_t end,
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
        double deriv = OutputLayer::get_activation(neuron_index).activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(neuron_index));
        gradients[neuron_index] = deltas[neuron_index] * deriv;
      }

      Logger::trace([&]()
      {
        std::ostringstream ss;
        ss << "[FFOutputLayer::run_output_gradients] b=" << b
           << ", neuron=" << neuron_index
           << ", target=" << target_outputs[neuron_index]
           << ", given=" << given_outputs[neuron_index]
           << ", delta=" << deltas[neuron_index]
           << ", grad=" << gradients[neuron_index];
        return ss.str();
      });
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
        _task_queue_pool->enqueue([start, end, N_prev, N_this, this]()
          { 
            run_gemm(start, end, N_prev, N_this); 
          });
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
        _task_queue_pool->enqueue([start,end, N_this, &batch_gradients_and_outputs, &batch_residual_output_values, &batch_hidden_states, is_training, this]()
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
  const size_t start,
  const size_t end,
  const size_t N_this,
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

    // Activation
    std::copy(current_pre_act, current_pre_act + N_this, output_row.begin());
    for (unsigned int i = 0; i < OutputLayer::number_output_layers(); ++i)
    {
      const auto& b_range = OutputLayer::layer_bounds(i);
      OutputLayer::get_activation(b_range.start).activate(output_row.data() + b_range.start, output_row.data() + b_range.end + 1);
    }

    // Dropout and Output
    const auto output_ptr = batch_gradients_and_outputs[b].get_outputs_raw(get_layer_index());
    for (size_t j = 0; j < N_this; j++)
    {
      const auto& neuron = get_neuron((unsigned)j);
      double output = output_row[j];

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

void FFOutputLayer::calculate_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  unsigned layer_number = 0;
  for (const auto& output_layer_detail : output_layer_details())
  {
    const auto error_calculation_type = output_layer_detail.get_output_error_calculation_type();
    const auto activation_method = output_layer_detail.get_activation().get_method();
    const auto evaluation_config = output_layer_detail.get_error_evaluation_config();
    const auto& bounds = layer_bounds(layer_number);
    Layer::calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type, evaluation_config, activation_method, bounds.start, bounds.end);
    ++layer_number;
  }
}

std::vector<std::vector<NeuralNetworkHelperMetrics>> FFOutputLayer::calculate_output_metrics(
  const std::vector<ErrorCalculation::type>& error_types,
  const std::vector<std::vector<double>>& predictions,
  const std::vector<std::vector<double>>& checking_outputs
) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  std::vector<std::vector<NeuralNetworkHelperMetrics>> errors;
  errors.reserve(number_output_layers());

  const size_t batch_size = predictions.size();
#if VALIDATE_DATA == 1
  if (batch_size != checking_outputs.size())
  {
    Logger::panic("The number of predictions is not the same as the number of given outputs!");
  }
#endif
  if (number_output_layers() == 1)
  {
    // only one output, fast path
    std::vector<NeuralNetworkHelperMetrics> layer_errors;
    layer_errors.reserve(error_types.size());

    constexpr unsigned output_layer_index = 0; // layer zero, fast track only one layer...
    const auto& configs = evaluation_config(output_layer_index);
    const auto& activation = OutputLayer::get_activation(output_layer_index);
    const auto& activation_method = activation.get_method();

    for (const auto& error_type : error_types)
    {
      layer_errors.emplace_back(
        ErrorCalculation::calculate_error(error_type, checking_outputs, predictions, configs, activation_method),
        error_type);
    }
    errors.emplace_back(std::move(layer_errors));
    return errors;
  }

  // Multi-output path
  std::vector<std::vector<double>> sliced_predictions(batch_size);
  std::vector<std::vector<double>> sliced_checking_outputs(batch_size);
  std::vector<NeuralNetworkHelperMetrics> layer_errors;
  layer_errors.reserve(error_types.size());

  for (unsigned output_layer_index = 0; output_layer_index < number_output_layers(); ++output_layer_index)
  {
    const auto& activation = OutputLayer::get_activation(output_layer_index);
    const auto& activation_method = activation.get_method();

    const auto& bounds = layer_bounds(output_layer_index);
    const auto& configs = evaluation_config(output_layer_index);
    const size_t num_neurons = bounds.end - bounds.start + 1;

    for (size_t b = 0; b < batch_size; ++b)
    {
      sliced_predictions[b].assign(predictions[b].begin() + bounds.start, predictions[b].begin() + bounds.start + num_neurons);
      sliced_checking_outputs[b].assign(checking_outputs[b].begin() + bounds.start, checking_outputs[b].begin() + bounds.start + num_neurons);
    }

    layer_errors.clear();
    for (const auto& error_type : error_types)
    {
      layer_errors.emplace_back(
        ErrorCalculation::calculate_error(error_type, std::span(sliced_checking_outputs), std::span(sliced_predictions), configs, activation_method),
        error_type);
    }
    errors.emplace_back(layer_errors);
  }
  return errors;
}

void FFOutputLayer::apply_stored_gradients(double learning_rate, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");

  // If we have no output layer details, we can't do much.
  if (output_layer_details().empty())
  {
    FFLayer::apply_stored_gradients(learning_rate, clipping_scale);
    return;
  }

  // We need to apply the gradients for each head separately.
  // This is because each head can have its own optimizer.
  unsigned current_output_neuron = 0;
  const unsigned num_inputs = get_number_input_neurons();
  const unsigned num_outputs = get_number_output_neurons();

  for (const auto& detail : output_layer_details())
  {
    const unsigned section_size = detail.get_size();
    const OptimiserType optimiser_type = detail.get_optimiser_type();

    // 1. Update weights for this head
    for (unsigned i = 0; i < num_inputs; ++i)
    {
      for (unsigned s = 0; s < section_size; ++s)
      {
        const unsigned j = current_output_neuron + s;
        const unsigned weight_index = i * num_outputs + j;
        
        // Use the Layer base class method, passing the head-specific optimizer
        Layer::apply_weight_gradient(_w_grads[weight_index], learning_rate, false, weight_index, clipping_scale, optimiser_type);
      }
    }

    // 2. Update biases for this head (if they exist)
    if (has_bias())
    {
      for (unsigned s = 0; s < section_size; ++s)
      {
        const unsigned j = current_output_neuron + s;
        Layer::apply_weight_gradient(_b_grads[j], learning_rate, true, j, clipping_scale, optimiser_type);
      }
    }

    current_output_neuron += section_size;
  }
}
