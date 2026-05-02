#include "./libraries/instrumentor.h"
#include "ffoutputlayer.h"
#include "logger.h"
#include "neuralnetworkhelpermetrics.h"

FFOutputLayer::FFOutputLayer(
  unsigned layer_index,
  const std::vector<OutputLayerDetails>& output_layer_details,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  int number_of_threads,
  bool has_bias
) :
  FFLayer(
    layer_index,
    create_weight_decays(num_neurons_in_previous_layer, num_neurons_in_this_layer, output_layer_details),
    Role::Output,
    create_layer_activation_helper(num_neurons_in_previous_layer, num_neurons_in_this_layer, output_layer_details),
    OptimiserType::None,
    -1,       //  no residual layer
    0.0,      //  no dropout for output layer
    nullptr,  //  no residual projector
    number_of_threads,
    has_bias, 
    0.0),
  OutputLayer(output_layer_details)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
}

layer_activation_helper FFOutputLayer::create_layer_activation_helper(unsigned num_inputs,
  unsigned num_neurons_in_this_layer,
  const std::vector<OutputLayerDetails>& output_layer_details)
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  layer_activation_helper lah(output_layer_details.front().get_activation(), num_inputs, num_neurons_in_this_layer);
  unsigned start = 0;
  unsigned end = 0;
  for (const auto& detail : output_layer_details)
  {
    end = start + detail.get_size();
    lah.set_bounds(detail.get_activation(), start, end);
    start = end;
  }
  return lah;
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
    Role::Output,
    OptimiserType::None,
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
    number_of_threads,
    create_layer_activation_helper(number_input_neurons, number_output_neurons, output_layer_details),
    0.0
    ),
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
  const auto& details = output_layer_details();
  const auto& ranges = _layer_activation_helper.ranges();
  const size_t num_time_steps = batch_hidden_states[0].at(get_layer_index()).size();

  for (size_t b = start; b < end; b++)
  {
    const auto& target_outputs = *(target_outputs_begin + b);
    const auto& layer_states = batch_hidden_states[b].at(get_layer_index());
    std::vector<double> rnn_grads_row(num_time_steps * num_neurons, 0.0);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const auto& current_hidden_state = layer_states[t];
      const auto& given_outputs = current_hidden_state.get_hidden_state_values();
      std::vector<double> given_outputs_vec(given_outputs.begin(), given_outputs.end());
      
      // Determine target for this time step
      std::vector<double> current_target;
      if (target_outputs.size() == num_time_steps * num_neurons)
      {
        current_target.assign(target_outputs.begin() + t * num_neurons, target_outputs.begin() + (t + 1) * num_neurons);
      }
      else if (t == num_time_steps - 1)
      {
        // Only one target provided, apply to the last step
        current_target = target_outputs;
      }
      else
      {
        // No target for this step
        continue;
      }

      std::vector<double> deltas(num_neurons, 0.0);
      calculate_error_deltas(deltas, current_target, given_outputs_vec);

      for (size_t h = 0; h < ranges.size(); ++h)
      {
        const auto& r = ranges[h];
        const auto& detail = details[h];
        const auto& activation = r.activation_method;

        const bool skip_derivative = (activation.get_method() == activation::method::softmax) ||
          is_not_using_activation_derivative(activation.get_method(), detail.get_output_error_calculation_type());

        if (skip_derivative)
        {
          for (unsigned i = r.start; i < r.end; ++i)
          {
            double mask = current_hidden_state.get_cell_state_value_at_neuron(i);
            rnn_grads_row[t * num_neurons + i] = deltas[i] * mask;
          }
        }
        else
        {
          for (unsigned i = r.start; i < r.end; ++i)
          {
            double deriv = activation.activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(i));
            double mask = current_hidden_state.get_cell_state_value_at_neuron(i);
            rnn_grads_row[t * num_neurons + i] = deltas[i] * deriv * mask;
          }
        }
      }
    }

    batch_gradients_and_outputs[b].set_rnn_gradients(get_layer_index(), rnn_grads_row);
    
    std::vector<double> last_step_grads(num_neurons);
    std::copy(rnn_grads_row.end() - num_neurons, rnn_grads_row.end(), last_step_grads.begin());
    batch_gradients_and_outputs[b].set_gradients(get_layer_index(), last_step_grads);
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
  if (batch_size == 0) return;

  const auto N_prev = get_number_input_neurons();
  const auto N_this = get_number_neurons();
  const unsigned prev_layer_index = previous_layer.get_layer_index();

  // 1. Determine sequence length and flatten inputs
  size_t num_time_steps = 1;
  for (size_t b = 0; b < batch_size; ++b)
  {
      const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
      if (!rnn_in.empty()) { num_time_steps = rnn_in.size() / N_prev; break; }
  }

  const size_t effective_batch_size = batch_size * num_time_steps;
  std::vector<double> batch_inputs_buffer(effective_batch_size * N_prev);
  
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& rnn_in = batch_gradients_and_outputs[b].get_rnn_outputs(prev_layer_index);
    if (!rnn_in.empty()) std::copy(rnn_in.begin(), rnn_in.end(), batch_inputs_buffer.begin() + b * num_time_steps * N_prev);
    else
    {
        const auto std_in = batch_gradients_and_outputs[b].get_outputs(prev_layer_index);
        for (size_t t = 0; t < num_time_steps; ++t) std::copy(std_in.begin(), std_in.end(), batch_inputs_buffer.begin() + (b * num_time_steps + t) * N_prev);
    }
  }

  std::vector<double> batch_pre_activation_sums_buffer(effective_batch_size * N_this, 0.0);

  // 2. Initialize with bias values
  if (has_bias())
  {
    for (size_t eb = 0; eb < effective_batch_size; eb++)
    {
      double* dest = &batch_pre_activation_sums_buffer[eb * N_this];
      for (size_t j = 0; j < N_this; j++) dest[j] = get_bias_value((unsigned)j);
    }
  }

  // 3. Batched GEMM
  run_gemm(0, effective_batch_size, N_prev, N_this, batch_inputs_buffer, batch_pre_activation_sums_buffer);

  // 4. Activation
  run_post_gemm(0, batch_size, num_time_steps, N_this, batch_gradients_and_outputs, batch_residual_output_values, batch_hidden_states, batch_inputs_buffer, batch_pre_activation_sums_buffer, is_training);
}

void FFOutputLayer::run_post_gemm(
  size_t start,
  size_t end,
  size_t num_time_steps,
  size_t N_this,
  std::vector<GradientsAndOutputs>& batch_gradients_and_outputs,
  const std::vector<std::vector<double>>& batch_residual_output_values,
  std::vector<HiddenStates>& batch_hidden_states,
  const std::vector<double>& /*batch_inputs_buffer*/,
  std::vector<double>& batch_pre_activation_sums_buffer,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  
  for (size_t b = start; b < end; b++)
  {
    std::vector<double> output_row_seq(num_time_steps * N_this);
    if (batch_hidden_states[b].at(get_layer_index()).size() != num_time_steps)
    {
      batch_hidden_states[b].assign(get_layer_index(), num_time_steps, {}, get_pre_activation_multiplier());
    }
    auto& layer_states_ref = batch_hidden_states[b].at(get_layer_index());

    for (size_t t = 0; t < num_time_steps; ++t)
    {
        double* current_pre_act = &batch_pre_activation_sums_buffer[(b * num_time_steps + t) * N_this];
        double* current_output_row = &output_row_seq[t * N_this];

        if (!batch_residual_output_values.empty() && batch_residual_output_values[b].size() == N_this)
        {
          if (num_time_steps == 1 || t == num_time_steps - 1)
            for (size_t j = 0; j < N_this; j++) current_pre_act[j] += batch_residual_output_values[b][j];
        }

        layer_states_ref[t].set_pre_activation_sums(std::vector<double>(current_pre_act, current_pre_act + N_this));

        std::vector<double> mask(N_this, 1.0);
        for (const auto& r : _layer_activation_helper.ranges())
        {
          r.activation_method.activate(current_pre_act + r.start, current_pre_act + r.end, is_training);
          for (size_t j = r.start; j < r.end; j++)
          {
            const auto& neuron = get_neuron((unsigned)j);
            double output = current_pre_act[j];
            if (is_training && neuron.is_dropout())
            {
              if (neuron.must_randomly_drop())
              {
                output = 0.0;
                mask[j] = 0.0;
              }
              else
              {
                double scale = 1.0 / (1.0 - neuron.get_dropout_rate());
                output *= scale;
                mask[j] = scale;
              }
            }
            current_output_row[j] = output;
          }
        }
        layer_states_ref[t].set_cell_state_values(mask);
        layer_states_ref[t].set_hidden_state_values(std::vector<double>(current_output_row, current_output_row + N_this));
    }

    batch_gradients_and_outputs[b].set_rnn_outputs(get_layer_index(), output_row_seq);
    std::vector<double> last_step_output(N_this);
    std::copy(output_row_seq.end() - N_this, output_row_seq.end(), last_step_output.begin());
    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), last_step_output);
  }
}

double FFOutputLayer::get_momentum(unsigned neuron_number) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFOutputLayer");
  const auto& details = OutputLayer::output_layer_details();
  if (number_output_layers() == 1)
  {
    return details[0].get_momentum();
  }

  unsigned number_neurons = 0;
  for (const auto& detail : details)
  {
    number_neurons += detail.get_size();
    if (neuron_number < number_neurons)
    {
      return detail.get_momentum();
    }
  }
  Logger::panic("Trying to get a neuron detail past the number of neurons");
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
  const std::vector<std::vector<double>>& checking_outputs,
  const std::vector<std::vector<double>>& predictions
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

  const unsigned total_outputs = get_number_neurons();

  for (unsigned output_layer_index = 0; output_layer_index < number_output_layers(); ++output_layer_index)
  {
    std::vector<NeuralNetworkHelperMetrics> layer_errors;
    layer_errors.reserve(error_types.size());

    const auto& activation = output_layer_details()[output_layer_index].get_activation();
    const auto& activation_method = activation.get_method();

    const auto& bounds = layer_bounds(output_layer_index);
    const auto& configs = evaluation_config(output_layer_index);
    const size_t num_neurons = bounds.end - bounds.start + 1;

    // Unroll sequences: treat each time step of each batch item as an independent sample for metrics.
    // This ensures that ErrorCalculation (which works on samples) correctly handles Softmax max-indices, etc.
    std::vector<std::vector<double>> unrolled_predictions;
    std::vector<std::vector<double>> unrolled_checking_outputs;

    for (size_t b = 0; b < batch_size; ++b)
    {
      const size_t p_total = predictions[b].size();
      const size_t c_total = checking_outputs[b].size();

      if (total_outputs == 0)
      {
        continue;
      }

      const size_t p_steps = p_total / total_outputs;
      const size_t c_steps = c_total / total_outputs;
      const size_t num_steps = std::min(p_steps, c_steps);

      // Align at the end. For example, if c_steps=1 and p_steps=10, we take the last prediction step.
      const size_t p_offset = (p_steps > num_steps) ? (p_steps - num_steps) : 0;
      const size_t c_offset = (c_steps > num_steps) ? (c_steps - num_steps) : 0;

      for (size_t t = 0; t < num_steps; ++t)
      {
        std::vector<double> p_slice(num_neurons);
        std::vector<double> c_slice(num_neurons);

        const auto p_start = predictions[b].begin() + (t + p_offset) * total_outputs + bounds.start;
        std::copy(p_start, p_start + num_neurons, p_slice.begin());

        const auto c_start = checking_outputs[b].begin() + (t + c_offset) * total_outputs + bounds.start;
        std::copy(c_start, c_start + num_neurons, c_slice.begin());

        unrolled_predictions.push_back(std::move(p_slice));
        unrolled_checking_outputs.push_back(std::move(c_slice));
      }
    }

    for (const auto& error_type : error_types)
    {
      layer_errors.emplace_back(
        ErrorCalculation::calculate_error(error_type, unrolled_checking_outputs, unrolled_predictions, configs, activation_method),
        error_type);
    }
    errors.emplace_back(std::move(layer_errors));
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
        const unsigned neuron_number = current_output_neuron + s;
        const unsigned weight_index = i * num_outputs + neuron_number;

        // Use the Layer base class method, passing the head-specific optimizer and state vectors
        apply_update_to_weight(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, weight_index, _w_grads[weight_index], learning_rate, clipping_scale, optimiser_type, neuron_number);
      }
    }

    // 2. Update biases for this head (if they exist)
    if (has_bias())
    {
      for (unsigned s = 0; s < section_size; ++s)
      {
        const unsigned neuron_number = current_output_neuron + s;
        apply_update_to_weight(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, neuron_number, _b_grads[neuron_number], learning_rate, clipping_scale, optimiser_type, neuron_number);
      }
    }

    current_output_neuron += section_size;
  }

  // Clear gradients
  std::fill(_w_grads.begin(), _w_grads.end(), 0.0);
  std::fill(_b_grads.begin(), _b_grads.end(), 0.0);
}
