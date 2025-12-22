#include "./libraries/instrumentor.h"
#include "fflayer.h"
#include "logger.h"

#include <iostream>
#include <numeric>

constexpr bool _has_bias_neuron = true;

FFLayer::FFLayer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer,
  unsigned num_neurons_in_this_layer,
  double weight_decay,
  LayerType layer_type,
  const activation::method& activation_method,
  const OptimiserType& optimiser_type,
  int residual_layer_number,
  double dropout_rate,
  ResidualProjector* residual_projector
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
    residual_projector)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(const FFLayer& src) noexcept :
  Layer(src)
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

FFLayer::~FFLayer() = default;

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

  std::vector<std::vector<double>> batch_pre_activation_sums(batch_size, std::vector<double>(N_this, 0.0));

  // Initialize with bias values
  if (has_bias())
  {
    for (size_t b = 0; b < batch_size; b++)
    {
      for (size_t j = 0; j < N_this; j++)
      {
        batch_pre_activation_sums[b][j] = get_bias_value((unsigned)j);
      }
    }
  }

  // Multiply and accumulate weights and inputs (Matrix-Matrix pattern)
  // Optimization: Loop order (i, b, j) ensures weights are in the inner loop and reused across the batch.
  for (size_t i = 0; i < N_prev; i++)
  {
    for (size_t b = 0; b < batch_size; b++)
    {
      const double input_val = batch_gradients_and_outputs[b].get_output(get_layer_index() - 1, (unsigned)i);
      if (input_val == 0.0) continue;
      for (size_t j = 0; j < N_this; j++)
      {
        batch_pre_activation_sums[b][j] += input_val * get_weight_value((unsigned)i, (unsigned)j);
      }
    }
  }

  if (!batch_residual_output_values.empty())
  {
    for (size_t b = 0; b < batch_size; b++)
    {
      if (batch_residual_output_values[b].size() == N_this)
      {
        for (size_t j = 0; j < N_this; j++)
        {
          batch_pre_activation_sums[b][j] += batch_residual_output_values[b][j];
        }
      }
    }
  }

  for (size_t b = 0; b < batch_size; b++)
  {
    std::vector<double> output_row(N_this);
    for (size_t j = 0; j < N_this; j++)
    {
      const auto& neuron = get_neuron((unsigned)j);
      double output = get_activation().activate(batch_pre_activation_sums[b][j]);

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
    }
    
    if(!batch_hidden_states.empty())
    {
      batch_hidden_states[b].at(get_layer_index())[0].set_pre_activation_sums(batch_pre_activation_sums[b]);
      batch_hidden_states[b].at(get_layer_index())[0].set_hidden_state_values(output_row);
    }
    batch_gradients_and_outputs[b].set_outputs(get_layer_index(), output_row);
  }
}

void FFLayer::calculate_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  switch (error_calculation_type)
  {
  case ErrorCalculation::type::mse:
    return calculate_mse_error_deltas(deltas, target_outputs, given_outputs);
  case ErrorCalculation::type::bce_loss:
    return calculate_bce_error_deltas(deltas, target_outputs, given_outputs);
  default:
    Logger::panic("ErrorCalculation type is not supported for FFLayer!");
  }
}

void FFLayer::calculate_bce_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t N_total = get_number_neurons();
  const double denom = static_cast<double>(N_total);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) / denom;
  }
}

void FFLayer::calculate_mse_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t N_total = get_number_neurons();
  const double denom = static_cast<double>(N_total);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) / denom;
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

  for (size_t b = 0; b < batch_size; b++)
  {
    std::vector<double> gradients(N_total, 0.0);
    std::vector<double> deltas(N_total, 0.0);
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

  for (size_t b = 0; b < batch_size; b++)
  {
    std::vector<double> grad_matrix(N_this, 0.0);
    const auto& next_grad_matrix = batch_next_grad_matrix[b];
    const auto& current_hidden_state = batch_hidden_states[b].at(get_layer_index())[0];

    // G_this = (G_next * W_next^T)
    for (unsigned i = 0; i < N_this; i++)
    {
      double weighted_sum = 0.0;
      for (size_t j = 0; j < N_next; j++)
      {
        weighted_sum += next_grad_matrix[j] * next_layer.get_weight_value(i, (unsigned)j);
      }
      double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(i));
      grad_matrix[i] = weighted_sum * deriv;
    }
    batch_gradients_and_outputs[b].set_gradients(get_layer_index(), grad_matrix);
  }
}

Layer* FFLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return new FFLayer(*this);
}
