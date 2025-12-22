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
    create_weights(weight_decay, layer_type, activation_method, num_neurons_in_previous_layer, num_neurons_in_this_layer),
    create_bias_weights(true, activation_method, num_neurons_in_this_layer),
    residual_projector)
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
  const std::vector<Neuron>& neurons,
  unsigned number_input_neurons,
  LayerType layer_type,
  OptimiserType optimiser_type,
  int residual_layer_number,
  const activation::method& activation_method,
  const std::vector<std::vector<WeightParam>>& weights,
  const std::vector<WeightParam>& bias_weights,
  const std::vector<std::vector<WeightParam>>& residual_weights
) : 
  Layer(
    layer_index,
    layer_type,
    activation_method,
    optimiser_type,
    residual_layer_number,
    number_input_neurons,
    static_cast<unsigned>(neurons.size()),
    neurons,
    weights,
    bias_weights,
    ResidualProjector::create(residual_weights))
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

std::vector<double> FFLayer::calculate_forward_feed(
  GradientsAndOutputs& gradients_and_outputs,
  const Layer& previous_layer,
  const std::vector<double>& previous_layer_inputs,
  const std::vector<double>& residual_output_values,
  std::vector<HiddenState>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t N_prev = previous_layer.get_number_neurons();
  const size_t N_this = get_number_neurons();

  std::vector<double> output_row(N_this, 0.0);
  std::vector<double> pre_activation_sums(N_this, 0.0);

  for (size_t j = 0; j < N_this; j++)
  {
    if (!get_bias_weight_params().empty())
    {
      pre_activation_sums[j] = get_bias_weight_param(j).get_value();
    }
  }

  for (size_t i = 0; i < N_prev; i++)
  {
    double input_val = previous_layer_inputs[i];
    if (input_val == 0.0) continue;
    const auto& weight_row = get_weight_param(i);
    for (size_t j = 0; j < N_this; j++)
    {
      pre_activation_sums[j] += input_val * weight_row[j].get_value();
    }
  }

  if (!residual_output_values.empty())
  {
    if (residual_output_values.size() != N_this)
    {
		  Logger::warning("Residual output values size mismatch. Expected ", N_this, " but got ", residual_output_values.size());
	  }
    else
    {
		  for (size_t j = 0; j < N_this; j++)
		  {
			  pre_activation_sums[j] += residual_output_values[j];
		  }
	  }
  }

  for (size_t j = 0; j < N_this; j++)
  {
    const auto& neuron = get_neuron((unsigned)j);
    double output = get_activation().activate(pre_activation_sums[j]);

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
  
  if(!hidden_states.empty())
  {
    hidden_states[0].set_pre_activation_sums(pre_activation_sums);
    hidden_states[0].set_hidden_state_values(output_row);
  }
  gradients_and_outputs.set_outputs(get_layer_index(), output_row);
  return output_row;
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
  GradientsAndOutputs& gradients_and_outputs,
  const std::vector<double>& target_outputs,
  const std::vector<HiddenState>& hidden_states,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t N_total = get_number_neurons();

  std::vector<double> gradients(N_total, 0.0);
  std::vector<double> deltas(N_total, 0.0);
  calculate_error_deltas(deltas, target_outputs, gradients_and_outputs.get_outputs(get_layer_index()), error_calculation_type);

  if (error_calculation_type == ErrorCalculation::type::bce_loss && get_activation().get_method() == activation::method::sigmoid)
  {
    // For BCE with Sigmoid, the derivative of the loss w.r.t. pre-activation is just (a - y)
    for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
    {
      gradients[neuron_index] = deltas[neuron_index];
    }
  }
  else
  {
    // For other cases (like MSE), we need to multiply by the activation's derivative
    const auto& current_hidden_state = hidden_states[0];
    for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
    {
      double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(neuron_index));
      gradients[neuron_index] = deltas[neuron_index] * deriv;
    }
  }
  gradients_and_outputs.set_gradients(get_layer_index(), gradients);
}

void FFLayer::calculate_hidden_gradients(
  GradientsAndOutputs& gradients_and_outputs,
  const Layer& next_layer,
  const std::vector<double>& next_grad_matrix,
  const std::vector<double>&,
  const std::vector<HiddenState>& hidden_states,
  int /*bptt_max_ticks*/) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t N_this = get_number_neurons();
  const size_t N_next = next_layer.get_number_neurons();

  std::vector<double> grad_matrix(N_this, 0.0);

  const auto& current_hidden_state = hidden_states[0];

  for (unsigned i = 0; i < N_this; i++)
  {
    double weighted_sum = 0.0;
    for (size_t j = 0; j < N_next; j++)
    {
      weighted_sum += next_grad_matrix[j] * next_layer.get_weight_param(i, j).get_value();
    }
    double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(i));
    grad_matrix[i] = weighted_sum * deriv;
  }
  gradients_and_outputs.set_gradients(get_layer_index(), grad_matrix);
}

Layer* FFLayer::clone() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return new FFLayer(*this);
}

// Implementations for get_residual_weight_params()
const std::vector<std::vector<WeightParam>>& FFLayer::get_residual_weight_params() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  static const std::vector<std::vector<WeightParam>> empty_vec_2d;
  return empty_vec_2d;
}

std::vector<std::vector<WeightParam>>& FFLayer::get_residual_weight_params()
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  static std::vector<std::vector<WeightParam>> empty_vec_2d; // Non-const version
  return empty_vec_2d;
}
