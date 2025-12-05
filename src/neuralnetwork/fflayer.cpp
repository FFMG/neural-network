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
  unsigned num_neurons_in_next_layer, 
  double weight_decay,
  LayerType layer_type, 
  const activation::method& activation_method,
  const OptimiserType& optimiser_type, 
  double dropout_rate
  ) :
  _layer_index(layer_index),
  _number_input_neurons(num_neurons_in_previous_layer),
  _number_output_neurons(num_neurons_in_this_layer),
  _layer_type(layer_type),
  _optimiser_type(optimiser_type),
  _activation(activation_method)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (num_neurons_in_this_layer == 0) 
  {
    Logger::warning("Warning: Creating a layer with 0 neurons.");
    throw std::invalid_argument("Warning: Creating a layer with 0 neurons.");
  }
  if (layer_type != LayerType::Input && num_neurons_in_previous_layer == 0) 
  {
    Logger::warning("Warning: Non-input layer created with 0 inputs.");
  }

  // create the weights
  resize_weights(_activation, _number_input_neurons, _number_output_neurons, weight_decay);

  // We have a new layer, now fill it with neurons, and add a bias neuron in each layer.
  _neurons.reserve(_number_output_neurons+1); // for bias
  for (unsigned neuron_number = 0; neuron_number < _number_output_neurons; ++neuron_number)
  {
    // force the bias node's output to 1.0
    auto neuron = Neuron(
      layer_type == LayerType::Input ? 0 : num_neurons_in_previous_layer,  //  previous
      _number_output_neurons,         //  current 
      layer_type == LayerType::Output ? 0 : num_neurons_in_next_layer+1,      //  next
      neuron_number, 
      dropout_rate == 0.0 ? Neuron::Type::Normal : Neuron::Type::Dropout,
      dropout_rate);
    _neurons.emplace_back(neuron);
  }
}

FFLayer::FFLayer(const FFLayer& src) noexcept :
  _layer_index(src._layer_index),
  _neurons(src._neurons),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _layer_type(src._layer_type),
  _weights(src._weights),
  _bias_weights(src._bias_weights),
  _optimiser_type(src._optimiser_type),
  _activation(src._activation)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(
  unsigned layer_index,
  const std::vector<Neuron>& neurons,
  unsigned number_input_neurons,
  LayerType layer_type,
  OptimiserType optimiser_type,
  const activation::method& activation_method,
  const std::vector<std::vector<WeightParam>>& weights,
  const std::vector<WeightParam>& bias_weights
) : 
  _layer_index(layer_index),
  _neurons(neurons),
  _number_input_neurons(number_input_neurons),
  _number_output_neurons( static_cast<unsigned>(neurons.size())),
  _layer_type(layer_type),
  _optimiser_type(optimiser_type),
  _activation(activation_method),
  _weights(weights),
  _bias_weights(bias_weights)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
}

FFLayer::FFLayer(FFLayer&& src) noexcept :
  _layer_index(src._layer_index),
  _neurons(std::move(src._neurons)),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _layer_type(src._layer_type),
  _weights(std::move(src._weights)),
  _bias_weights(std::move(src._bias_weights)),
  _optimiser_type(std::move(src._optimiser_type)),
  _activation(std::move(src._activation))
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  src._layer_index = 0;
  src._number_output_neurons = 0;
  src._number_input_neurons = 0;
  src._optimiser_type = OptimiserType::None;
}

FFLayer& FFLayer::operator=(const FFLayer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if(this != &src)
  {
    _layer_index = src._layer_index;
    _neurons = src._neurons;
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _weights = src._weights;
    _bias_weights = src._bias_weights;
    _optimiser_type = src._optimiser_type;
    _activation = src._activation;
  }
  return *this;
}

FFLayer& FFLayer::operator=(FFLayer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if(this != &src)
  {
    _layer_index = src._layer_index;
    _neurons = std::move(src._neurons);
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _weights = std::move(src._weights);
    _bias_weights = std::move(src._bias_weights);
    _optimiser_type = std::move(src._optimiser_type);
    _activation = std::move(src._activation);
   
    src._number_output_neurons = 0;
    src._number_input_neurons = 0;
    src._optimiser_type = OptimiserType::None;
  }
  return *this;
}

FFLayer::~FFLayer() = default;

void FFLayer::resize_weights(
  const activation& activation_method,
  unsigned number_input_neurons,
  unsigned number_output_neurons, 
  double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (has_bias())
  {
    _bias_weights.reserve(number_output_neurons);
    auto weights = activation_method.weight_initialization(number_output_neurons, 1);
    for (unsigned o = 0; o < number_output_neurons; ++o)
    {
      // bias has no weight decay.
      const auto& weight = weights[o];
      _bias_weights.emplace_back(WeightParam(weight, 0.0, 0.0, 0.0, 0.0));
    }
  }

  if (number_input_neurons == 0)
  {
    assert(_layer_type == LayerType::Input);
    return;
  }

  _weights.resize(number_input_neurons);
  for (unsigned i = 0; i < number_input_neurons; ++i)
  {
    auto weights = activation_method.weight_initialization(number_output_neurons, number_input_neurons);
    assert(weights.size() == number_output_neurons);
    _weights[i].reserve(number_output_neurons);
    for (unsigned o = 0; o < number_output_neurons; ++o)
    {
      const auto& weight = weights[o];
      _weights[i].emplace_back(WeightParam(weight, 0.0, 0.0, 0.0, weight_decay));
    }
  }
}

bool FFLayer::has_bias() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _has_bias_neuron;
}

unsigned FFLayer::number_neurons() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _number_output_neurons;
}

const std::vector<Neuron>& FFLayer::get_neurons() const noexcept
{ 
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _neurons;
}

std::vector<Neuron>& FFLayer::get_neurons() noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _neurons;
}

const Neuron& FFLayer::get_neuron(unsigned index) const 
{ 
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (index >= _neurons.size()) 
  {
    Logger::error("Index out of bounds in FFLayer::get_neuron.");
    throw std::out_of_range("Index out of bounds in FFLayer::get_neuron.");
  }
  return _neurons[index];
}

Neuron& FFLayer::get_neuron(unsigned index) 
{ 
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (index >= _neurons.size()) 
  {
    Logger::panic("Index out of bounds in FFLayer::get_neuron.");
  }
  return _neurons[index];
}

std::vector<std::vector<double>> FFLayer::calculate_forward_feed(
  const BaseLayer& previous_layer,
  const std::vector<std::vector<double>>& previous_layer_inputs,
  const std::vector<std::vector<double>>&, // residual_output_values is not used
  std::vector<std::vector<HiddenState>>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t batch_size = previous_layer_inputs.size();
  const size_t N_prev = previous_layer.number_neurons();
  const size_t N_this = number_neurons();

  std::vector<std::vector<double>> output_matrix(batch_size, std::vector<double>(N_this, 0.0));

  for (size_t b = 0; b < batch_size; b++)
  {
    const auto& prev_row = previous_layer_inputs[b];
    auto& output_row = output_matrix[b];
    std::vector<double> pre_activation_sums(N_this, 0.0);

    for (size_t j = 0; j < N_this; j++)
    {
      if (!_bias_weights.empty()) 
      {
        pre_activation_sums[j] = _bias_weights[j].get_value();
      }
    }

    for (size_t i = 0; i < N_prev; i++)
    {
      double input_val = prev_row[i];
      if (input_val == 0.0) continue;
      const auto& weight_row = _weights[i];
      for (size_t j = 0; j < N_this; j++)
      {
        pre_activation_sums[j] += input_val * weight_row[j].get_value();
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
      hidden_states[b][get_layer_index()].set_pre_activation_sums(pre_activation_sums);
      hidden_states[b][get_layer_index()].set_hidden_state_values(output_row);
    }
  }
  return output_matrix;
}

void FFLayer::calculate_error_deltas(
  std::vector<std::vector<double>>& deltas,
  const std::vector<std::vector<double>>& target_outputs,
  const std::vector<std::vector<double>>& given_outputs,
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
  std::vector<std::vector<double>>& deltas,
  const std::vector<std::vector<double>>& target_outputs,
  const std::vector<std::vector<double>>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t B = target_outputs.size();
  const size_t N_total = number_neurons();

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    for (size_t b = 0; b < B; ++b)
    {
      deltas[b][neuron_index] = given_outputs[b][neuron_index] - target_outputs[b][neuron_index];
    }
  }
}

void FFLayer::calculate_mse_error_deltas(
  std::vector<std::vector<double>>& deltas,
  const std::vector<std::vector<double>>& target_outputs,
  const std::vector<std::vector<double>>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t B = target_outputs.size();
  const size_t N_total = number_neurons();

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    for (size_t b = 0; b < B; ++b)
    {
      deltas[b][neuron_index] = given_outputs[b][neuron_index] - target_outputs[b][neuron_index];
    }
  }
}

std::vector<std::vector<double>> FFLayer::calculate_output_gradients(
  const std::vector<std::vector<double>>& target_outputs,
  const std::vector<std::vector<double>>& given_outputs,
  const std::vector<std::vector<HiddenState>>&,
  double gradient_clip_threshold,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t B = target_outputs.size();
  const size_t N_total = number_neurons();

  std::vector<std::vector<double>> gradients(B, std::vector<double>(N_total, 0.0));
  std::vector<std::vector<double>> deltas(B, std::vector<double>(N_total, 0.0));
  calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    for (size_t b = 0; b < B; ++b)
    {
      double grad = deltas[b][neuron_index];
      gradients[b][neuron_index] = clip_gradient(grad, gradient_clip_threshold);
    }
  }
  return gradients;
}

std::vector<std::vector<double>> FFLayer::calculate_hidden_gradients(
  const BaseLayer& next_layer,
  const std::vector<std::vector<double>>& next_grad_matrix,
  const std::vector<std::vector<double>>&,
  const std::vector<std::vector<HiddenState>>& hidden_states,
  double gradient_clip_threshold) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t B = next_grad_matrix.size();
  const size_t N_this = number_neurons();
  const size_t N_next = next_layer.number_neurons();

  std::vector<std::vector<double>> grad_matrix(B, std::vector<double>(N_this, 0.0));

  for (size_t b = 0; b < B; b++)
  {
    const auto& current_hidden_state = hidden_states[b][get_layer_index()];

    for (unsigned i = 0; i < N_this; i++)
    {
      double weighted_sum = 0.0;
      for (size_t j = 0; j < N_next; j++)
      {
        weighted_sum += next_grad_matrix[b][j] * next_layer.get_weight_param(i, j).get_value();
      }
      double deriv = get_activation().activate_derivative(current_hidden_state.get_pre_activation_sum_at_neuron(i));
      double g = weighted_sum * deriv;
      grad_matrix[b][i] = clip_gradient(g, gradient_clip_threshold);
    }
  }
  return grad_matrix;
}

void FFLayer::apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  auto clipped_gradient = clipping_scale <= 0.0 ? clip_gradient(gradient, gradient_clip_threshold) : gradient * clipping_scale;
  
  double final_gradient = clipped_gradient;
  if (!is_bias && weight_param.get_weight_decay() > 0.0)
  {
    final_gradient += weight_param.get_weight_decay() * weight_param.get_value();
  }

  double new_weight = weight_param.get_value() - learning_rate * final_gradient;
  weight_param.set_raw_gradient(clipped_gradient);
  weight_param.set_value(new_weight);
}

double FFLayer::clip_gradient(double gradient, double gradient_clip_threshold)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (!std::isfinite(gradient))
  {
    Logger::error("Gradient is not finite.");
    throw std::invalid_argument("Gradient is not finite.");
  }

  if (gradient > gradient_clip_threshold)
  {
    return gradient_clip_threshold;
  }
  if (gradient < -gradient_clip_threshold)
  {
    return -gradient_clip_threshold;
  }
  return gradient;
}

unsigned int FFLayer::get_layer_index() const noexcept
{
    return _layer_index;
}

BaseLayer::LayerType FFLayer::layer_type() const
{
    return _layer_type;
}

unsigned int FFLayer::number_input_neurons(bool add_bias) const noexcept
{
    return _number_input_neurons + (add_bias ? 1 : 0);
}

const std::vector<std::vector<WeightParam>>& FFLayer::get_weight_params() const
{
    return _weights;
}

const WeightParam& FFLayer::get_weight_param(unsigned int input_neuron_number, unsigned int neuron_index) const
{
    return _weights[input_neuron_number][neuron_index];
}

WeightParam& FFLayer::get_weight_param(unsigned int input_neuron_number, unsigned int neuron_index)
{
    return _weights[input_neuron_number][neuron_index];
}

const std::vector<WeightParam>& FFLayer::get_bias_weight_params() const
{
    return _bias_weights;
}

const activation& FFLayer::get_activation() const noexcept
{
    return _activation;
}

WeightParam& FFLayer::get_bias_weight_param(unsigned int neuron_index)
{
    return _bias_weights[neuron_index];
}

unsigned int FFLayer::get_number_output_neurons() const
{
    return _number_output_neurons;
}

const OptimiserType FFLayer::get_optimiser_type() const noexcept
{
    return _optimiser_type;
}

int FFLayer::residual_layer_number() const
{
    return -1;
}

BaseLayer* FFLayer::clone() const
{
    return new FFLayer(*this);
}

// Implementations for get_residual_weight_params()
const std::vector<std::vector<WeightParam>>& FFLayer::get_residual_weight_params() const
{
    static const std::vector<std::vector<WeightParam>> empty_vec_2d;
    return empty_vec_2d;
}

std::vector<std::vector<WeightParam>>& FFLayer::get_residual_weight_params()
{
    static std::vector<std::vector<WeightParam>> empty_vec_2d; // Non-const version
    return empty_vec_2d;
}

std::vector<WeightParam>& FFLayer::get_residual_weight_params(unsigned neuron_index)
{
    static std::vector<WeightParam> empty_vec_1d; // Non-const version
    return empty_vec_1d;
}
