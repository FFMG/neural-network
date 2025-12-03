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

FFLayer FFLayer::create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return FFLayer(0, 0, num_neurons_in_this_layer, num_neurons_in_next_layer, weight_decay, LayerType::Input, activation::method::linear, OptimiserType::None, 0.0);
}

FFLayer FFLayer::create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay, const FFLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, double dropout_rate)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return FFLayer(previous_layer.get_layer_index() + 1, previous_layer._number_output_neurons, num_neurons_in_this_layer, num_neurons_in_next_layer, weight_decay, LayerType::Hidden, activation, optimiser_type, dropout_rate);
}

FFLayer FFLayer::create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const FFLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return FFLayer(previous_layer.get_layer_index()+1, previous_layer._number_output_neurons, num_neurons_in_this_layer, 0, weight_decay, LayerType::Output, activation, optimiser_type, 0.0);
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
  const std::vector<std::vector<double>>& residual_output_values, // This parameter is no longer used
  std::vector<std::vector<HiddenState>>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t batch_size = previous_layer_inputs.size();
  const size_t N_prev = previous_layer.number_neurons();  // INPUT SIZE
  const size_t N_this = number_neurons();                 // OUTPUT SIZE

#if VALIDATE_DATA == 1
  // 1. Validate previous layer input shape
  for (size_t batch_index = 0; batch_index < batch_size; batch_index++)
  {
    if (previous_layer_inputs[batch_index].size() != N_prev)
    {
      Logger::error("previous_layer_inputs wrong shape");
      throw std::runtime_error("previous_layer_inputs wrong shape");
    }
  }

  // 3. Validate THIS layer's weights
  if (_weights.size() != N_prev)
  {
    Logger::panic("FFLayer::_weights row count != N_prev");
  }

  for (size_t i = 0; i < N_prev; i++)
  {
    if (_weights[i].size() != N_this)
    {
      Logger::panic("FFLayer::_weights column count != N_this");
    }
  }

  if (!_bias_weights.empty() && _bias_weights.size() != N_this)
  {
    Logger::panic("FFLayer::_bias_weights size != N_this");
  }
#endif

  // Initialize with 0.0
  std::vector<std::vector<double>> output_matrix(batch_size, std::vector<double>(N_this, 0.0));
  for (size_t b = 0; b < batch_size; b++)
  {
    const auto& prev_row = previous_layer_inputs[b];
    auto& output_row = output_matrix[b];

    // -------------------------------------------------------
    // PHASE 1: Initialize Sums (Bias & Constants)
    // -------------------------------------------------------
    for (size_t j = 0; j < N_this; j++)
    {
      if (!_bias_weights.empty()) 
      {
        output_row[j] = _bias_weights[j].get_value(); // Additive Bias
      }
    }

    // -------------------------------------------------------
    // PHASE 2: Matrix Multiplication (The Fast Way)
    // Iterate Inputs (i) -> Outputs (j) to hit memory linearly
    // -------------------------------------------------------
    for (size_t i = 0; i < N_prev; i++)
    {
      double input_val = prev_row[i];
      if (input_val == 0.0)
      {
        continue; // Optimization for sparse inputs/ReLU
      }

      const auto& weight_row = _weights[i]; // Vector of size N_this

      // Compiler can vectorise this inner loop easily (SIMD)
      for (size_t j = 0; j < N_this; j++)
      {
        output_row[j] += input_val * weight_row[j].get_value();
      }
    }

    // -------------------------------------------------------
    // PHASE 3: Activation & Post-Processing
    // -------------------------------------------------------
    for (size_t j = 0; j < N_this; j++)
    {
      const auto& neuron = get_neuron((unsigned)j);
      double sum = output_row[j];

      // Activation
      double output = get_activation().activate(sum);

      // Dropout
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

      Logger::trace([&] {
          return Logger::factory("Layer ", _layer_index, " Neuron ", j, ": sum=", sum, ", output=", output, ", saved_pre_activation_sum=", sum);
        });

      // Save Hidden State
      if (!hidden_states.empty()) {
        hidden_states[b][j].set_pre_activation_sum(sum);
      }
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
  MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
  switch (error_calculation_type)
  {
  case ErrorCalculation::type::none:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::huber_loss:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::mae:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::mse:
    return calculate_mse_error_deltas(deltas, target_outputs, given_outputs);

  case ErrorCalculation::type::rmse:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::nrmse:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::mape:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::wape:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::smape:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::directional_accuracy:
    Logger::panic("ErrorCalculation type is not supported!");

  case ErrorCalculation::type::bce_loss:
    return calculate_bce_error_deltas(deltas, target_outputs, given_outputs);
  }

  Logger::panic("Unknown ErrorCalculation type!");
}

void FFLayer::calculate_bce_error_deltas(
  std::vector<std::vector<double>>& deltas,
  const std::vector<std::vector<double>>& target_outputs,
  const std::vector<std::vector<double>>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t B = target_outputs.size();
  const size_t N_total = number_neurons();

  // Small epsilon to prevent log(0) and division by zero
  const double epsilon = 1e-8; 

  size_t out_col = 0;
  for (unsigned neuron_index = 0; neuron_index < static_cast<unsigned>(N_total); ++neuron_index)
  {
    for (size_t b = 0; b < B; ++b)
    {
      const double target = target_outputs[b][out_col];
      const double output = given_outputs[b][out_col];

      // Derivative of Binary Cross-Entropy Loss w.r.t. output
      // For sigmoid activation, dL/d_output = (output - target)
      // This formula is often simplified because d_output/d_sum for sigmoid is output * (1 - output)
      // and dL/d_sum = (output - target) when combined with sigmoid's derivative.
      // So, the delta here is actually dL/d_output, not dL/d_sum directly.
      // But since we're chaining it with activate_derivative(pre_activation_sum) later,
      // and for sigmoid activate_derivative(sum) is output * (1-output)
      // we need the term (output - target)
      deltas[b][out_col] = (output - target);

      Logger::trace([&] {
        return Logger::factory("Layer ", _layer_index, " BCE Delta: b=", b, ", neuron=", neuron_index, ", target=", target, ", output=", output, ", delta=", deltas[b][out_col]);
      });
    }
    out_col++;
  }
}

void FFLayer::calculate_mse_error_deltas(
  std::vector<std::vector<double>>& deltas,
  const std::vector<std::vector<double>>& target_outputs,
  const std::vector<std::vector<double>>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t B = target_outputs.size();   // batch size
  const size_t N_total = number_neurons();  // includes bias if present

  size_t out_col = 0;
  for (unsigned neuron_index = 0; neuron_index < static_cast<unsigned>(N_total); ++neuron_index)
  {
    for (size_t b = 0; b < B; ++b)
    {
      const double target = target_outputs[b][out_col];
      const double output = given_outputs[b][out_col];
      deltas[b][out_col] = output - target;
    }
    out_col++;
  }
}

std::vector<std::vector<double>> FFLayer::calculate_output_gradients(
  const std::vector<std::vector<double>>& target_outputs,
  const std::vector<std::vector<double>>& given_outputs,
  const std::vector<std::vector<HiddenState>>& hidden_states,
  double gradient_clip_threshold,
  ErrorCalculation::type error_calculation_type) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t B = target_outputs.size();   // batch size
  const size_t N_total = number_neurons();  // includes bias if present

  // Validate given and target shapes
  assert(B == given_outputs.size());
  for (size_t b = 0; b < B; ++b)
  {
    assert(target_outputs[b].size() == N_total);
    assert(given_outputs[b].size() == N_total);
  }

  // Allocate gradient matrix: only for NON-BIAS neurons
  std::vector<std::vector<double>> gradients(B, std::vector<double>(N_total, 0.0));

  // Get deltas from the calculate_error function
  std::vector<std::vector<double>> deltas(B, std::vector<double>(N_total, 0.0));
  calculate_error_deltas(deltas, target_outputs, given_outputs, error_calculation_type);

  // Map output index (no bias) to neuron index (with bias)
  size_t out_col = 0;  // index in target/given arrays

  for (unsigned neuron_index = 0; neuron_index < static_cast<unsigned>(N_total); ++neuron_index)
  {
    for (size_t b = 0; b < B; ++b)
    {
      double delta = deltas[b][out_col]; // Use delta from calculate_error

      // derivative is applied to the *pre-activation sum*
      double grad = delta;

      Logger::trace([&] {
        return Logger::factory("Layer ", _layer_index, " Output Grad: b=", b, ", neuron=", neuron_index, ", pre_clip_grad=", grad);
      });

      grad = clip_gradient(grad, gradient_clip_threshold);

      Logger::trace([&] {
        return Logger::factory("Layer ", _layer_index, " Output Grad: b=", b, ", neuron=", neuron_index, ", post_clip_grad=", grad);
      });

      if (!std::isfinite(grad))
      {
        Logger::error("Error while calculating output gradients.");
        throw std::invalid_argument("Error while calculating output gradients.");
      }

      gradients[b][out_col] = grad;
    }

    out_col++;  // move to next non-bias neuron output column
  }

  return gradients;
}

std::vector<std::vector<double>> FFLayer::calculate_hidden_gradients(
  const BaseLayer& next_layer,
  const std::vector<std::vector<double>>& next_grad_matrix,
  const std::vector<std::vector<double>>& output_matrix,
  const std::vector<std::vector<HiddenState>>& hidden_states,
  double gradient_clip_threshold) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  const size_t B = next_grad_matrix.size();   // batch size
  const size_t N_this = number_neurons();
  const size_t N_next_total = next_layer.number_neurons();

  // ---- Count NON-BIAS neurons in next layer ----
  size_t N_next_no_bias = 0;
  std::vector<size_t> next_nonbias_index;
  next_nonbias_index.reserve(N_next_total);

  for (size_t j = 0; j < N_next_total; ++j)
  {
    next_nonbias_index.push_back(j);
    ++N_next_no_bias;
  }

  // ---- Validate shapes ----
  for (size_t b = 0; b < B; b++)
  {
    if (next_grad_matrix[b].size() != N_next_no_bias)
    {
      Logger::panic("next_grad_matrix wrong shape");
    }

    if (output_matrix[b].size() != N_this)
    {
      Logger::panic("output_matrix wrong shape");
    }
  }

  // ---- Allocate gradient output (same size as this layer neurons) ----
  std::vector<std::vector<double>> grad_matrix(B, std::vector<double>(N_this, 0.0));

  // ---- Compute hidden layer gradients ----
  //
  // grad[b][i] = E_j ( next_grad[b][j] * W(i->j) )
  // grad[b][i] *= activation_derivative(output[b][i])
  //
  for (size_t b = 0; b < B; b++)
  {
    for (unsigned i = 0; i < static_cast<unsigned>(N_this); i++)
    {
      double weighted_sum = 0.0;
      Logger::trace([&] {
        return Logger::factory("Layer ", _layer_index, " Hidden Grad: b=", b, ", neuron=", i, ", initial_weighted_sum=", weighted_sum);
      });

      // ---- Sum over NON-BIAS next-layer neurons ----
      for (size_t k = 0; k < N_next_no_bias; k++)
      {
        unsigned j = static_cast<unsigned>(next_nonbias_index[k]);  // actual next neuron index

        double w = next_layer.get_weight_param(i, j).get_value();
        weighted_sum += next_grad_matrix[b][k] * w;
      }
      Logger::trace([&] {
        return Logger::factory("Layer ", _layer_index, " Hidden Grad: b=", b, ", neuron=", i, ", after_sum_weighted_sum=", weighted_sum);
      });

      double deriv = get_activation().activate_derivative(hidden_states[b][i].get_pre_activation_sum());
      Logger::trace([&] {
        return Logger::factory("Layer ", _layer_index, " Hidden Grad: b=", b, ", neuron=", i, ", deriv=", deriv);
      });

      double g = weighted_sum * deriv;
      Logger::trace([&] {
        return Logger::factory("Layer ", _layer_index, " Hidden Grad: b=", b, ", neuron=", i, ", pre_clip_grad=", g);
      });

      g = clip_gradient(g, gradient_clip_threshold);
      Logger::trace([&] {
        return Logger::factory("Layer ", _layer_index, " Hidden Grad: b=", b, ", neuron=", i, ", post_clip_grad=", g);
      });

      if (!std::isfinite(g))
        throw std::runtime_error("Hidden gradient is not finite.");

      grad_matrix[b][i] = g;
    }
  }

  return grad_matrix;
}

void FFLayer::apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  if (!std::isfinite(gradient))
  {
    Logger::panic("Error while calculating input weigh gradient it invalid.");
  }
  
  if (clipping_scale < 0.0)
  {
    // If clipping scale is negative, we clip the gradient to a fixed range
    // This is useful for debugging or when we want to ensure gradients are not too large.
    Logger::warning("Clipping gradient to a fixed range.");
  }

  auto clipped_gradient = clipping_scale <= 0.0 ? clip_gradient(gradient, gradient_clip_threshold) : gradient * clipping_scale;
  
  // Apply L2 regularization (weight decay) if not bias and weight_decay is set
  double final_gradient = clipped_gradient;
  if (!is_bias && weight_param.get_weight_decay() > 0.0)
  {
    final_gradient += weight_param.get_weight_decay() * weight_param.get_value();
  }

  // Basic update (equivalent to SGD without momentum/adaptive rates)
  double new_weight = weight_param.get_value() - learning_rate * final_gradient;
  weight_param.set_raw_gradient(clipped_gradient); // Store the raw (clipped) gradient
  weight_param.set_value(new_weight);
}

// TODO the clip threshold used is not the one we have in the config/options!
double FFLayer::clip_gradient(double gradient, double gradient_clip_threshold)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  // TODO remove this hardcoded value.
  // constexpr double gradient_clip_threshold = 1.0;
  if (!std::isfinite(gradient))
  {
    Logger::error("Gradient is not finite.");
    throw std::invalid_argument("Gradient is not finite.");
  }

  if (gradient > gradient_clip_threshold)
  {
    gradient = gradient_clip_threshold;
  }
  else if (gradient < -gradient_clip_threshold)
  {
    gradient = -gradient_clip_threshold;
  }
  return gradient;
}

unsigned FFLayer::get_layer_index() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _layer_index;
}

BaseLayer::LayerType FFLayer::layer_type() const {
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _layer_type;
}

unsigned FFLayer::number_input_neurons(bool add_bias) const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _number_input_neurons + (add_bias ? 1 : 0);
}

const std::vector<std::vector<WeightParam>>& FFLayer::get_weight_params() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _weights;
}

const std::vector<WeightParam>& FFLayer::get_bias_weight_params() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _bias_weights;
}

const activation& FFLayer::get_activation() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _activation;
}

const WeightParam& FFLayer::get_weight_param(unsigned input_neuron_number, unsigned neuron_index) const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  // TODO Valiate
  return _weights[input_neuron_number][neuron_index];
}

WeightParam& FFLayer::get_weight_param(unsigned input_neuron_number, unsigned neuron_index)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  // TODO Valiate
  return _weights[input_neuron_number][neuron_index];
}

WeightParam& FFLayer::get_bias_weight_param(unsigned neuron_index)
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  // TODO Valiate
  return _bias_weights[neuron_index];
}

unsigned FFLayer::get_number_output_neurons() const
{
  MYODDWEB_PROFILE_FUNCTION("FFLayer");
  return _number_output_neurons;
}