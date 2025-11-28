#include "./libraries/instrumentor.h"
#include "layer.h"
#include "logger.h"

#include <iostream>
#include <numeric>

constexpr bool _has_bias_neuron = true;

Layer::Layer(LayerType layer_type) :
  _layer_index(0),
  _number_input_neurons(0),
  _number_output_neurons(0),
  _residual_layer_number(-1),
  _residual_projector(nullptr),
  _layer_type(layer_type),
  _optimiser_type(OptimiserType::None),
  _activation_method(activation::method::linear)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
}

Layer::Layer(
  unsigned layer_index,
  unsigned num_neurons_in_previous_layer, 
  unsigned num_neurons_in_this_layer, 
  unsigned num_neurons_in_next_layer, 
  double weight_decay,
  int residual_layer_number, 
  LayerType layer_type, 
  const activation::method& activation, 
  const OptimiserType& optimiser_type, 
  double dropout_rate) :
  _layer_index(layer_index),
  _number_input_neurons(num_neurons_in_previous_layer),
  _number_output_neurons(num_neurons_in_this_layer),
  _residual_layer_number(residual_layer_number),
  _residual_projector(nullptr),
  _layer_type(layer_type),
  _optimiser_type(optimiser_type),
  _activation_method(activation)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
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
  resize_weights(activation, _number_input_neurons, _number_output_neurons, weight_decay);

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
      activation,
      optimiser_type,
      dropout_rate == 0.0 ? Neuron::Type::Normal : Neuron::Type::Dropout,
      dropout_rate);
    _neurons.emplace_back(neuron);
  }
}

Layer::Layer(const Layer& src) noexcept :
  _layer_index(src._layer_index),
  _neurons(src._neurons),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _residual_layer_number(src._residual_layer_number),
  _residual_projector(nullptr),
  _layer_type(src._layer_type),
  _weights(src._weights),
  _bias_weights(src._bias_weights),
  _optimiser_type(src._optimiser_type),
  _activation_method(src._activation_method)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(src._residual_projector != nullptr)
  {
    _residual_projector = new ResidualProjector(*src._residual_projector);
  }
}

Layer::Layer(
  unsigned layer_index,
  const std::vector<Neuron>& neurons,
  unsigned number_input_neurons,
  int residual_layer_number,
  LayerType layer_type,
  OptimiserType optimiser_type,
  activation::method activation_method,
  const std::vector<std::vector<WeightParam>>& weights,
  const std::vector<WeightParam>& bias_weights,
  const std::vector<std::vector<WeightParam>>& residual_weights
) : 
  _layer_index(layer_index),
  _neurons(neurons),
  _number_input_neurons(number_input_neurons),
  _number_output_neurons( static_cast<unsigned>(neurons.size())),
  _residual_layer_number(residual_layer_number),
  _layer_type(layer_type),
  _activation_method(activation_method),
  _weights(weights),
  _bias_weights(bias_weights)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (!residual_weights.empty())
  {
    _residual_projector = new ResidualProjector(residual_weights);
  }
}

Layer::Layer(Layer&& src) noexcept :
  _layer_index(src._layer_index),
  _neurons(std::move(src._neurons)),
  _number_input_neurons(src._number_input_neurons),
  _number_output_neurons(src._number_output_neurons),
  _residual_layer_number(src._residual_layer_number),
  _residual_projector(src._residual_projector),
  _layer_type(src._layer_type),
  _weights(std::move(src._weights)),
  _bias_weights(std::move(src._bias_weights)),
  _optimiser_type(src._optimiser_type),
  _activation_method(src._activation_method)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  src._layer_index = 0;
  src._number_output_neurons = 0;
  src._number_input_neurons = 0;
  src._residual_layer_number = -1;
  src._residual_projector = nullptr;
  src._optimiser_type = OptimiserType::None;
}

Layer& Layer::operator=(const Layer& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(this != &src)
  {
    clean();
    _layer_index = src._layer_index;
    _neurons = src._neurons;
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _residual_layer_number = src._residual_layer_number;
    if (src._residual_projector != nullptr)
    {
      _residual_projector = new ResidualProjector(*src._residual_projector);
    }
    _layer_type = src._layer_type;
    _weights = src._weights;
    _bias_weights = src._bias_weights;
    _optimiser_type = src._optimiser_type;
    _activation_method = src._activation_method;
  }
  return *this;
}

Layer& Layer::operator=(Layer&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(this != &src)
  {
    clean();
    _layer_index = src._layer_index;
    _neurons = std::move(src._neurons);
    _number_input_neurons = src._number_input_neurons;
    _number_output_neurons = src._number_output_neurons;
    _layer_type = src._layer_type;
    _residual_layer_number = src._residual_layer_number;
    _residual_projector = src._residual_projector;
    _weights = std::move(src._weights);
    _bias_weights = std::move(src._bias_weights);
    _optimiser_type = src._optimiser_type;
    _activation_method = src._activation_method;
   
    src._number_output_neurons = 0;
    src._number_input_neurons = 0;
    src._residual_layer_number = -1;
    src._residual_projector = nullptr;
    src._optimiser_type = OptimiserType::None;
    src._activation_method = activation(activation::method::linear);
  }
  return *this;
}

Layer::~Layer()
{
  clean();
}

void Layer::clean()
{
  delete _residual_projector;
  _residual_projector = nullptr;
}

void Layer::resize_weights(
  const activation& activation_method,
  unsigned number_input_neurons,
  unsigned number_output_neurons, 
  double weight_decay)
{
  if (has_bias())
  {
    _bias_weights.reserve(number_output_neurons);
    auto weights = activation_method.weight_initialization(number_output_neurons, number_input_neurons);
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
    // TODO the activation function is using the wrong variable names here.
    //      we should use input and output in that function and return exctly number_outputs_neurons weights.
    //      weight_initialization( ... ) parm names are just confusing
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

void Layer::move_residual_projector(ResidualProjector* residual_projector)
{
  if(residual_projector != _residual_projector)
  {
    delete _residual_projector;
    _residual_projector = residual_projector;
  }
}

bool Layer::has_bias() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _has_bias_neuron;
}

unsigned Layer::number_neurons() const noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _number_output_neurons;
}

unsigned Layer::number_neurons_with_bias() const noexcept
{
  // use this function in case we do not want to have bias.
  // so we do not adjust number_neurons()
  return number_neurons() + (has_bias() ? 1 : 0);
}


Layer Layer::create_input_layer(const std::vector<Neuron>& neurons, double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (neurons.size() <= 1) 
  {
    Logger::error("Creating a layer with 1 neurons, (bias is needed).");
    throw std::invalid_argument("Warning: Creating a layer with 1 neurons, (bias is needed).");
  }
  auto layer = Layer(LayerType::Input);
  layer._layer_index = 0;
  layer._number_input_neurons = 0;
  layer._number_output_neurons = static_cast<unsigned>(neurons.size()) -1; // remove bias
  layer._neurons = neurons;
  layer._residual_layer_number = -1;
  return layer;
}

Layer Layer::create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(0, 0, num_neurons_in_this_layer, num_neurons_in_next_layer, weight_decay, -1, LayerType::Input, activation::method::linear, OptimiserType::None, 0.0);
}

Layer Layer::create_hidden_layer(
  unsigned layer_index, 
  const std::vector<Neuron>& neurons, 
  unsigned num_neurons_in_previous_layer, 
  double weight_decay,
  int residual_layer_number, 
  const std::vector<std::vector<WeightParam>>& residual_weight_params)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (neurons.size() <= 1) 
  {
    Logger::error("Creating a layer with 1 neurons, (bias is needed).");
    throw std::invalid_argument("Warning: Creating a layer with 1 neurons, (bias is needed).");
  }
  auto layer = Layer(LayerType::Hidden);
  layer._layer_index = layer_index;
  layer._number_input_neurons = num_neurons_in_previous_layer;
  layer._number_output_neurons = static_cast<unsigned>(neurons.size()) -1; // remove bias
  layer._neurons = neurons;
  layer._residual_layer_number = residual_layer_number;
  if(residual_weight_params.size() > 0 )
  {
    layer._residual_projector = new Layer::ResidualProjector(residual_weight_params);
  }
  return layer;
}

Layer Layer::create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number, double dropout_rate)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(previous_layer.get_layer_index() + 1, previous_layer._number_output_neurons, num_neurons_in_this_layer, num_neurons_in_next_layer, weight_decay, residual_layer_number, LayerType::Hidden, activation, optimiser_type, dropout_rate);
}

Layer Layer::create_output_layer(unsigned layer_index, const std::vector<Neuron>& neurons, double weight_decay, unsigned num_neurons_in_previous_layer, int residual_layer_number, const std::vector<std::vector<WeightParam>>& residual_weight_params)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (neurons.size() <= 1) 
  {
    Logger::error("Creating a layer with 1 neurons, (bias is needed).");
    throw std::invalid_argument("Warning: Creating a layer with 1 neurons, (bias is needed).");
  }
  auto layer = Layer(LayerType::Output);
  layer._layer_index = layer_index;
  layer._number_input_neurons = num_neurons_in_previous_layer;
  layer._number_output_neurons = static_cast<unsigned>(neurons.size()) -1; // remove bias
  layer._neurons = neurons;
  layer._residual_layer_number = residual_layer_number;
  if(residual_weight_params.size() > 0 )
  {
    layer._residual_projector = new Layer::ResidualProjector(residual_weight_params);
  }
  return layer;
}

Layer Layer::create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, int residual_layer_number)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return Layer(previous_layer.get_layer_index()+1, previous_layer._number_output_neurons, num_neurons_in_this_layer, 0, weight_decay, residual_layer_number, LayerType::Output, activation, optimiser_type, 0.0);
}

const std::vector<Neuron>& Layer::get_neurons() const noexcept
{ 
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _neurons;
}

std::vector<Neuron>& Layer::get_neurons() noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  return _neurons;
}

const Neuron& Layer::get_neuron(unsigned index) const 
{ 
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (index >= _neurons.size()) 
  {
    Logger::error("Index out of bounds in Layer::get_neuron.");
    throw std::out_of_range("Index out of bounds in Layer::get_neuron.");
  }
  return _neurons[index];
}

Neuron& Layer::get_neuron(unsigned index) 
{ 
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (index >= _neurons.size()) 
  {
    Logger::error("Index out of bounds in Layer::get_neuron.");
    throw std::out_of_range("Index out of bounds in Layer::get_neuron.");
  }
  return _neurons[index];
}

std::vector<std::vector<double>> Layer::project_residual_output_values(const std::vector<std::vector<double>>& residual_layer_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (residual_layer_outputs.empty())
  {
    return {};
  }

  std::vector<std::vector<double>> projected;
  projected.reserve(residual_layer_outputs.size());
  for (const auto& batch : residual_layer_outputs)
  {
    projected.emplace_back(project_residual_output_values(batch));
  }
  return projected;
}

std::vector<double> Layer::project_residual_output_values(const std::vector<double>& residual_layer_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(nullptr == _residual_projector)
  {
    return {};
  }
  return _residual_projector->project(residual_layer_outputs);
}

WeightParam& Layer::residual_weight_param(unsigned residual_source_index, unsigned target_neuron_index)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(nullptr == _residual_projector)
  {
    Logger::error("Trying to get residual weights for a layer that does not have any!");
    throw std::invalid_argument("Trying to get residual weights for a layer that does not have any!");
  }
  return _residual_projector->get_weight_params(residual_source_index, target_neuron_index);
}

const std::vector<std::vector<WeightParam>>& Layer::residual_weight_params() const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if(nullptr == _residual_projector)
  {
    Logger::error("Trying to get residual weights for a layer that does not have any!");
    throw std::invalid_argument("Trying to get residual weights for a layer that does not have any!");
  }
  return _residual_projector->get_weight_params();
}

std::vector<std::vector<double>> Layer::calculate_forward_feed(
  const Layer& previous_layer,
  const std::vector<std::vector<double>>& previous_layer_inputs,
  const std::vector<std::vector<double>>& residual_output_values,
  std::vector<std::vector<HiddenState>>& hidden_states,
  bool is_training) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const size_t batch_size = previous_layer_inputs.size();
  const size_t N_prev = previous_layer.number_neurons();  // INPUT SIZE
  const size_t N_this = number_neurons();                 // OUTPUT SIZE

  const auto projected_residual_output_values = project_residual_output_values(residual_output_values);

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

  // 2. Validate residuals
  if (!residual_output_values.empty())
  {
    if (residual_output_values.size() != batch_size)
    {
      Logger::error("residual_output_values wrong batch size");
      throw std::runtime_error("residual_output_values wrong batch size");
    }

    for (size_t b = 0; b < batch_size; b++)
    {
      if (!projected_residual_output_values[b].empty())
      {
        if (projected_residual_output_values[b].size() != N_this)
        {
          Logger::error("residual_output_values row wrong size");
          throw std::runtime_error("residual_output_values row wrong size");
        }
      }
    }
  }

  // 3. Validate THIS layer's weights
  if (_weights.size() != N_prev)
  {
    Logger::panic("Layer::_weights row count != N_prev");
  }

  for (size_t i = 0; i < N_prev; i++)
  {
    if (_weights[i].size() != N_this)
    {
      Logger::panic("Layer::_weights column count != N_this");
    }
  }

  if (!_bias_weights.empty() && _bias_weights.size() != N_this)
  {
    Logger::panic("Layer::_bias_weights size != N_this");
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
      const auto& neuron = get_neuron((unsigned)j);
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

      // Residual
      if (!projected_residual_output_values.empty() && !projected_residual_output_values[b].empty())
      {
        sum += projected_residual_output_values[b][j];
      }


      // Activation
      double output = neuron.get_activation_method().activate(sum);

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

      // Save Hidden State
      if (!hidden_states[b].empty()) {
        hidden_states[b][j].queue_output(output);
      }
    }
  }

  return output_matrix;
}

// forward for entire layer, batch-aware, matrix-based
std::vector<std::vector<double>> Layer::calculate_output_gradients(
  const std::vector<std::vector<double>>& target_outputs,
  const std::vector<std::vector<double>>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
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

  // Map output index (no bias) to neuron index (with bias)
  size_t out_col = 0;  // index in target/given arrays

  for (unsigned neuron_index = 0; neuron_index < static_cast<unsigned>(N_total); ++neuron_index)
  {
    const auto& neuron = get_neuron(neuron_index);
    for (size_t b = 0; b < B; ++b)
    {
      const double target = target_outputs[b][out_col];
      const double output = given_outputs[b][out_col];

      double delta = output - target;

      // derivative is applied to the *post-activation output*
      double grad = delta * neuron.get_activation_method().activate_derivative(output);

      grad = clip_gradient(grad);

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

std::vector<std::vector<double>> Layer::calculate_hidden_gradients(
  const Layer& next_layer,
  const std::vector<std::vector<double>>& next_grad_matrix,
  const std::vector<std::vector<double>>& output_matrix) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
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
      const auto& neuron = get_neuron(i);

      double weighted_sum = 0.0;

      // ---- Sum over NON-BIAS next-layer neurons ----
      for (size_t k = 0; k < N_next_no_bias; k++)
      {
        size_t j = next_nonbias_index[k];  // actual next neuron index

        double w = next_layer.get_weight_param(i, j).get_value();
        weighted_sum += next_grad_matrix[b][k] * w;
      }

      // ---- Multiply by activation derivative ----
      double deriv = neuron.get_activation_method().activate_derivative(output_matrix[b][i]);

      double g = weighted_sum * deriv;

      g = clip_gradient(g);

      if (!std::isfinite(g))
        throw std::runtime_error("Hidden gradient is not finite.");

      grad_matrix[b][i] = g;
    }
  }

  return grad_matrix;
}

void Layer::apply_nadam_update(
  WeightParam& weight_param,
  double raw_gradient,
  double learning_rate,
  double beta1,
  double beta2,
  double epsilon
)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  // Update timestep
  weight_param.increment_timestep();
  const auto& time_step = weight_param.get_timestep();

  // These moment estimate updates are identical to Adam
  weight_param.set_first_moment_estimate(beta1 * weight_param.get_first_moment_estimate() + (1.0 - beta1) * raw_gradient);
  weight_param.set_second_moment_estimate(beta2 * weight_param.get_second_moment_estimate() + (1.0 - beta2) * (raw_gradient * raw_gradient));

  double m_hat = weight_param.get_first_moment_estimate() / (1.0 - std::pow(beta1, time_step));
  double v_hat = weight_param.get_second_moment_estimate() / (1.0 - std::pow(beta2, time_step));

  // Nadam's key difference:
  // It combines the momentum from the historical gradient (m_hat) with the
  // momentum from the current gradient.
  double corrected_gradient = (beta1 * m_hat) + ((1.0 - beta1) * raw_gradient) / (1.0 - std::pow(beta1, time_step));

  // The denominator is the same as Adam's
  double weight_update = learning_rate * (corrected_gradient / (std::sqrt(v_hat) + epsilon));

  // Apply the final update (No decoupled weight decay)
  double new_weight = weight_param.get_value() - weight_update;

  weight_param.set_value(new_weight);
  weight_param.set_raw_gradient(raw_gradient);
}

void Layer::apply_nadamw_update(
  WeightParam& weight_param,
  double raw_gradient,
  double learning_rate,
  double beta1,
  double beta2,
  double epsilon,
  bool is_bias
)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  // 1. Increment timestep
  weight_param.increment_timestep();
  const long long time_step = weight_param.get_timestep();

  // 2. Update biased moment estimates
  double first_moment = beta1 * weight_param.get_first_moment_estimate()
    + (1.0 - beta1) * raw_gradient;
  double second_moment = beta2 * weight_param.get_second_moment_estimate()
    + (1.0 - beta2) * (raw_gradient * raw_gradient);

  weight_param.set_first_moment_estimate(first_moment);
  weight_param.set_second_moment_estimate(second_moment);

  // 3. Bias corrections
  double m_hat = first_moment / (1.0 - std::pow(beta1, time_step));
  double v_hat = second_moment / (1.0 - std::pow(beta2, time_step));

  // 4. NAdam momentum blend (with bias correction for gradient term)
  double m_nadam = beta1 * m_hat
    + ((1.0 - beta1) * raw_gradient) / (1.0 - std::pow(beta1, time_step));

  // 5. Adaptive step
  double step = m_nadam / (std::sqrt(v_hat) + epsilon);

  // 6. Decoupled weight decay
  double w = weight_param.get_value();
  if (!is_bias)
  {
    w *= (1.0 - learning_rate * weight_param.get_weight_decay());
  }

  // 7. Apply update
  w -= learning_rate * step;
  weight_param.set_value(w);
  weight_param.set_raw_gradient(raw_gradient);
}

void Layer::apply_adamw_update(
  WeightParam& weight_param,
  double raw_gradient,           // unclipped, averaged over batch
  double learning_rate,
  double beta1,
  double beta2,
  double epsilon
)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  // Update timestep
  weight_param.increment_timestep();
  const auto& time_step = weight_param.get_timestep();

  // Update biased first and second moment estimates
  weight_param.set_first_moment_estimate(beta1 * weight_param.get_first_moment_estimate() + (1.0 - beta1) * raw_gradient);
  weight_param.set_second_moment_estimate(beta2 * weight_param.get_second_moment_estimate() + (1.0 - beta2) * (raw_gradient * raw_gradient));

  // Compute bias-corrected moments
  double first_moment_estimate = weight_param.get_first_moment_estimate();
  double m_hat = first_moment_estimate / (1.0 - std::pow(beta1, time_step));

  auto second_moment_estimate = weight_param.get_second_moment_estimate();
  double v_hat = second_moment_estimate / (1.0 - std::pow(beta2, time_step));

  // AdamW update rule
  double weight_update = learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));

  // Decoupled weight decay
  auto new_weight = weight_param.get_value();
  new_weight *= (1.0 - learning_rate * weight_param.get_weight_decay());

  // Apply update
  new_weight -= weight_update;

  weight_param.set_value(new_weight);
  weight_param.set_raw_gradient(raw_gradient);
}

void Layer::apply_adam_update(WeightParam& weight_param, double raw_gradient, double learning_rate, double beta1, double beta2, double epsilon, bool is_bias)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  // Update timestep
  weight_param.increment_timestep();
  const auto& time_step = weight_param.get_timestep();

  // Update moments
  double m = beta1 * weight_param.get_first_moment_estimate() + (1.0 - beta1) * raw_gradient;
  double v = beta2 * weight_param.get_second_moment_estimate() + (1.0 - beta2) * raw_gradient * raw_gradient;

  weight_param.set_first_moment_estimate(m);
  weight_param.set_second_moment_estimate(v);

  double m_hat = m / (1.0 - std::pow(beta1, time_step));
  double v_hat = v / (1.0 - std::pow(beta2, time_step));

  double adam_update = learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

  // Apply decoupled weight decay (skip if bias)
  double decayed_weight = is_bias ? weight_param.get_value()
    : weight_param.get_value() * (1.0 - learning_rate * weight_param.get_weight_decay());

  double new_weight = decayed_weight - adam_update;

  weight_param.set_value(new_weight);
  weight_param.set_raw_gradient(raw_gradient);
}

void Layer::apply_none_update(WeightParam& weight_param, double raw_gradient, double learning_rate)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  double new_weight = weight_param.get_value() - learning_rate * raw_gradient;
  weight_param.set_raw_gradient(raw_gradient);
  weight_param.set_value(new_weight);
}

void Layer::apply_sgd_update(WeightParam& weight_param, double raw_gradient, double learning_rate, double momentum, bool is_bias)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  double g = raw_gradient;

  // --------------------------
  // 1. Apply weight decay (L2)
  // --------------------------
  if (!is_bias) 
  {
    g += weight_param.get_weight_decay() * weight_param.get_value();
  }

  // --------------------------
  // 2. Momentum update
  // --------------------------
  double prev_v = weight_param.get_velocity();
  double v = momentum * prev_v - learning_rate * g;

  // --------------------------
  // 3. Weight update
  // --------------------------
  double new_weight = weight_param.get_value() + v;

  weight_param.set_velocity(v);
  weight_param.set_raw_gradient(raw_gradient); // store only raw
  weight_param.set_value(new_weight);
}

void Layer::apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  if (!std::isfinite(gradient))
  {
    Logger::panic("Error while calculating input weigh gradient it invalid.");
  }
  auto old_velocity = weight_param.get_velocity();
  if (!std::isfinite(old_velocity))
  {
    Logger::panic("Error while calculating input weigh old velocity is invalid.");
  }

  if (clipping_scale < 0.0)
  {
    // If clipping scale is negative, we clip the gradient to a fixed range
    // This is useful for debugging or when we want to ensure gradients are not too large.
    Logger::warning("Clipping gradient to a fixed range.");
  }

  auto clipped_gradient = clipping_scale <= 0.0 ? clip_gradient(gradient) : gradient * clipping_scale;
  switch (_optimiser_type)
  {
  case OptimiserType::None:
    Layer::apply_none_update(weight_param, clipped_gradient, learning_rate);
    break;

  case OptimiserType::SGD:
    Layer::apply_sgd_update(weight_param, clipped_gradient, learning_rate, _activation_method.momentum(), is_bias);
    break;

  case OptimiserType::Adam:
    Layer::apply_adam_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias);
    break;

  case OptimiserType::AdamW:
    Layer::apply_adamw_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8);
    break;

  case OptimiserType::Nadam:
    Layer::apply_nadam_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8);
    break;

  case OptimiserType::NadamW:
    Layer::apply_nadamw_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias);
    break;

  default:
    throw std::runtime_error("Unknown optimizer type.");
  }
}

// TODO the clip threshold used is not the one we have in the config/options!
double Layer::clip_gradient(double gradient)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  constexpr double gradient_clip_threshold = 1.0;
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
