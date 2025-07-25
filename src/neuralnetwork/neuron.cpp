#include "neuron.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

Neuron::Neuron(
  unsigned /*num_neurons_prev_layer*/,
  unsigned num_neurons_current_layer,
  unsigned num_neurons_next_layer,
  unsigned index,
  const activation& activation,
  const OptimiserType& optimiser_type,
  const Type& type,
  const Logger& logger
) :
  _index(index),
  _activation_method(activation),
  _optimiser_type(optimiser_type),
  _alpha(LEARNING_ALPHA),
  _type(type),
  _logger(logger)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  _weight_params.reserve(num_neurons_next_layer);
  auto weights = _activation_method.weight_initialization(num_neurons_next_layer, num_neurons_current_layer);
  for (const auto& weight : weights)
  {
    _weight_params.emplace_back(WeightParam(weight, 0.0, 0.0, logger));
  }
}

Neuron::Neuron(
  unsigned index,
  const activation& activation,
  const std::vector<WeightParam>& weights_params,
  const OptimiserType& optimiser_type,
  const Type& type,
  const Logger& logger
) :
  _index(index),
  _activation_method(activation),
  _optimiser_type(optimiser_type),
  _alpha(LEARNING_ALPHA),
  _type(type),
  _logger(logger)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  _weight_params.reserve(weights_params.size());
  for (auto& weights_param : weights_params)
  {
    _weight_params.emplace_back(weights_param);
  }
}

Neuron::Neuron(const Neuron& src)  noexcept : 
  _index(src._index),
  _activation_method(src._activation_method),
  _weight_params({}),
  _optimiser_type(src._optimiser_type),
  _alpha(LEARNING_ALPHA),
  _type(src._type),
  _logger(src._logger)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  _weight_params = src._weight_params;
}

Neuron& Neuron::operator=(const Neuron& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _activation_method = src._activation_method;
    _weight_params = src._weight_params;
    _optimiser_type = src._optimiser_type;
    _type = src._type;
    _logger = src._logger;
  }
  return *this;
}

Neuron::Neuron(Neuron&& src) noexcept :
  _index(src._index),
  _activation_method(src._activation_method),
  _weight_params(std::move(src._weight_params)),
  _optimiser_type(src._optimiser_type),
  _alpha(LEARNING_ALPHA),
  _type(src._type),
  _logger(src._logger)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");

  src._optimiser_type = OptimiserType::None;
  src._index = 0;
  src._type = Neuron::Type::Normal;
}

Neuron& Neuron::operator=(Neuron&& src) noexcept
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (this != &src)
  {
    Clean();

    _index = src._index;
    _activation_method = src._activation_method;
    _weight_params = std::move(src._weight_params);
    _optimiser_type = src._optimiser_type;
    _logger = src._logger;
    _type = src._type;

    src._optimiser_type = OptimiserType::None;
    src._index = 0;
    src._type = Neuron::Type::Normal;
  }
  return *this;
}

Neuron::~Neuron()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  Clean();
}

const std::vector<Neuron::WeightParam>& Neuron::get_weight_params() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _weight_params;
}

void Neuron::Clean()
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
}

unsigned Neuron::get_index() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _index;
}

void Neuron::apply_weight_gradient(const double gradient, const double learning_rate, const Neuron& previous_layer_neuron, WeightParam& weight_param)
{
    if (!std::isfinite(gradient))
    {
      _logger.log_error("Error while calculating input weigh gradient it invalid.");
      throw std::invalid_argument("Error while calculating input weight.");
    }
  auto old_velocity = weight_param.velocity();
  if (!std::isfinite(old_velocity))
  {
    _logger.log_error("Error while calculating input weigh old velocity is invalid.");
    throw std::invalid_argument("Error while calculating input weigh old velocity is invalid.");
  }

    const auto& is_bias = previous_layer_neuron.is_bias();
    auto clipped_gradient = clip_gradient(gradient);

    switch( _optimiser_type)
    {
    case OptimiserType::None:
      apply_none_update(weight_param, clipped_gradient, learning_rate);
      break;

    case OptimiserType::SGD:
      apply_sgd_update(weight_param, clipped_gradient, learning_rate, _activation_method.momentum(), is_bias);
      break;

    case OptimiserType::Adam:
      apply_adam_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias);
      break;

    case OptimiserType::AdamW:
      apply_adamw_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8);
      break;

    case OptimiserType::Nadam:
      apply_nadam_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8);
      break;

    case OptimiserType::NadamW:
      apply_nadamw_update(weight_param, clipped_gradient, learning_rate, 0.9, 0.999, 1e-8, is_bias);
      break;

    default:
      throw std::runtime_error("Unknown optimizer type.");
    }
}

void Neuron::apply_weight_gradients(Layer& previous_layer, const std::vector<double>& gradients, const double learning_rate, unsigned /*epoch*/)
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");

  // we should not be doing anything with out weights.
  assert(!is_bias());
  assert(gradients.size() == previous_layer.number_neurons());
  for (size_t i = 0; i < gradients.size(); ++i)
  {
    auto& previous_layer_neuron = previous_layer.get_neuron(static_cast<unsigned>(i));
    auto& weight_param = previous_layer_neuron._weight_params[get_index()];

    const auto& gradient = gradients[i];         // from prev layer, averaged over batch
    if (!std::isfinite(gradient))
    {
      _logger.log_error("Error while calculating input weigh gradient it invalid.");
      throw std::invalid_argument("Error while calculating input weight.");
    }
    apply_weight_gradient(gradient, learning_rate, previous_layer_neuron, weight_param);
  }
}

void Neuron::apply_nadam_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
) const
{
  // Update timestep
  weight_param.increment_timestep();
  const auto& time_step = weight_param.timestep();

  // These moment estimate updates are identical to Adam
  weight_param.set_first_moment_estimate(beta1 * weight_param.first_moment_estimate() + (1.0 - beta1) * raw_gradient);
  weight_param.set_second_moment_estimate(beta2 * weight_param.second_moment_estimate() + (1.0 - beta2) * (raw_gradient * raw_gradient));

  double m_hat = weight_param.first_moment_estimate() / (1.0 - std::pow(beta1, time_step));
  double v_hat = weight_param.second_moment_estimate() / (1.0 - std::pow(beta2, time_step));

  // Nadam's key difference:
  // It combines the momentum from the historical gradient (m_hat) with the
  // momentum from the current gradient.
  double corrected_gradient = (beta1 * m_hat) + ((1.0 - beta1) * raw_gradient) / (1.0 - std::pow(beta1, time_step));

  // The denominator is the same as Adam's
  double weight_update = learning_rate * (corrected_gradient / (std::sqrt(v_hat) + epsilon));

// Apply the final update (No decoupled weight decay)
  double new_weight = weight_param.value() - weight_update;

  weight_param.set_value(new_weight);
  weight_param.set_gradient(raw_gradient);
}

void Neuron::apply_nadamw_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    bool is_bias
) const
{
  // 1. Increment timestep
  weight_param.increment_timestep();
  const long long time_step = weight_param.timestep();

  // 2. Update biased first and second moment estimates
  const double first_moment = beta1 * weight_param.first_moment_estimate() +
                              (1.0 - beta1) * raw_gradient;
  const double second_moment = beta2 * weight_param.second_moment_estimate() +
                                (1.0 - beta2) * (raw_gradient * raw_gradient);

  weight_param.set_first_moment_estimate(first_moment);
  weight_param.set_second_moment_estimate(second_moment);

  // 3. Bias-corrected moments
  const double bias_correction1 = 1.0 - std::pow(beta1, time_step);
  const double bias_correction2 = 1.0 - std::pow(beta2, time_step);

  const double m_hat = first_moment / bias_correction1;
  const double v_hat = second_moment / bias_correction2;

  // 4. NAdam momentum blend with decoupled weight decay (NAdamW)
  const double blended_gradient = (beta1 * m_hat) + ((1.0 - beta1) * raw_gradient);
  const double adaptive_step = blended_gradient / (std::sqrt(v_hat) + epsilon);
  const double weight_update = learning_rate * adaptive_step;

  // 5. Apply weight decay (decoupled)
  const double decayed_weight = 
    is_bias ?
    weight_param.value()
      :
    weight_param.value() * (1.0 - learning_rate * weight_param.weight_decay());
  const double new_weight = decayed_weight - weight_update;

  // 6. Store new values
  weight_param.set_value(new_weight);
  weight_param.set_gradient(raw_gradient);  // raw gradient, in case needed elsewhere
}

void Neuron::apply_adamw_update(
  WeightParam& weight_param,
  double raw_gradient,           // unclipped, averaged over batch
  double learning_rate,
  double beta1,
  double beta2,
  double epsilon
) const
{
  // Update timestep
  weight_param.increment_timestep();
  const auto& time_step = weight_param.timestep();

  // Update biased first and second moment estimates
  weight_param.set_first_moment_estimate(beta1 * weight_param.first_moment_estimate() + (1.0 - beta1) * raw_gradient);
  weight_param.set_second_moment_estimate(beta2 * weight_param.second_moment_estimate() + (1.0 - beta2) * (raw_gradient * raw_gradient));

  // Compute bias-corrected moments
  double first_moment_estimate = weight_param.first_moment_estimate();
  double m_hat = first_moment_estimate / (1.0 - std::pow(beta1, time_step));

  auto second_moment_estimate = weight_param.second_moment_estimate();
  double v_hat = second_moment_estimate / (1.0 - std::pow(beta2, time_step));

  // AdamW update rule
  double weight_update = learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon));

  // Decoupled weight decay
  auto new_weight = weight_param.value();
  new_weight *= (1.0 - learning_rate * weight_param.weight_decay());

  // Apply update
  new_weight -= weight_update;

  weight_param.set_value(new_weight);
  weight_param.set_gradient(raw_gradient);
}

void Neuron::apply_adam_update(WeightParam& weight_param, double raw_gradient, double learning_rate, double beta1, double beta2, double epsilon,  bool is_bias) const
{
  // Update timestep
  weight_param.increment_timestep();
  const auto& time_step = weight_param.timestep();

 // Update moments
  double m = beta1 * weight_param.first_moment_estimate() + (1.0 - beta1) * raw_gradient;
  double v = beta2 * weight_param.second_moment_estimate() + (1.0 - beta2) * raw_gradient * raw_gradient;

  weight_param.set_first_moment_estimate(m);
  weight_param.set_second_moment_estimate(v);

  double m_hat = m / (1.0 - std::pow(beta1, time_step));
  double v_hat = v / (1.0 - std::pow(beta2, time_step));

  double adam_update = learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

  // Apply decoupled weight decay (skip if bias)
  double decayed_weight = is_bias ? weight_param.value()
                                   : weight_param.value() * (1.0 - learning_rate * weight_param.weight_decay());

  double new_weight = decayed_weight - adam_update;

  weight_param.set_value(new_weight);
  weight_param.set_gradient(raw_gradient);
}

void Neuron::apply_none_update(WeightParam& weight_param, double raw_gradient, double learning_rate) const
{
  double new_weight = weight_param.value() - learning_rate * raw_gradient;
  weight_param.set_gradient(raw_gradient);
  weight_param.set_value(new_weight);
}


void Neuron::apply_sgd_update(WeightParam& weight_param, double raw_gradient, double learning_rate, double momentum, bool is_bias) const
{
  double previous_velocity = weight_param.velocity();

  double velocity = momentum * previous_velocity - learning_rate * raw_gradient;

  double decayed_weight = is_bias
        ? weight_param.value()
        : weight_param.value() * (1.0 - learning_rate * weight_param.weight_decay());

  double new_weight = decayed_weight + velocity;
  weight_param.set_velocity(velocity);
  weight_param.set_gradient(raw_gradient);
  weight_param.set_value(new_weight);
}

double Neuron::get_output_weight(int index) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  assert(index < (int)_weight_params.size());
  return _weight_params[index].value();
}

double Neuron::sum_of_derivatives_of_weights(const Layer& next_layer, const std::vector<double>& activation_gradients) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if(is_bias())
  {
    //  bias neuron has not gradient.
    return 0.0;
  }
  assert(activation_gradients.size() == next_layer.number_neurons());

  double sum = 0.0;
  const size_t num_next_neurons = next_layer.number_neurons();

  for (unsigned neuron_index = 0; neuron_index < num_next_neurons; ++neuron_index) 
  {
    auto weights_and_gradients = get_output_weight(neuron_index) * activation_gradients[neuron_index];
    sum += std::isinf(weights_and_gradients) ? std::numeric_limits<double>::infinity() : weights_and_gradients;
  }
  if (!std::isfinite(sum))
  {
    _logger.log_error("Error while calculating sum of the derivatives of the weights.");
    throw std::invalid_argument("Error while calculating sum of the derivatives of the weights.");
    return std::numeric_limits<double>::quiet_NaN();
  }
  return sum;
}

double Neuron::clip_gradient(double gradient) const
{
  constexpr double gradient_clip_threshold = 1.0;
  if (!std::isfinite(gradient))
  {
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

double Neuron::calculate_output_gradients(double target_value, double output_value) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if (is_bias())
  {
    return 0.0;
  }

  double delta = output_value - target_value;
  auto gradient = delta * _activation_method.activate_derivative(output_value);
  gradient = clip_gradient(gradient);
  if (!std::isfinite(gradient))
  {
    _logger.log_error("Error while calculating output gradients.");
    throw std::invalid_argument("Error while calculating output gradients.");
    return std::numeric_limits<double>::quiet_NaN();
  }
  return gradient;
}

double Neuron::calculate_hidden_gradients(const Layer& next_layer, const std::vector<double>& activation_gradients, double output_value) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  auto derivatives_of_weights = sum_of_derivatives_of_weights(next_layer, activation_gradients);
  auto gradient = derivatives_of_weights * _activation_method.activate_derivative(output_value);
  gradient = clip_gradient(gradient);
  if (!std::isfinite(gradient))
  {
    _logger.log_error("Error while calculating hidden gradients.");
    throw std::invalid_argument("Error while calculating hidden gradients.");
    return std::numeric_limits<double>::quiet_NaN();
  }  
  return gradient;
}

double Neuron::calculate_forward_feed(const Layer& previous_layer, const std::vector<double>& previous_layer_output_values) const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  if(is_bias())
  {
    return 1.0;  // Bias neurons always output 1
  }

  double sum = 0.0;

  // Sum the previous layer's outputs (which are our inputs)
  // Include the bias node from the previous layer.
  assert(previous_layer_output_values.size() == previous_layer.number_neurons());
  for (unsigned neuron_index = 0; neuron_index < previous_layer.number_neurons(); ++neuron_index) 
  {
    const auto& previous_layer_neuron = previous_layer.get_neuron(neuron_index);
    const auto output_weight = previous_layer_neuron.get_output_weight(get_index());
    if (std::abs(output_weight) > 1e5)
    {
      _logger.log_error("Exploding weight detected");
      throw std::runtime_error("Exploding weight detected");
    }

    // get that neuron 
    const auto output_value  = previous_layer_output_values[neuron_index];
    sum +=  output_value * output_weight;
    if (!std::isfinite(sum))
    {
      _logger.log_error("Error while calculating forward feed.");
      throw std::invalid_argument("Error while calculating forward feed.");
    }
  }
  return _activation_method.activate(sum);
}

const OptimiserType& Neuron::get_optimiser_type() const 
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _optimiser_type; 
}

bool Neuron::is_bias() const
{
  MYODDWEB_PROFILE_FUNCTION("Neuron");
  return _type == Neuron::Type::Bias;
}
