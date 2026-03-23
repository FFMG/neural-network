#include "layer.h"
#include "logger.h"

void Layer::calculate_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  ErrorCalculation::type error_calculation_type,
  const ErrorCalculation::EvaluationConfig& evaluation_config) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  switch (error_calculation_type)
  {
  case ErrorCalculation::type::huber_loss:
    return calculate_huber_loss_error_deltas(deltas, target_outputs, given_outputs, evaluation_config);
  case ErrorCalculation::type::huber_direction_loss:
    return calculate_huber_direction_loss_error_deltas(deltas, target_outputs, given_outputs, evaluation_config);
  case ErrorCalculation::type::mse:
    return calculate_mse_error_deltas(deltas, target_outputs, given_outputs);
  case ErrorCalculation::type::rmse:
    return calculate_rmse_error_deltas(deltas, target_outputs, given_outputs);
  case ErrorCalculation::type::bce_loss:
    return calculate_bce_error_deltas(deltas, target_outputs, given_outputs);
  case ErrorCalculation::type::cross_entropy:
    return calculate_cross_entropy_error_deltas(deltas, target_outputs, given_outputs);
  case ErrorCalculation::type::log_cosh:
    return calculate_log_cosh_error_deltas(deltas, target_outputs, given_outputs);
  default:
    Logger::panic("Error calculation type, ", ErrorCalculation::type_to_string(error_calculation_type)," is not supported for Layer!");
  }
}

void Layer::calculate_huber_direction_loss_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const ErrorCalculation::EvaluationConfig& evaluation_config) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");

  const size_t N_total = get_number_neurons();

  const double& delta = evaluation_config.huber_delta;                   // Huber threshold
  const double& lambda = evaluation_config.direction_lambda;             // Direction penalty strength
  const double& neutral_tolerance = evaluation_config.neutral_tolerance; // Ignore tiny targets
  const bool use_direction_penalty = evaluation_config.use_direction_penalty;

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    const double target = target_outputs[neuron_index];
    const double output = given_outputs[neuron_index];

    const double error = output - target;
    const double abs_error = std::abs(error);

    // --- Base Huber gradient ---
    double grad;
    if (abs_error <= delta)
    {
      grad = error;
    }
    else
    {
      grad = (error > 0.0 ? delta : -delta);
    }

    // --- Optional Direction Penalty ---
    if (use_direction_penalty)
    {
      // Only apply if target is meaningful (not noise)
      if (std::abs(target) > neutral_tolerance)
      {
        const bool sign_mismatch =
          (target > 0.0 && output < 0.0) ||
          (target < 0.0 && output > 0.0);

        if (sign_mismatch)
        {
          // Smooth directional push (prevents collapse to zero)
          const double direction_grad = output;

          // Stronger penalty for stronger signals
          const double strength = std::abs(target);

          grad += lambda * strength * direction_grad;
        }
      }
    }

    deltas[neuron_index] = grad * _inv_num_neurons;
  }
}

void Layer::calculate_huber_loss_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const ErrorCalculation::EvaluationConfig& evaluation_config) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");

  const size_t N_total = get_number_neurons();
  const double& delta = evaluation_config.huber_delta;

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    const double error = given_outputs[neuron_index] - target_outputs[neuron_index];
    const double abs_error = std::abs(error);

    double grad;
    if (abs_error <= delta)
    {
      grad = error;
    }
    else
    {
      grad = (error > 0.0 ? delta : -delta);
    }

    deltas[neuron_index] = grad * _inv_num_neurons;
  }
}

void Layer::calculate_cross_entropy_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const size_t N_total = get_number_neurons();
  
  // This delta calculation assumes Softmax activation is used at the output layer.
  // dL/dz = y_pred - y_true
  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) * _inv_num_neurons;
  }
}

void Layer::calculate_bce_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const size_t N_total = get_number_neurons();
  const double denom = static_cast<double>(N_total);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) * _inv_num_neurons;
  }
}

void Layer::calculate_mse_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const size_t N_total = get_number_neurons();
  const double denom = static_cast<double>(N_total);

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) * _inv_num_neurons;
  }
}

void Layer::calculate_rmse_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const size_t N_total = get_number_neurons();

  // 1. Calculate MSE sum
  double sum_squared_error = 0.0;
  for (unsigned i = 0; i < N_total; ++i)
  {
    const double diff = given_outputs[i] - target_outputs[i];
    sum_squared_error += diff * diff;
  }

  // 2. Calculate RMSE
  // Avoid division by zero if RMSE is 0 (perfect prediction)
  const double mse = sum_squared_error * _inv_num_neurons;
  const double rmse = std::sqrt(mse);
  const double epsilon = 1e-12;
  const double divisor = (rmse < epsilon) ? epsilon : rmse;

  // 3. Calculate deltas
  // dE/dy = (1 / (N * RMSE)) * (y - t)
  const double factor = _inv_num_neurons / divisor;

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) * factor;
  }
}

void Layer::calculate_log_cosh_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const size_t N_total = get_number_neurons();

  for (unsigned neuron_index = 0; neuron_index < N_total; ++neuron_index)
  {
    const double x = given_outputs[neuron_index] - target_outputs[neuron_index];
    // d/dx log(cosh(x)) = tanh(x)
    deltas[neuron_index] = std::tanh(x) * _inv_num_neurons;
  }
}
