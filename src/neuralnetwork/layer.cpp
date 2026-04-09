#include <algorithm>
#include "layer.h"
#include "logger.h"

void Layer::calculate_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  ErrorCalculation::type error_calculation_type,
  const EvaluationConfig& evaluation_config,
  const activation::method activation_method,
  unsigned start_neuron,
  unsigned end_neuron) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");

#if VALIDATE_DATA == 1
  if (end_neuron < start_neuron )
  {
    Logger::panic("end neuron cannot be less than start neuron!");
  }
  if (end_neuron > get_number_neurons() - 1)
  {
    Logger::panic("end neuron Cannot be greater than ", get_number_neurons() -1, "!");
  }
#endif

  std::span<Neuron> neurons_span(const_cast<Neuron*>(&_neurons[start_neuron]), end_neuron - start_neuron + 1);

  switch (error_calculation_type)
  {
  case ErrorCalculation::type::huber_loss:
    return calculate_huber_loss_error_deltas(deltas, target_outputs, given_outputs, evaluation_config, activation_method, neurons_span);
  case ErrorCalculation::type::huber_direction_loss:
    return calculate_huber_direction_loss_error_deltas(deltas, target_outputs, given_outputs, evaluation_config, activation_method, neurons_span);
  case ErrorCalculation::type::mse:
    return calculate_mse_error_deltas(deltas, target_outputs, given_outputs, activation_method, neurons_span);
  case ErrorCalculation::type::rmse:
    return calculate_rmse_error_deltas(deltas, target_outputs, given_outputs, activation_method, neurons_span);
  case ErrorCalculation::type::bce_loss:
    return calculate_bce_error_deltas(deltas, target_outputs, given_outputs, evaluation_config, activation_method, neurons_span);
  case ErrorCalculation::type::cross_entropy:
    return calculate_cross_entropy_error_deltas(deltas, target_outputs, given_outputs, evaluation_config, activation_method, neurons_span);
  case ErrorCalculation::type::log_cosh:
    return calculate_log_cosh_error_deltas(deltas, target_outputs, given_outputs, activation_method, neurons_span);
  default:
    Logger::panic("Error calculation type, ", ErrorCalculation::type_to_string(error_calculation_type), " is not supported for Layer!");
  }
}

void Layer::calculate_huber_direction_loss_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const EvaluationConfig& evaluation_config,
  const activation::method activation_method,
  std::span<Neuron> neurons) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");

  const double& delta = evaluation_config.huber_delta();                   // Huber threshold
  const double& lambda = evaluation_config.direction_lambda();             // Direction penalty strength
  const double& neutral_tolerance = evaluation_config.neutral_tolerance(); // Ignore tiny targets
  const bool use_direction_penalty = evaluation_config.use_direction_penalty();
  const double inv_num_neurons = neurons.empty() ? 0.0 : 1.0 / static_cast<double>(neurons.size());

  for (const auto& neuron : neurons)
  {
    const unsigned neuron_index = neuron.get_index();
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

    deltas[neuron_index] = grad * inv_num_neurons;
  }
}

void Layer::calculate_huber_loss_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const EvaluationConfig& evaluation_config,
  const activation::method activation_method,
  std::span<Neuron> neurons) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");

  const double& delta = evaluation_config.huber_delta();
  const double inv_num_neurons = neurons.empty() ? 0.0 : 1.0 / static_cast<double>(neurons.size());

  for (const auto& neuron : neurons)
  {
    const unsigned neuron_index = neuron.get_index();
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

    deltas[neuron_index] = grad * inv_num_neurons;
  }
}

void Layer::calculate_cross_entropy_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const EvaluationConfig& evaluation_config,
  const activation::method activation_method,
  std::span<Neuron> neurons) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  
  const double dir_lambda = evaluation_config.direction_lambda();
  const bool   use_dir = evaluation_config.use_direction_penalty();
  const double ce_lambda = evaluation_config.cross_entropy_lambda();

  // --- Optional directional boost ---
  int pred_dir = 0;
  int gt_dir = 0;
  if (use_dir && activation_method == activation::method::softmax && !neurons.empty())
  {
    // For multiclass (Softmax), we use a midpoint to define direction.
    // In composite layers, we must only consider the current span of neurons.
    const size_t num_classes = neurons.size();
    const double mid = (static_cast<double>(num_classes) - 1.0) / 2.0;

    const unsigned start_idx = neurons.front().get_index();
    const unsigned end_idx = start_idx + static_cast<unsigned>(num_classes);

    // Determine predicted winning index (ArgMax) for this slice
    auto max_pred_it = std::max_element(given_outputs.begin() + start_idx, given_outputs.begin() + end_idx);
    const size_t pred_idx = std::distance(given_outputs.begin() + start_idx, max_pred_it);

    // Determine ground truth index for this slice
    auto max_gt_it = std::max_element(target_outputs.begin() + start_idx, target_outputs.begin() + end_idx);
    const size_t gt_idx = std::distance(target_outputs.begin() + start_idx, max_gt_it);

    pred_dir = (static_cast<double>(pred_idx) > mid) ? 1 : (static_cast<double>(pred_idx) < mid ? -1 : 0);
    gt_dir = (static_cast<double>(gt_idx) > mid) ? 1 : (static_cast<double>(gt_idx) < mid ? -1 : 0);
  }

  // This delta calculation assumes Softmax activation is used at the output layer.
  // dL/dz = y_pred - y_true
  for (const auto& neuron : neurons)
  {
    const unsigned neuron_index = neuron.get_index();
    const double target = target_outputs[neuron_index];
    const double output = given_outputs[neuron_index];

    double grad = (output - target);

    if (use_dir && activation_method == activation::method::softmax && gt_dir != 0 && pred_dir != gt_dir)
    {
      grad *= (1.0 + dir_lambda);
    }

    // --- Apply Cross Entropy scaling ---
    grad *= ce_lambda;

    deltas[neuron_index] = grad;
  }
}

void Layer::calculate_bce_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const EvaluationConfig& evaluation_config,
  const activation::method activation_method,
  std::span<Neuron> neurons) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");

  const double dir_lambda = evaluation_config.direction_lambda();
  const bool   use_dir = evaluation_config.use_direction_penalty();
  const double ce_lambda = evaluation_config.cross_entropy_lambda();
  const double inv_num_neurons = neurons.empty() ? 0.0 : 1.0 / static_cast<double>(neurons.size());

  // --- Optional directional boost ---
  int pred_dir = 0;
  int gt_dir = 0;
  if (use_dir && activation_method == activation::method::softmax && !neurons.empty())
  {
    // For multiclass (Softmax), we use a midpoint to define direction.
    // In composite layers, we must only consider the current span of neurons.
    const size_t num_classes = neurons.size();
    const double mid = (static_cast<double>(num_classes) - 1.0) / 2.0;

    const unsigned start_idx = neurons.front().get_index();
    const unsigned end_idx = start_idx + static_cast<unsigned>(num_classes);

    // Determine predicted winning index (ArgMax) for this slice
    auto max_pred_it = std::max_element(given_outputs.begin() + start_idx, given_outputs.begin() + end_idx);
    const size_t pred_idx = std::distance(given_outputs.begin() + start_idx, max_pred_it);

    // Determine ground truth index for this slice
    auto max_gt_it = std::max_element(target_outputs.begin() + start_idx, target_outputs.begin() + end_idx);
    const size_t gt_idx = std::distance(target_outputs.begin() + start_idx, max_gt_it);

    pred_dir = (static_cast<double>(pred_idx) > mid) ? 1 : (static_cast<double>(pred_idx) < mid ? -1 : 0);
    gt_dir = (static_cast<double>(gt_idx) > mid) ? 1 : (static_cast<double>(gt_idx) < mid ? -1 : 0);
  }

  for (const auto& neuron : neurons)
  {
    const unsigned idx = neuron.get_index();

    const double target = target_outputs[idx];
    const double output = given_outputs[idx];

    // --- Standard BCE gradient (correct for sigmoid output) ---
    double grad = (output - target);

    // --- Optional directional boost ---
    if (use_dir)
    {
      if (activation_method == activation::method::softmax)
      {
        if (gt_dir != 0 && pred_dir != gt_dir)
        {
          grad *= (1.0 + dir_lambda);
        }
      }
      else
      {
        const int direction = (target > 0.5) ? 1 : -1;
        const int predicted_dir = (output > 0.5) ? 1 : -1;

        if (direction != predicted_dir)
        {
          grad *= (1.0 + dir_lambda);
        }
      }
    }

    // --- Apply Cross Entropy scaling ---
    grad *= ce_lambda;

    // --- Normalize ---
    deltas[idx] = grad * inv_num_neurons;
  }
}

void Layer::calculate_mse_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const activation::method activation_method,
  std::span<Neuron> neurons) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const double inv_num_neurons = neurons.empty() ? 0.0 : 1.0 / static_cast<double>(neurons.size());

  for (const auto& neuron : neurons)
  {
    const unsigned neuron_index = neuron.get_index();
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) * inv_num_neurons;
  }
}

void Layer::calculate_rmse_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const activation::method activation_method,
  std::span<Neuron> neurons) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const double inv_num_neurons = neurons.empty() ? 0.0 : 1.0 / static_cast<double>(neurons.size());

  // 1. Calculate MSE sum
  double sum_squared_error = 0.0;
  for (const auto& neuron : neurons)
  {
    const unsigned i = neuron.get_index();
    const double diff = given_outputs[i] - target_outputs[i];
    sum_squared_error += diff * diff;
  }

  // 2. Calculate RMSE
  // Avoid division by zero if RMSE is 0 (perfect prediction)
  const double mse = sum_squared_error * inv_num_neurons;
  const double rmse = std::sqrt(mse);
  const double epsilon = 1e-12;
  const double divisor = (rmse < epsilon) ? epsilon : rmse;

  // 3. Calculate deltas
  // dE/dy = (1 / (N * RMSE)) * (y - t)
  const double factor = inv_num_neurons / divisor;

  for (const auto& neuron : neurons)
  {
    const unsigned neuron_index = neuron.get_index();
    deltas[neuron_index] = (given_outputs[neuron_index] - target_outputs[neuron_index]) * factor;
  }
}

void Layer::calculate_log_cosh_error_deltas(
  std::vector<double>& deltas,
  const std::vector<double>& target_outputs,
  const std::vector<double>& given_outputs,
  const activation::method activation_method,
  std::span<Neuron> neurons) const
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const double inv_num_neurons = neurons.empty() ? 0.0 : 1.0 / static_cast<double>(neurons.size());

  for (const auto& neuron : neurons)
  {
    const unsigned neuron_index = neuron.get_index();
    const double x = given_outputs[neuron_index] - target_outputs[neuron_index];
    // d/dx log(cosh(x)) = tanh(x)
    deltas[neuron_index] = std::tanh(x) * inv_num_neurons;
  }
}
