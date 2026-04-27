#include <algorithm>
#include "layer.h"
#include "simd_utils.h"
#include "logger.h"
#include "layerdetails.h"
#include "fflayer.h"
#include "elmanrnnlayer.h"
#include "grurnnlayer.h"
#include "lstmlayer.h"

std::unique_ptr<Layer> Layer::create_hidden_layer(
  unsigned layer_index,
  unsigned number_input_neurons,
  const LayerDetails& ld,
  int number_of_threads,
  bool has_bias,
  int residual_layer_number,
  ResidualProjector* residual_projector
)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  switch (ld.get_layer_architecture())
  {
  case Layer::Architecture::FF:
    return std::make_unique<FFLayer>(
      layer_index,
      number_input_neurons,
      ld.get_size(),
      ld.get_weight_decay(),
      Role::Hidden,
      ld.get_activation(),
      ld.get_optimiser_type(),
      residual_layer_number,
      ld.get_dropout(),
      residual_projector,
      number_of_threads,
      has_bias,
      ld.get_momentum()
    );

  case Layer::Architecture::Elman:
    return std::make_unique<ElmanRNNLayer>(
      layer_index,
      number_input_neurons,
      ld.get_size(),
      ld.get_weight_decay(),
      Role::Hidden,
      ld.get_activation(),
      ld.get_optimiser_type(),
      residual_layer_number,
      ld.get_dropout(),
      residual_projector,
      number_of_threads,
      has_bias,
      ld.get_momentum()
    );

  case Layer::Architecture::Gru:
    return std::make_unique<GRURNNLayer>(
      layer_index,
      number_input_neurons,
      ld.get_size(),
      ld.get_weight_decay(),
      Role::Hidden,
      ld.get_activation(),
      ld.get_optimiser_type(),
      residual_layer_number,
      ld.get_dropout(),
      residual_projector,
      number_of_threads,
      has_bias,
      ld.get_momentum()
    );

  case Layer::Architecture::Lstm:
    return std::make_unique<LSTMLayer>(
      layer_index,
      number_input_neurons,
      ld.get_size(),
      ld.get_weight_decay(),
      Role::Hidden,
      ld.get_activation(),
      ld.get_optimiser_type(),
      residual_layer_number,
      ld.get_dropout(),
      residual_projector,
      number_of_threads,
      has_bias,
      ld.get_momentum()
    );

  default:
    Logger::panic("Unknown Layer architecture: ", (int)ld.get_layer_architecture());
    return nullptr;
  }
}

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

  // This delta calculation assumes Softmax activation is used at the output layer.
  // dL/dz = y_pred - y_true
  for (const auto& neuron : neurons)
  {
    const unsigned neuron_index = neuron.get_index();
    const double target = target_outputs[neuron_index];
    const double output = given_outputs[neuron_index];

    // Standard Cross-Entropy + Softmax gradient: (given - target)
    double grad = (output - target);

    // If softmax is used, temperature scaling applies to the gradient
    if (activation_method == activation::method::softmax)
    {
      double temperature = get_activation(neuron_index).get_temperature();
      grad /= temperature;
    }

    if (use_dir && activation_method == activation::method::softmax && gt_dir != 0 && pred_dir != gt_dir)
    {
      grad *= (1.0 + dir_lambda);
    }

    // Apply Cross Entropy scaling
    grad *= ce_lambda;

    if (!std::isfinite(grad))
    {
        Logger::panic("CRITICAL: Non-finite gradient detected at neuron ", neuron_index);
    }

    deltas[neuron_index] = grad * inv_num_neurons;
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

void Layer::apply_update_to_vector(
    std::vector<double>& values,
    std::vector<double>& grads,
    std::vector<double>& velocities,
    std::vector<double>& m1,
    std::vector<double>& m2,
    std::vector<long long>& timesteps,
    const std::vector<double>& decays,
    double learning_rate,
    double clipping_scale,
    bool is_bias,
    OptimiserType optimiser_type)
{
  MYODDWEB_PROFILE_FUNCTION("Layer");
  const size_t n = values.size();
  if (n == 0)
  {
    return;
  }

  switch (optimiser_type)
  {
  case OptimiserType::None:
    for (size_t i = 0; i < n; ++i)
    {
      double grad = grads[i] * clipping_scale;
      values[i] -= learning_rate * grad;
      grads[i] = grad;
    }
    break;

  case OptimiserType::SGD:
  {
    const auto& momentum = get_momentum();
    for (size_t i = 0; i < n; ++i)
    {
      double grad = grads[i] * clipping_scale;
      if (!is_bias && i < decays.size() && decays[i] > 0.0)
      {
        grad += decays[i] * values[i];
      }
      velocities[i] = momentum * velocities[i] + grad;
      values[i] -= learning_rate * velocities[i];
      grads[i] = grad;
    }
  }
  break;

  case OptimiserType::Adam:
  case OptimiserType::AdamW:
  {
    const double beta1 = get_momentum();
    const double beta2 = 0.999;
    const double epsilon = 1e-8;

    if (clipping_scale != 1.0)
    {
      for (double& g : grads) g *= clipping_scale;
    }

    for (auto& t : timesteps) ++t;
    const double p1 = 1.0 - std::pow(beta1, timesteps[0]);
    const double p2 = 1.0 - std::pow(beta2, timesteps[0]);
    const double* decay_ptr = (optimiser_type == OptimiserType::AdamW && !is_bias && decays.size() >= n) ? decays.data() : nullptr;

    simd::adam_step(values.data(), grads.data(), m1.data(), m2.data(), beta1, beta2, p1, p2, learning_rate, epsilon, n, decay_ptr);
  }
  break;

  case OptimiserType::Nadam:
  case OptimiserType::NadamW:
  {
    const double beta1 = get_momentum();
    const double beta2 = 0.999;
    const double epsilon = 1e-8;

    if (clipping_scale != 1.0)
    {
      for (double& g : grads) g *= clipping_scale;
    }

    for (auto& t : timesteps) ++t;
    const double p1 = 1.0 - std::pow(beta1, timesteps[0]);
    const double p2 = 1.0 - std::pow(beta2, timesteps[0]);
    const double* decay_ptr = (optimiser_type == OptimiserType::NadamW && !is_bias && decays.size() >= n) ? decays.data() : nullptr;

    simd::nadam_step(values.data(), grads.data(), m1.data(), m2.data(), beta1, beta2, p1, p2, learning_rate, epsilon, n, decay_ptr);
  }
  break;

  default:
    Logger::panic("Unknown optimizer type:", (int)optimiser_type);
  }
}

void Layer::apply_update_to_weight(
    std::vector<double>& values,
    std::vector<double>& grads,
    std::vector<double>& velocities,
    std::vector<double>& m1,
    std::vector<double>& m2,
    std::vector<long long>& timesteps,
    const std::vector<double>& decays,
    unsigned idx,
    double gradient,
    double learning_rate,
    double clipping_scale,
    OptimiserType optimiser_type,
    unsigned neuron_number)
{
    MYODDWEB_PROFILE_FUNCTION("Layer");

    // validation
    if (!std::isfinite(gradient))
    {
      Logger::panic("Error while calculating input weigh gradient it invalid.");
    }

    if (clipping_scale < 0.0)
    {
      // If clipping scale is negative, we clip the gradient to a fixed range
      Logger::warning("Clipping gradient to a fixed range.");
    }

    double final_gradient = gradient * clipping_scale;

    // Detect gradient explosions before they impact weights
    if (std::abs(final_gradient) > 1e6)
    {
      Logger::panic("CRITICAL: Gradient too large! Grad: ", final_gradient, " lr: ", learning_rate);
    }
    else if (!std::isfinite(final_gradient))
    {
      Logger::panic("CRITICAL: Non-finite gradient detected! Grad: ", final_gradient);
    }

    // Log trace for some updates to avoid flooding
    if (idx == 0 && (timesteps.empty() || timesteps[idx] % 50 == 0))
    {
      Logger::trace([&]()
        {
          std::ostringstream ss;
          ss << "[Layer::apply_update_to_weight] layer=" << _layer_index
            << ", idx=" << idx
            << ", grad=" << gradient
            << ", final_grad=" << final_gradient
            << ", lr=" << learning_rate
            << ", val_before=" << values[idx];
          return ss.str();
        });
    }

    switch (optimiser_type)
    {
    case OptimiserType::None:
      values[idx] -= learning_rate * final_gradient;
      grads[idx] = final_gradient;
      break;

    case OptimiserType::SGD:
    {
      double grad = final_gradient;
      if (!is_bias_index(values) && decays.size() > idx && decays[idx] > 0.0)
      {
        grad += decays[idx] * values[idx];
      }
      double previous_velocity = velocities[idx];
      double velocity = get_momentum(neuron_number) * previous_velocity + grad;
      values[idx] -= learning_rate * velocity;
      velocities[idx] = velocity;
      grads[idx] = grad;
    }
    break;

    case OptimiserType::Adam:
    case OptimiserType::AdamW:
    {
      const double beta1 = get_momentum(neuron_number);
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      const long long time_step = ++timesteps[idx];

      m1[idx] = beta1 * m1[idx] + (1.0 - beta1) * final_gradient;
      m2[idx] = beta2 * m2[idx] + (1.0 - beta2) * (final_gradient * final_gradient);

      double m_hat = m1[idx] / (1.0 - std::pow(beta1, time_step));
      double v_hat = m2[idx] / (1.0 - std::pow(beta2, time_step));

      double update_step = m_hat / (std::sqrt(v_hat) + epsilon);

      double current_weight = values[idx];
      if (optimiser_type == OptimiserType::AdamW && !is_bias_index(values) && decays.size() > idx)
      {
        current_weight *= (1.0 - learning_rate * decays[idx]);
      }

      values[idx] = current_weight - learning_rate * update_step;
      grads[idx] = final_gradient;
    }
    break;

    case OptimiserType::Nadam:
    case OptimiserType::NadamW:
    {
      const double beta1 = get_momentum(neuron_number);
      const double beta2 = 0.999;
      const double epsilon = 1e-8;

      const long long time_step = ++timesteps[idx];

      m1[idx] = beta1 * m1[idx] + (1.0 - beta1) * final_gradient;
      m2[idx] = beta2 * m2[idx] + (1.0 - beta2) * (final_gradient * final_gradient);

      double m_hat = m1[idx] / (1.0 - std::pow(beta1, time_step));
      double v_hat = m2[idx] / (1.0 - std::pow(beta2, time_step));

      double m_nadam = beta1 * m_hat + ((1.0 - beta1) * final_gradient) / (1.0 - std::pow(beta1, time_step));
      double update_step = m_nadam / (std::sqrt(v_hat) + epsilon);

      double current_weight = values[idx];
      if (optimiser_type == OptimiserType::NadamW && !is_bias_index(values) && decays.size() > idx)
      {
        current_weight *= (1.0 - learning_rate * decays[idx]);
      }

      values[idx] = current_weight - learning_rate * update_step;
      grads[idx] = final_gradient;
    }
    break;

    default:
      Logger::panic("Unknown optimizer type:", (int)optimiser_type);
    }
}

void Layer::apply_weight_gradient(double gradient, double learning_rate, bool is_bias, unsigned weight_index, double clipping_scale, OptimiserType optimiser_type, unsigned neuron_number)
{
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (is_bias)
    {
        apply_update_to_weight(_b_values, _b_grads, _b_velocities, _b_m1, _b_m2, _b_timesteps, _b_decays, weight_index, gradient, learning_rate, clipping_scale, optimiser_type, neuron_number);
    }
    else
    {
        apply_update_to_weight(_w_values, _w_grads, _w_velocities, _w_m1, _w_m2, _w_timesteps, _w_decays, weight_index, gradient, learning_rate, clipping_scale, optimiser_type, neuron_number);
    }
}
