#pragma once
#include "./libraries/instrumentor.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <vector>

#include "activation.h"
#include "evaluationconfig.h"
#include "logger.h"

class ErrorCalculation
{
public:
  enum class type
  {
    none,
    huber_loss,
    huber_direction_loss,
    mae,
    mse,
    rmse,
    nrmse,
    mape,
    smape,
    wape,
    directional_accuracy,
    bce_loss,
    cross_entropy,
    log_cosh,
    directional_confidence_score,
    prediction_coverage
  };
public:
  inline static std::string type_to_string(const ErrorCalculation::type& type)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    switch (type)
    {
    case type::none:
      return "none";
    case type::huber_loss: 
      return "huber-loss";
    case type::huber_direction_loss:
      return "huber-direction-loss";
    case type::mae: 
      return "mae";
    case type::mse: 
      return "mse";
    case type::rmse: 
      return "rmse";
    case type::nrmse: 
      return "nrmse";
    case type::mape: 
      return "mape";
    case type::smape: 
      return "smape";
    case type::wape: 
      return "wape";
    case type::directional_accuracy: 
      return "directional-accuracy";
    case type::bce_loss:
      return "bce-loss";
    case type::cross_entropy:
      return "cross-entropy";
    case type::log_cosh:
      return "log-cosh";
    case type::directional_confidence_score:
      return "directional-confidence-score";
    case type::prediction_coverage:
      return "prediction-coverage";
    }
    Logger::panic("Unknown activation type!");
  }

  inline static type string_to_type(const std::string& str)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
      [](unsigned char c) { return std::tolower(c); });

    if (lower_str == "none")
    {
      return type::none;
    }
    if (lower_str == "huber-loss")
    {
      return type::huber_loss;
    }
    if (lower_str == "huber-direction-loss")
    {
      return type::huber_direction_loss;
    }
    if (lower_str == "mae")
    {
      return type::mae;
    }
    if (lower_str == "mse")
    {
      return type::mse;
    }
    if (lower_str == "rmse")
    {
      return type::rmse;
    }
    if (lower_str == "nrmse")
    {
      return type::nrmse;
    }
    if (lower_str == "mape")
    {
      return type::mape;
    }
    if (lower_str == "smape")
    {
      return type::smape;
    }
    if (lower_str == "wape")
    {
      return type::wape;
    }
    if (lower_str == "directional-accuracy")
    {
      return type::directional_accuracy;
    }
    if (lower_str == "directional-confidence-score")
    {
      return type::directional_confidence_score;
    }
    if (lower_str == "prediction-coverage")
    {
      return type::prediction_coverage;
    }
    if (lower_str == "bce-loss")
    {
      return type::bce_loss;
    }
    if (lower_str == "cross-entropy")
    {
      return type::cross_entropy;
    }
    if (lower_str == "log-cosh")
    {
      return type::log_cosh;
    }
    Logger::panic("Unknown error type: ", str);

  }

  static double calculate_error(type error_type, std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config, const activation::method& activation_method )
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
#if VALIDATE_DATA == 1
    if (predictions.size() != ground_truths.size() || predictions.empty())
    {
      Logger::panic("Input vectors must have the same, non-zero size.");
    }
#endif
    switch (error_type)
    {
    case type::none:
      return 0.0;

    case type::huber_loss:
      return calculate_huber_loss_error(ground_truths, predictions, evaluation_config);

    case type::huber_direction_loss:
      return calculate_huber_direction_loss(ground_truths, predictions, evaluation_config);

    case type::mae:
      return calculate_mae_error(ground_truths, predictions);

    case type::mse:
      return calculate_mse_error(ground_truths, predictions);

    case type::rmse:
      return calculate_rmse_error(ground_truths, predictions);

    case type::nrmse:
      return calculate_nrmse_error(ground_truths, predictions);

    case type::mape:
      return calculate_forecast_mape(ground_truths, predictions);

    case type::wape:
      return calculate_forecast_wape(ground_truths, predictions);

    case type::smape:
      return calculate_forecast_smape(ground_truths, predictions);

    case type::directional_accuracy:
      if (activation_method == activation::method::softmax)
      {
        return calculate_softmax_directional_accuracy(ground_truths, predictions, evaluation_config);
      }
      return calculate_directional_accuracy(ground_truths, predictions, evaluation_config);

    case type::directional_confidence_score:
      if (activation_method == activation::method::softmax)
      {
        return calculate_softmax_directional_confidence_score(ground_truths, predictions, evaluation_config);
      }
      return calculate_directional_confidence_score(ground_truths, predictions, evaluation_config);

    case type::bce_loss:
      return calculate_bce_loss(ground_truths, predictions);

    case type::cross_entropy:
      return calculate_cross_entropy(ground_truths, predictions);

    case type::log_cosh:
      return calculate_log_cosh(ground_truths, predictions);

    case type::prediction_coverage:
      if (activation_method == activation::method::softmax)
      {
        return calculate_softmax_prediction_coverage(predictions, evaluation_config);
      }
      return calculate_prediction_coverage(predictions, evaluation_config);
    }

    Logger::panic("Unknown ErrorCalculation type!");
  }

  static double calculate_huber_loss_error(std::span<const std::vector<double>> ground_truth, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const auto& delta = evaluation_config.huber_delta();

    double total_loss = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
      if (ground_truth[i].size() != predictions[i].size())
      {
        Logger::panic("Mismatched vector sizes at index ", i);
      }

      for (size_t j = 0; j < ground_truth[i].size(); ++j)
      {
        double error = ground_truth[i][j] - predictions[i][j];
        double abs_error = std::abs(error);

        if (abs_error <= delta)
        {
          total_loss += 0.5 * error * error;
        }
        else
        {
          total_loss += delta * (abs_error - 0.5 * delta);
        }
        ++count;
      }
    }
    return (count > 0) ? (total_loss / count) : 0.0;
  }

  static double calculate_huber_direction_loss(std::span<const std::vector<double>> ground_truth, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const auto& lambda = evaluation_config.direction_lambda();
    const auto& delta = evaluation_config.huber_delta();

    double total_loss = 0.0;
    size_t count = 0;

    const double scale = 100.0; // important

    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
      for (size_t j = 0; j < ground_truth[i].size(); ++j)
      {
        const double target = ground_truth[i][j];
        const double output = predictions[i][j];

        const double error = target - output;
        const double abs_error = std::abs(error);

        // --- Huber loss ---
        double loss = 0.0;
        if (abs_error <= delta)
        {
          loss = 0.5 * error * error;
        }
        else
        {
          loss = delta * (abs_error - 0.5 * delta);
        }

        if (std::abs(target) > 1e-6) // ignore noise
        {
          const double x = -scale * target * output;

          // log(1 + exp(x)) (numerically stable version optional)
          const double direction_loss = std::log(1.0 + std::exp(x));

          loss += lambda * direction_loss;
        }

        total_loss += loss;
        ++count;
      }
    }

    return (count > 0) ? (total_loss / count) : 0.0;
  }

  static double calculate_mae_error(std::span<const std::vector<double>> ground_truth, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_abs_error = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
      if (ground_truth[i].size() != predictions[i].size())
      {
        Logger::panic("Mismatched vector sizes at index ", i);
      }
      for (size_t j = 0; j < ground_truth[i].size(); ++j)
      {
        total_abs_error += std::abs(ground_truth[i][j] - predictions[i][j]);
        ++count;
      }
    }
    return (count > 0) ? (total_abs_error / count) : 0.0;
  }

  static double calculate_mse_error(std::span<const std::vector<double>> ground_truth, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double mean_squared_error = 0.0;
    size_t valid_count = 0;

    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
      const auto& true_output = ground_truth[i];
      const auto& predicted_output = predictions[i];

      if (true_output.size() != predicted_output.size())
      {
        Logger::warning("Mismatch in output vector sizes at index ", i);
        continue;
      }

      for (size_t j = 0; j < true_output.size(); ++j)
      {
        double error = predicted_output[j] - true_output[j];

        if (!std::isfinite(error))
        {
          continue;
        }

        double squared_error = error * error;
        if (!std::isfinite(squared_error))
        {
          continue;
        }
        ++valid_count;
        mean_squared_error += (squared_error - mean_squared_error) / valid_count;
      }
    }

    if (valid_count == 0)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return mean_squared_error;
  }

  static double calculate_rmse_error(std::span<const std::vector<double>> ground_truths,std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_rmse = 0.0;
    size_t sequence_count = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
        continue;

      double mse = 0.0;
      for (size_t i = 0; i < gt.size(); ++i)
      {
        double diff = gt[i] - pred[i];
        mse += diff * diff;
      }

      mse /= gt.size();
      total_rmse += std::sqrt(mse);
      ++sequence_count;
    }

    return (sequence_count == 0) ? 0.0 : (total_rmse / sequence_count);
  }

  static double calculate_nrmse_error(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_nrmse = 0.0;
    size_t sequence_count = 0;
    const double eps = 1e-12; // small value to avoid division by zero

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
        continue;

      double mse = 0.0;
      double min_val = gt[0], max_val = gt[0], mean_abs = 0.0;

      for (size_t i = 0; i < gt.size(); ++i)
      {
        double diff = gt[i] - pred[i];
        mse += diff * diff;

        min_val = std::min(min_val, gt[i]);
        max_val = std::max(max_val, gt[i]);
        mean_abs += std::abs(gt[i]);
      }

      mse /= gt.size();
      double rmse = std::sqrt(mse);
      mean_abs /= gt.size();

      // Auto-select normalization
      double denom = max_val - min_val;         // primary: range
      if (denom < eps) denom = mean_abs;        // fallback: mean magnitude
      if (denom < eps) continue;                // skip if both tiny

      total_nrmse += rmse / denom;
      ++sequence_count;
    }

    return (sequence_count == 0) ? 0.0 : (total_nrmse / sequence_count);
  }

  // TODO epsilon should be a common const rather than a param, it is never passed.
  static double calculate_forecast_mape(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, double epsilon = 1e-8)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_mape = 0.0;
    size_t sequence_count = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        continue; // skip empty or mismatched
      }

      double seq_error_sum = 0.0;
      size_t count = 0;
              
      for (size_t i = 0; i < gt.size(); ++i)
      {
        double denom = std::abs(gt[i]);
        if (denom < epsilon) continue; // skip tiny values
        seq_error_sum += std::abs((gt[i] - pred[i]) / denom);
        ++count;
      }
              
      if (count > 0)
      {
        total_mape += seq_error_sum / count;
        ++sequence_count;
      }

      Logger::trace([&]
      {
        std::ostringstream ss;
        ss << "[MAPE_DEBUG] After sequence " << seq_idx << ": total_mape=" << total_mape
            << ", sequence_count=" << sequence_count;
        return ss.str();
      });
    }
    return (sequence_count == 0) ? 0.0 : (total_mape / sequence_count);
  }

  static double calculate_forecast_wape(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_absolute_error = 0.0;
    double total_absolute_actuals = 0.0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      // Skip mismatched or empty sequences
      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      // Sum the errors and actuals for this sequence
      for (size_t i = 0; i < gt.size(); ++i)
      {
        total_absolute_error += std::abs(gt[i] - pred[i]);
        total_absolute_actuals += std::abs(gt[i]);
      }
    }

    // Perform a single division at the end
    // Check if the total sum of actuals is zero
    if (total_absolute_actuals == 0.0)
    {
      // If total actuals are 0, error is 0 only if total error is also 0.
      // Otherwise, it's undefined. We can return 0 if both are 0, 
      // or 1.0 (100% error) if we predicted non-zero values for all-zero actuals.
      return (total_absolute_error == 0.0) ? 0.0 : 1.0;
    }

    // WAPE formula
    return total_absolute_error / total_absolute_actuals;
  }

  // TODO epsilon should be a common const rather than a param, it is never passed.
  static double calculate_forecast_smape(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, double epsilon = 1e-8)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_smape = 0.0;
    size_t sequence_count = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty()) {
        continue; // skip empty or mismatched
      }

      double seq_error_sum = 0.0;
      size_t count = 0;

      for (size_t i = 0; i < gt.size(); ++i)
      {
        double denom = (std::abs(gt[i]) + std::abs(pred[i])) / 2.0;
        if (denom < epsilon) continue; // skip both near-zero
        seq_error_sum += std::abs(gt[i] - pred[i]) / denom;
        ++count;
      }

      if (count > 0)
      {
        total_smape += seq_error_sum / count;
        ++sequence_count;
      }
    }
    return (sequence_count == 0) ? 0.0 : (total_smape / sequence_count);
  }

  static double calculate_softmax_prediction_coverage(std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    size_t confident = 0;
    size_t total = 0;

    for (const auto& seq : predictions)
    {
      if (seq.empty())
      {
        continue;
      }

      // In Softmax, coverage is based on the confidence of the winning class.
      auto max_it = std::max_element(seq.begin(), seq.end());
      if (*max_it > evaluation_config.confidence_threshold())
      {
        ++confident;
      }

      ++total;
    }
    return (total == 0) ? 0.0 : static_cast<double>(confident) / total;
  }

  static double calculate_softmax_directional_confidence_score(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    size_t correct = 0;
    size_t total = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.empty() || pred.empty()) continue;

      // 1. Determine Midpoint for generic "Directional" logic
      // For N classes: indices < mid are DOWN, indices > mid are UP, index == mid is NEUTRAL.
      const size_t num_classes = pred.size();
      const double mid = (static_cast<double>(num_classes) - 1.0) / 2.0;

      // 2. Get Predicted Winning Index (ArgMax)
      auto max_pred_it = std::max_element(pred.begin(), pred.end());
      size_t pred_idx = std::distance(pred.begin(), max_pred_it);
      double confidence = *max_pred_it;

      // 3. Get Truth Winning Index
      auto max_gt_it = std::max_element(gt.begin(), gt.end());
      size_t gt_idx = std::distance(gt.begin(), max_gt_it);

      // 4. Filter by confidence and neutral predictions
      const bool is_pred_neutral = std::abs(static_cast<double>(pred_idx) - mid) < 0.1;
      if (is_pred_neutral || confidence < evaluation_config.confidence_threshold())
      {
        continue;
      }

      // 5. Filter by neutral truth
      const bool is_gt_neutral = std::abs(static_cast<double>(gt_idx) - mid) < 0.1;
      if (is_gt_neutral)
      {
        continue;
      }

      // 6. Directional Logic:
      bool predicted_up = (static_cast<double>(pred_idx) > mid);
      bool actual_up = (static_cast<double>(gt_idx) > mid);

      if (predicted_up == actual_up)
      {
        ++correct;
      }

      ++total;
    }

    return (total == 0) ? 0.0 : (static_cast<double>(correct) / total);
  }

  static double calculate_directional_confidence_score( std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    size_t correct = 0;
    size_t total = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        Logger::panic("Ground truth size mismatch.");
      }

      for (size_t i = 0; i < gt.size(); ++i)
      {
        double gt_val = gt[i];
        double pred_val = pred[i];

        // Ignore tiny real moves
        if (std::abs(gt_val) < evaluation_config.neutral_tolerance())
        {
          continue;
        }

        // Ignore weak predictions (confidence filter)
        if (std::abs(pred_val) < evaluation_config.confidence_threshold())
        {
          continue;
        }

        if ((gt_val * pred_val) > 0.0)
        {
          ++correct;
        }

        ++total;
      }
    }

    return (total == 0) ? 0.0 : (static_cast<double>(correct) / total);
  }

  static double calculate_softmax_directional_accuracy(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions,const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    size_t correct = 0;
    size_t total = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        Logger::panic("Dimension mismatch or empty vectors in accuracy calculation.");
      }

      const size_t num_classes = pred.size();
      const double mid = (static_cast<double>(num_classes) - 1.0) / 2.0;

      // 1. Find the index of the highest predicted probability
      auto max_pred_it = std::max_element(pred.begin(), pred.end());
      size_t pred_idx = std::distance(pred.begin(), max_pred_it);

      // 2. Find the index of the actual target
      auto max_gt_it = std::max_element(gt.begin(), gt.end());
      size_t gt_idx = std::distance(gt.begin(), max_gt_it);

      // 3. Skip samples where the ground truth is neutral
      const bool is_gt_neutral = std::abs(static_cast<double>(gt_idx) - mid) < 0.1;
      if (is_gt_neutral)
      {
        continue;
      }

      // 4. Directional Match: Are they in the same directional group?
      bool predicted_up = (static_cast<double>(pred_idx) > mid);
      bool actual_up = (static_cast<double>(gt_idx) > mid);

      if (predicted_up == actual_up)
      {
        ++correct;
      }

      ++total;
    }

    return (total == 0) ? 0.0 : (static_cast<double>(correct) / total);
  }

  static double calculate_directional_accuracy( std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const auto& neutral_tolerance = evaluation_config.neutral_tolerance();
    size_t correct = 0;
    size_t total = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        Logger::panic("The provided ground truth for directional accuracy is either not the correct size or is empty.");
      }

      for (size_t i = 0; i < gt.size(); ++i)
      {
        double gt_val = gt[i];
        double pred_val = pred[i];

        // Ignore small ground truth moves
        if (std::abs(gt_val) < neutral_tolerance)
        {
          continue;
        }

        if ((gt_val * pred_val) > 0.0)
        {
          ++correct;
        }

        ++total;
      }
    }

    return (total == 0) ? 0.0 : (static_cast<double>(correct) / total);
  }

  static double calculate_bce_loss( std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_bce = 0.0;
    size_t count = 0;

    // small epsilon to avoid log(0)
    const double eps = 1e-12;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      for (size_t i = 0; i < gt.size(); ++i)
      {
        // clip predictions to [eps, 1 - eps]
        double p = std::max(eps, std::min(1.0 - eps, pred[i]));
        double y = gt[i];

        total_bce += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        ++count;
      }
    }

    return (count == 0) ? 0.0 : (total_bce / count);
  }

  static double calculate_log_cosh( std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_log_cosh = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < ground_truths.size(); ++i)
    {
      if (ground_truths[i].size() != predictions[i].size())
      {
        Logger::panic("Mismatched vector sizes at index ", i);
      }

      for (size_t j = 0; j < ground_truths[i].size(); ++j)
      {
        const double x = predictions[i][j] - ground_truths[i][j];
        const double abs_x = std::abs(x);

        // Log-cosh(x) = ln(cosh(x))
        // For numerical stability:
        // ln(cosh(x)) = ln((e^x + e^-x) / 2) = ln(e^abs(x) * (1 + e^(-2*abs(x))) / 2)
        //             = abs(x) + ln(1 + e^(-2*abs(x))) - ln(2)
        // We use std::log1p(y) which is ln(1+y) for better precision with small y.
        total_log_cosh += abs_x + std::log1p(std::exp(-2.0 * abs_x)) - std::log(2.0);
        ++count;
      }
    }
    return (count > 0) ? (total_log_cosh / count) : 0.0;
  }

  static double calculate_cross_entropy( std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_loss = 0.0;
    size_t sequence_count = 0;
    const double eps = 1e-12;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      double sample_loss = 0.0;
      for (size_t i = 0; i < gt.size(); ++i)
      {
        if (gt[i] > 0.0)
        {
           double p = std::max(eps, std::min(1.0 - eps, pred[i]));
           sample_loss += -gt[i] * std::log(p);
        }
      }
      total_loss += sample_loss;
      ++sequence_count;
    }

    return (sequence_count == 0) ? 0.0 : (total_loss / sequence_count);
  }

  static double calculate_prediction_coverage( std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    size_t confident = 0;
    size_t total = 0;

    for (const auto& seq : predictions)
    {
      if (seq.empty())
      {
        Logger::panic("Prediction sequence cannot be empty.");
      }

      for (double v : seq)
      {
        if (std::abs(v) > evaluation_config.confidence_threshold())
        {
          ++confident;
        }

        ++total;
      }
    }
    return (total == 0) ? 0.0 : static_cast<double>(confident) / total;
  }
};