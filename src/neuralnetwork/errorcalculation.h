#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include "./libraries/instrumentor.h"

#include <vector>
#include "logger.h"

class ErrorCalculation
{
public:
  enum class type
  {
    none,
    huber_loss,
    mae,
    mse,
    rmse,
    nrmse,
    mape,
    smape,
    wape,
    directional_accuracy,
    bce_loss
  };
public:

  static double calculate_error(type error_type, const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    switch (error_type)
    {
    case type::none:
      return 0.0;

    case type::huber_loss:
      return calculate_huber_loss_error(ground_truth, predictions);

    case type::mae:
      return calculate_mae_error(ground_truth, predictions);

    case type::mse:
      return calculate_mse_error(ground_truth, predictions);

    case type::rmse:
      return calculate_rmse_error(ground_truth, predictions);

    case type::nrmse:
      return calculate_nrmse_error(ground_truth, predictions);

    case type::mape:
      return calculate_forecast_mape(ground_truth, predictions);

    case type::wape:
      return calculate_forecast_wape(ground_truth, predictions);

    case type::smape:
      return calculate_forecast_smape(ground_truth, predictions);

    case type::directional_accuracy:
      return calculate_directional_accuracy(ground_truth, predictions);

    case type::bce_loss:
      return calculate_bce_loss(ground_truth, predictions);
    }

    Logger::panic("Unknown ErrorCalculation type!");
  }

  static double calculate_huber_loss_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions, double delta = 1.0)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
#if VALIDATE_DATA == 1
    if (ground_truth.size() != predictions.size())
    {
      Logger::panic("Mismatched number of samples");
    }
#endif

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

  static double calculate_mae_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (ground_truth.size() != predictions.size())
    {
      Logger::panic("Mismatched number of samples");
    }


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

  static double calculate_mse_error(const std::vector<std::vector<double>>& ground_truth, const std::vector<std::vector<double>>& predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (ground_truth.size() != predictions.size())
    {
      Logger::panic("Mismatch in batch sizes.");
    }

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

  static double calculate_rmse_error(const std::vector<std::vector<double>>& ground_truths,const std::vector<std::vector<double>>& predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.size() != ground_truths.size() || predictions.empty())
    {
      Logger::panic("Mismatched number of samples");
    }

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

  static double calculate_nrmse_error(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.size() != ground_truths.size() || predictions.empty())
    {
      Logger::panic("Input vectors must have the same, non-zero size.");
    }

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
  static double calculate_forecast_mape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon = 1e-8)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.size() != ground_truths.size() || predictions.empty())
    {
      Logger::panic("Input vectors must have the same, non-zero size.");
    }

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
    }
    return (sequence_count == 0) ? 0.0 : (total_mape / sequence_count);
  }

  static double calculate_forecast_wape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.size() != ground_truths.size() || predictions.empty())
    {
      Logger::panic("Input vectors must have the same, non-zero size.");
    }

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
  static double calculate_forecast_smape(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double epsilon = 1e-8)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.size() != ground_truths.size() || predictions.empty())
    {
      Logger::panic("Input vectors must have the same, non-zero size.");
    }

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

  static double calculate_directional_accuracy(const std::vector<std::vector<double>>& ground_truths, const std::vector<std::vector<double>>& predictions, double neutral_tolerance = 0.001 /* threshold below which movement is ignored*/ )
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.size() != ground_truths.size() || predictions.empty())
    {
      Logger::panic("Input vectors must have the same, non-zero size.");
    }

    size_t correct = 0;
    size_t total = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.size() < 2)
      {
        continue; // Skip sequences that are too short
      }

      for (size_t i = 1; i < gt.size(); ++i)
      {
        double gt_diff = gt[i] - gt[i - 1];
        double pred_diff = pred[i] - pred[i - 1];

        // Ignore negligible ground truth movements (noise)
        if (std::abs(gt_diff) < neutral_tolerance)
        {
          continue;
        }

        if ((gt_diff * pred_diff) > 0.0)
        {
          ++correct;
        }
        ++total;
      }
    }
    return (total == 0) ? 0.0 : (static_cast<double>(correct) / total);
  }

  static double calculate_bce_loss(
    const std::vector<std::vector<double>>& ground_truths,
    const std::vector<std::vector<double>>& predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.size() != ground_truths.size() || predictions.empty())
    {
      Logger::panic("Mismatched number of samples");
    }

    double total_bce = 0.0;
    size_t sequence_count = 0;

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

      double bce = 0.0;
      for (size_t i = 0; i < gt.size(); ++i)
      {
        // clip predictions to [eps, 1 - eps]
        double p = std::max(eps, std::min(1.0 - eps, pred[i]));
        double y = gt[i];

        bce += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
      }

      bce /= gt.size();    // average over outputs in sequence
      total_bce += bce;
      ++sequence_count;
    }

    return (sequence_count == 0) ? 0.0 : (total_bce / sequence_count);
  }
};