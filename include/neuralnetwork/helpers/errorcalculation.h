#pragma once
#include "../libraries/instrumentor.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <vector>

#include "../common/activation.h"
#include "../common/evaluationconfig.h"
#include "../common/logger.h"


namespace myoddweb::nn
{
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
    prediction_coverage,
    quantile_loss,
    sharpe_ratio_loss,
    sortino_ratio_loss
  };
private:
  [[nodiscard]] inline static bool iequals(const std::string& str, const char* lit)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    size_t i = 0;
    while (str[i] != '\0' && lit[i] != '\0')
    {
      if (static_cast<char>(std::tolower(static_cast<unsigned char>(str[i]))) != lit[i])
      {
        return false;
      }
      ++i;
    }
    return str[i] == '\0' && lit[i] == '\0';
  }
  [[nodiscard]] inline static std::vector<double> calculate_portfolio_returns(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, double transaction_cost_penalty)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const size_t num_samples = ground_truths.size();
    std::vector<double> returns(num_samples, 0.0);
    for (size_t t = 0; t < num_samples; ++t)
    {
      const auto& gt = ground_truths[t];
      const auto& pred = predictions[t];
      const size_t num_assets = std::min(gt.size(), pred.size());
      if (num_assets == 0)
      {
        continue;
      }

      double step_return = 0.0;
      for (size_t j = 0; j < num_assets; ++j)
      {
        const double pos = pred[j];
        const double prev_pos = (t > 0 && j < predictions[t - 1].size()) ? predictions[t - 1][j] : 0.0;
        const double cost = (t > 0) ? (transaction_cost_penalty * std::abs(pos - prev_pos)) : 0.0;
        step_return += (pos * gt[j] - cost);
      }
      returns[t] = step_return / static_cast<double>(num_assets);
    }
    return returns;
  }
public:
  [[nodiscard]] inline static std::string type_to_string(const ErrorCalculation::type& type)
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
    case type::quantile_loss:
      return "quantile-loss";
    case type::sharpe_ratio_loss:
      return "sharpe-ratio-loss";
    case type::sortino_ratio_loss:
      return "sortino-ratio-loss";
    }
    Logger::panic("Unknown ErrorCalculation type!");
  }

  [[nodiscard]] inline static type string_to_type(const std::string& str)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (iequals(str, "none"))
    {
      return type::none;
    }
    if (iequals(str, "huber-loss"))
    {
      return type::huber_loss;
    }
    if (iequals(str, "huber-direction-loss"))
    {
      return type::huber_direction_loss;
    }
    if (iequals(str, "mae"))
    {
      return type::mae;
    }
    if (iequals(str, "mse"))
    {
      return type::mse;
    }
    if (iequals(str, "rmse"))
    {
      return type::rmse;
    }
    if (iequals(str, "nrmse"))
    {
      return type::nrmse;
    }
    if (iequals(str, "mape"))
    {
      return type::mape;
    }
    if (iequals(str, "smape"))
    {
      return type::smape;
    }
    if (iequals(str, "wape"))
    {
      return type::wape;
    }
    if (iequals(str, "directional-accuracy"))
    {
      return type::directional_accuracy;
    }
    if (iequals(str, "directional-confidence-score"))
    {
      return type::directional_confidence_score;
    }
    if (iequals(str, "prediction-coverage"))
    {
      return type::prediction_coverage;
    }
    if (iequals(str, "bce-loss"))
    {
      return type::bce_loss;
    }
    if (iequals(str, "cross-entropy"))
    {
      return type::cross_entropy;
    }
    if (iequals(str, "log-cosh"))
    {
      return type::log_cosh;
    }
    if (iequals(str, "quantile-loss") || iequals(str, "pinball-loss") || iequals(str, "quantile"))
    {
      return type::quantile_loss;
    }
    if (iequals(str, "sharpe-ratio-loss") || iequals(str, "sharpe-loss") || iequals(str, "sharpe"))
    {
      return type::sharpe_ratio_loss;
    }
    if (iequals(str, "sortino-ratio-loss") || iequals(str, "sortino-loss") || iequals(str, "sortino"))
    {
      return type::sortino_ratio_loss;
    }
    Logger::panic("Unknown error type: ", str);

  }

  [[nodiscard]] static double calculate_error(type error_type, std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config, const activation::method& activation_method )
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
      return calculate_forecast_mape(ground_truths, predictions, evaluation_config);

    case type::wape:
      return calculate_forecast_wape(ground_truths, predictions);

    case type::smape:
      return calculate_forecast_smape(ground_truths, predictions, evaluation_config);

    case type::directional_accuracy:
      if (activation_method == activation::method::softmax)
      {
        return calculate_softmax_directional_accuracy(ground_truths, predictions);
      }
      return calculate_directional_accuracy(ground_truths, predictions, evaluation_config, activation_method);

    case type::directional_confidence_score:
      if (activation_method == activation::method::softmax)
      {
        return calculate_softmax_directional_confidence_score(ground_truths, predictions, evaluation_config);
      }
      return calculate_directional_confidence_score(ground_truths, predictions, evaluation_config, activation_method);

    case type::bce_loss:
      return calculate_bce_loss(ground_truths, predictions, evaluation_config);

    case type::cross_entropy:
      return calculate_cross_entropy(ground_truths, predictions, evaluation_config);

    case type::log_cosh:
      return calculate_log_cosh(ground_truths, predictions);

    case type::prediction_coverage:
      if (activation_method == activation::method::softmax)
      {
        return calculate_softmax_prediction_coverage(predictions, evaluation_config);
      }
      return calculate_prediction_coverage(predictions, evaluation_config, activation_method);

    case type::quantile_loss:
      return calculate_quantile_loss(ground_truths, predictions, evaluation_config);

    case type::sharpe_ratio_loss:
      return calculate_sharpe_ratio_loss(ground_truths, predictions, evaluation_config);

    case type::sortino_ratio_loss:
      return calculate_sortino_ratio_loss(ground_truths, predictions, evaluation_config);
    }

    Logger::panic("Unknown ErrorCalculation type!");
  }

  static double calculate_quantile_loss(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const auto& quantiles = evaluation_config.quantiles();
    double total_loss = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < ground_truths.size(); ++i)
    {
      const auto& gt_vec = ground_truths[i];
      const auto& pred_vec = predictions[i];

      if (gt_vec.size() != pred_vec.size())
      {
        Logger::panic("Mismatched vector sizes at index ", i);
      }

      const double* gt_ptr = gt_vec.data();
      const double* pred_ptr = pred_vec.data();
      const size_t vec_len = gt_vec.size();

      for (size_t j = 0; j < vec_len; ++j)
      {
        const double q = (j < quantiles.size()) ? quantiles[j] : (quantiles.empty() ? 0.5 : quantiles.back());
        const double error = gt_ptr[j] - pred_ptr[j];
        const double loss = (error >= 0.0) ? (q * error) : ((q - 1.0) * error);
        total_loss += loss;
        ++count;
      }
    }
    return (count > 0) ? (total_loss / static_cast<double>(count)) : 0.0;
  }

  static double calculate_sharpe_ratio_std_dev(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const double eps = evaluation_config.epsilon();
    if (ground_truths.empty() || predictions.empty() || ground_truths.size() != predictions.size())
    {
      return std::sqrt(eps);
    }

    const auto returns = calculate_portfolio_returns(ground_truths, predictions, evaluation_config.transaction_cost_penalty());
    const size_t num_samples = returns.size();

    double sum_returns = 0.0;
    for (const auto r : returns)
    {
      sum_returns += r;
    }
    const double mean_return = sum_returns / static_cast<double>(num_samples);

    double sum_sq_diff = 0.0;
    for (const auto r : returns)
    {
      const double diff = r - mean_return;
      sum_sq_diff += diff * diff;
    }

    const double variance = sum_sq_diff / static_cast<double>(num_samples);
    return std::sqrt(variance + eps);
  }

  static double calculate_sharpe_ratio(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (ground_truths.empty() || predictions.empty() || ground_truths.size() != predictions.size())
    {
      return 0.0;
    }

    const auto returns = calculate_portfolio_returns(ground_truths, predictions, evaluation_config.transaction_cost_penalty());
    const size_t num_samples = returns.size();

    double sum_returns = 0.0;
    for (const auto r : returns)
    {
      sum_returns += r;
    }
    const double mean_return = sum_returns / static_cast<double>(num_samples);
    const double std_dev = calculate_sharpe_ratio_std_dev(ground_truths, predictions, evaluation_config);

    return mean_return / std_dev;
  }

  static double calculate_sharpe_ratio_loss(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    return -calculate_sharpe_ratio(ground_truths, predictions, evaluation_config);
  }

  static double calculate_sortino_ratio_downside_std_dev(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const double eps = evaluation_config.epsilon();
    if (ground_truths.empty() || predictions.empty() || ground_truths.size() != predictions.size())
    {
      return std::sqrt(eps);
    }

    const auto returns = calculate_portfolio_returns(ground_truths, predictions, evaluation_config.transaction_cost_penalty());
    const size_t num_samples = returns.size();
    const double tau = evaluation_config.sortino_target_return();

    double sum_downside_sq = 0.0;
    for (const auto r : returns)
    {
      const double downside = std::min(0.0, r - tau);
      sum_downside_sq += downside * downside;
    }

    const double downside_variance = sum_downside_sq / static_cast<double>(num_samples);
    return std::sqrt(downside_variance + eps);
  }

  static double calculate_sortino_ratio(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (ground_truths.empty() || predictions.empty() || ground_truths.size() != predictions.size())
    {
      return 0.0;
    }

    const auto returns = calculate_portfolio_returns(ground_truths, predictions, evaluation_config.transaction_cost_penalty());
    const size_t num_samples = returns.size();
    const double tau = evaluation_config.sortino_target_return();

    double sum_returns = 0.0;
    for (const auto r : returns)
    {
      sum_returns += r;
    }
    const double mean_return = sum_returns / static_cast<double>(num_samples);
    const double downside_std_dev = calculate_sortino_ratio_downside_std_dev(ground_truths, predictions, evaluation_config);

    return (mean_return - tau) / downside_std_dev;
  }

  static double calculate_sortino_ratio_loss(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    return -calculate_sortino_ratio(ground_truths, predictions, evaluation_config);
  }

  static double calculate_huber_loss_error(std::span<const std::vector<double>> ground_truth, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const auto delta = evaluation_config.huber_delta();

    double total_loss = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
      const auto& gt_vec = ground_truth[i];
      const auto& pred_vec = predictions[i];

      if (gt_vec.size() != pred_vec.size())
      {
        Logger::panic("Mismatched vector sizes at index ", i);
      }

      const double* gt_ptr = gt_vec.data();
      const double* pred_ptr = pred_vec.data();
      const size_t vec_len = gt_vec.size();

      for (size_t j = 0; j < vec_len; ++j)
      {
        const double target = gt_ptr[j];
        const double output = pred_ptr[j];

        const double error = target - output;
        const double abs_error = std::abs(error);

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
    return (count > 0) ? (total_loss / static_cast<double>(count)) : 0.0;
  }

  static double calculate_huber_direction_loss(std::span<const std::vector<double>> ground_truth, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    const auto lambda = evaluation_config.direction_lambda();
    const auto delta = evaluation_config.huber_delta();
    const bool use_penalty = evaluation_config.use_direction_penalty();

    double total_loss = 0.0;
    size_t count = 0;

    const double scale = 100.0;

    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
      const auto& gt_vec = ground_truth[i];
      const auto& pred_vec = predictions[i];

      if (gt_vec.size() != pred_vec.size())
      {
        Logger::panic("Mismatched vector sizes at index ", i);
      }

      const double* gt_ptr = gt_vec.data();
      const double* pred_ptr = pred_vec.data();
      const size_t vec_len = gt_vec.size();

      for (size_t j = 0; j < vec_len; ++j)
      {
        const double target = gt_ptr[j];
        const double output = pred_ptr[j];

        const double error = target - output;
        const double abs_error = std::abs(error);

        double loss = 0.0;
        if (abs_error <= delta)
        {
          loss = 0.5 * error * error;
        }
        else
        {
          loss = delta * (abs_error - 0.5 * delta);
        }

        if (use_penalty && std::abs(target) > 1e-6)
        {
          const double x = -scale * target * output;
          const double direction_loss = (x > 0.0) ? (x + std::log1p(std::exp(-x))) : std::log1p(std::exp(x));
          loss += lambda * direction_loss;
        }

        total_loss += loss;
        ++count;
      }
    }

    return (count > 0) ? (total_loss / static_cast<double>(count)) : 0.0;
  }

  static double calculate_mae_error(std::span<const std::vector<double>> ground_truth, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_abs_error = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
      const auto& gt_vec = ground_truth[i];
      const auto& pred_vec = predictions[i];

      if (gt_vec.size() != pred_vec.size())
      {
        Logger::panic("Mismatched vector sizes at index ", i);
      }

      const double* gt_ptr = gt_vec.data();
      const double* pred_ptr = pred_vec.data();
      const size_t vec_len = gt_vec.size();

      for (size_t j = 0; j < vec_len; ++j)
      {
        total_abs_error += std::abs(gt_ptr[j] - pred_ptr[j]);
        ++count;
      }
    }
    return (count > 0) ? (total_abs_error / static_cast<double>(count)) : 0.0;
  }

  static double calculate_mse_error(std::span<const std::vector<double>> ground_truth, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_squared_error = 0.0;
    size_t valid_count = 0;
    size_t skipped_non_finite_count = 0;

    for (size_t i = 0; i < ground_truth.size(); ++i)
    {
      const auto& true_output = ground_truth[i];
      const auto& predicted_output = predictions[i];

      if (true_output.size() != predicted_output.size())
      {
        Logger::warning("Mismatch in output vector sizes at index ", i);
        continue;
      }

      const double* true_ptr = true_output.data();
      const double* pred_ptr = predicted_output.data();
      const size_t vec_len = true_output.size();

      for (size_t j = 0; j < vec_len; ++j)
      {
        const double error = pred_ptr[j] - true_ptr[j];

        if (!std::isfinite(error))
        {
          ++skipped_non_finite_count;
          continue;
        }

        const double squared_error = error * error;
        if (!std::isfinite(squared_error))
        {
          ++skipped_non_finite_count;
          continue;
        }
        total_squared_error += squared_error;
        ++valid_count;
      }
    }

    // Logged once per call (not per value) so a run of diverged/NaN predictions doesn't
    // flood the log; without this, non-finite predictions were silently excluded from both
    // the sum and the count, understating the reported error with no visible indication.
    if (skipped_non_finite_count > 0)
    {
      Logger::warning("calculate_mse_error skipped ", skipped_non_finite_count, " non-finite prediction error(s) out of ", (valid_count + skipped_non_finite_count), " total value(s).");
    }

    if (valid_count == 0)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return total_squared_error / static_cast<double>(valid_count);
  }

  static double calculate_rmse_error(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_rmse = 0.0;
    size_t sequence_count = 0;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      double mse = 0.0;
      for (size_t i = 0; i < vec_len; ++i)
      {
        const double diff = gt_ptr[i] - pred_ptr[i];
        mse += diff * diff;
      }

      mse /= static_cast<double>(vec_len);
      total_rmse += std::sqrt(mse);
      ++sequence_count;
    }

    return (sequence_count == 0) ? 0.0 : (total_rmse / static_cast<double>(sequence_count));
  }

  static double calculate_nrmse_error(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_nrmse = 0.0;
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

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      double mse = 0.0;
      double min_val = gt_ptr[0];
      double max_val = gt_ptr[0];
      double mean_abs = 0.0;

      for (size_t i = 0; i < vec_len; ++i)
      {
        const double val = gt_ptr[i];
        const double diff = val - pred_ptr[i];
        mse += diff * diff;

        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        mean_abs += std::abs(val);
      }

      mse /= static_cast<double>(vec_len);
      double rmse = std::sqrt(mse);
      mean_abs /= static_cast<double>(vec_len);

      double denom = max_val - min_val;
      if (denom < eps)
      {
        denom = mean_abs;
      }
      if (denom < eps)
      {
        continue;
      }

      total_nrmse += rmse / denom;
      ++sequence_count;
    }

    return (sequence_count == 0) ? 0.0 : (total_nrmse / static_cast<double>(sequence_count));
  }

  static double calculate_forecast_mape(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_mape = 0.0;
    size_t sequence_count = 0;
    const double eps = evaluation_config.epsilon();
    const bool can_trace_log = Logger::can_trace();

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      double seq_error_sum = 0.0;
      size_t count = 0;

      for (size_t i = 0; i < vec_len; ++i)
      {
        const double denom = std::abs(gt_ptr[i]);
        if (denom < eps)
        {
          continue;
        }
        seq_error_sum += std::abs((gt_ptr[i] - pred_ptr[i]) / denom);
        ++count;
      }

      if (count > 0)
      {
        total_mape += seq_error_sum / static_cast<double>(count);
        ++sequence_count;
      }

      if (can_trace_log)
      {
        Logger::trace("[MAPE_DEBUG] After sequence ", seq_idx, ": total_mape=", total_mape, ", sequence_count=", sequence_count);
      }
    }
    return (sequence_count == 0) ? 0.0 : (total_mape / static_cast<double>(sequence_count));
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

      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      for (size_t i = 0; i < vec_len; ++i)
      {
        total_absolute_error += std::abs(gt_ptr[i] - pred_ptr[i]);
        total_absolute_actuals += std::abs(gt_ptr[i]);
      }
    }

    if (total_absolute_actuals == 0.0)
    {
      return (total_absolute_error == 0.0) ? 0.0 : 1.0;
    }

    return total_absolute_error / total_absolute_actuals;
  }

  static double calculate_forecast_smape(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_smape = 0.0;
    size_t sequence_count = 0;
    const double eps = evaluation_config.epsilon();

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      double seq_error_sum = 0.0;
      size_t count = 0;

      for (size_t i = 0; i < vec_len; ++i)
      {
        const double denom = (std::abs(gt_ptr[i]) + std::abs(pred_ptr[i])) / 2.0;
        if (denom < eps)
        {
          continue;
        }
        seq_error_sum += std::abs(gt_ptr[i] - pred_ptr[i]) / denom;
        ++count;
      }

      if (count > 0)
      {
        total_smape += seq_error_sum / static_cast<double>(count);
        ++sequence_count;
      }
    }
    return (sequence_count == 0) ? 0.0 : (total_smape / static_cast<double>(sequence_count));
  }

  static double calculate_softmax_prediction_coverage(std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    size_t confident = 0;
    size_t total = 0;
    const double threshold = evaluation_config.confidence_threshold();

    for (const auto& seq : predictions)
    {
      if (seq.empty())
      {
        continue;
      }

      auto max_it = std::max_element(seq.begin(), seq.end());
      if (*max_it > threshold)
      {
        ++confident;
      }

      ++total;
    }
    return (total == 0) ? 0.0 : static_cast<double>(confident) / static_cast<double>(total);
  }

  static double calculate_softmax_directional_confidence_score(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    size_t correct = 0;
    size_t total = 0;
    const double threshold = evaluation_config.confidence_threshold();

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.empty() || pred.empty())
      {
        continue;
      }

      const size_t num_classes = pred.size();
      const double mid = (static_cast<double>(num_classes) - 1.0) / 2.0;

      auto max_pred_it = std::max_element(pred.begin(), pred.end());
      const size_t pred_idx = std::distance(pred.begin(), max_pred_it);
      const double confidence = *max_pred_it;

      auto max_gt_it = std::max_element(gt.begin(), gt.end());
      const size_t gt_idx = std::distance(gt.begin(), max_gt_it);

      const bool is_pred_neutral = std::abs(static_cast<double>(pred_idx) - mid) < 0.1;
      if (is_pred_neutral || confidence < threshold)
      {
        continue;
      }

      const bool is_gt_neutral = std::abs(static_cast<double>(gt_idx) - mid) < 0.1;
      if (is_gt_neutral)
      {
        continue;
      }

      const bool predicted_up = (static_cast<double>(pred_idx) > mid);
      const bool actual_up = (static_cast<double>(gt_idx) > mid);
      const bool match = (predicted_up == actual_up);

      if (match)
      {
        ++correct;
      }

      ++total;

      if (Logger::can_trace())
      {
        Logger::trace("[ErrorCalculation::calculate_softmax_directional_confidence_score] seq=", seq_idx, ", pred_idx=", pred_idx, ", gt_idx=", gt_idx, ", mid=", mid, ", match=", (match ? "OK" : "FAIL"));
      }
    }

    return (total == 0) ? 0.0 : (static_cast<double>(correct) / static_cast<double>(total));
  }

  static double calculate_directional_confidence_score(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config, const activation::method activation_method)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.empty())
    {
      return 0.0;
    }

    size_t confident = 0;
    size_t total = 0;

    if (activation_method == activation::method::softmax)
    {
      return calculate_softmax_directional_confidence_score(ground_truths, predictions, evaluation_config);
    }

    const double baseline = (activation_method == activation::method::sigmoid) ? 0.5 : 0.0;
    const double neutral_tolerance = evaluation_config.neutral_tolerance();
    const double confidence_threshold = evaluation_config.confidence_threshold();
    const bool can_trace_log = Logger::can_trace();

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        Logger::panic("Ground truth size mismatch.");
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      for (size_t i = 0; i < vec_len; ++i)
      {
        const double gt_val = gt_ptr[i];
        const double pred_val = pred_ptr[i];

        if (std::abs(gt_val - baseline) < neutral_tolerance)
        {
          continue;
        }

        if (std::abs(pred_val - baseline) < confidence_threshold)
        {
          continue;
        }

        const bool actual_up = (gt_val > baseline);
        const bool predicted_up = (pred_val > baseline);
        const bool match = (actual_up == predicted_up);

        if (match)
        {
          ++confident;
        }

        ++total;

        if (can_trace_log)
        {
          Logger::trace("[ErrorCalculation::calculate_directional_confidence_score] gt=", gt_val, ", pred=", pred_val, ", baseline=", baseline, ", actual_up=", (actual_up ? "Y" : "N"), ", pred_up=", (predicted_up ? "Y" : "N"), ", match=", (match ? "OK" : "FAIL"));
        }
      }
    }

    return (total == 0) ? 0.0 : (static_cast<double>(confident) / static_cast<double>(total));
  }

  static double calculate_softmax_directional_accuracy(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
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

      auto max_pred_it = std::max_element(pred.begin(), pred.end());
      const size_t pred_idx = std::distance(pred.begin(), max_pred_it);

      auto max_gt_it = std::max_element(gt.begin(), gt.end());
      const size_t gt_idx = std::distance(gt.begin(), max_gt_it);

      const bool is_gt_neutral = std::abs(static_cast<double>(gt_idx) - mid) < 0.1;
      if (is_gt_neutral)
      {
        continue;
      }

      const bool is_pred_neutral = std::abs(static_cast<double>(pred_idx) - mid) < 0.1;
      const bool predicted_up = (static_cast<double>(pred_idx) > mid);
      const bool actual_up = (static_cast<double>(gt_idx) > mid);

      const bool match = !is_pred_neutral && (predicted_up == actual_up);

      if (match)
      {
        ++correct;
      }

      ++total;

      if (Logger::can_trace())
      {
        Logger::trace("[ErrorCalculation::calculate_softmax_directional_accuracy] seq=", seq_idx, ", pred_idx=", pred_idx, ", gt_idx=", gt_idx, ", mid=", mid, ", match=", (match ? "OK" : "FAIL"));
      }
    }

    return (total == 0) ? 0.0 : (static_cast<double>(correct) / static_cast<double>(total));
  }

  static double calculate_directional_accuracy(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config, const activation::method activation_method)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    size_t correct = 0;
    size_t total = 0;

    const double baseline = (activation_method == activation::method::sigmoid) ? 0.5 : 0.0;
    const double neutral_tolerance = evaluation_config.neutral_tolerance();
    const bool can_trace_log = Logger::can_trace();

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        Logger::panic("The provided ground truth for directional accuracy is either not the correct size or is empty.");
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      for (size_t i = 0; i < vec_len; ++i)
      {
        const double gt_val = gt_ptr[i];
        const double pred_val = pred_ptr[i];

        if (std::abs(gt_val - baseline) < neutral_tolerance)
        {
          continue;
        }

        const bool actual_up = (gt_val > baseline);
        const bool predicted_up = (pred_val > baseline);
        const bool match = (actual_up == predicted_up);

        if (match)
        {
          ++correct;
        }

        ++total;

        if (can_trace_log)
        {
          Logger::trace("[ErrorCalculation::calculate_directional_accuracy] gt=", gt_val, ", pred=", pred_val, ", baseline=", baseline, ", actual_up=", (actual_up ? "Y" : "N"), ", pred_up=", (predicted_up ? "Y" : "N"), ", match=", (match ? "OK" : "FAIL"));
        }
      }
    }

    return (total == 0) ? 0.0 : (static_cast<double>(correct) / static_cast<double>(total));
  }

  [[nodiscard]] static std::vector<double> smooth_labels(std::span<const double> targets, double label_smoothing)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (label_smoothing < 0.0 || label_smoothing >= 1.0)
    {
      Logger::panic("The label smoothing factor must be in the range [0.0, 1.0)!");
    }
    std::vector<double> smoothed(targets.size(), 0.0);
    smooth_labels(targets, std::span<double>(smoothed), label_smoothing);
    return smoothed;
  }

  static void smooth_labels(std::span<const double> targets, std::span<double> destination, double label_smoothing)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (label_smoothing < 0.0 || label_smoothing >= 1.0)
    {
      Logger::panic("The label smoothing factor must be in the range [0.0, 1.0)!");
    }
    if (targets.size() != destination.size())
    {
      Logger::panic("Targets and destination spans must have the same size.");
    }
    if (targets.empty())
    {
      return;
    }
    if (label_smoothing == 0.0)
    {
      std::copy(targets.begin(), targets.end(), destination.begin());
      return;
    }
    const size_t num_classes = targets.size();
    const double smooth_prior = label_smoothing / static_cast<double>(num_classes);
    const double one_minus_smoothing = 1.0 - label_smoothing;
    for (size_t i = 0; i < num_classes; ++i)
    {
      destination[i] = targets[i] * one_minus_smoothing + smooth_prior;
    }
  }

  static double calculate_bce_loss(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_bce = 0.0;
    size_t count = 0;

    const auto eps = evaluation_config.epsilon();
    const double one_minus_eps = 1.0 - eps;
    const double label_smoothing = evaluation_config.label_smoothing();
    const double one_minus_smoothing = 1.0 - label_smoothing;
    const double half_smoothing = 0.5 * label_smoothing;

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      for (size_t i = 0; i < vec_len; ++i)
      {
        const auto p = std::clamp(pred_ptr[i], eps, one_minus_eps);
        const auto raw_y = gt_ptr[i];
        const auto y = (label_smoothing > 0.0) ? (raw_y * one_minus_smoothing + half_smoothing) : raw_y;

        total_bce += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        ++count;
      }
    }

    return (count == 0) ? 0.0 : (total_bce / static_cast<double>(count));
  }

  static double calculate_log_cosh(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_log_cosh = 0.0;
    size_t count = 0;
    const double log2_val = std::log(2.0);

    for (size_t i = 0; i < ground_truths.size(); ++i)
    {
      const auto& gt = ground_truths[i];
      const auto& pred = predictions[i];

      if (gt.size() != pred.size())
      {
        Logger::panic("Mismatched vector sizes at index ", i);
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();

      for (size_t j = 0; j < vec_len; ++j)
      {
        const double x = pred_ptr[j] - gt_ptr[j];
        const double abs_x = std::abs(x);

        total_log_cosh += abs_x + std::log1p(std::exp(-2.0 * abs_x)) - log2_val;
        ++count;
      }
    }
    return (count > 0) ? (total_log_cosh / static_cast<double>(count)) : 0.0;
  }

  static double calculate_cross_entropy(std::span<const std::vector<double>> ground_truths, std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    double total_loss = 0.0;
    size_t sequence_count = 0;
    const double eps = evaluation_config.epsilon();
    const double one_minus_eps = 1.0 - eps;
    const double cross_entropy_lambda = evaluation_config.cross_entropy_lambda();
    const double label_smoothing = evaluation_config.label_smoothing();

    for (size_t seq_idx = 0; seq_idx < ground_truths.size(); ++seq_idx)
    {
      const auto& gt = ground_truths[seq_idx];
      const auto& pred = predictions[seq_idx];

      if (gt.size() != pred.size() || gt.empty())
      {
        continue;
      }

      const double* gt_ptr = gt.data();
      const double* pred_ptr = pred.data();
      const size_t vec_len = gt.size();
      const double smooth_prior = (vec_len > 0 && label_smoothing > 0.0) ? (label_smoothing / static_cast<double>(vec_len)) : 0.0;
      const double one_minus_smoothing = 1.0 - label_smoothing;

      double sample_loss = 0.0;
      for (size_t i = 0; i < vec_len; ++i)
      {
        const double y = (label_smoothing > 0.0)
          ? (gt_ptr[i] * one_minus_smoothing + smooth_prior)
          : gt_ptr[i];

        if (y > 0.0)
        {
          const double p = std::clamp(pred_ptr[i], eps, one_minus_eps);
          sample_loss += -y * std::log(p);
        }
      }
      total_loss += sample_loss;
      ++sequence_count;
    }

    return (sequence_count == 0) ? 0.0 : (total_loss / static_cast<double>(sequence_count)) * cross_entropy_lambda;
  }

  static double calculate_prediction_coverage(std::span<const std::vector<double>> predictions, const EvaluationConfig& evaluation_config, const activation::method activation_method)
  {
    MYODDWEB_PROFILE_FUNCTION("ErrorCalculation");
    if (predictions.empty())
    {
      return 0.0;
    }

    size_t predicted = 0;
    size_t total = 0;
    const double threshold = evaluation_config.confidence_threshold();
    // Match the baseline convention used by calculate_directional_confidence_score/
    // calculate_directional_accuracy: sigmoid's neutral point is 0.5, everything else is 0.0.
    // Without this, "confidence" for a sigmoid head would be measured as raw magnitude
    // instead of distance from neutral, silently undercounting confident low-value (near-0,
    // i.e. confidently "down") predictions.
    const double baseline = (activation_method == activation::method::sigmoid) ? 0.5 : 0.0;

    for (const auto& seq : predictions)
    {
      if (seq.empty())
      {
        Logger::panic("Prediction sequence cannot be empty.");
      }

      if (activation_method == activation::method::softmax)
      {
        auto max_it = std::max_element(seq.begin(), seq.end());
        if (*max_it > threshold)
        {
          ++predicted;
        }
      }
      else
      {
        const double* seq_ptr = seq.data();
        const size_t seq_len = seq.size();
        bool any_confident = false;

        for (size_t j = 0; j < seq_len; ++j)
        {
          if (std::abs(seq_ptr[j] - baseline) > threshold)
          {
            any_confident = true;
            break;
          }
        }
        if (any_confident)
        {
          ++predicted;
        }
      }
      ++total;
    }
    return (total == 0) ? 0.0 : static_cast<double>(predicted) / static_cast<double>(total);
  }
};
} // namespace myoddweb::nn
