#pragma once

#include "../libraries/instrumentor.h"
#include "logger.h"
#include <cmath>
#include <vector>


namespace myoddweb::nn
{
class EvaluationConfig final
{
public:
  EvaluationConfig() noexcept :
    EvaluationConfig(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0)
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
  }

  EvaluationConfig(
    double neutral_tolerance,
    double confidence_threshold,
    double huber_delta,
    double direction_lambda,
    bool   use_direction_penalty,
    double cross_entropy_lambda,
    double epsilon,
    double label_smoothing,
    const std::vector<double>& quantiles,
    double transaction_cost_penalty,
    double sortino_target_return) :
    _neutral_tolerance(neutral_tolerance),
    _confidence_threshold(confidence_threshold),
    _huber_delta(huber_delta),
    _direction_lambda(direction_lambda),
    _use_direction_penalty(use_direction_penalty),
    _cross_entropy_lambda(cross_entropy_lambda),
    _epsilon(epsilon),
    _label_smoothing(label_smoothing),
    _quantiles(quantiles.empty() ? std::vector<double>{ 0.5 } : quantiles),
    _transaction_cost_penalty(transaction_cost_penalty),
    _sortino_target_return(sortino_target_return)
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    if (_label_smoothing < 0.0 || _label_smoothing >= 1.0)
    {
      Logger::panic("The label smoothing factor must be in the range [0.0, 1.0)!");
    }
    if (_transaction_cost_penalty < 0.0 || !std::isfinite(_transaction_cost_penalty))
    {
      Logger::panic("The transaction cost penalty must be non-negative!");
    }
    if (!std::isfinite(_sortino_target_return))
    {
      Logger::panic("The sortino target return must be a finite number!");
    }
    for (const auto q : _quantiles)
    {
      if (q <= 0.0 || q >= 1.0 || !std::isfinite(q))
      {
        Logger::panic("The quantile must be in the range (0.0, 1.0)!");
      }
    }
  }

  EvaluationConfig(const EvaluationConfig& src) noexcept :
    _neutral_tolerance(src._neutral_tolerance),
    _confidence_threshold(src._confidence_threshold),
    _huber_delta(src._huber_delta),
    _direction_lambda(src._direction_lambda),
    _use_direction_penalty(src._use_direction_penalty),
    _cross_entropy_lambda(src._cross_entropy_lambda),
    _epsilon(src._epsilon),
    _label_smoothing(src._label_smoothing),
    _quantiles(src._quantiles),
    _transaction_cost_penalty(src._transaction_cost_penalty),
    _sortino_target_return(src._sortino_target_return)
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
  }

  EvaluationConfig(EvaluationConfig&& src) noexcept :
    _neutral_tolerance(src._neutral_tolerance),
    _confidence_threshold(src._confidence_threshold),
    _huber_delta(src._huber_delta),
    _direction_lambda(src._direction_lambda),
    _use_direction_penalty(src._use_direction_penalty),
    _cross_entropy_lambda(src._cross_entropy_lambda),
    _epsilon(src._epsilon),
    _label_smoothing(src._label_smoothing),
    _quantiles(std::move(src._quantiles)),
    _transaction_cost_penalty(src._transaction_cost_penalty),
    _sortino_target_return(src._sortino_target_return)
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
  }

  EvaluationConfig& operator=(const EvaluationConfig& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    if (this != &src)
    {
      _neutral_tolerance = src._neutral_tolerance;
      _confidence_threshold = src._confidence_threshold;
      _huber_delta = src._huber_delta;
      _direction_lambda = src._direction_lambda;
      _use_direction_penalty = src._use_direction_penalty;
      _cross_entropy_lambda = src._cross_entropy_lambda;
      _epsilon = src._epsilon;
      _label_smoothing = src._label_smoothing;
      _quantiles = src._quantiles;
      _transaction_cost_penalty = src._transaction_cost_penalty;
      _sortino_target_return = src._sortino_target_return;
    }
    return *this;
  }

  EvaluationConfig& operator=(EvaluationConfig&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    if (this != &src)
    {
      _neutral_tolerance = src._neutral_tolerance;
      _confidence_threshold = src._confidence_threshold;
      _huber_delta = src._huber_delta;
      _direction_lambda = src._direction_lambda;
      _use_direction_penalty = src._use_direction_penalty;
      _cross_entropy_lambda = src._cross_entropy_lambda;
      _epsilon = src._epsilon;
      _label_smoothing = src._label_smoothing;
      _quantiles = std::move(src._quantiles);
      _transaction_cost_penalty = src._transaction_cost_penalty;
      _sortino_target_return = src._sortino_target_return;
    }
    return *this;
  }

  [[nodiscard]] inline double neutral_tolerance() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _neutral_tolerance;
  }
  [[nodiscard]] inline double confidence_threshold() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _confidence_threshold;
  }
  [[nodiscard]] inline double huber_delta() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _huber_delta;
  }
  [[nodiscard]] inline double direction_lambda() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _direction_lambda;
  }
  [[nodiscard]] inline bool use_direction_penalty() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _use_direction_penalty;
  }
  [[nodiscard]] inline double cross_entropy_lambda() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _cross_entropy_lambda;
  }
  [[nodiscard]] inline double epsilon() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _epsilon;
  }
  [[nodiscard]] inline double label_smoothing() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _label_smoothing;
  }
  [[nodiscard]] inline const std::vector<double>& quantiles() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _quantiles;
  }
  [[nodiscard]] inline double transaction_cost_penalty() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _transaction_cost_penalty;
  }
  [[nodiscard]] inline double sortino_target_return() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
    return _sortino_target_return;
  }
private:
  double _neutral_tolerance;
  double _confidence_threshold;
  double _huber_delta;
  double _direction_lambda;
  bool _use_direction_penalty;
  double _cross_entropy_lambda;
  double _epsilon;
  double _label_smoothing;
  std::vector<double> _quantiles;
  double _transaction_cost_penalty;
  double _sortino_target_return;
};

} // namespace myoddweb::nn

