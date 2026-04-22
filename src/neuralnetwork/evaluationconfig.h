#pragma once

#include "./libraries/instrumentor.h"

class EvaluationConfig final
{
public:
  EvaluationConfig() noexcept :
    EvaluationConfig(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12)
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
    double epsilon = 1e-12) noexcept :
    _neutral_tolerance(neutral_tolerance),
    _confidence_threshold(confidence_threshold),
    _huber_delta(huber_delta),
    _direction_lambda(direction_lambda),
    _use_direction_penalty(use_direction_penalty),
    _cross_entropy_lambda(cross_entropy_lambda),
    _epsilon(epsilon)
  {
    MYODDWEB_PROFILE_FUNCTION("EvaluationConfig");
  }

  EvaluationConfig(const EvaluationConfig& src) noexcept :
    _neutral_tolerance(src._neutral_tolerance),
    _confidence_threshold(src._confidence_threshold),
    _huber_delta(src._huber_delta),
    _direction_lambda(src._direction_lambda),
    _use_direction_penalty(src._use_direction_penalty),
    _cross_entropy_lambda(src._cross_entropy_lambda),
    _epsilon(src._epsilon)
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
private:
  double _neutral_tolerance;
  double _confidence_threshold;
  double _huber_delta;
  double _direction_lambda;
  bool _use_direction_penalty;
  double _cross_entropy_lambda;
  double _epsilon = 1e-8;
};
