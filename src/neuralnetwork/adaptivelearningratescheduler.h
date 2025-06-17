#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <numeric>
#include <vector>

#include "logger.h"

class AdaptiveLearningRateScheduler 
{
private:
  static constexpr int CoolDownExploding = 2;
  static constexpr int CoolDownPlateau = 10;
  static constexpr int CoolDownDecreasing = 20;
  static constexpr int CoolDownIncrease = 10;

  enum RateState
  {
    Stable, 
    Decreasing,
    Plateauing,
    Increasing,
    Exploding
  };
public:
  AdaptiveLearningRateScheduler(
    Logger& logger,
    size_t history_size = 10,
    double min_percent_change = 0.005, // percent (0 <> 1)
    double adjustment_rate = 0.1)
    : 
    _logger(logger),
    _history_size(history_size),
    _min_percent_change(min_percent_change),
    _adjustmentRate(adjustment_rate),
    _cool_down(0),
    _max_learning_rate(0.0)
  {
    assert(min_percent_change >= 0 && min_percent_change <= 1.0);
  }

  ~AdaptiveLearningRateScheduler() = default;
  AdaptiveLearningRateScheduler(const AdaptiveLearningRateScheduler&) = delete;
  AdaptiveLearningRateScheduler& operator=(const AdaptiveLearningRateScheduler&) = delete;
  AdaptiveLearningRateScheduler(AdaptiveLearningRateScheduler&&) = delete;
  AdaptiveLearningRateScheduler& operator=(AdaptiveLearningRateScheduler&&) = delete;

  double update(double currentError, double current_learning_rate) 
  {
    if (_max_learning_rate == 0)
    {
      //  set the max learning rate.
      _max_learning_rate = 10* current_learning_rate;
    }

    // Store error history
    _error_history.push_back(currentError);
    if (_error_history.size() > _history_size)
    {
      _error_history.pop_front();
    }

    if (_error_history.size() < _history_size)
    {
      return current_learning_rate; // Not enough data yet
    }

    if (_cool_down > 0)
    {
      --_cool_down;
      return current_learning_rate; // Cooling down
    }

    double new_learning_rate = current_learning_rate;
    switch (get_rate_change())
    {
    case RateState::Stable:
      return current_learning_rate;

    case RateState::Decreasing:
    {
      double new_learning_rate = clamp_learning_rate(current_learning_rate * (1.0 + (_adjustmentRate / 2.0)));
      if (!will_change(current_learning_rate, new_learning_rate))
      {
        return current_learning_rate;
      }
      _cool_down = CoolDownDecreasing;
      _logger.log_info("Learning is improving. Changing learning rate from "
        , std::fixed, std::setprecision(15), current_learning_rate
        , " to "
        , std::fixed, std::setprecision(15), new_learning_rate);
      _logger.log_info("Cooldown set to ", _cool_down);
      return new_learning_rate;
    }

    case RateState::Plateauing:
    {
      double new_learning_rate = clamp_learning_rate(current_learning_rate * (1.0 - _adjustmentRate)); // Mild reduce
      if (!will_change(current_learning_rate, new_learning_rate))
      {
        return current_learning_rate;
      }
      _cool_down = CoolDownPlateau;
      _logger.log_info("Learning is plateauing. Decreasing learning rate from "
        , std::fixed, std::setprecision(15), current_learning_rate
        , " to "
        , std::fixed, std::setprecision(15), new_learning_rate);
      _logger.log_info("Cooldown set to ", _cool_down);
      return new_learning_rate;
    }

    case RateState::Increasing:
    {
      double new_learning_rate = clamp_learning_rate(current_learning_rate * (1.0 - _adjustmentRate * 1.5)); // Reduce faster
      if (!will_change(current_learning_rate, new_learning_rate))
      {
        return current_learning_rate;
      }
      _cool_down = CoolDownIncrease;
      _logger.log_warning("Learning is increasing! Changing learning rate from "
        , std::fixed, std::setprecision(15), current_learning_rate
        , " to "
        , std::fixed, std::setprecision(15), new_learning_rate);
      _logger.log_info("Cooldown set to ", _cool_down);
      return new_learning_rate;
    }

    case RateState::Exploding:
    {
      double new_learning_rate = clamp_learning_rate(current_learning_rate * (1.0 - _adjustmentRate * 2.0)); // Reduce even faster
      if (!will_change(current_learning_rate, new_learning_rate))
      {
        return current_learning_rate;
      }
      _cool_down = CoolDownExploding;
      _logger.log_warning("Learning is increasing! Changing learning rate from "
        , std::fixed, std::setprecision(15), current_learning_rate
        , " to "
        , std::fixed, std::setprecision(15), new_learning_rate);
      _logger.log_info("Cooldown set to ", _cool_down);
      return new_learning_rate;
    }

    default:
      _logger.log_error("Learning Rate Scheduler: unknown state!");
    }
    return current_learning_rate;
  }

private:
  Logger& _logger;
  std::deque<double> _error_history;
  size_t _history_size;
  double _min_percent_change;  // Minimum % change to consider it significant
  double _adjustmentRate;
  int _cool_down;
  double _max_learning_rate;

  double clamp_learning_rate(double new_learning_rate) const
  {
    new_learning_rate = std::clamp(new_learning_rate, 1e-6, _max_learning_rate);
    return new_learning_rate;
  }
  
  bool will_change(double current_rate, double new_rate) const
  {
    return current_rate != new_rate;
  }

  RateState get_rate_change() const
  {
    const size_t error_size = _error_history.size();
    if (error_size < 2)
    {
      return RateState::Stable;
    }

    size_t decreaseCount = 0;
    size_t plateauCount = 0;
    size_t increaseCount = 0;
    size_t explodingCount = 0;
    size_t explodingPattern = 0;
    size_t comparisons = error_size / 2;

    for (size_t i = error_size - comparisons; i < error_size - 1; ++i)
    {
      auto change = percent_change(_error_history[i], _error_history[i + 1]);

      // increasing or decreasing less than the percent request
      if (std::fabs(change) <= _min_percent_change)
      {
        ++plateauCount;
      }

      // decreasing
      if (change < 0 && change <= -_min_percent_change)
      {
        ++decreaseCount;
      }

      // increasing
      if (change > 0 && change >= _min_percent_change)
      {
        ++increaseCount;
      }

      // exploding
      if (change > 0 && change >= 2*_min_percent_change)
      {
        ++explodingCount;
      }

      // also exploding if the error is > than a certain number
      if (_error_history[i + 1] > 5.00)
      {
        ++explodingPattern;
      }
    }
    if (explodingPattern >= comparisons - 1)
    {
      _logger.log_warning("Exploding pattern detected!");
      return RateState::Exploding;
    }
    if (explodingCount >= comparisons - 1)
    {
      return RateState::Exploding;
    }
    if (increaseCount >= comparisons - 1)
    {
      return RateState::Increasing;
    }
    if (decreaseCount >= comparisons - 1)
    {
      return RateState::Decreasing;
    }
    if (plateauCount >= comparisons - 1)
    {
      return RateState::Plateauing;
    }
    // no change.
    return RateState::Stable;
  }

  double percent_change(double oldValue, double newValue) const
  {
    // Handle the edge case where the old value is 0 to avoid division by zero.
    if (oldValue == 0.0) 
    {
      if (newValue == 0.0) 
      {
        // The change from 0 to 0 is 0%.
        return 0.0;
      }
      else 
      {
        // Any change from 0 to a non-zero number is technically an infinite percent change.
        return std::numeric_limits<double>::infinity();
      }
    }

    // Apply the standard formula for percent change.
    double change = newValue - oldValue;
    return (change / oldValue) * 100.0;
  }
};