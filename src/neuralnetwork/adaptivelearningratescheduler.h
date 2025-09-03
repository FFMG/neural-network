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
  static constexpr size_t CoolDownExploding = 1;
  static constexpr size_t CoolDownIncrease = 10;
  static constexpr size_t CoolDownPlateau = 3;
  static constexpr size_t CoolDownDecreasing = 10;

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
    size_t history_size = 25,
    double min_plateau_percent_change = 0.0005, // percent (0 <> 1)
    double min_percent_change = 0.005, // percent (0 <> 1)
    double adjustment_rate = 0.1)
    : 
    _history_size(history_size),
    _min_plateau_percent_change(min_plateau_percent_change),
    _min_percent_change(min_percent_change),
    _adjustment_rate(adjustment_rate),
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

  double update(double currentError, double current_learning_rate, int epoch, int number_of_epoch)
  {
    if (_max_learning_rate == 0)
    {
      //  set the max learning rate.
      _max_learning_rate = std::clamp(2* current_learning_rate, current_learning_rate, 0.99);
      Logger::debug("Adaptive Learning Rate max value set to: ", std::fixed, std::setprecision(15), _max_learning_rate);
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

    switch (get_rate_change())
    {
    case RateState::Stable:
      return current_learning_rate;

    case RateState::Decreasing:
    {
      double new_learning_rate = clamp_learning_rate(current_learning_rate * (1.0 + (_adjustment_rate / 2.0)));
      if (!will_change(current_learning_rate, new_learning_rate))
      {
        return current_learning_rate;
      }
      _cool_down = static_cast<int>( CoolDownDecreasing * _history_size);
      Logger::info("Learning is improving. Changing learning rate from "
        , std::fixed, std::setprecision(15), current_learning_rate
        , " to "
        , std::fixed, std::setprecision(15), new_learning_rate);
      Logger::debug("Cooldown set to ", _cool_down);
      return new_learning_rate;
    }

    case RateState::Plateauing:
    {
      double new_learning_rate = clamp_learning_rate(linear_decay(current_learning_rate, epoch, number_of_epoch)); // Mild reduce
      if (!will_change(current_learning_rate, new_learning_rate))
      {
        return current_learning_rate;
      }
      _cool_down = static_cast<int>(CoolDownPlateau * _history_size);
      Logger::info("Learning is plateauing. Changing learning down rate from "
        , std::fixed, std::setprecision(15), current_learning_rate
        , " to "
        , std::fixed, std::setprecision(15), new_learning_rate);
      Logger::debug("Cooldown set to ", _cool_down);
      return new_learning_rate;
    }

    case RateState::Increasing:
    {
      double new_learning_rate = clamp_learning_rate(current_learning_rate * (1.0 - _adjustment_rate * 1.5)); // Reduce faster
      if (!will_change(current_learning_rate, new_learning_rate))
      {
        return current_learning_rate;
      }
      _cool_down = static_cast<int>(CoolDownIncrease * _history_size);
      Logger::warning("Learning is increasing! Changing learning rate from "
        , std::fixed, std::setprecision(15), current_learning_rate
        , " to "
        , std::fixed, std::setprecision(15), new_learning_rate);
      Logger::debug("Cooldown set to ", _cool_down);
      return new_learning_rate;
    }

    case RateState::Exploding:
    {
      double new_learning_rate = clamp_learning_rate(current_learning_rate * (1.0 - _adjustment_rate * 2.0)); // Reduce even faster
      if (!will_change(current_learning_rate, new_learning_rate))
      {
        return current_learning_rate;
      }
      _cool_down = static_cast<int>(CoolDownExploding * _history_size);
      Logger::error("Exploding pattern detected! Changing learning rate from "
        , std::fixed, std::setprecision(15), current_learning_rate
        , " to "
        , std::fixed, std::setprecision(15), new_learning_rate);
      Logger::debug("Cooldown set to ", _cool_down);
      return new_learning_rate;
    }

    default:
      Logger::error("Learning Rate Scheduler: unknown state!");
    }
    return current_learning_rate;
  }

private:
  std::deque<double> _error_history;
  size_t _history_size;
  double _min_plateau_percent_change;
  double _min_percent_change;  // Minimum % change to consider it significant
  double _adjustment_rate;
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
    if (error_size < _history_size)
    {
      return RateState::Stable;
    }

    size_t decreaseCount = 0;
    size_t plateau = 0;
    size_t increaseCount = 0;
    size_t explodingCount = 0;
    size_t explodingPattern = 0;
    size_t comparisons = error_size / 2;
    size_t plateau_comparisons = error_size / 4;
    plateau_comparisons = std::max(plateau_comparisons, size_t(4));

    for (size_t i = error_size - comparisons; i < error_size - 1; ++i)
    {
      auto change = percent_change(_error_history[i], _error_history[i + 1]);

      // increasing or decreasing less than the percent request
      if (std::fabs(change) <= _min_plateau_percent_change)
      {
        ++plateau;
      }

      // decreasing
      // for decrease to cause a change we must be doing really well.
      // so we will be looking for 2x the percent change
      if (change < 0 && change <= (- 2 * _min_percent_change))
      {
        ++decreaseCount;
      }

      // increasing
      if (change > 0 && change >= _min_percent_change)
      {
        ++increaseCount;
      }

      // exploding
      if (change > 0 && change >= (2*_min_percent_change))
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
    if (plateau >= plateau_comparisons - 1)
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

  double linear_decay(double current_learning_rate, int epoch, int number_of_epoch)
  {
    return current_learning_rate * (1.0 - static_cast<double>(epoch) / number_of_epoch);
  }
};