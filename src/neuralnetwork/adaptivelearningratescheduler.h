#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

class AdaptiveLearningRateScheduler 
{
public:
  AdaptiveLearningRateScheduler(
    double max_learning_rate,
    size_t history_size = 10,
    double min_percent_change = 0.5, // percent
    double adjustment_rate = 0.1)
    : _history_size(history_size),
    _min_percent_change(min_percent_change),
    _adjustmentRate(adjustment_rate),
    _cool_down(0),
    _max_learning_rate(max_learning_rate)
  {
  }

  ~AdaptiveLearningRateScheduler() = default;
  AdaptiveLearningRateScheduler(const AdaptiveLearningRateScheduler&) = delete;
  AdaptiveLearningRateScheduler& operator=(const AdaptiveLearningRateScheduler&) = delete;

  double update(double currentError, double current_learning_rate) 
  {
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

    // Analyze trend
    bool decreasing = is_decreasing();
    if (decreasing) 
    {
      if (is_improving())
      {
        double new_learning_rate = current_learning_rate;
        if (current_learning_rate >= _max_learning_rate)
        {
          return _max_learning_rate;
        }

        // Increase learning rate if training is improving steadily
        new_learning_rate *= (1.0 + (_adjustmentRate / 2.0));
        if (new_learning_rate >= _max_learning_rate)
        {
          new_learning_rate = _max_learning_rate;
        }
        std::cout << "Learning is increasing! Changing learning rate from "
          << std::fixed << std::setprecision(15) << current_learning_rate
          << " to "
          << std::fixed << std::setprecision(15) << new_learning_rate
          << std::endl;
        _cool_down = 10;
        return new_learning_rate;
      }
      return current_learning_rate; // Good progress
    }

    double new_learning_rate = current_learning_rate;
    bool increasing = is_increasing();
    bool plateauing = is_plateauing();
    if (plateauing) 
    {
      new_learning_rate *= (1.0 - _adjustmentRate); // Mild reduce
    }
    else if (increasing) 
    {
      new_learning_rate *= (1.0 - _adjustmentRate * 1.5); // Reduce faster
    }
    else
    {
      // no change ...
      return current_learning_rate;
    }

    _cool_down = 10;
    new_learning_rate = std::clamp(new_learning_rate, 1e-6, 1.0);
    if (plateauing)
    {
      std::cout << "Learning is plateauing. Decreasing learning rate from "
                << std::fixed << std::setprecision(15) << current_learning_rate
                << " to "
                << std::fixed << std::setprecision(15) << new_learning_rate
                << std::endl;
      return new_learning_rate;
    }

    std::cout << "Learning is increasing! Changing learning rate from "
      << std::fixed << std::setprecision(15) << current_learning_rate
      << " to "
      << std::fixed << std::setprecision(15) << new_learning_rate
      << std::endl;
    return new_learning_rate;
  }

private:
  std::deque<double> _error_history;
  size_t _history_size;
  double _min_percent_change;  // Minimum % change to consider it significant
  double _adjustmentRate;
  int _cool_down;
  const double _max_learning_rate;

  double percent_change(double from, double to) const
  {
    if (from == 0.0) return 0.0;
    return ((from - to) / from) * 100.0;
  }

  bool is_improving() const
  {
    const size_t error_size = _error_history.size();
    if (error_size < 2)
    {
      return false;
    }

    size_t plateauCount = 0;
    size_t comparisons = error_size / 2;

    for (size_t i = error_size - comparisons; i < error_size - 1; ++i)
    {
      double change = percent_change(_error_history[i], _error_history[i + 1]);
      if (change > _min_percent_change)
      {
        ++plateauCount;
      }
    }
    return plateauCount == comparisons;
  }

  bool is_decreasing() const
  {
    return percent_change(_error_history.front(), _error_history.back()) > _min_percent_change;
  }

  bool is_increasing() const
  {
    return percent_change(_error_history.back(), _error_history.front()) > _min_percent_change;
  }

  bool is_plateauing() const
  {
    const size_t error_size = _error_history.size();
    if (error_size < 2)
    {
      return false;
    }

    size_t plateauCount = 0;
    size_t comparisons = error_size / 2;

    for (size_t i = error_size - comparisons; i < error_size - 1; ++i) 
    {
      double change = percent_change(_error_history[i], _error_history[i + 1]);
      if (std::fabs(change) <= _min_percent_change) 
      {
        ++plateauCount;
      }
    }
    return plateauCount == comparisons;
  }
};