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
  /**
   * @brief Constructor for the scheduler.
   * @param patience The number of past errors to consider ('x').
   * @param reduction_factor Factor to multiply LR by on a plateau (e.g., 0.5).
   * @param min_improvement The minimum error decrease to not be considered a plateau.
   * @param min_learning_rate The lowest the learning rate can go.
   * @param explosion_factor If current error is this many times the average, it's growing (e.g., 2.0).
   */
  AdaptiveLearningRateScheduler(size_t patience          = 10,
                                double reduction_factor  = 0.9,
                                double min_improvement   = 0.000001,
                                double min_learning_rate = 0.000001,
                                double explosion_factor  = 2.0) : 
      _patience(patience),
      _reduction_factor(reduction_factor),
      _min_improvement(min_improvement),
      _min_learning_rate(min_learning_rate),
      _explosion_factor(explosion_factor)
  {
    if (_patience < 2) {
        _patience = 2; // Need at least 2 errors to compare.
    }
  }

  ~AdaptiveLearningRateScheduler() = default;
  AdaptiveLearningRateScheduler(const AdaptiveLearningRateScheduler&) = delete;
  AdaptiveLearningRateScheduler& operator=(const AdaptiveLearningRateScheduler&) = delete;

  /**
   * @brief Main method to update the learning rate.
   * @param current_rmse The RMSE from the latest training step/batch.
   * @param current_learning_rate The learning rate used for the latest step.
   * @return The updated (or same) learning rate for the next step.
   */
  double update(double current_rmse, double current_learning_rate) 
  {
    if (!std::isfinite(current_rmse))
    {
      std::cerr << "[Scheduler] WARNING: Invalid Error: " << std::fixed << std::setprecision(15) << current_rmse << std::endl;
      return current_learning_rate;
    }
    // Add current error to our history
    _error_history.push_back(current_rmse);
        
    // Keep the history at the desired size ('patience')
    if (_error_history.size() > _patience) 
    {
      _error_history.pop_front();
    }

    // --- We need enough history to make a decision ---
    if (_error_history.size() < _patience) 
    {
      // [Scheduler] Gathering history... (" << _error_history.size() << "/" << _patience << ")"
      return current_learning_rate;
    }

    // --- Rule 1: Check for growing/exploding error ---
    // Average of all errors *except* the most recent one.
    double sum_of_past_errors = std::accumulate( _error_history.begin(), std::prev(_error_history.end()), 0.0);
    double avg_past_error = sum_of_past_errors / (_patience - 1);

    if (current_rmse > avg_past_error * _explosion_factor) 
    {
      // Make a more drastic cut for explosions
      double new_learning_rate = current_learning_rate * 0.1; // Aggressive reduction
      if (new_learning_rate < _min_learning_rate)
      {
        new_learning_rate = _min_learning_rate;
        std::cerr << "[Scheduler] WARNING: Error is growing but has reached minimums: " << std::fixed << std::setprecision(15) << new_learning_rate << std::endl;
      }
      else
      {
        std::cerr << "[Scheduler] WARNING: Error is growing! Drastically reducing learning rate to: " << std::fixed << std::setprecision(15) << new_learning_rate << std::endl;
      }
      return new_learning_rate;
    }

    // --- Rule 2: Check for stalled learning (plateau) ---
    auto best_error_in_history = *(std::min_element(_error_history.begin(), _error_history.end()));
        
    // Improvement is how much lower the current error is than the best historical error
    auto improvement = best_error_in_history - current_rmse;

    if (improvement < _min_improvement) 
    {
      if (_min_learning_rate == current_learning_rate)
      {
        return _min_learning_rate;
      }
      double new_learning_rate = current_learning_rate * _reduction_factor;
      if (new_learning_rate < _min_learning_rate)
      {
        new_learning_rate = _min_learning_rate;
        std::cout << "[Scheduler] Learning has stalled (plateau) but rate has reached minimum to: " << std::fixed << std::setprecision(15) << new_learning_rate << std::endl;
      }
      else
      {
        std::cout << "[Scheduler] Learning has stalled (plateau), reducing learning rate to: " << std::fixed << std::setprecision(15) << new_learning_rate << std::endl;
        _error_history.clear();
      }      
      return new_learning_rate;
    }
    // [Scheduler] Error is improving. Keeping current learning rate at: " << std::fixed << std::setprecision(15) << current_learning_rate
    return current_learning_rate;
  }

private:
  std::deque<double> _error_history;
  size_t _patience;
  double _reduction_factor;
  double _min_improvement;
  double _min_learning_rate;
  double _explosion_factor;
};