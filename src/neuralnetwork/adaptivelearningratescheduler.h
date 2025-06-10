#include <iostream>
#include <vector>
#include <deque>
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::min_element and std::max
#include <functional> // For std::function

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
  AdaptiveLearningRateScheduler(size_t patience = 10,
                                double reduction_factor = 0.5,
                                double min_improvement = 1e-4,
                                double min_learning_rate = 1e-6,
                                double explosion_factor = 2.0) : 
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
      std::cout << "[Scheduler] Gathering history... ("
                << _error_history.size() << "/" << _patience << ")\n";
      return current_learning_rate;
    }

    // --- Rule 1: Check for growing/exploding error ---
    // Average of all errors *except* the most recent one.
    double sum_of_past_errors = std::accumulate( _error_history.begin(), std::prev(_error_history.end()), 0.0);
    double avg_past_error = sum_of_past_errors / (_patience - 1);

    if (current_rmse > avg_past_error * _explosion_factor) 
    {
      std::cout << "[Scheduler] WARNING: Error is growing! Drastically reducing LR.\n";
      // Make a more drastic cut for explosions
      double new_lr = current_learning_rate * 0.1; // Aggressive reduction
      return std::max(new_lr, _min_learning_rate);
    }

    // --- Rule 2: Check for stalled learning (plateau) ---
    auto best_error_in_history = *(std::min_element(_error_history.begin(), _error_history.end()));
        
    // Improvement is how much lower the current error is than the best historical error
    auto improvement = best_error_in_history - current_rmse;

    if (improvement < _min_improvement) 
    {
      std::cout << "[Scheduler] Learning has stalled (plateau). Reducing LR.\n";
      double new_lr = current_learning_rate * _reduction_factor;
      // Don't let the learning rate go below the defined minimum
      return std::max(new_lr, _min_learning_rate);
    }
        
    // --- If neither rule is met, learning is progressing well ---
    std::cout << "[Scheduler] Error is improving. Keeping current LR.\n";
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