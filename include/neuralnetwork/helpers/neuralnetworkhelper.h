#pragma once
#include <chrono>
#include <memory>
#include <vector>

#include "../libraries/instrumentor.h"
#include "errorcalculation.h"
#include "neuralnetworkhelpermetrics.h"
#include "trainingmonitor.h"

namespace myoddweb::nn
{
class NeuralNetwork;
class NeuralNetworkHelper
{
public:
  NeuralNetworkHelper() = delete;
  NeuralNetworkHelper(const NeuralNetworkHelper& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    *this = src;
  }
  NeuralNetworkHelper(NeuralNetworkHelper&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    *this = std::move(src);
  }
  NeuralNetworkHelper& operator=(const NeuralNetworkHelper& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    if (this != &src)
    {
      _neural_network = src._neural_network;
      _learning_rate = src._learning_rate;
      _number_of_epoch = src._number_of_epoch;
      _epoch = src._epoch;
      _percent_complete = src._percent_complete;
      _training_inputs = src._training_inputs;
      _training_outputs = src._training_outputs;
      _training_indexes = src._training_indexes;
      _checking_indexes = src._checking_indexes;
      _final_check_indexes = src._final_check_indexes;
      _training_monitors = src._training_monitors;
      _duration_ms = src._duration_ms;
      _last_epoch_time = src._last_epoch_time;
      _epoch_durations = src._epoch_durations;
      _max_history_size = src._max_history_size;
      _duration_sum = src._duration_sum;
      _ring_buffer_index = src._ring_buffer_index;
    }
    return *this;
  }

  NeuralNetworkHelper& operator=(NeuralNetworkHelper&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    if (this != &src)
    {
      _neural_network = src._neural_network;
      _learning_rate = src._learning_rate;
      _number_of_epoch = src._number_of_epoch;
      _epoch = src._epoch;
      _percent_complete = src._percent_complete;
      _training_inputs = src._training_inputs;
      _training_outputs = src._training_outputs;
      _training_indexes = std::move(src._training_indexes);
      _checking_indexes = std::move(src._checking_indexes);
      _final_check_indexes = std::move(src._final_check_indexes);
      _training_monitors = std::move(src._training_monitors);
      _duration_ms = src._duration_ms;
      _last_epoch_time = std::move(src._last_epoch_time);
      _epoch_durations = std::move(src._epoch_durations);
      _max_history_size = src._max_history_size;
      _duration_sum = src._duration_sum;
      _ring_buffer_index = src._ring_buffer_index;
      src._neural_network = nullptr;
      src._learning_rate = 0;
      src._number_of_epoch = 0;
      src._epoch = 0;
      src._percent_complete = 0;
      src._duration_ms = 0.0;
      src._max_history_size = 10;
      src._duration_sum = 0.0;
      src._ring_buffer_index = 0;
      src._training_inputs = nullptr;
      src._training_outputs = nullptr;
    }
    return *this;
  }
  virtual ~NeuralNetworkHelper() = default;

  [[nodiscard]] inline double learning_rate() const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _learning_rate; 
  }
  void set_learning_rate(double learning_rate) noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    _learning_rate = learning_rate; 
  }

  [[nodiscard]] inline double duration_ms() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _duration_ms;
  }

  [[nodiscard]] inline unsigned number_of_epoch() const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _number_of_epoch; 
  }
  [[nodiscard]] inline unsigned epoch() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _epoch; 
  }
  [[nodiscard]] inline double percent_complete() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _percent_complete; 
  }
  [[nodiscard]] bool is_at_epoch_interval(double percent) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    if (percent <= 0.0)
    {
      return false;
    }
    const auto update_interval = std::max<unsigned>(1, static_cast<unsigned>(std::round(percent * _number_of_epoch)));
    return _epoch % update_interval == 0;
  }
  [[nodiscard]] inline size_t sample_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return _training_inputs != nullptr ? _training_inputs->size() : 0;
  }

  /**
   * @brief Gets the training monitor for a specific output layer.
   * @param output_layer The index of the output layer (0-based).
   * @return A const reference to the TrainingMonitor.
   * @throws std::runtime_error (via Logger::panic) if output_layer is out of bounds.
   */
  [[nodiscard]] inline const TrainingMonitor& training_monitor(unsigned output_layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    if (output_layer >= _training_monitors.size())
    {
      Logger::panic("Output layer index is out of bounds for the training monitors.");
    }
    return _training_monitors[output_layer];
  }

  /**
   * @brief Gets the mutable training monitor for a specific output layer.
   * @param output_layer The index of the output layer (0-based).
   * @return A reference to the TrainingMonitor.
   * @throws std::runtime_error (via Logger::panic) if output_layer is out of bounds.
   */
  [[nodiscard]] inline TrainingMonitor& training_monitor(unsigned output_layer)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    if (output_layer >= _training_monitors.size())
    {
      Logger::panic("Output layer index is out of bounds for the training monitors.");
    }
    return _training_monitors[output_layer];
  }

  std::vector<NeuralNetworkHelperMetrics> calculate_forecast_metric(ErrorCalculation::type error_type) const;
  std::vector<std::vector<NeuralNetworkHelperMetrics>> calculate_forecast_metrics(const std::vector<ErrorCalculation::type>& error_types, bool in_sample = true) const;

  NeuralNetworkHelper(
    NeuralNetwork& neural_network,
    double learning_rate,
    unsigned number_of_epoch,
    const std::vector<std::vector<double>>& training_inputs,
    const std::vector<std::vector<double>>& training_outputs
  ) noexcept;

  void set_epoch(unsigned epoch) noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    if (_last_epoch_time == std::chrono::steady_clock::time_point{})
    {
      _last_epoch_time = std::chrono::steady_clock::now();
      _duration_ms = 0.0;
      _duration_sum = 0.0;
      _ring_buffer_index = 0;
      _epoch_durations.clear();
      _epoch = epoch;
      _percent_complete = _number_of_epoch == 0 ? 0.0 : static_cast<double>(_epoch) / _number_of_epoch;
      return;
    }

    if (_epoch != epoch)
    {
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration<double, std::milli>(now - _last_epoch_time).count();

      if (_epoch_durations.size() < _max_history_size)
      {
        _epoch_durations.push_back(elapsed);
        _duration_sum += elapsed;
      }
      else
      {
        _duration_sum -= _epoch_durations[_ring_buffer_index];
        _epoch_durations[_ring_buffer_index] = elapsed;
        _duration_sum += elapsed;
        _ring_buffer_index = (_ring_buffer_index + 1) % _max_history_size;
      }

      _duration_ms = _duration_sum / _epoch_durations.size();
      _last_epoch_time = now;

      _epoch = epoch;
      _percent_complete = _number_of_epoch == 0 ? 0.0 : static_cast<double>(_epoch) / _number_of_epoch;
    }
  }

  void move_indexes (
    std::vector<size_t>&& training_indexes,
    std::vector<size_t>&& checking_indexes,
    std::vector<size_t>&& final_check_indexes
  )
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    _training_indexes = std::make_shared<std::vector<size_t>>(std::move(training_indexes));
    _checking_indexes = std::make_shared<std::vector<size_t>>(std::move(checking_indexes));
    _final_check_indexes = std::make_shared<std::vector<size_t>>(std::move(final_check_indexes));
  }

  void move_training_indexes(std::vector<size_t>&& training_indexes)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    _training_indexes = std::make_shared<std::vector<size_t>>(std::move(training_indexes));
  }
  [[nodiscard]] const std::vector<size_t>& training_indexes() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return *_training_indexes; 
  }
  [[nodiscard]] const std::vector<size_t>& checking_indexes() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return *_checking_indexes; 
  }
  [[nodiscard]] const std::vector<size_t>& final_check_indexes() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return *_final_check_indexes; 
  }
  [[nodiscard]] const std::vector<std::vector<double>>& training_inputs() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return *_training_inputs; 
  }
  [[nodiscard]] const std::vector<std::vector<double>>& training_outputs() const noexcept {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelper");
    return *_training_outputs; 
  }

  friend class NeuralNetwork;

private:
  NeuralNetwork* _neural_network;
  double _learning_rate;
  unsigned _number_of_epoch;
  unsigned _epoch;
  double _percent_complete;
  const std::vector<std::vector<double>>* _training_inputs;
  const std::vector<std::vector<double>>* _training_outputs;
  std::shared_ptr<std::vector<size_t>> _training_indexes;
  std::shared_ptr<std::vector<size_t>> _checking_indexes;
  std::shared_ptr<std::vector<size_t>> _final_check_indexes;
  std::vector<TrainingMonitor> _training_monitors;
  double _duration_ms;
  std::chrono::steady_clock::time_point _last_epoch_time;
  std::vector<double> _epoch_durations;
  size_t _max_history_size;
  double _duration_sum;
  size_t _ring_buffer_index;

};

} // namespace myoddweb::nn
