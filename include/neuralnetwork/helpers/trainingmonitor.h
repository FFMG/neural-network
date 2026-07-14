#pragma once

#include <unordered_map>
#include "errorcalculation.h"
#include "../common/logger.h"
#include "../libraries/instrumentor.h"


namespace myoddweb::nn
{
class TrainingMonitor
{
public:
  enum class TrainingStatus
  {
    OnTrack,
    Stuck,
    Diverging
  };

  TrainingMonitor(size_t window_size = 5,
    double da_weight = 0.6,
    double rmse_weight = 0.4,
    double da_threshold = 0.52,
    double rmse_tolerance = 1e-3) noexcept
    : 
    _min_window(window_size),
    _da_weight(da_weight),
    _rmse_weight(rmse_weight),
    _da_threshold(da_threshold),
    _rmse_tolerance(rmse_tolerance)
  {
    MYODDWEB_PROFILE_FUNCTION("TrainingMonitor");
  }

  TrainingMonitor(const TrainingMonitor& src) noexcept :
    _metrics(src._metrics),
    _min_window(src._min_window),
    _da_weight(src._da_weight),
    _rmse_weight(src._rmse_weight),
    _da_threshold(src._da_threshold),
    _rmse_tolerance(src._rmse_tolerance)
  {
    MYODDWEB_PROFILE_FUNCTION("TrainingMonitor");
  }

  TrainingMonitor(TrainingMonitor&& src) noexcept :
    _metrics(std::move(src._metrics)),
    _min_window(src._min_window),
    _da_weight(src._da_weight),
    _rmse_weight(src._rmse_weight),
    _da_threshold(src._da_threshold),
    _rmse_tolerance(src._rmse_tolerance)
  {
    MYODDWEB_PROFILE_FUNCTION("TrainingMonitor");
  }

  TrainingMonitor& operator=(const TrainingMonitor& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TrainingMonitor");
    if (this != &src)
    {
      _metrics = src._metrics;
      _min_window = src._min_window;
      _da_weight = src._da_weight;
      _rmse_weight = src._rmse_weight;
      _da_threshold = src._da_threshold;
      _rmse_tolerance = src._rmse_tolerance;
    }
    return *this;
  }

  TrainingMonitor& operator=(TrainingMonitor&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TrainingMonitor");
    if (this != &src)
    {
      _metrics = std::move(src._metrics);
      _min_window = src._min_window;
      _da_weight = src._da_weight;
      _rmse_weight = src._rmse_weight;
      _da_threshold = src._da_threshold;
      _rmse_tolerance = src._rmse_tolerance;
    }
    return *this;
  }

  // Add a new checkpoint value for a metric
  void add_metric(ErrorCalculation::type type, double value)
  {
    MYODDWEB_PROFILE_FUNCTION("TrainingMonitor");
    _metrics[type].push_back(value);
    // Limit the size to window + extra for history
    if (_metrics[type].size() > _min_window * 5)
    {
      _metrics[type].erase(_metrics[type].begin());
    }
  }

  // Evaluate training status based on recent window
  TrainingStatus evaluate() const
  {
    MYODDWEB_PROFILE_FUNCTION("TrainingMonitor");
    auto rmse_win_it = _metrics.find(ErrorCalculation::type::rmse);
    auto da_win_it = _metrics.find(ErrorCalculation::type::directional_accuracy);

    if (rmse_win_it == _metrics.end() || da_win_it == _metrics.end())
    {
      Logger::error("RMSE or DA metrics missing for training status.");
      return TrainingStatus::OnTrack;
    }

    const auto& rmse_vals = rmse_win_it->second;
    const auto& da_vals = da_win_it->second;

    if (rmse_vals.size() < _min_window || da_vals.size() < _min_window)
    {
      Logger::trace("Not enough data for training status yet.");
      return TrainingStatus::OnTrack; // Not enough data
    }

    double rmse_slope = (rmse_vals.back() - rmse_vals[rmse_vals.size() - _min_window]) / _min_window;
    double da_slope = (da_vals.back() - da_vals[da_vals.size() - _min_window]) / _min_window;

    // Weighted score for trends: positive = improving
    double score = (-rmse_slope * _rmse_weight) + (da_slope * _da_weight);

    if (score < 0.0)
    {
      return TrainingStatus::Diverging;
    }

    if (da_vals.back() < _da_threshold || score < 0.01)
    {
      return TrainingStatus::Stuck;
    }

    return TrainingStatus::OnTrack;
  }

  static const char* monitor_status_to_string(TrainingMonitor::TrainingStatus status) 
  {
    MYODDWEB_PROFILE_FUNCTION("TrainingMonitor");
    switch (status) 
    {
    case TrainingMonitor::TrainingStatus::OnTrack:   
      return "on-track";

    case TrainingMonitor::TrainingStatus::Stuck:
      return "stuck";

    case TrainingMonitor::TrainingStatus::Diverging:
      return "diverging";

    default:                                        
      return "unknown";
    }
  }

private:
  std::unordered_map<ErrorCalculation::type, std::vector<double>> _metrics;
  size_t _min_window;
  double _da_weight;
  double _rmse_weight;
  double _da_threshold;
  double _rmse_tolerance;
};
} // namespace myoddweb::nn
