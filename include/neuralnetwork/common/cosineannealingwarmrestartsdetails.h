#pragma once

#include "../libraries/instrumentor.h"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace myoddweb::nn
{
class CosineAnnealingWarmRestartsDetails
{
public:
  CosineAnnealingWarmRestartsDetails(
    bool enabled,
    int first_cycle_epochs,
    double cycle_multiplier,
    double minimum_learning_rate,
    double restart_decay) noexcept :
    _enabled(enabled),
    _first_cycle_epochs(first_cycle_epochs),
    _cycle_multiplier(cycle_multiplier),
    _minimum_learning_rate(minimum_learning_rate),
    _restart_decay(restart_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
  }

  CosineAnnealingWarmRestartsDetails(const CosineAnnealingWarmRestartsDetails& src) noexcept :
    _enabled(src._enabled),
    _first_cycle_epochs(src._first_cycle_epochs),
    _cycle_multiplier(src._cycle_multiplier),
    _minimum_learning_rate(src._minimum_learning_rate),
    _restart_decay(src._restart_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
  }

  CosineAnnealingWarmRestartsDetails(CosineAnnealingWarmRestartsDetails&& src) noexcept :
    _enabled(src._enabled),
    _first_cycle_epochs(src._first_cycle_epochs),
    _cycle_multiplier(src._cycle_multiplier),
    _minimum_learning_rate(src._minimum_learning_rate),
    _restart_decay(src._restart_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    src._enabled = false;
    src._first_cycle_epochs = 1;
    src._cycle_multiplier = 1.0;
    src._minimum_learning_rate = 0.0;
    src._restart_decay = 1.0;
  }

  CosineAnnealingWarmRestartsDetails& operator=(const CosineAnnealingWarmRestartsDetails& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    if (this != &src)
    {
      _enabled = src._enabled;
      _first_cycle_epochs = src._first_cycle_epochs;
      _cycle_multiplier = src._cycle_multiplier;
      _minimum_learning_rate = src._minimum_learning_rate;
      _restart_decay = src._restart_decay;
    }
    return *this;
  }

  CosineAnnealingWarmRestartsDetails& operator=(CosineAnnealingWarmRestartsDetails&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    if (this != &src)
    {
      _enabled = src._enabled;
      _first_cycle_epochs = src._first_cycle_epochs;
      _cycle_multiplier = src._cycle_multiplier;
      _minimum_learning_rate = src._minimum_learning_rate;
      _restart_decay = src._restart_decay;
      src._enabled = false;
      src._first_cycle_epochs = 1;
      src._cycle_multiplier = 1.0;
      src._minimum_learning_rate = 0.0;
      src._restart_decay = 1.0;
    }
    return *this;
  }

  virtual ~CosineAnnealingWarmRestartsDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
  }

  [[nodiscard]] inline bool enabled() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    return _enabled;
  }

  [[nodiscard]] inline int first_cycle_epochs() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    return _first_cycle_epochs;
  }

  [[nodiscard]] inline double cycle_multiplier() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    return _cycle_multiplier;
  }

  [[nodiscard]] inline double minimum_learning_rate() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    return _minimum_learning_rate;
  }

  [[nodiscard]] inline double restart_decay() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    return _restart_decay;
  }

  [[nodiscard]] double calculate_learning_rate(int relative_epoch, double base_learning_rate) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("CosineAnnealingWarmRestartsDetails");
    if (!_enabled || _first_cycle_epochs <= 0 || relative_epoch < 0)
    {
      return base_learning_rate;
    }

    constexpr double pi = 3.14159265358979323846;
    double cur_t_i = static_cast<double>(_first_cycle_epochs);
    double cur_eta_max = base_learning_rate;
    double t_remaining = static_cast<double>(relative_epoch);

    if (_cycle_multiplier <= 1.0)
    {
      int cycle_idx = static_cast<int>(std::floor(t_remaining / cur_t_i));
      double t_cur = t_remaining - cycle_idx * cur_t_i;
      double eta_max_cycle = std::max(_minimum_learning_rate, base_learning_rate * std::pow(_restart_decay, cycle_idx));
      double cos_val = std::cos((t_cur / cur_t_i) * pi);
      return _minimum_learning_rate + 0.5 * (eta_max_cycle - _minimum_learning_rate) * (1.0 + cos_val);
    }

    while (t_remaining >= cur_t_i)
    {
      t_remaining -= cur_t_i;
      // +1.0 guards against std::round leaving cur_t_i unchanged for a cycle_multiplier
      // close to 1.0, which would otherwise stall cycle growth and make this loop O(N).
      cur_t_i = std::max(cur_t_i + 1.0, std::round(cur_t_i * _cycle_multiplier));
      cur_eta_max *= _restart_decay;
    }

    double t_cur = t_remaining;
    double eta_max_cycle = std::max(_minimum_learning_rate, cur_eta_max);
    double cos_val = std::cos((t_cur / cur_t_i) * pi);
    return _minimum_learning_rate + 0.5 * (eta_max_cycle - _minimum_learning_rate) * (1.0 + cos_val);
  }

private:
  bool _enabled;
  int _first_cycle_epochs;
  double _cycle_multiplier;
  double _minimum_learning_rate;
  double _restart_decay;
};

using cosineannealingwarmrestartsdetails = CosineAnnealingWarmRestartsDetails;
using CosineAnnealingDetails = CosineAnnealingWarmRestartsDetails;
using cosineannealingdetails = CosineAnnealingWarmRestartsDetails;

} // namespace myoddweb::nn
