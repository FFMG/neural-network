#pragma once

#include "../libraries/instrumentor.h"


namespace myoddweb::nn
{
class StochasticWeightAveragingDetails
{
public:
  StochasticWeightAveragingDetails(
    bool swa_enabled,
    double swa_start_percent,
    double swa_update_percent) noexcept :
    _swa_enabled(swa_enabled),
    _swa_start_percent(swa_start_percent),
    _swa_update_percent(swa_update_percent)
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
  }

  StochasticWeightAveragingDetails(const StochasticWeightAveragingDetails& src) noexcept :
    _swa_enabled(src._swa_enabled),
    _swa_start_percent(src._swa_start_percent),
    _swa_update_percent(src._swa_update_percent)
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
  }

  StochasticWeightAveragingDetails(StochasticWeightAveragingDetails&& src) noexcept :
    _swa_enabled(src._swa_enabled),
    _swa_start_percent(src._swa_start_percent),
    _swa_update_percent(src._swa_update_percent)
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    src._swa_enabled = false;
    src._swa_start_percent = 0.0;
    src._swa_update_percent = 0.0;
  }

  StochasticWeightAveragingDetails& operator=(const StochasticWeightAveragingDetails& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    if (this != &src)
    {
      _swa_enabled = src._swa_enabled;
      _swa_start_percent = src._swa_start_percent;
      _swa_update_percent = src._swa_update_percent;
    }
    return *this;
  }

  StochasticWeightAveragingDetails& operator=(StochasticWeightAveragingDetails&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    if (this != &src)
    {
      _swa_enabled = src._swa_enabled;
      _swa_start_percent = src._swa_start_percent;
      _swa_update_percent = src._swa_update_percent;
      src._swa_enabled = false;
      src._swa_start_percent = 0.0;
      src._swa_update_percent = 0.0;
    }
    return *this;
  }

  virtual ~StochasticWeightAveragingDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
  }

  [[nodiscard]] inline bool enabled() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    return _swa_enabled;
  }

  [[nodiscard]] inline bool swa_enabled() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    return _swa_enabled;
  }

  [[nodiscard]] inline double start_percent() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    return _swa_start_percent;
  }

  [[nodiscard]] inline double swa_start_percent() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    return _swa_start_percent;
  }

  [[nodiscard]] inline double update_percent() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    return _swa_update_percent;
  }

  [[nodiscard]] inline double swa_update_percent() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetails");
    return _swa_update_percent;
  }

private:
  bool _swa_enabled;
  double _swa_start_percent;
  double _swa_update_percent;
};

using stochasticweightaveragingdetails = StochasticWeightAveragingDetails;
using SwaDetails = StochasticWeightAveragingDetails;
using swadetails = StochasticWeightAveragingDetails;

} // namespace myoddweb::nn
