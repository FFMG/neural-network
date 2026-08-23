#pragma once

#include "../libraries/instrumentor.h"


namespace myoddweb::nn
{
class LookaheadDetails
{
public:
  LookaheadDetails(
    bool enabled,
    int synchronisation_period,
    double slow_weights_step_size) noexcept :
    _enabled(enabled),
    _synchronisation_period(synchronisation_period),
    _slow_weights_step_size(slow_weights_step_size)
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
  }

  LookaheadDetails(const LookaheadDetails& src) noexcept :
    _enabled(src._enabled),
    _synchronisation_period(src._synchronisation_period),
    _slow_weights_step_size(src._slow_weights_step_size)
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
  }

  LookaheadDetails(LookaheadDetails&& src) noexcept :
    _enabled(src._enabled),
    _synchronisation_period(src._synchronisation_period),
    _slow_weights_step_size(src._slow_weights_step_size)
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
    src._enabled = false;
    src._synchronisation_period = 5;
    src._slow_weights_step_size = 0.5;
  }

  LookaheadDetails& operator=(const LookaheadDetails& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
    if (this != &src)
    {
      _enabled = src._enabled;
      _synchronisation_period = src._synchronisation_period;
      _slow_weights_step_size = src._slow_weights_step_size;
    }
    return *this;
  }

  LookaheadDetails& operator=(LookaheadDetails&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
    if (this != &src)
    {
      _enabled = src._enabled;
      _synchronisation_period = src._synchronisation_period;
      _slow_weights_step_size = src._slow_weights_step_size;
      src._enabled = false;
      src._synchronisation_period = 5;
      src._slow_weights_step_size = 0.5;
    }
    return *this;
  }

  virtual ~LookaheadDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
  }

  [[nodiscard]] inline bool enabled() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
    return _enabled;
  }

  [[nodiscard]] inline bool lookahead_enabled() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
    return _enabled;
  }

  [[nodiscard]] inline int synchronisation_period() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
    return _synchronisation_period;
  }

  [[nodiscard]] inline double slow_weights_step_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
    return _slow_weights_step_size;
  }

  [[nodiscard]] inline double slow_step_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LookaheadDetails");
    return _slow_weights_step_size;
  }

private:
  bool _enabled;
  int _synchronisation_period;
  double _slow_weights_step_size;
};

} // namespace myoddweb::nn
