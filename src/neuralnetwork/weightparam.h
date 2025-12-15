#pragma once
#include "logger.h"
#include "./libraries/instrumentor.h"

#include <cmath>

class WeightParam
{
public:
  WeightParam(
    double value,
    double raw_gradient,
    double velocity,
    double first_moment_estimate,
    double second_moment_estimate,
    long long time_step,
    double weight_decay) noexcept :
    _value(value),
    _raw_gradient(raw_gradient),
    _velocity(velocity),
    _first_moment_estimate(first_moment_estimate),
    _second_moment_estimate(second_moment_estimate),
    _time_step(time_step),
    _weight_decay(weight_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
  }

  WeightParam(double value, double raw_gradient, double velocity, double weight_decay) noexcept :
    WeightParam(value, raw_gradient, velocity, 0.0, 0.0, 0, weight_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
  }

  WeightParam(const WeightParam& src) noexcept :
    _value(src._value),
    _raw_gradient(src._raw_gradient),
    _velocity(src._velocity),
    _first_moment_estimate(src._first_moment_estimate),
    _second_moment_estimate(src._second_moment_estimate),
    _time_step(src._time_step),
    _weight_decay(src._weight_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
  }

  WeightParam(WeightParam&& src) noexcept: 
    _value(src._value),
    _raw_gradient(src._raw_gradient),
    _velocity(src._velocity),
    _first_moment_estimate(src._first_moment_estimate),
    _second_moment_estimate(src._second_moment_estimate),
    _time_step(src._time_step),
    _weight_decay(src._weight_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    src._value = 0.0;
    src._raw_gradient = 0.0;
    src._velocity = 0.0;
    src._first_moment_estimate = 0.0;
    src._second_moment_estimate = 0.0;
    src._time_step = 0;
    src._weight_decay = 0.0;
  }

  WeightParam& operator=(const WeightParam& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (this != &src)
    {
      _value = src._value;
      _raw_gradient = src._raw_gradient;
      _velocity = src._velocity;
      _first_moment_estimate = src._first_moment_estimate;
      _second_moment_estimate = src._second_moment_estimate;
      _time_step = src._time_step;
      _weight_decay = src._weight_decay;
    }
    return *this;
  }
  WeightParam& operator=(WeightParam&& src)  noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (this != &src)
    {
      _value = src._value;
      _raw_gradient = src._raw_gradient;
      _velocity = src._velocity;
      _first_moment_estimate = src._first_moment_estimate;
      _second_moment_estimate = src._second_moment_estimate;
      _time_step = src._time_step;
      _weight_decay = src._weight_decay;
      src._value = 0.0;
      src._raw_gradient = 0.0;
      src._velocity = 0.0;
      src._first_moment_estimate = 0.0;
      src._second_moment_estimate = 0.0;
      src._time_step = 0;
      src._weight_decay = 0.0;
    }
    return *this;
  }
  virtual ~WeightParam() = default;

  inline double get_value() const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _value; 
  };
  inline double get_raw_gradient() const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _raw_gradient;
  }

  inline double get_velocity() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _velocity;
  }
  inline long long get_timestep() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _time_step;
  }
  inline double get_first_moment_estimate() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _first_moment_estimate;
  }
  inline double get_second_moment_estimate() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _second_moment_estimate;
  }
  inline double get_weight_decay() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _weight_decay;
  }
  inline void set_value( double value) 
  { 
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(value))
    {
      Logger::panic("Error while setting value.");
      return;
    }
    _value = value; 
  }
  inline void set_raw_gradient(double raw_gradient)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(raw_gradient))
    {
      Logger::panic("Error while setting gradient.");
      return;
    }
    _raw_gradient = raw_gradient;
  };

  inline void set_velocity(double velocity)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(velocity))
    {
      Logger::panic("Error while setting velocity.");
      return;
    }
    _velocity = velocity;
  }
  inline void set_first_moment_estimate(double first_moment_estimate)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(first_moment_estimate))
    {
      Logger::panic("Error while setting first_moment_estimate.");
      return;
    }
    _first_moment_estimate = first_moment_estimate;
  }
  inline void set_second_moment_estimate(double second_moment_estimate)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(second_moment_estimate))
    {
      Logger::panic("Error while setting second_moment_estimate.");
      return;
    }
    _second_moment_estimate = second_moment_estimate;
  }

  inline void increment_timestep() noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    ++_time_step;
  }


  inline double clip_gradient(double g) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    constexpr double gradient_clip_threshold = 1.0;
    // Maximum absolute gradient allowed (before scaling)
    const double max_clip = gradient_clip_threshold;  // e.g. set in constructor or defaults to large value

    if (max_clip <= 0.0)
    {
      return g; // clipping disabled
    }

    if (g > max_clip) 
      return  max_clip;
    if (g < -max_clip) 
      return -max_clip;

    return g;
  }
private:
  double _value;
  double _raw_gradient;
  double _velocity;

  // For Adam:
  double _first_moment_estimate = 0.0;
  double _second_moment_estimate = 0.0;
  long long _time_step = 0;
  double _weight_decay = 0.0;
};