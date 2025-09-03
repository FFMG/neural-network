#pragma once
#include "logger.h"
#include "./libraries/instrumentor.h"

#include <cmath>

class WeightParam
{
public:
  WeightParam(
    double value, 
    double gradient, 
    double velocity, 
    double first_moment_estimate,
    double second_moment_estimate,
    long long time_step,
    double weight_decay) noexcept :
    _value(value),
    _gradient(gradient),
    _velocity(velocity),
    _first_moment_estimate(first_moment_estimate),
    _second_moment_estimate(second_moment_estimate),
    _time_step(time_step),
    _weight_decay(weight_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
  }

  WeightParam(double value, double gradient, double velocity) noexcept :
    WeightParam(value, gradient, velocity, 0.0, 0.0, 0, 0.01)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
  }

  WeightParam(const WeightParam& src) noexcept :
    _value(src._value),
    _gradient(src._gradient),
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
    _gradient(src._gradient),
    _velocity(src._velocity),
    _first_moment_estimate(src._first_moment_estimate),
    _second_moment_estimate(src._second_moment_estimate),
    _time_step(src._time_step)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    src._value = 0.0;
    src._gradient = 0.0;
    src._velocity = 0.0;
    src._first_moment_estimate = 0.0;
    src._second_moment_estimate = 0.0;
    src._time_step = 0;
  }

  WeightParam& operator=(const WeightParam& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (this != &src)
    {
      _value = src._value;
      _gradient = src._gradient;
      _velocity = src._velocity;
      _first_moment_estimate = src._first_moment_estimate;
      _second_moment_estimate = src._second_moment_estimate;
      _time_step = src._time_step;
    }
    return *this;
  }
  WeightParam& operator=(WeightParam&& src)  noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (this != &src)
    {
      _value = src._value;
      _gradient = src._gradient;
      _velocity = src._velocity;
      _first_moment_estimate = src._first_moment_estimate;
      _second_moment_estimate = src._second_moment_estimate;
      _time_step = src._time_step;
      src._value = 0.0;
      src._gradient = 0.0;
      src._velocity = 0.0;
      src._first_moment_estimate = 0.0;
      src._second_moment_estimate = 0.0;
      src._time_step = 0;
    }
    return *this;
  }
  virtual ~WeightParam() = default;

  double value() const 
  { 
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _value; 
  };
  double gradient() const 
  { 
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _gradient; 
  }
  double velocity() const
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _velocity;
  }
  long long timestep() const
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _time_step;
  }
  double first_moment_estimate() const
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _first_moment_estimate;
  }
  double second_moment_estimate() const
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _second_moment_estimate;
  }

  double weight_decay() const
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    return _weight_decay;
  }
  void set_value( double value) 
  { 
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(value))
    {
      Logger::error("Error while setting value.");
      throw std::invalid_argument("Error while setting value.");
      return;
    }
    _value = value; 
  }
  void set_gradient(double gradient)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(gradient))
    {
      Logger::error("Error while setting gradient.");
      throw std::invalid_argument("Error while setting gradient.");
      return;
    }
    _gradient = gradient; 
  };
  void set_velocity(double velocity)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(velocity))
    {
      Logger::error("Error while setting velocity.");
      throw std::invalid_argument("Error while setting velocity.");
      return;
    }
    _velocity = velocity;
  }
  void set_first_moment_estimate(double first_moment_estimate)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(first_moment_estimate))
    {
      Logger::error("Error while setting first_moment_estimate.");
      throw std::invalid_argument("Error while setting first_moment_estimate.");
      return;
    }
    _first_moment_estimate = first_moment_estimate;
  }
  void set_second_moment_estimate(double second_moment_estimate)
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    if (!std::isfinite(second_moment_estimate))
    {
      Logger::error("Error while setting second_moment_estimate.");
      throw std::invalid_argument("Error while setting second_moment_estimate.");
      return;
    }
    _second_moment_estimate = second_moment_estimate;
  }

  void increment_timestep()
  {
    MYODDWEB_PROFILE_FUNCTION("WeightParam");
    ++_time_step;
  }

private:
  double _value;
  double _gradient;
  double _velocity;

  // For Adam:
  double _first_moment_estimate = 0.0;
  double _second_moment_estimate = 0.0;
  long long _time_step = 0;
  double _weight_decay = 0.01;
};