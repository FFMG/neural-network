#pragma once
#include "activation.h"
#include "layer.h"
#include "logger.h"
#include "optimiser.h"
#include "./libraries/instrumentor.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#define LEARNING_ALPHA double(0.5)    // momentum, multiplier of last gradient, [0.0..1.0]

class Layer;
class Neuron
{
private:
  class WeightParam
  {
  public:
    WeightParam(double value, double gradient, double velocity, const Logger& logger) : 
      _value(value), 
      _gradient(gradient),
      _velocity(velocity),
      _logger(logger),
      _first_moment_estimate(0.0),
      _second_moment_estimate(0.0),
      _time_step(0)
    {
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
    }
    WeightParam(const WeightParam& src) noexcept :
      _value(src._value),
      _gradient(src._gradient),
      _velocity(src._velocity),
      _logger(src._logger),
      _first_moment_estimate(src._first_moment_estimate),
      _second_moment_estimate(src._second_moment_estimate),
      _time_step(src._time_step)
    {
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
    }
    WeightParam(WeightParam&& src) noexcept: 
      _value(src._value),
      _gradient(src._gradient),
      _velocity(src._velocity),
      _logger(src._logger),
      _first_moment_estimate(src._first_moment_estimate),
      _second_moment_estimate(src._second_moment_estimate),
      _time_step(src._time_step)
    {
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
      src._value = 0.0;
      src._gradient = 0.0;
      src._velocity = 0.0;
      _first_moment_estimate = 0.0;
      _second_moment_estimate = 0.0;
      _time_step = 0;
    }
    WeightParam& operator=(const WeightParam& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
      if (this != &src)
      {
        _value = src._value;
        _gradient = src._gradient;
        _velocity = src._velocity;
        _logger = src._logger;
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
        _logger = src._logger;
        _first_moment_estimate = src._first_moment_estimate;
        _second_moment_estimate = src._second_moment_estimate;
        _time_step = src._time_step;
        src._value = 0.0;
        src._gradient = 0.0;
        src._velocity = 0.0;
        _first_moment_estimate = 0.0;
        _second_moment_estimate = 0.0;
        _time_step = 0;
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
    long timestep() const
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
    void set_value( double value) 
    { 
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
      if (!std::isfinite(value))
      {
        _logger.log_error("Error while setting value.");
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
        _logger.log_error("Error while setting gradient.");
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
        _logger.log_error("Error while setting velocity.");
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
        _logger.log_error("Error while setting first_moment_estimate.");
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
        _logger.log_error("Error while setting second_moment_estimate.");
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
    Logger _logger;

    // For Adam:
    double _first_moment_estimate = 0.0;
    double _second_moment_estimate = 0.0;
    long _time_step = 0;
  };

public:
  Neuron(
    unsigned index, 
    double output_value,
    const activation& activation,
    const std::vector<std::array<double,2>>& weights_params,
    const OptimiserType& optimiser_type,
    const Logger& logger
    );
    
  Neuron(
    unsigned num_neurons_prev_layer,
    unsigned num_neurons_current_layer,
    unsigned index, 
    const activation& activation,
    const OptimiserType& optimiser_type,
    const Logger& logger
    );

  Neuron(const Neuron& src) noexcept;
  Neuron& operator=(const Neuron& src) noexcept;
  Neuron(Neuron&& src) noexcept;
  Neuron& operator=(Neuron&& src) noexcept;

  virtual ~Neuron();

  void set_output_value(double val);
  double get_output_value() const;
  
  double calculate_forward_feed(const Layer& , const std::vector<double>& previous_layer_output_values ) const;
  void forward_feed(const Layer& previous_layer);
  
  double calculate_output_gradients(double target_value, double output_value) const;
  
  double calculate_hidden_gradients(const Layer& next_layer, const std::vector<double>& activation_gradients, double output_value) const;

  void apply_weight_gradients(Layer& previous_layer, const std::vector<double>& gradients, const double learning_rate, unsigned epoch);

  unsigned get_index() const;
  std::vector<std::array<double, 2>> get_weight_params() const;

  const OptimiserType& get_optimiser_type() const { return _optimiser_type; }

private:
  void Clean();
  double sum_of_derivatives_of_weights(const Layer& next_layer, const std::vector<double>& activation_gradients) const;
  double get_output_weight(int index) const;

  // optimisers
  void apply_sgd_update(WeightParam& weight_param, double clipped_gradient, double learning_rate, double momentum, double weight_decay) const;
  void apply_adam_update(WeightParam& weight_param, double clipped_gradient, double learning_rate, double weight_decay) const;
  void apply_adamw_update(
    WeightParam& weight_param,
    double raw_gradient,
    double learning_rate,
    double weight_decay,
    double beta1,
    double beta2,
    double epsilon,
    int time_step
  ) const;
  
  std::pair<double, double> clip_gradient(double gradient) const;
  
  // data to save...
  unsigned _index;
  double _output_value;
  activation _activation_method;
  std::vector<WeightParam> _weight_params;
  OptimiserType _optimiser_type;

  const double _alpha; // [0.0..n] multiplier of last weight change (momentum)
  Logger _logger;
};