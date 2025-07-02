#pragma once
#include "activation.h"
#include "layer.h"
#include "logger.h"
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
    WeightParam(double value, double gradient, const Logger& logger) : 
      _value(value), 
      _gradient(gradient),
      _logger(logger)
    {
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
    }
    WeightParam(const WeightParam& src) noexcept :
      _value(src._value),
      _gradient(src._gradient),
      _logger(src._logger)
    {
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
    }
    WeightParam(WeightParam&& src) noexcept: 
      _value(src._value),
      _gradient(src._gradient),
      _logger(src._logger)
    {
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
      src._value = 0.0;
      src._gradient = 0.0;
    }
    WeightParam& operator=(const WeightParam& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("WeightParam");
      if (this != &src)
      {
        _value = src._value;
        _gradient = src._gradient;
        _logger = src._logger;
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
        src._value = 0.0;
        src._gradient = 0.0;
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
    };
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
    };
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

  private:
    double _value;
    double _gradient;
    Logger _logger;
  };

public:
  Neuron(
    unsigned index, 
    double output_value,
    const activation& activation,
    const std::vector<std::array<double,2>>& weights_params,
    const Logger& logger
    );
    
  Neuron(
    unsigned num_neurons_prev_layer,
    unsigned num_neurons_current_layer,
    unsigned index, 
    const activation& activation,
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

  void update_input_weights(Layer& previous_layer, const std::vector<double>& weights_gradients, double learning_rate);

  unsigned get_index() const;
  std::vector<std::array<double, 2>> get_weight_params() const;

private:
  void Clean();
  double sum_of_derivatives_of_weights(const Layer& next_layer, const std::vector<double>& activation_gradients) const;
  double get_output_weight(int index) const;

  double clip_gradient(double gradient) const;
  
  // data to save...
  unsigned _index;
  double _output_value;
  activation _activation_method;
  std::vector<WeightParam> _weight_params;

  const double _alpha; // [0.0..n] multiplier of last weight change (momentum)
  Logger _logger;
};