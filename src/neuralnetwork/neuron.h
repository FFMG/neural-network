#pragma once
#include "activation.h"
#include "layer.h"
#include "logger.h"
#include "./libraries/instrumentor.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#define LEARNING_ALPHA double(0.5)    // momentum, multiplier of last deltaWeight, [0.0..1.0]

class Layer;
class Neuron
{
private:
  class Connection
  {
  public:
    Connection(double weight, double delta_weight, Logger& logger) : 
      _weight(weight), 
      _delta_weight(delta_weight),
      _logger(logger)
    {
      MYODDWEB_PROFILE_FUNCTION("Connection");
    }
    Connection(const Connection& connection) : 
      _weight(connection._weight),
      _delta_weight(connection._delta_weight),
      _logger(connection._logger)
    {
      MYODDWEB_PROFILE_FUNCTION("Connection");
    }
    Connection(Connection&& connection) : 
      _weight(connection._weight),
      _delta_weight(connection._delta_weight),
      _logger(connection._logger)
    {
      MYODDWEB_PROFILE_FUNCTION("Connection");
      connection._weight = 0.0;
      connection._delta_weight = 0.0;
    }
    Connection& operator=(const Connection& connection)
    {
      MYODDWEB_PROFILE_FUNCTION("Connection");
      if (this != &connection)
      {
        _weight = connection._weight;
        _delta_weight = connection._delta_weight;
        _logger = connection._logger;
      }
      return *this;
    }
    Connection& operator=(Connection&& connection)
    {
      MYODDWEB_PROFILE_FUNCTION("Connection");
      if (this != &connection)
      {
        _weight = connection._weight;
        _delta_weight = connection._delta_weight;
        _logger = connection._logger;
        connection._weight = 0.0;
        connection._delta_weight = 0.0;
      }
      return *this;
    }
    virtual ~Connection() = default;

    double weight() const 
    { 
      MYODDWEB_PROFILE_FUNCTION("Connection");
      return _weight; 
    };
    double delta_weight() const 
    { 
      MYODDWEB_PROFILE_FUNCTION("Connection");
      return _delta_weight; 
    };
    void set_weight( double weight) 
    { 
      MYODDWEB_PROFILE_FUNCTION("Connection");
      if (!std::isfinite(weight))
      {
        _logger.log_error("Error while setting weight.");
        throw std::invalid_argument("Error while setting weight.");
        return;
      }
      _weight = weight; 
    };
    void set_delta_weight(double delta_weight)
    {
      MYODDWEB_PROFILE_FUNCTION("Connection");
      if (!std::isfinite(delta_weight))
      {
        _logger.log_error("Error while setting delta weight.");
        throw std::invalid_argument("Error while setting delta weight.");
        return;
      }
      _delta_weight = delta_weight; 
    };

  private:
    double _weight;
    double _delta_weight;
    Logger& _logger;
  };

public:
  Neuron(
    unsigned index, 
    double output_value,
    const activation& activation,
    const std::vector<std::array<double,2>>& output_weights,
    Logger& logger
    );
    
  Neuron(
    unsigned num_neurons_prev_layer,
    unsigned num_neurons_current_layer,
    unsigned index, 
    const activation& activation,
    Logger& logger
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

  unsigned get_index() const 
  {
    MYODDWEB_PROFILE_FUNCTION("Neuron");
    return _index;
  }
  std::vector<std::array<double, 2>> get_weights() const;

private:
  void Clean();
  double sum_of_derivatives_of_weights(const Layer& next_layer, const std::vector<double>& activation_gradients) const;
  double get_output_weight(int index) const;

  static double clip_gradient(double val, double clip_val);
  
  // data to save...
  unsigned _index;
  double _output_value;
  activation _activation_method;
  std::vector<Connection> _output_weights;

  const double _alpha; // [0.0..n] multiplier of last weight change (momentum)
  Logger& _logger;
};