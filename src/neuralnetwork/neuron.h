#pragma once
#include "activation.h"

#include <array>
#include <vector>
#include <cmath>

#define LEARNING_RATE double(0.15)    // overall net learning rate, [0.0..1.0]
                                      // the smaller the learning rate, the more epoch are needed.
                                      // but the bigger the learning rate, the more likely are exploding weights.
#define LEARNING_ALPHA double(0.5)    // momentum, multiplier of last deltaWeight, [0.0..1.0]

class Neuron
{
private:
  class Connection
  {
  public:
    Connection(double weight) : 
      _weight(weight), 
      _delta_weight(0.0)
    {
    }
    Connection(const Connection& connection) : 
      _weight(connection._weight),
      _delta_weight(connection._delta_weight)
    {

    }
    Connection& operator=(const Connection& connection)
    {
      if (this != &connection)
      {
        set_weight(connection._weight);
        set_delta_weight( connection._delta_weight);
      }
      return *this;
    }
    virtual ~Connection() = default;

    double weight() const { return _weight; };
    double delta_weight() const { return _delta_weight; };
    void set_weight( double weight) 
    { 
      if (!std::isfinite(weight))
      {
        return;
      }
      _weight = weight; 
    };
    void set_delta_weight(double delta_weight)
    {
      if (!std::isfinite(delta_weight))
      {
        return;
      }
      _delta_weight = delta_weight; 
    };

  private:
    double _weight;
    double _delta_weight;
  };

public:
  typedef std::vector<Neuron> Layer;

  Neuron(
    unsigned index, 
    double output_value,
    double gradient,
    const activation::method& activation,
    const std::vector<std::array<double,2>>& output_weights,
    double learning_rate = LEARNING_RATE
    );
    
  Neuron(unsigned numOutputs, 
    unsigned index, 
    const activation::method& activation,
    double learning_rate = LEARNING_RATE
    );
  Neuron(const Neuron& src);
  const Neuron& operator=(const Neuron& src);
  virtual ~Neuron();

  void set_output_value(double val);
  double get_output_value() const;
  void forward_feed(const Layer& prevLayer);
  void calculate_output_gradients(double targetVal);
  void calculate_hidden_gradients(const Layer& nextLayer);
  void update_input_weights(Layer& previous_layer);

  double get_gradient() const {
    return _gradient;
  }

  unsigned get_index() const {
    return _index;
  }
  double get_learning_rate() const {
    return _learning_rate;
  }
  std::vector<std::array<double, 2>> get_weights() const;
private:
  void Clean();
  double sumDOW(const Layer& nextLayer) const;

  std::vector<double> he_initialization(int num_neurons_prev_layer);
  
  // data to save...
  unsigned _index;
  double _output_value;
  double _gradient;
  activation::method _activation_method;
  std::vector<Connection>* _output_weights;
  double _learning_rate;   // [0.0..1.0] overall net training rate
  const double _alpha; // [0.0..n] multiplier of last weight change (momentum)
};