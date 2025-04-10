#pragma once
#include "activation.h"
#include <vector>

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
        _weight = connection._weight;
        _delta_weight = connection._delta_weight;
      }
      return *this;
    }

    double weight() const { return _weight; };
    double delta_weight() const { return _delta_weight; };
    void set_weight( double weight) { _weight = weight; };
    void set_delta_weight(double delta_weight){ _delta_weight = delta_weight; };

  private:
    double _weight;
    double _delta_weight;
  };

public:
  typedef std::vector<Neuron> Layer;

  Neuron(
    unsigned index, 
    double outputVal,
    double gradient,
    const activation::method& activation,
    const std::vector<Connection>& output_weights
  );
  Neuron(unsigned numOutputs, unsigned index, const activation::method& activation);
  Neuron(const Neuron& src);
  const Neuron& operator=(const Neuron& src);
  virtual ~Neuron();

  void set_output_value(double val);
  double get_output_value() const;
  void forward_feed(const Layer& prevLayer);
  void calculate_output_gradients(double targetVal);
  void calculate_hidden_gradients(const Layer& nextLayer);
  void update_input_weights(Layer& previous_layer);

private:
  
  void Clean();
  static double eta;   // [0.0..1.0] overall net training rate
  static double alpha; // [0.0..n] multiplier of last weight change (momentum)
  double sumDOW(const Layer& nextLayer) const;

  std::vector<double> he_initialization(int num_neurons_prev_layer);
  
  // data to save...
  unsigned _index;
  double _output_value;
  double _gradient;
  std::vector<Connection>* _output_weights;
  activation::method _activation_method;
};