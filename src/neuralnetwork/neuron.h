#pragma once
#include "activation.h"
#include <vector>

class Neuron
{
private:
  struct Connection
  {
    double weight;
    double delta_weight;
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

  void setOutputVal(double val) { _outputVal = val; }
  double getOutputVal(void) const { return _outputVal; }
  void forward_feed(const Layer& prevLayer);
  void calcOutputGradients(double targetVal);
  void calcHiddenGradients(const Layer& nextLayer);
  void updateInputWeights(Layer& prevLayer);

private:
  void Clean();
  static double eta;   // [0.0..1.0] overall net training rate
  static double alpha; // [0.0..n] multiplier of last weight change (momentum)
  static double randomWeight(void) { return rand() / double(RAND_MAX); }
  double sumDOW(const Layer& nextLayer) const;
  
  const unsigned _index;
  double _outputVal;
  double _gradient;
  std::vector<Connection>* _output_weights;
  const activation::method _activation_method;
};