#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif
#include "activation.h"
#include "layer.h"
#include "hiddenstate.h"
#include "optimiser.h"
#include "weightparam.h"
#include "./libraries/instrumentor.h"

#include <vector>

#define LEARNING_ALPHA double(0.5)    // momentum, multiplier of last gradient, [0.0..1.0]

class Layer;
class Neuron
{
public:
  enum class Type
  {
    Normal,
    Dropout,
  };

public:
  Neuron(
    unsigned index, 
    const activation& activation,
    const OptimiserType& optimiser_type,
    const Type& type,
    const double dropout_rate,
    const double recurrent_weight
    );
    
  Neuron(
    unsigned num_neurons_prev_layer,
    unsigned num_neurons_current_layer,
    unsigned num_neurons_next_layer,
    unsigned index, 
    const activation& activation,
    const OptimiserType& optimiser_type,
    const Type& type,
    const double dropout_rate
    );

  Neuron(const Neuron& src) noexcept;
  Neuron& operator=(const Neuron& src) noexcept;
  Neuron(Neuron&& src) noexcept;
  Neuron& operator=(Neuron&& src) noexcept;

  virtual ~Neuron();
  
  void apply_weight_gradients(Layer& previous_layer, const std::vector<double>& gradients, const double learning_rate, unsigned epoch, double clipping_scale);
  void apply_residual_weight_gradients(Layer& layer, Layer& residual_layer, const std::vector<double>& residual_outputs, const std::vector<double>& gradients, double learning_rate, double clipping_scale);

  unsigned get_index() const;

  const OptimiserType& get_optimiser_type() const noexcept;
  const Type& get_type() const noexcept;
  bool is_dropout() const noexcept;
  double get_dropout_rate() const noexcept;

  inline const activation& get_activation_method() const noexcept { return _activation_method; }
  bool must_randomly_drop() const;

  const WeightParam& get_recurrent_weight() const noexcept;

private:

  void Clean();

  void apply_weight_gradient(const double gradient, const double learning_rate, bool is_bias, WeightParam& weight_param, double clipping_scale);
  
  unsigned _index;
  activation _activation_method;
  OptimiserType _optimiser_type;
  
  const double _alpha; // [0.0..n] multiplier of last weight change (momentum)
  Type _type;
  double _dropout_rate;
  WeightParam _recurrent_weight;
};