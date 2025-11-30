#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif
#include "layer.h"
#include "hiddenstate.h"
#include "weightparam.h"
#include "./libraries/instrumentor.h"

#include <vector>

class Layer;
class Neuron
{
public:
  enum class Type
  {
    Normal,
    Dropout
  };

public:
  Neuron(
    unsigned index, 
    const Type& type,
    const double dropout_rate
    );
    
  Neuron(
    unsigned num_neurons_prev_layer,
    unsigned num_neurons_current_layer,
    unsigned num_neurons_next_layer,
    unsigned index, 
    const Type& type,
    const double dropout_rate
    );

  Neuron(const Neuron& src) noexcept;
  Neuron& operator=(const Neuron& src) noexcept;
  Neuron(Neuron&& src) noexcept;
  Neuron& operator=(Neuron&& src) noexcept;

  virtual ~Neuron();

  unsigned get_index() const;

  const Type& get_type() const noexcept;
  bool is_dropout() const noexcept;
  double get_dropout_rate() const noexcept;

  bool must_randomly_drop() const;

private:

  void Clean();
  unsigned _index;
  
  Type _type;
  double _dropout_rate;
};