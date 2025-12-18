#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif
#include "layer.h"
#include "residualprojector.h"
#include <memory>
#include "optimiser.h"

class Layers
{
public:
  Layers(
    const std::vector<unsigned>& topology, 
    double weight_decay,
    const std::vector<unsigned>& recurrent_layers,
    const std::vector<double>& dropout_layers,
    const activation::method& hidden_activation,
    const activation::method& output_activation,
    const OptimiserType& optimiser_type,
    int residual_layer_jump) noexcept;
  Layers(const Layers& layers) noexcept;
  Layers(Layers&& layers) noexcept;

  Layers& operator=(const Layers& layers) noexcept;
  Layers& operator=(Layers&& layers) noexcept;
  
  virtual ~Layers();

  const std::vector<std::unique_ptr<Layer>>& get_layers() const;
  std::vector<std::unique_ptr<Layer>>& get_layers();

  const Layer& operator[](unsigned index) const;
  Layer& operator[](unsigned index);

  int get_residual_layer_number(unsigned index) const noexcept;
  const ResidualProjector* get_residual_layer_projector(unsigned index) const noexcept;

  inline size_t size() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _layers.size();
  }

  inline const Layer& input_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.front();
  }

  inline const Layer& output_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.back();
  }

private:
  ResidualProjector* create_residual_projector(const activation& activation_method, int residual_layer_number, int number_of_neurons_in_current_layer);
  static std::unique_ptr<Layer> create_input_layer(unsigned num_neurons_in_this_layer, double weight_decay, int residual_layer_number);
  std::unique_ptr<Layer> create_hidden_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, const std::vector<unsigned>& recurrent_layers, int residual_layer_number, double dropout_rate);
  std::unique_ptr<Layer> create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, const std::vector<unsigned>& recurrent_layers, int residual_layer_number);

  int compute_residual_layer(int current_layer_index, int residual_layer_jump) const;

  std::vector<std::unique_ptr<Layer>> _layers;
  double _weight_decay;
  std::vector<unsigned> _recurrent_layers;

public:
  const std::vector<unsigned>& recurrent_layers() const noexcept;
};