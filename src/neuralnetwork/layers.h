#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif
#include "baselayer.h"
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

  const std::vector<std::unique_ptr<BaseLayer>>& get_layers() const;
  std::vector<std::unique_ptr<BaseLayer>>& get_layers();

  const BaseLayer& operator[](unsigned index) const;
  BaseLayer& operator[](unsigned index);

  int residual_layer_number(unsigned index) const;

  inline size_t size() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _layers.size();
  }

  inline const BaseLayer& input_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.front();
  }

  inline const BaseLayer& output_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return *_layers.back();
  }

private:
  static std::unique_ptr<BaseLayer> create_input_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay, int residual_layer_number);
  static std::unique_ptr<BaseLayer> create_hidden_layer(unsigned num_neurons_in_this_layer, unsigned num_neurons_in_next_layer, double weight_decay, const BaseLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, const std::vector<unsigned>& recurrent_layers, int residual_layer_number, double dropout_rate);
  static std::unique_ptr<BaseLayer> create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const BaseLayer& previous_layer, const activation::method& activation, const OptimiserType& optimiser_type, const std::vector<unsigned>& recurrent_layers, int residual_layer_number);

  void add_residual_layer(BaseLayer& layer, const activation::method& activation_method) const;
  int compute_residual_layer(int current_layer_index, int residual_layer_jump) const;

  std::vector<std::unique_ptr<BaseLayer>> _layers;
  std::vector<int> _residual_layer_numbers;
  double _weight_decay;
  std::vector<unsigned> _recurrent_layers; // New member variable

public:
  const std::vector<unsigned>& recurrent_layers() const noexcept;
};