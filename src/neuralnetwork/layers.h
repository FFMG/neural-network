#pragma once
#include "activation.h"
#include "errorcalculation.h"
#include "layer.h"
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
    int residual_layer_jump,
    ErrorCalculation::type error_calculation_type) noexcept;
  Layers(const std::vector<Layer>& layers) noexcept;
  Layers(const Layers& layers) noexcept;
  Layers(Layers&& layers) noexcept;

  Layers& operator=(const Layers& layers) noexcept;
  Layers& operator=(Layers&& layers) noexcept;
  
  virtual ~Layers() = default;

  const std::vector<Layer>& get_layers() const;
  std::vector<Layer>& get_layers();

  const Layer& operator[](unsigned index) const;
  Layer& operator[](unsigned index);

  int residual_layer_number(unsigned index) const;

  inline size_t size() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _layers.size();
  }

  inline const Layer& input_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _layers.front();
  }

  inline const Layer& output_layer() const
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _layers.back();
  }

private:
  void add_residual_layer(Layer& layer, const activation::method& activation_method) const;
  int compute_residual_layer(int current_layer_index, int residual_layer_jump) const;

  std::vector<Layer> _layers;
};