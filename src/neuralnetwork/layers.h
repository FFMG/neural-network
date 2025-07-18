#pragma once
#include "activation.h"
#include "layer.h"
#include "logger.h"
#include "optimiser.h"

class Layers
{
public:
  Layers(const std::vector<unsigned>& topology, 
    const activation::method& hidden_activation,
    const activation::method& output_activation,
    const OptimiserType& optimiser_type,
    const Logger& logger);
  Layers(const std::vector<Layer>& layers);
  Layers(const Layers& layers);
  Layers(Layers&& layers);

  const std::vector<Layer>& get_layers() const;
  std::vector<Layer>& get_layers();

  const Layer& operator[](unsigned index ) const;

  inline size_t size() const {
    return _layers.size();
  }

private:
  std::vector<Layer> _layers;
};