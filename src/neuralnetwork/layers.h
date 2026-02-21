#pragma once
#ifndef VALIDATE_DATA
  #if !defined(NDEBUG)
    #define VALIDATE_DATA 1
  #else
    #define VALIDATE_DATA 0
  #endif
#endif

#include "layer.h"
#include "layerdetails.h"
#include "residualprojector.h"
#include <memory>
#include "optimiser.h"

class Layers
{
public:
  Layers(
    const std::vector<unsigned>& topology,
    const std::vector<LayerDetails>& hidden_layers, //  TODO: topolody and hidden layers should be put together.
    double weight_decay,
    const std::vector<double>& dropout_layers,
    const activation& output_activation,
    const OptimiserType& optimiser_type,
    int residual_layer_jump,
    int number_of_threads) noexcept;
  Layers(const Layers& layers) noexcept;
  Layers(Layers&& layers) noexcept;
  
  Layers& operator=(const Layers& layers) noexcept;
  Layers& operator=(Layers&& layers) noexcept;

  Layers(
    const std::vector<std::unique_ptr<Layer>>& layers,
    double weight_decay
    ) noexcept;
  
  virtual ~Layers();

  const std::vector<std::unique_ptr<Layer>>& get_layers() const;
  std::vector<std::unique_ptr<Layer>>& get_layers();

  const Layer& operator[](unsigned index) const;
  Layer& operator[](unsigned index);

  int get_residual_layer_number(unsigned index) const noexcept;
  const ResidualProjector* get_residual_layer_projector(unsigned index) const noexcept;

  inline size_t size() const noexcept
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

  inline double get_weight_decay() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("Layers");
    return _weight_decay;
  }

private:
  ResidualProjector* create_residual_projector(const activation& activation_method, int residual_layer_number, int number_of_neurons_in_current_layer, double weight_decay);
  static std::unique_ptr<Layer> create_input_layer(unsigned num_neurons_in_this_layer, double weight_decay, int residual_layer_number, int number_of_threads);
  std::unique_ptr<Layer> create_hidden_layer(double weight_decay, const Layer& previous_layer, const OptimiserType& optimiser_type, int residual_layer_number, double dropout_rate, const LayerDetails& layer_details, int number_of_threads);
  std::unique_ptr<Layer> create_output_layer(unsigned num_neurons_in_this_layer, double weight_decay, const Layer& previous_layer, const activation& activation, const OptimiserType& optimiser_type, int residual_layer_number, int number_of_threads);

  int compute_residual_layer(int current_layer_index, int residual_layer_jump) const;

  std::vector<std::unique_ptr<Layer>> _layers;
  double _weight_decay;
};