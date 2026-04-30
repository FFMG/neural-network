#pragma once

#include "./libraries/instrumentor.h"

#include "layer.h"
#include "logger.h"

#include "activation.h"
#include "optimiser.h"
#include "outputlayerdetails.h"
#include <string>

class LayerDetails
{
public:
  LayerDetails( 
    Layer::Architecture layer_architecture,
    unsigned layer_size, 
    const activation& activation, 
    double dropout, 
    double weight_decay, 
    OptimiserType optimiser_type, 
    double momentum) noexcept :
    _layer_architecture(layer_architecture),
    _layer_size(layer_size),
    _activation(activation),
    _dropout(dropout),
    _weight_decay(weight_decay),
    _optimiser_type(optimiser_type),
    _momentum(momentum)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  LayerDetails(const LayerDetails& src) noexcept :
    _layer_architecture(src._layer_architecture),
    _layer_size(src._layer_size),
    _activation(src._activation),
    _dropout(src._dropout),
    _weight_decay(src._weight_decay),
    _optimiser_type(src._optimiser_type),
    _momentum(src._momentum)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  LayerDetails(LayerDetails&& src) noexcept :
    _layer_architecture(src._layer_architecture),
    _layer_size(src._layer_size),
    _activation(std::move(src._activation)),
    _dropout(src._dropout),
    _weight_decay(src._weight_decay),
    _optimiser_type( src._optimiser_type),
    _momentum( src._momentum)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    src._layer_size = 0;
  }

  LayerDetails& operator=(const LayerDetails& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    if (this != &src)
    {
      _layer_architecture = src._layer_architecture;
      _layer_size = src._layer_size;
      _activation = src._activation;
      _dropout = src._dropout;
      _weight_decay = src._weight_decay;
      _optimiser_type = src._optimiser_type;
      _momentum = src._momentum;
    }
    return *this;
  }

  LayerDetails& operator=(LayerDetails&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    if (this != &src)
    {
      _layer_architecture = src._layer_architecture;
      _layer_size = src._layer_size;
      _activation = std::move(src._activation);
      _dropout = src._dropout;
      _weight_decay = src._weight_decay;
      _optimiser_type = src._optimiser_type;
      _momentum = src._momentum;

      src._layer_architecture = Layer::Architecture::None;
      src._layer_size = 0;
      src._dropout = 0.0;
      src._weight_decay = 0;
    }
    return *this;
  }

  virtual ~LayerDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }
  [[nodiscard]] inline const Layer::Architecture& get_layer_architecture() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _layer_architecture;
  }
  [[nodiscard]] inline unsigned get_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _layer_size;
  }
  [[nodiscard]] inline const activation& get_activation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _activation;
  }
  [[nodiscard]] inline double get_dropout() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _dropout;
  }
  [[nodiscard]] inline double get_weight_decay() const  noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _weight_decay;
  }
  [[nodiscard]] inline OptimiserType get_optimiser_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _optimiser_type;
  }
  [[nodiscard]] inline double get_momentum() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _momentum;
  }
private:
  Layer::Architecture _layer_architecture;
  unsigned _layer_size;
  activation _activation;
  double _dropout;
  double _weight_decay;
  OptimiserType _optimiser_type;
  double _momentum;
};