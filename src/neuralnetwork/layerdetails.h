#pragma once

#include "./libraries/instrumentor.h"

#include "logger.h"

#include "activation.h"
#include "optimiser.h"
#include "outputlayerdetails.h"
#include <string>

class LayerDetails
{
public:
  enum class LayerType
  {
    None,
    FF,
    Elman,
    Gru,
    Branched
  };

public:
  struct BranchDetails
  {
    std::vector<LayerDetails> hidden_layers;
    OutputLayerDetails output_details;
  };

public:
  LayerDetails( 
    LayerType layer_type, 
    unsigned layer_size, 
    const activation& activation, 
    double dropout, 
    double weight_decay, 
    OptimiserType optimiser_type, 
    double momentum) noexcept :
    _layer_type(layer_type),
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
    _layer_type(src._layer_type),
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
    _layer_type(src._layer_type),
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
      _layer_type = src._layer_type;
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
      _layer_type = src._layer_type;
      _layer_size = src._layer_size;
      _activation = std::move(src._activation);
      _dropout = src._dropout;
      _weight_decay = src._weight_decay;
      _optimiser_type = src._optimiser_type;
      _momentum = src._momentum;

      src._layer_type = LayerType::None;
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
  [[nodiscard]] inline const LayerType& get_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _layer_type;
  }
  [[nodiscard]] inline unsigned get_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _layer_size;
  }
  [[nodiscard]] inline std::string get_type_string() const
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    switch (_layer_type)
    {
    case LayerType::None:
      return "None";

    case LayerType::FF:
      return "FF";

    case LayerType::Elman:
      return "Elman";

    case LayerType::Gru:
      return "Gru";

    default:
      Logger::panic("Unknown Layer type: ", (int)_layer_type);
    }
  }

  [[nodiscard]] inline static LayerType type_from_string(const std::string& str)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    std::string lower_str = str;
    // Convert the string to lowercase for case-insensitive comparison
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
      [](unsigned char c) { return std::tolower(c); });

    if (lower_str == "none")
    {
      return LayerType::None;
    }
    if (lower_str == "ff")
    {
      return LayerType::FF;
    }
    if (lower_str == "elman")
    {
      return LayerType::Elman;
    }
    if (lower_str == "gru")
    {
      return LayerType::Gru;
    }
    Logger::panic("Unknown Layer type: ", str);
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
  LayerType _layer_type;
  unsigned _layer_size;
  activation _activation;
  double _dropout;
  double _weight_decay;
  OptimiserType _optimiser_type;
  double _momentum;
};