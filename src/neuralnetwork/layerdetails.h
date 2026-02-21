#pragma once

#include "./libraries/instrumentor.h"

#include "logger.h"

#include <string>
#include "activation.h"

class LayerDetails
{
public:
  enum class LayerType
  {
    None,
    FF,
    Elman,
    Gru
  };

public:
  LayerDetails( LayerType layer_type, unsigned layer_size, const activation& activation, double dropout) noexcept :
    _layer_type(layer_type),
    _layer_size(layer_size),
    _activation(activation),
    _dropout(dropout)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  LayerDetails(const LayerDetails& src) noexcept :
    _layer_type(src._layer_type),
    _layer_size(src._layer_size),
    _activation(src._activation),
    _dropout(src._dropout)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  LayerDetails(LayerDetails&& src) noexcept :
    _layer_type(src._layer_type),
    _layer_size(src._layer_size),
    _activation(std::move(src._activation)),
    _dropout(src._dropout)
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

      src._layer_type = LayerType::None;
      src._layer_size = 0;
      src._dropout = 0.0;
    }
    return *this;
  }

  virtual ~LayerDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }
  inline const LayerType& get_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _layer_type;
  }
  inline unsigned get_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _layer_size;
  }
  inline std::string get_type_string() const 
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

  inline static LayerType type_from_string(const std::string& str)
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
  inline const activation& get_activation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _activation;
  }
  inline double get_dropout() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    return _dropout;
  }
private:
  LayerType _layer_type;
  unsigned _layer_size;
  activation _activation;
  double _dropout;
};