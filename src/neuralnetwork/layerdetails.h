#pragma once

#include "./libraries/instrumentor.h"

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
  LayerDetails( LayerType layer_type, unsigned layer_size) noexcept :
    _layer_type(layer_type),
    _layer_size(layer_size)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  LayerDetails(const LayerDetails& src) noexcept :
    _layer_type(src._layer_type),
    _layer_size(src._layer_size)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  LayerDetails(LayerDetails&& src) noexcept :
    _layer_type(src._layer_type),
    _layer_size(src._layer_size)
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    src._layer_type = LayerType::None;
    src._layer_size = 0;
  }

  LayerDetails& operator=(const LayerDetails& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
    if (this != &src)
    {
      _layer_type = src._layer_type;
      _layer_size = src._layer_size;
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

      src._layer_type = LayerType::None;
      src._layer_size = 0;

    }
    return *this;
  }

  virtual ~LayerDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("LayerDetails");
  }

  inline LayerType get_type() const noexcept
  {
    return _layer_type;
  }
  inline unsigned get_size() const noexcept
  {
    return _layer_size;
  }
private:
  LayerType _layer_type;
  unsigned _layer_size;
};