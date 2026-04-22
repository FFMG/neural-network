#pragma once

#include "./libraries/instrumentor.h"

#include "logger.h"

#include "layerdetails.h"
#include "outputlayerdetails.h"

#include <vector>

class MultiOutputLayerDetails
{

public:
  MultiOutputLayerDetails(
    const std::vector<LayerDetails>& hidden_layers,
    const OutputLayerDetails& output_details ) noexcept :
    _hidden_layers(hidden_layers),
    _output_details(output_details)
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
  }

  MultiOutputLayerDetails(const MultiOutputLayerDetails& src) noexcept :
    _hidden_layers(src._hidden_layers),
    _output_details(src._output_details)
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
  }

  MultiOutputLayerDetails(MultiOutputLayerDetails&& src) noexcept :
    _hidden_layers(std::move(src._hidden_layers)),
    _output_details(std::move(src._output_details))

  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
  }

  MultiOutputLayerDetails& operator=(const MultiOutputLayerDetails& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
    if (this != &src)
    {
      _hidden_layers = src._hidden_layers;
      _output_details = src._output_details;
    }
    return *this;
  }

  MultiOutputLayerDetails& operator=(MultiOutputLayerDetails&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
    if (this != &src)
    {
      _hidden_layers = std::move(src._hidden_layers);
      _output_details = std::move(src._output_details);
    }
    return *this;
  }

  virtual ~MultiOutputLayerDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
  }
  [[nodiscard]] inline const OutputLayerDetails& get_output_details() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
    return _output_details;
  }
  [[nodiscard]] inline const std::vector<LayerDetails>& get_hidden_layers() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
    return _hidden_layers;
  }
  [[nodiscard]] inline const LayerDetails& get_hidden_layer(unsigned index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("MultiOutputLayerDetails");
#if VALIDATE_DATA == 1
    if (index >= _hidden_layers.size())
    {
      Logger::panic("Trying to get a Multi Output Hiden Layer Detail past the number of hidden layers!");
    }
#endif
    return _hidden_layers[index];
  }

private:
  std::vector<LayerDetails> _hidden_layers;
  OutputLayerDetails _output_details;
};