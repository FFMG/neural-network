#pragma once

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "errorcalculation.h"

class OutputLayerDetails
{
public:
  OutputLayerDetails(unsigned layer_size, const activation& activation, const ErrorCalculation::type& output_error_calculation_type) noexcept :
    _layer_size(layer_size),
    _activation(activation),
    _output_error_calculation_type(output_error_calculation_type)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
  }

  OutputLayerDetails(const OutputLayerDetails& src) noexcept :
    _layer_size(src._layer_size),
    _activation(src._activation),
    _output_error_calculation_type(src._output_error_calculation_type)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
  }

  OutputLayerDetails(OutputLayerDetails&& src) noexcept :
    _layer_size(src._layer_size),
    _activation(std::move(src._activation)),
    _output_error_calculation_type(src._output_error_calculation_type)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    src._layer_size = 0;
  }

  OutputLayerDetails& operator=(const OutputLayerDetails& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    if (this != &src)
    {
      _layer_size = src._layer_size;
      _activation = src._activation;
      _output_error_calculation_type = src._output_error_calculation_type;
    }
    return *this;
  }

  OutputLayerDetails& operator=(OutputLayerDetails&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    if (this != &src)
    {
      _layer_size = src._layer_size;
      _activation = std::move(src._activation);
      _output_error_calculation_type = src._output_error_calculation_type;
      src._layer_size = 0;
    }
    return *this;
  }
  virtual ~OutputLayerDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
  }

  inline unsigned get_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    return _layer_size;
  }
  inline const activation& get_activation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    return _activation;
  }
  inline ErrorCalculation::type get_output_error_calculation_type() const noexcept 
  { 
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails"); 
    return _output_error_calculation_type; 
  }
private:
  unsigned _layer_size;
  activation _activation;
  ErrorCalculation::type _output_error_calculation_type;
};