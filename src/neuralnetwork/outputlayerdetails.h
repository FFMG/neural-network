#pragma once

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "errorcalculation.h"

class OutputLayerDetails
{
public:
  OutputLayerDetails(unsigned layer_size, const activation& activation, const ErrorCalculation::type& output_error_calculation_type, const ErrorCalculation::EvaluationConfig& error_evaluation_config) noexcept :
    _layer_size(layer_size),
    _activation(activation),
    _output_error_calculation_type(output_error_calculation_type),
    _error_evaluation_config(error_evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
  }

  OutputLayerDetails(const OutputLayerDetails& src) noexcept :
    _layer_size(src._layer_size),
    _activation(src._activation),
    _output_error_calculation_type(src._output_error_calculation_type),
    _error_evaluation_config(src._error_evaluation_config)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
  }

  OutputLayerDetails(OutputLayerDetails&& src) noexcept :
    _layer_size(src._layer_size),
    _activation(std::move(src._activation)),
    _output_error_calculation_type(src._output_error_calculation_type),
    _error_evaluation_config(std::move(src._error_evaluation_config))
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
      _error_evaluation_config = src._error_evaluation_config;
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
      _error_evaluation_config = std::move(src._error_evaluation_config);
      src._layer_size = 0;
    }
    return *this;
  }
  virtual ~OutputLayerDetails()
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
  }

  
  [[nodiscard]] inline unsigned get_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    return _layer_size;
  }
  
  [[nodiscard]] inline const activation& get_activation() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    return _activation;
  }
  
  [[nodiscard]] inline ErrorCalculation::type get_output_error_calculation_type() const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails"); 
    return _output_error_calculation_type; 
  }

  [[nodiscard]] inline const ErrorCalculation::EvaluationConfig& get_error_evaluation_config() const  noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    return _error_evaluation_config;
  }
private:
  unsigned _layer_size;
  activation _activation;
  ErrorCalculation::type _output_error_calculation_type;
  ErrorCalculation::EvaluationConfig _error_evaluation_config;
};