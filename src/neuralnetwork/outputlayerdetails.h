#pragma once

#include "./libraries/instrumentor.h"

#include "activation.h"
#include "errorcalculation.h"
#include "evaluationconfig.h"

class OutputLayerDetails
{
public:
  OutputLayerDetails(
    unsigned layer_size, 
    const activation& activation, 
    const ErrorCalculation::type& output_error_calculation_type, 
    const EvaluationConfig& error_evaluation_config,
    double weight_decay) noexcept :
    _layer_size(layer_size),
    _activation(activation),
    _output_error_calculation_type(output_error_calculation_type),
    _error_evaluation_config(error_evaluation_config),
    _weight_decay(weight_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    if (weight_decay < 0)
    {
      Logger::panic("The weight decay cannot be -ve!");
    }
  }

  OutputLayerDetails(const OutputLayerDetails& src) noexcept :
    _layer_size(src._layer_size),
    _activation(src._activation),
    _output_error_calculation_type(src._output_error_calculation_type),
    _error_evaluation_config(src._error_evaluation_config),
    _weight_decay(src._weight_decay)
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
  }

  OutputLayerDetails(OutputLayerDetails&& src) noexcept :
    _layer_size(src._layer_size),
    _activation(std::move(src._activation)),
    _output_error_calculation_type(src._output_error_calculation_type),
    _error_evaluation_config(std::move(src._error_evaluation_config)),
    _weight_decay(src._weight_decay)
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
      _weight_decay = src._weight_decay;
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
      _weight_decay = src._weight_decay;
      src._weight_decay = 0;
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

  [[nodiscard]] inline const EvaluationConfig& get_error_evaluation_config() const  noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    return _error_evaluation_config;
  }
  [[nodiscard]] inline double get_weight_decay() const  noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("OutputLayerDetails");
    return _weight_decay;
  }
private:
  unsigned _layer_size;
  activation _activation;
  ErrorCalculation::type _output_error_calculation_type;
  EvaluationConfig _error_evaluation_config;
  double _weight_decay;
};