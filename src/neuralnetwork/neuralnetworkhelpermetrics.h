#pragma once

#include "./libraries/instrumentor.h"
#include "errorcalculation.h"

class NeuralNetworkHelperMetrics final
{
public:
  [[nodiscard]] inline long double error() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    return _error;
  }

  [[nodiscard]] inline ErrorCalculation::type error_type() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    return _error_type;
  }

  virtual ~NeuralNetworkHelperMetrics() = default;

  NeuralNetworkHelperMetrics(const NeuralNetworkHelperMetrics& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    *this = src;
  }
  NeuralNetworkHelperMetrics& operator=(const NeuralNetworkHelperMetrics& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    if (this != &src)
    {
      _error = src._error;
      _error_type = src._error_type;
    }
    return *this;
  }

  NeuralNetworkHelperMetrics(NeuralNetworkHelperMetrics&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    *this = std::move(src);
  }

  NeuralNetworkHelperMetrics& operator=(NeuralNetworkHelperMetrics&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    if (this != &src)
    {
      _error = src._error;
      _error_type = src._error_type;
      src._error = 0.0;
      src._error_type = ErrorCalculation::type::none;
    }
    return *this;
  }

  NeuralNetworkHelperMetrics(long double error, ErrorCalculation::type error_type) noexcept :
    _error(error),
    _error_type(error_type)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
  }

protected:
  long double _error;
  ErrorCalculation::type _error_type;
};
