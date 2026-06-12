#pragma once

#include "../libraries/instrumentor.h"
#include "errorcalculation.h"


namespace myoddweb::nn
{
class NeuralNetworkHelperMetrics final
{
public:
  [[nodiscard]] inline double error() const noexcept
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

  NeuralNetworkHelperMetrics(double error, ErrorCalculation::type error_type) noexcept :
    _error(error),
    _error_type(error_type)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
  }

protected:
  double _error;
  ErrorCalculation::type _error_type;
};

} // namespace myoddweb::nn
