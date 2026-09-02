#pragma once

#include <optional>

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

  [[nodiscard]] inline std::optional<size_t> numerator() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    return _numerator;
  }

  [[nodiscard]] inline std::optional<size_t> denominator() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    return _denominator;
  }

  NeuralNetworkHelperMetrics() noexcept :
    _error(0.0),
    _error_type(ErrorCalculation::type::none)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
  }

  NeuralNetworkHelperMetrics(const NeuralNetworkHelperMetrics& src) noexcept :
    _error(src._error),
    _error_type(src._error_type),
    _numerator(src._numerator),
    _denominator(src._denominator)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
  }
  NeuralNetworkHelperMetrics& operator=(const NeuralNetworkHelperMetrics& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    if (this != &src)
    {
      _error = src._error;
      _error_type = src._error_type;
      _numerator = src._numerator;
      _denominator = src._denominator;
    }
    return *this;
  }

  NeuralNetworkHelperMetrics(NeuralNetworkHelperMetrics&& src) noexcept :
    _error(src._error),
    _error_type(src._error_type),
    _numerator(src._numerator),
    _denominator(src._denominator)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    src._error = 0.0;
    src._error_type = ErrorCalculation::type::none;
    src._numerator = std::nullopt;
    src._denominator = std::nullopt;
  }

  NeuralNetworkHelperMetrics& operator=(NeuralNetworkHelperMetrics&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
    if (this != &src)
    {
      _error = src._error;
      _error_type = src._error_type;
      _numerator = src._numerator;
      _denominator = src._denominator;
      src._error = 0.0;
      src._error_type = ErrorCalculation::type::none;
      src._numerator = std::nullopt;
      src._denominator = std::nullopt;
    }
    return *this;
  }

  NeuralNetworkHelperMetrics(double error, ErrorCalculation::type error_type) noexcept :
    _error(error),
    _error_type(error_type)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
  }

  NeuralNetworkHelperMetrics(double error, ErrorCalculation::type error_type, std::optional<size_t> numerator, std::optional<size_t> denominator) noexcept :
    _error(error),
    _error_type(error_type),
    _numerator(numerator),
    _denominator(denominator)
  {
    MYODDWEB_PROFILE_FUNCTION("NeuralNetworkHelperMetrics");
  }

private:
  double _error;
  ErrorCalculation::type _error_type;
  std::optional<size_t> _numerator;
  std::optional<size_t> _denominator;
};

} // namespace myoddweb::nn
