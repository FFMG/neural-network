#pragma once

#include <string>

#define NEURALNETWORK_VERSION_MAJOR 0
#define NEURALNETWORK_VERSION_MINOR 0
#define NEURALNETWORK_VERSION_PATCH 71
#define NEURALNETWORK_VERSION_STRING "0.0.71"
#define NEURALNETWORK_VERSION_CODE ((NEURALNETWORK_VERSION_MAJOR << 16) | (NEURALNETWORK_VERSION_MINOR << 8) | (NEURALNETWORK_VERSION_PATCH))

namespace myoddweb::nn
{
class Version
{
public:
  [[nodiscard]] static constexpr unsigned major() noexcept
  {
    return NEURALNETWORK_VERSION_MAJOR;
  }

  [[nodiscard]] static constexpr unsigned minor() noexcept
  {
    return NEURALNETWORK_VERSION_MINOR;
  }

  [[nodiscard]] static constexpr unsigned patch() noexcept
  {
    return NEURALNETWORK_VERSION_PATCH;
  }

  [[nodiscard]] static constexpr unsigned code() noexcept
  {
    return NEURALNETWORK_VERSION_CODE;
  }

  [[nodiscard]] static constexpr const char* string() noexcept
  {
    return NEURALNETWORK_VERSION_STRING;
  }
};
} // namespace myoddweb::nn
