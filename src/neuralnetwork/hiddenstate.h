#pragma once
#include "./libraries/instrumentor.h"
#include "logger.h"

#include <vector>
#include <span>


namespace myoddweb::nn
{
class HiddenState
{
public:
  HiddenState() = default;

  HiddenState(double* pre_activation_sums, double* hidden_state_values, double* cell_state_values, unsigned int num_neurons, unsigned int num_pre_activations) :
    _pre_activation_sums(pre_activation_sums, num_pre_activations),
    _hidden_state_values(hidden_state_values, num_neurons),
    _cell_state_values(cell_state_values, num_neurons)
  {
  }

  inline void set_pre_activation_sum(unsigned index, double value) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    _pre_activation_sums[index] = value;
  }

  inline void set_pre_activation_sums(const std::vector<double>& sums) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    const auto size = std::min<size_t>(sums.size(), _pre_activation_sums.size());
    std::copy(sums.begin(), sums.begin() + size, _pre_activation_sums.begin());
  }
    
  inline void set_hidden_state_value(unsigned index, double value) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    if (index < _hidden_state_values.size())
    {
      _hidden_state_values[index] = value;
    }
  }

  inline void set_hidden_state_values(const std::vector<double>& values) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    const auto size = std::min<size_t>(values.size(), _hidden_state_values.size());
    std::copy(values.begin(), values.begin() + size, _hidden_state_values.begin());
  }

  inline void set_cell_state_value(unsigned index, double value) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    if (index < _cell_state_values.size())
    {
      _cell_state_values[index] = value;
    }
  }

  inline void set_cell_state_values(const std::vector<double>& values) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    const auto size = std::min<size_t>(values.size(), _cell_state_values.size());
    std::copy(values.begin(), values.begin() + size, _cell_state_values.begin());
  }

  [[nodiscard]] inline double get_pre_activation_sum_at_neuron(unsigned neuron_index) const
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    if (neuron_index >= _pre_activation_sums.size())
    {
      return 0.0;
    }
    return _pre_activation_sums[neuron_index];
  }

  [[nodiscard]] inline double& get_pre_activation_sum(unsigned index) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    if (index >= _pre_activation_sums.size())
    {
       return _dummy;
    }
    return _pre_activation_sums[index];
  }

  [[nodiscard]] inline std::span<double> get_pre_activation_sums() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    return _pre_activation_sums;
  }

  [[nodiscard]] inline double get_hidden_state_value_at_neuron(unsigned neuron_index) const
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    if (neuron_index >= _hidden_state_values.size())
    {
      return 0.0;
    }
    return _hidden_state_values[neuron_index];
  }

  [[nodiscard]] inline std::span<double> get_hidden_state_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    return _hidden_state_values;
  }

  [[nodiscard]] inline double get_cell_state_value_at_neuron(unsigned neuron_index) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    if (neuron_index >= _cell_state_values.size())
    {
      return 0.0;
    }
    return _cell_state_values[neuron_index];
  }

  [[nodiscard]] inline std::span<double> get_cell_state_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenState");
    return _cell_state_values;
  }

private:
  std::span<double> _pre_activation_sums;
  std::span<double> _hidden_state_values;
  std::span<double> _cell_state_values;
  double _dummy = 0.0;
};

} // namespace myoddweb::nn
