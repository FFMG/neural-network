#pragma once
#include "./libraries/instrumentor.h"

#include <vector>
#include <cassert>

class HiddenState
{
public:
    HiddenState() = default;

    HiddenState(unsigned num_neurons) noexcept
      : _pre_activation_sums(num_neurons, 0.0),
        _hidden_state_values(num_neurons, 0.0)
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
    }

    HiddenState(const HiddenState& src) noexcept :
      _pre_activation_sums(src._pre_activation_sums),
      _hidden_state_values(src._hidden_state_values)
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
    }

    HiddenState(HiddenState&& src) noexcept :
      _pre_activation_sums(std::move(src._pre_activation_sums)),
      _hidden_state_values(std::move(src._hidden_state_values))
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
    }

    HiddenState& operator=(const HiddenState& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
      if (this != &src)
      {
        _pre_activation_sums = src._pre_activation_sums;
        _hidden_state_values = src._hidden_state_values;
      }
      return *this;
    }

    HiddenState& operator=(HiddenState&& src) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
      if (this != &src)
      {
        _pre_activation_sums = std::move(src._pre_activation_sums);
        _hidden_state_values = std::move(src._hidden_state_values);
      }
      return *this;
    }

    inline void set_pre_activation_sums(const std::vector<double>& sums) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
      _pre_activation_sums = sums;
    }
    
    inline void set_hidden_state_values(const std::vector<double>& values) noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
      _hidden_state_values = values;
    }

    inline double get_pre_activation_sum_at_neuron(unsigned neuron_index) const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
#if VALIDATE_DATA == 1
      assert(neuron_index < _pre_activation_sums.size());
#endif
      return _pre_activation_sums[neuron_index];
    }

    inline const std::vector<double>& get_pre_activation_sums() const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
      return _pre_activation_sums;
    }

    inline double get_hidden_state_value_at_neuron(unsigned neuron_index) const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
#if VALIDATE_DATA == 1
      assert(neuron_index < _hidden_state_values.size());
#endif
      return _hidden_state_values[neuron_index];
    }

    inline const std::vector<double>& get_hidden_state_values() const noexcept
    {
      MYODDWEB_PROFILE_FUNCTION("HiddenState");
      return _hidden_state_values;
    }

private:
    std::vector<double> _pre_activation_sums;
    std::vector<double> _hidden_state_values;
};