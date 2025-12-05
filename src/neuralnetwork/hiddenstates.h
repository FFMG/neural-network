#pragma once

#include "logger.h"
#include "hiddenstate.h" // New include
#include <exception>
#include <vector>
#include <deque>

class HiddenStates 
{
public:
  HiddenStates(
    const std::vector<unsigned>& topology
  )
  {
    // The 'sequences' parameter from OldHiddenStates constructor is removed
    // as HiddenState does not use sequence_length.
    // Assuming topology provides enough information to size HiddenState.

    auto layer_index = 0;
    for (const unsigned& layer_size : topology)
    {
      std::vector<HiddenState> states_for_layer;
      states_for_layer.reserve(layer_size); // Pre-allocate
      for (unsigned neuron = 0; neuron < layer_size; ++neuron)
      {
        // HiddenState constructor takes num_neurons, not sequence_length
        states_for_layer.emplace_back(HiddenState(layer_size)); // Initialize with layer_size
      }
      _states.push_back(std::move(states_for_layer));
      ++layer_index;
    }
  }

  HiddenStates(const HiddenStates& src) noexcept :
    _states(src._states)
  {
  }

  HiddenStates(HiddenStates&& src) noexcept :
    _states(std::move(src._states))
  {
  }

  HiddenStates& operator=(HiddenStates&& src) noexcept
  {
    if (this != &src)
    {
      _states = std::move(src._states);
    }
    return *this;
  }

  HiddenStates& operator=(const HiddenStates& src) noexcept
  {
    if (this != &src)
    {
      _states = src._states;
    }
    return *this;
  }

 

  std::vector<HiddenState>& at(size_t layer_number)
  {
    return _states[layer_number];
  }
  
  const std::vector<HiddenState>& at(size_t layer_number) const
  {
    return _states[layer_number];
  }

  HiddenState& at(size_t layer_number, size_t neuron_number)
  {
    return _states[layer_number][neuron_number];
  }
  const HiddenState& at(size_t layer_number, size_t neuron_number) const
  {
    return _states[layer_number][neuron_number];
  }

private:
  std::vector<std::vector<HiddenState>> _states;
};