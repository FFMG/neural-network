#pragma once
#include "./libraries/instrumentor.h"
#include <vector>
#include <deque>

#include "hiddenstate.h"
#include "logger.h"

class HiddenStates 
{
public:
  HiddenStates( const std::vector<unsigned>& topology )
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    for (const unsigned& layer_size : topology)
    {
      std::vector<HiddenState> states_for_layer;
      _states.push_back(std::move(states_for_layer));
    }
  }

  HiddenStates(const HiddenStates& src) noexcept :
    _states(src._states)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
  }

  HiddenStates(HiddenStates&& src) noexcept :
    _states(std::move(src._states))
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
  }

  HiddenStates& operator=(HiddenStates&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    if (this != &src)
    {
      _states = std::move(src._states);
    }
    return *this;
  }

  HiddenStates& operator=(const HiddenStates& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    if (this != &src)
    {
      _states = src._states;
    }
    return *this;
  }
  
  std::vector<HiddenState>& at(size_t layer_number)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
#if VALIDATE_DATA == 1
    assert(layer_number < _states.size());
#endif
    return _states[layer_number];
  }
  
  const std::vector<HiddenState>& at(size_t layer_number) const
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
#if VALIDATE_DATA == 1
    assert(layer_number < _states.size());
#endif
    return _states[layer_number];
  }

  HiddenState& at(size_t layer_number, size_t neuron_number)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
#if VALIDATE_DATA == 1
    assert(layer_number < _states.size());
    assert(neuron_number < _states[layer_number].size());
#endif
    return _states[layer_number][neuron_number];
  }
  const HiddenState& at(size_t layer_number, size_t neuron_number) const
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
#if VALIDATE_DATA == 1
    assert(layer_number < _states.size());
    assert(neuron_number < _states[layer_number].size());
#endif
    return _states[layer_number][neuron_number];
  }

private:
  std::vector<std::vector<HiddenState>> _states;
};