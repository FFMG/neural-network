#pragma once

#include "logger.h"
#include <exception>
#include <vector>
#include <deque>

// This is a minimal recreation of the original HiddenState object
// that was coupled with the HiddenStates class.
class OldHiddenState
{
public:
  OldHiddenState(unsigned sequence_length) noexcept
    : 
    _sequence_length(sequence_length),
    _pre_activation_sum(0.0)
  {
  }

  void set_pre_activation_sum(double sum)
  {
    _pre_activation_sum = sum;
  }

  double get_pre_activation_sum() const
  {
    return _pre_activation_sum;
  }

  void clear()
  {
    _pre_activation_sum = 0.0;
  }

private:
  unsigned _sequence_length;
  double _pre_activation_sum;
};

class OldHiddenStates // Renamed from HiddenStates
{
public:
  OldHiddenStates(
    const std::vector<unsigned>& topology,
    const std::vector<unsigned>& sequences
  )
  {
    if (topology.size() != sequences.size())
    {
      Logger::error("The topology size does not match the sequence size!");
      throw std::invalid_argument("The topology size does not match the sequence size!");
    }

    auto layer_index = 0;
    for (const unsigned& layer_size : topology)
    {
      std::vector<OldHiddenState> states;
      for (unsigned neuron = 0; neuron < layer_size; ++neuron)
      {
        states.push_back(OldHiddenState(sequences[layer_index]));
      }
      _states.push_back(std::move(states));
      ++layer_index;
    }
  }

  OldHiddenStates(const OldHiddenStates& src) noexcept :
    _states(src._states)
  {
  }

  OldHiddenStates(OldHiddenStates&& src) noexcept :
    _states(std::move(src._states))
  {
  }

  OldHiddenStates& operator=(OldHiddenStates&& src) noexcept
  {
    if (this != &src)
    {
      _states = std::move(src._states);
    }
    return *this;
  }

  OldHiddenStates& operator=(const OldHiddenStates& src) noexcept
  {
    if (this != &src)
    {
      _states = src._states;
    }
    return *this;
  }

  void clear()
  {
    for (auto& layer : _states)
    {
      for (auto& neuron : layer)
      {
        neuron.clear();
      }
    }
  }

  std::vector<OldHiddenState>& at(size_t layer_number)
  {
    return _states[layer_number];
  }
  
  const std::vector<OldHiddenState>& at(size_t layer_number) const
  {
    return _states[layer_number];
  }

  OldHiddenState& at(size_t layer_number, size_t neuron_number)
  {
    return _states[layer_number][neuron_number];
  }
  const OldHiddenState& at(size_t layer_number, size_t neuron_number) const
  {
    return _states[layer_number][neuron_number];
  }

private:
  std::vector<std::vector<OldHiddenState>> _states;
};