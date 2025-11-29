#pragma once

#include "logger.h"
#include <exception>
#include <vector>
#include <deque>

class HiddenState
{
public:
  HiddenState(unsigned sequence_length) noexcept
    : 
    _sequence_length(sequence_length)
  {
  }

  HiddenState(const HiddenState& hs) noexcept
    :
    _sequence_length(hs._sequence_length)
  {
    _hidden_state_history = hs._hidden_state_history;
  }

  HiddenState(HiddenState&& hs) noexcept
    :
    _sequence_length(hs._sequence_length)
  {
    _hidden_state_history = std::move(hs._hidden_state_history);
    hs._sequence_length = 0;
  }

  HiddenState& operator=(const HiddenState& hs) noexcept
  {
    if (this != &hs)
    {
      _sequence_length = hs._sequence_length;
      _hidden_state_history = hs._hidden_state_history;
    }
    return *this;
  }

  HiddenState& operator=(HiddenState&& hs) noexcept
  {
    if (this != &hs)
    {
      _sequence_length = hs._sequence_length;
      _hidden_state_history = std::move(hs._hidden_state_history);
      hs._sequence_length = 0;
    }
    return *this;
  }

  inline bool is_recurrent_neural_network() const noexcept
  {
    return _sequence_length > 0;
  }

  double update_sum(double sum, double weight) const
  {
    const auto& hidden_state_history = _hidden_state_history;
    if (hidden_state_history.empty())
    {
      return sum;
    }

    sum += _hidden_state_history.back() * weight;
    return sum;
  }

  void queue_output(double output)
  {
    if (!is_recurrent_neural_network())
    {
      return;
    }
    _hidden_state_history.push_back(output);

    // Trim buffer to segment size
    if (_hidden_state_history.size() > static_cast<size_t>(_sequence_length))
    {
      _hidden_state_history.pop_front();
    }
  }

  void clear()
  {
    _hidden_state_history.clear();
  }

  unsigned get_sequence_length() const noexcept
  {
    return _sequence_length;
  }

private:
  unsigned _sequence_length;
  std::deque<double> _hidden_state_history;
};

class HiddenStates
{
public:
  HiddenStates(
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
    for (const unsigned& layer : topology)
    {
      std::vector<HiddenState> states;
      for (unsigned neuron = 0; neuron < layer; ++neuron)
      {
        states.push_back(HiddenState(sequences[layer_index]));
      }
      _states.push_back(std::move(states));
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

  std::vector<HiddenState>& at(size_t layer_number)
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
