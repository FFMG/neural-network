#pragma once
#include "./libraries/instrumentor.h"
#include <vector>
#include <deque>
#include <cassert>

#include "hiddenstate.h"
#include "logger.h"

class HiddenStates 
{
public:
  HiddenStates( const std::vector<unsigned>& topology )
    : _topology(topology)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    const size_t num_layers = topology.size();
    _pre_activation_sums.resize(num_layers);
    _hidden_state_values.resize(num_layers);
    _cell_state_values.resize(num_layers);
    _layer_views.resize(num_layers);
  }

  HiddenStates(const HiddenStates& src) = default;
  HiddenStates(HiddenStates&& src) noexcept = default;
  HiddenStates& operator=(HiddenStates&& src) noexcept = default;
  HiddenStates& operator=(const HiddenStates& src) = default;

  void zero()
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    for (auto& layer : _pre_activation_sums) std::fill(layer.begin(), layer.end(), 0.0);
    for (auto& layer : _hidden_state_values) std::fill(layer.begin(), layer.end(), 0.0);
    for (auto& layer : _cell_state_values) std::fill(layer.begin(), layer.end(), 0.0);
  }
  
  void assign(size_t layer_number, size_t num_time_steps, const HiddenState& /*ignored_proto*/)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    const size_t n = _topology[layer_number];
    const size_t total_size = num_time_steps * n;

    if (_pre_activation_sums[layer_number].size() != total_size)
    {
      _pre_activation_sums[layer_number].assign(total_size, 0.0);
      _hidden_state_values[layer_number].assign(total_size, 0.0);
      _cell_state_values[layer_number].assign(total_size, 0.0);
    }

    // Rebuild views for this layer
    auto& views = _layer_views[layer_number];
    views.clear();
    views.reserve(num_time_steps);
    
    double* p_pre = _pre_activation_sums[layer_number].data();
    double* p_hid = _hidden_state_values[layer_number].data();
    double* p_cel = _cell_state_values[layer_number].data();

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      views.emplace_back(&p_pre[t * n], &p_hid[t * n], &p_cel[t * n], static_cast<unsigned>(n));
    }
  }

  std::vector<HiddenState>& at(size_t layer_number)
  {
    return _layer_views[layer_number];
  }
  
  const std::vector<HiddenState>& at(size_t layer_number) const
  {
    return _layer_views[layer_number];
  }

  HiddenState& at(size_t layer_number, size_t time_step)
  {
    return _layer_views[layer_number][time_step];
  }

  const HiddenState& at(size_t layer_number, size_t time_step) const
  {
    return _layer_views[layer_number][time_step];
  }

private:
  std::vector<unsigned> _topology;
  std::vector<std::vector<double>> _pre_activation_sums; // [layer][time * neuron]
  std::vector<std::vector<double>> _hidden_state_values;
  std::vector<std::vector<double>> _cell_state_values;
  std::vector<std::vector<HiddenState>> _layer_views;   // [layer][time]
};
