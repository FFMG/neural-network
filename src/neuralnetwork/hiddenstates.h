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

  HiddenStates(const HiddenStates& src) :
    _topology(src._topology),
    _pre_activation_sums(src._pre_activation_sums),
    _hidden_state_values(src._hidden_state_values),
    _cell_state_values(src._cell_state_values)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    rebuild_views();
  }

  HiddenStates(HiddenStates&& src) noexcept :
    _topology(std::move(src._topology)),
    _pre_activation_sums(std::move(src._pre_activation_sums)),
    _hidden_state_values(std::move(src._hidden_state_values)),
    _cell_state_values(std::move(src._cell_state_values))
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    rebuild_views();
  }

  HiddenStates& operator=(const HiddenStates& src)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    if (this != &src)
    {
      _topology = src._topology;
      _pre_activation_sums = src._pre_activation_sums;
      _hidden_state_values = src._hidden_state_values;
      _cell_state_values = src._cell_state_values;
      rebuild_views();
    }
    return *this;
  }

  HiddenStates& operator=(HiddenStates&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    if (this != &src)
    {
      _topology = std::move(src._topology);
      _pre_activation_sums = std::move(src._pre_activation_sums);
      _hidden_state_values = std::move(src._hidden_state_values);
      _cell_state_values = std::move(src._cell_state_values);
      rebuild_views();
    }
    return *this;
  }

  void rebuild_views()
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    _layer_views.clear();
    const size_t num_layers = _topology.size();
    _layer_views.resize(num_layers);

    for (size_t layer_number = 0; layer_number < num_layers; ++layer_number)
    {
      const size_t n = _topology[layer_number];
      if (n == 0) continue;

      const size_t total_hid = _hidden_state_values[layer_number].size();
      const size_t num_time_steps = total_hid / n;
      if (num_time_steps == 0) continue;

      const size_t total_pre = _pre_activation_sums[layer_number].size();
      const unsigned multiplier = static_cast<unsigned>(total_pre / (num_time_steps * n));

      auto& views = _layer_views[layer_number];
      views.reserve(num_time_steps);

      double* p_pre = _pre_activation_sums[layer_number].data();
      double* p_hid = _hidden_state_values[layer_number].data();
      double* p_cel = _cell_state_values[layer_number].data();

      for (size_t t = 0; t < num_time_steps; ++t)
      {
        views.emplace_back(&p_pre[t * n * multiplier], &p_hid[t * n], &p_cel[t * n], static_cast<unsigned>(n), static_cast<unsigned>(n * multiplier));
      }
    }
  }

  void zero()
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    for (auto& layer : _pre_activation_sums) std::fill(layer.begin(), layer.end(), 0.0);
    for (auto& layer : _hidden_state_values) std::fill(layer.begin(), layer.end(), 0.0);
    for (auto& layer : _cell_state_values) std::fill(layer.begin(), layer.end(), 0.0);
  }
  
  void assign(size_t layer_number, size_t num_time_steps, const HiddenState& /*ignored_proto*/, unsigned multiplier = 1)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    const size_t n = _topology[layer_number];
    const size_t total_size = num_time_steps * n;
    const size_t pre_total_size = num_time_steps * n * multiplier;

    if (_pre_activation_sums[layer_number].size() != pre_total_size)
    {
      _pre_activation_sums[layer_number].assign(pre_total_size, 0.0);
    }
    if (_hidden_state_values[layer_number].size() != total_size)
    {
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
      views.emplace_back(&p_pre[t * n * multiplier], &p_hid[t * n], &p_cel[t * n], static_cast<unsigned>(n), static_cast<unsigned>(n * multiplier));
    }
  }

  std::vector<HiddenState>& at(size_t layer_number)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    return _layer_views[layer_number];
  }
  
  const std::vector<HiddenState>& at(size_t layer_number) const
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    return _layer_views[layer_number];
  }

  HiddenState& at(size_t layer_number, size_t time_step)
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    return _layer_views[layer_number][time_step];
  }

  const HiddenState& at(size_t layer_number, size_t time_step) const
  {
    MYODDWEB_PROFILE_FUNCTION("HiddenStates");
    return _layer_views[layer_number][time_step];
  }

private:
  std::vector<unsigned> _topology;
  std::vector<std::vector<double>> _pre_activation_sums; // [layer][time * neuron]
  std::vector<std::vector<double>> _hidden_state_values;
  std::vector<std::vector<double>> _cell_state_values;
  std::vector<std::vector<HiddenState>> _layer_views;   // [layer][time]
};
