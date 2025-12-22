#pragma once

#include <cassert>
#include <vector>

#include "libraries/instrumentor.h"

#include "activation.h"
#include "weightparam.h"

class ResidualProjector
{
public:
  ResidualProjector(
    unsigned input_size,       // size of residual_layer_outputs (e.g., 160)
    unsigned output_size,      // size of the target layer (e.g., 128)
    const activation& activation_method,
    double weight_decay
  )
    :
    _input_size(input_size),
    _output_size(output_size),
    _weights_cache_dirty(true)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    const size_t num_weights = static_cast<size_t>(input_size) * output_size;
    _w_values.resize(num_weights);
    auto initial_weights = activation_method.weight_initialization(output_size, input_size);
    for (size_t i = 0; i < input_size; ++i) {
      for (size_t j = 0; j < output_size; ++j) {
        _w_values[i * output_size + j] = initial_weights[j];
      }
    }

    _w_grads.assign(num_weights, 0.0);
    _w_velocities.assign(num_weights, 0.0);
    _w_m1.assign(num_weights, 0.0);
    _w_m2.assign(num_weights, 0.0);
    _w_timesteps.assign(num_weights, 0);
    _w_decays.assign(num_weights, weight_decay);
  }

  ResidualProjector(const std::vector<std::vector<WeightParam>>& weight_params) :
    _input_size(0),
    _output_size(0),
    _weights_cache_dirty(true)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    _output_size = static_cast<unsigned>(weight_params.size());
    _input_size = _output_size > 0 ? static_cast<unsigned>(weight_params.back().size()) : 0;
    
    const size_t num_weights = static_cast<size_t>(_input_size) * _output_size;
    _w_values.resize(num_weights);
    _w_grads.resize(num_weights);
    _w_velocities.resize(num_weights);
    _w_m1.resize(num_weights);
    _w_m2.resize(num_weights);
    _w_timesteps.resize(num_weights);
    _w_decays.resize(num_weights);

    for (unsigned j = 0; j < _output_size; ++j) {
      for (unsigned i = 0; i < _input_size; ++i) {
        const auto& wp = weight_params[j][i];
        const auto idx = i * _output_size + j;
        _w_values[idx] = wp.get_value();
        _w_grads[idx] = wp.get_raw_gradient();
        _w_velocities[idx] = wp.get_velocity();
        _w_m1[idx] = wp.get_first_moment_estimate();
        _w_m2[idx] = wp.get_second_moment_estimate();
        _w_timesteps[idx] = wp.get_timestep();
        _w_decays[idx] = wp.get_weight_decay();
      }
    }
  }

  ResidualProjector(const ResidualProjector& rp) :
    _input_size(rp._input_size),
    _output_size(rp._output_size),
    _w_values(rp._w_values),
    _w_grads(rp._w_grads),
    _w_velocities(rp._w_velocities),
    _w_m1(rp._w_m1),
    _w_m2(rp._w_m2),
    _w_timesteps(rp._w_timesteps),
    _w_decays(rp._w_decays),
    _weights_cache_dirty(true)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
  }

  ResidualProjector(ResidualProjector&& rp) noexcept :
    _input_size(rp._input_size),
    _output_size(rp._output_size),
    _w_values(std::move(rp._w_values)),
    _w_grads(std::move(rp._w_grads)),
    _w_velocities(std::move(rp._w_velocities)),
    _w_m1(std::move(rp._w_m1)),
    _w_m2(std::move(rp._w_m2)),
    _w_timesteps(std::move(rp._w_timesteps)),
    _w_decays(std::move(rp._w_decays)),
    _weights_cache_dirty(true)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
  }

  virtual ~ResidualProjector() = default;

  // Projects residual_layer_outputs (size = input_size) to a vector of size = output_size
  std::vector<double> project(const std::vector<double>& residual_layer_outputs) const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    assert(residual_layer_outputs.size() == _input_size);
    std::vector<double> projected(_output_size, 0.0);
    for (size_t in = 0; in < _input_size; ++in)
    {
      const double val = residual_layer_outputs[in];
      for (size_t out = 0; out < _output_size; ++out)
      {
        projected[out] += _w_values[in * _output_size + out] * val;
      }
    }
    return projected;
  }

  void apply_weight_gradient(double gradient, double learning_rate, unsigned in, unsigned out, double clipping_scale)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    const auto idx = in * _output_size + out;
    double final_gradient = gradient * clipping_scale;

    if (_w_decays[idx] > 0.0)
    {
      final_gradient += _w_decays[idx] * _w_values[idx];
    }

    _w_values[idx] -= learning_rate * final_gradient;
    _w_grads[idx] = final_gradient;
    _weights_cache_dirty = true;
  }

  const std::vector<std::vector<WeightParam>>& get_weights() const
  {
    return get_weight_params();
  }

  inline const std::vector<std::vector<WeightParam>>& get_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    if (_weights_cache_dirty) {
      _cached_weights.assign(_output_size, std::vector<WeightParam>(_input_size, WeightParam(0,0,0,0)));
      for (unsigned j = 0; j < _output_size; ++j) {
        for (unsigned i = 0; i < _input_size; ++i) {
          const auto idx = i * _output_size + j;
          _cached_weights[j][i] = WeightParam(
            _w_values[idx], _w_grads[idx], _w_velocities[idx],
            _w_m1[idx], _w_m2[idx], _w_timesteps[idx], _w_decays[idx]
          );
        }
      }
      _weights_cache_dirty = false;
    }
    return _cached_weights;
  }

  // This is a bit of a hack to maintain compatibility while refactoring.
  // We can't return a reference to a WeightParam that doesn't exist.
  // Instead, we'll return a WeightParam by value, but then apply_weight_gradient
  // won't be able to update it.
  // So we need to change how this is used in apply_weight_gradients.
  /*
  inline WeightParam& get_weight_params(unsigned out, unsigned in)
  {
    // ...
  }
  */

  void update_weight(size_t out, size_t in, double delta)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    assert(out < _output_size && in < _input_size);
    _w_values[in * _output_size + out] += delta;
    _weights_cache_dirty = true;
  }

  inline unsigned input_size() const {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _input_size;
  };
  inline unsigned output_size() const {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _output_size;
  };

  static ResidualProjector* create(const std::vector<std::vector<WeightParam>>& residual_weights)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (residual_weights.empty())
    {
      return nullptr;
    }
    return new ResidualProjector(residual_weights);
  }

  static ResidualProjector* create(
    int residual_layer_number,
    const activation& activation_method,
    unsigned input_size,
    unsigned  output_size,
    double weight_decay
  ) noexcept
  {
    if (residual_layer_number == 0)
    {
      return nullptr;
    }
    return new ResidualProjector(input_size, output_size, activation_method, weight_decay);
  }

private:
  unsigned _input_size;
  unsigned _output_size;
  
  std::vector<double> _w_values;
  std::vector<double> _w_grads;
  std::vector<double> _w_velocities;
  std::vector<double> _w_m1;
  std::vector<double> _w_m2;
  std::vector<long long> _w_timesteps;
  std::vector<double> _w_decays;

  mutable std::vector<std::vector<WeightParam>> _cached_weights;
  mutable bool _weights_cache_dirty;
};