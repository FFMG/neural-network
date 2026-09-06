#pragma once

#include <optional>
#include <vector>
#include <mutex>
#include <cstring>

#include "../libraries/instrumentor.h"

#include "../common/activation.h"
#include "../common/rng.h"
#include "../common/weightparam.h"
#include "../common/simd_utils.h"


namespace myoddweb::nn
{
class ResidualProjector
{
public:
  ResidualProjector(
    unsigned input_size,       // size of residual_layer_outputs (e.g., 160)
    unsigned output_size,      // size of the target layer (e.g., 128)
    const activation& activation_method,
    double weight_decay,
    std::optional<uint32_t> seed
  )
    :
    _input_size(input_size),
    _output_size(output_size),
    _weights_cache_dirty(true)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    const size_t num_weights = static_cast<size_t>(input_size) * output_size;
    _w_values.resize(num_weights);
    for (size_t i = 0; i < input_size; ++i)
    {
      for (size_t j = 0; j < output_size; ++j)
      {
        const size_t flat_index = i * output_size + j;
        const auto weight_seed = seed.has_value() ? std::optional<uint32_t>(static_cast<uint32_t>(Rng::derive(seed.value(), flat_index))) : std::nullopt;
        _w_values[flat_index] = activation_method.weight_initialization(input_size, output_size, weight_seed);
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

    for (unsigned j = 0; j < _output_size; ++j) 
    {
      for (unsigned i = 0; i < _input_size; ++i) 
      {
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

  ResidualProjector(
    unsigned input_size,
    unsigned output_size,
    const std::vector<double>& w_values,
    const std::vector<double>& w_grads,
    const std::vector<double>& w_velocities,
    const std::vector<double>& w_m1,
    const std::vector<double>& w_m2,
    const std::vector<long long>& w_timesteps,
    const std::vector<double>& w_decays
   ) noexcept :
    _input_size(input_size),
    _output_size(output_size),
    _w_values(w_values),
    _w_grads(w_grads),
    _w_velocities(w_velocities),
    _w_m1(w_m1),
    _w_m2(w_m2),
    _w_timesteps(w_timesteps),
    _w_decays(w_decays),
    _weights_cache_dirty(true)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
  }

  ResidualProjector& operator=(const ResidualProjector& rp)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    if (this != &rp)
    {
      std::scoped_lock lock(_cache_mutex, rp._cache_mutex);
      _input_size = rp._input_size;
      _output_size = rp._output_size;
      _w_values = rp._w_values;
      _w_grads = rp._w_grads;
      _w_velocities = rp._w_velocities;
      _w_m1 = rp._w_m1;
      _w_m2 = rp._w_m2;
      _w_timesteps = rp._w_timesteps;
      _w_decays = rp._w_decays;
      _weights_cache_dirty = true;
    }
    return *this;
  }

  ResidualProjector& operator=(ResidualProjector&& rp) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    if (this != &rp)
    {
      std::scoped_lock lock(_cache_mutex, rp._cache_mutex);
      _input_size = rp._input_size;
      _output_size = rp._output_size;
      _w_values = std::move(rp._w_values);
      _w_grads = std::move(rp._w_grads);
      _w_velocities = std::move(rp._w_velocities);
      _w_m1 = std::move(rp._w_m1);
      _w_m2 = std::move(rp._w_m2);
      _w_timesteps = std::move(rp._w_timesteps);
      _w_decays = std::move(rp._w_decays);
      _weights_cache_dirty = true;
    }
    return *this;
  }

  virtual ~ResidualProjector() = default;

  // Projects residual_layer_outputs (size = input_size) to a vector of size = output_size
  [[nodiscard]] std::vector<double> project(const std::vector<double>& residual_layer_outputs) const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
#if VALIDATE_DATA == 1
    if (residual_layer_outputs.size() != _input_size)
    {
      Logger::panic("The residual layer output size does not match the input size!");
    }
#endif
    std::vector<double> projected(_output_size, 0.0);
    if (_output_size == 0 || _input_size == 0)
    {
      return projected;
    }
    simd::gemm_one_batch(residual_layer_outputs.data(), _w_values.data(), projected.data(), _input_size, _output_size);
    return projected;
  }

  void project(const double* residual_layer_outputs, double* out) const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    if (_output_size == 0 || _input_size == 0 || residual_layer_outputs == nullptr || out == nullptr)
    {
      return;
    }
    std::memset(out, 0, _output_size * sizeof(double));
    simd::gemm_one_batch(residual_layer_outputs, _w_values.data(), out, _input_size, _output_size);
  }

  [[nodiscard]] std::vector<std::vector<double>> project_batch(const std::vector<std::vector<double>>& batch_residual_layer_outputs) const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    std::vector<std::vector<double>> batch_projected;
    project_batch_into(batch_residual_layer_outputs, batch_projected);
    return batch_projected;
  }

  [[nodiscard]] std::vector<std::vector<double>> project_batch(const std::vector<const double*>& batch_residual_layer_outputs) const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    std::vector<std::vector<double>> batch_projected;
    project_batch_into(batch_residual_layer_outputs, batch_projected);
    return batch_projected;
  }

  void project_batch(const double* batch_inputs, double* batch_out, size_t batch_size) const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    if (batch_size == 0 || _output_size == 0 || batch_inputs == nullptr || batch_out == nullptr)
    {
      return;
    }
    std::memset(batch_out, 0, batch_size * _output_size * sizeof(double));
    if (_input_size == 0)
    {
      return;
    }

    size_t b = 0;
    for (; b + 3 < batch_size; b += 4)
    {
      simd::gemm_four_batches(
        batch_inputs + b * _input_size,
        batch_inputs + (b + 1) * _input_size,
        batch_inputs + (b + 2) * _input_size,
        batch_inputs + (b + 3) * _input_size,
        _w_values.data(),
        batch_out + b * _output_size,
        batch_out + (b + 1) * _output_size,
        batch_out + (b + 2) * _output_size,
        batch_out + (b + 3) * _output_size,
        _input_size,
        _output_size
      );
    }
    for (; b + 1 < batch_size; b += 2)
    {
      simd::gemm_two_batches(
        batch_inputs + b * _input_size,
        batch_inputs + (b + 1) * _input_size,
        _w_values.data(),
        batch_out + b * _output_size,
        batch_out + (b + 1) * _output_size,
        _input_size,
        _output_size
      );
    }
    for (; b < batch_size; ++b)
    {
      simd::gemm_one_batch(
        batch_inputs + b * _input_size,
        _w_values.data(),
        batch_out + b * _output_size,
        _input_size,
        _output_size
      );
    }
  }

  void project_batch_into(const std::vector<std::vector<double>>& batch_residual_layer_outputs, std::vector<std::vector<double>>& out) const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    const size_t batch_size = batch_residual_layer_outputs.size();
    if (batch_size == 0)
    {
      out.clear();
      return;
    }
    if (_output_size == 0)
    {
      out.assign(batch_size, std::vector<double>());
      return;
    }
    if (out.size() != batch_size)
    {
      out.resize(batch_size, std::vector<double>(_output_size, 0.0));
    }
    for (size_t b = 0; b < batch_size; ++b)
    {
      if (out[b].size() != _output_size)
      {
        out[b].assign(_output_size, 0.0);
      }
      else
      {
        std::memset(out[b].data(), 0, _output_size * sizeof(double));
      }
    }
    if (_input_size == 0)
    {
      return;
    }

    size_t b = 0;
    for (; b + 3 < batch_size; b += 4)
    {
      simd::gemm_four_batches(
        batch_residual_layer_outputs[b].data(),
        batch_residual_layer_outputs[b + 1].data(),
        batch_residual_layer_outputs[b + 2].data(),
        batch_residual_layer_outputs[b + 3].data(),
        _w_values.data(),
        out[b].data(),
        out[b + 1].data(),
        out[b + 2].data(),
        out[b + 3].data(),
        _input_size,
        _output_size
      );
    }
    for (; b + 1 < batch_size; b += 2)
    {
      simd::gemm_two_batches(
        batch_residual_layer_outputs[b].data(),
        batch_residual_layer_outputs[b + 1].data(),
        _w_values.data(),
        out[b].data(),
        out[b + 1].data(),
        _input_size,
        _output_size
      );
    }
    for (; b < batch_size; ++b)
    {
      simd::gemm_one_batch(
        batch_residual_layer_outputs[b].data(),
        _w_values.data(),
        out[b].data(),
        _input_size,
        _output_size
      );
    }
  }

  void project_batch_into(const std::vector<const double*>& batch_residual_layer_outputs, std::vector<std::vector<double>>& out) const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    const size_t batch_size = batch_residual_layer_outputs.size();
    if (batch_size == 0)
    {
      out.clear();
      return;
    }
    if (_output_size == 0)
    {
      out.assign(batch_size, std::vector<double>());
      return;
    }
    if (out.size() != batch_size)
    {
      out.resize(batch_size, std::vector<double>(_output_size, 0.0));
    }
    for (size_t b = 0; b < batch_size; ++b)
    {
      if (out[b].size() != _output_size)
      {
        out[b].assign(_output_size, 0.0);
      }
      else
      {
        std::memset(out[b].data(), 0, _output_size * sizeof(double));
      }
    }
    if (_input_size == 0)
    {
      return;
    }

    size_t b = 0;
    for (; b + 3 < batch_size; b += 4)
    {
      simd::gemm_four_batches(
        batch_residual_layer_outputs[b],
        batch_residual_layer_outputs[b + 1],
        batch_residual_layer_outputs[b + 2],
        batch_residual_layer_outputs[b + 3],
        _w_values.data(),
        out[b].data(),
        out[b + 1].data(),
        out[b + 2].data(),
        out[b + 3].data(),
        _input_size,
        _output_size
      );
    }
    for (; b + 1 < batch_size; b += 2)
    {
      simd::gemm_two_batches(
        batch_residual_layer_outputs[b],
        batch_residual_layer_outputs[b + 1],
        _w_values.data(),
        out[b].data(),
        out[b + 1].data(),
        _input_size,
        _output_size
      );
    }
    for (; b < batch_size; ++b)
    {
      simd::gemm_one_batch(
        batch_residual_layer_outputs[b],
        _w_values.data(),
        out[b].data(),
        _input_size,
        _output_size
      );
    }
  }

  void apply_weight_gradient(double gradient, double learning_rate, unsigned in, unsigned out, double clipping_scale)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
#if VALIDATE_DATA == 1
    if (in >= _input_size || out >= _output_size)
    {
      Logger::panic("Trying to apply a weight gradient outside of the bounds!");
    }
#endif
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

  [[nodiscard]] inline const std::vector<std::vector<WeightParam>>& get_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    std::lock_guard<std::mutex> lock(_cache_mutex);
    if (_weights_cache_dirty) 
    {
      if (_cached_weights.size() != _output_size)
      {
        _cached_weights.resize(_output_size);
      }
      for (unsigned j = 0; j < _output_size; ++j) 
      {
        if (_cached_weights[j].size() != _input_size)
        {
          _cached_weights[j].resize(_input_size, WeightParam(0, 0, 0, 0));
        }
        for (unsigned i = 0; i < _input_size; ++i) 
        {
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

  // Standard incremental/running-mean SWA update, folding `snapshot`'s weight
  // values into this instance (already the average of `existing_swa_count`
  // prior snapshots). Never touches gradients/velocities/moments/timesteps/decays.
  void accumulate_swa_average(const ResidualProjector& snapshot, size_t existing_swa_count)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    const double denom = static_cast<double>(existing_swa_count + 1);
    const double alpha = 1.0 / denom;
    simd::swa_step(_w_values.data(), snapshot._w_values.data(), alpha, _w_values.size());
    _weights_cache_dirty = true;
  }

  void update_lookahead_slow_weights(ResidualProjector& fast_projector, double alpha)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    simd::lookahead_step(_w_values.data(), fast_projector._w_values.data(), alpha, _w_values.size());
    _weights_cache_dirty = true;
    fast_projector._weights_cache_dirty = true;
  }

  inline void update_weight(size_t out, size_t in, double delta)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
#if VALIDATE_DATA == 1
    if (out >= _output_size || in >= _input_size)
    {
      Logger::panic("Trying to update a weight outsize of the bounds!");
    }
#endif
    _w_values[in * _output_size + out] += delta;
    _weights_cache_dirty = true;
  }

  inline unsigned get_input_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _input_size;
  }
  inline unsigned get_output_size() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _output_size;
  }
  inline const std::vector<double>& get_w_values() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _w_values;
  }
  inline const std::vector<double>& get_w_grads() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _w_grads;
  }
  inline const std::vector<double>& get_w_velocities() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _w_velocities;
  }
  inline const std::vector<double>& get_w_m1() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _w_m1;
  }
  inline const std::vector<double>& get_w_m2() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _w_m2;
  }
  inline const std::vector<long long>& get_w_timesteps() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _w_timesteps;
  }
  inline const std::vector<double>& get_w_decays() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _w_decays;
  }

  static ResidualProjector* create(const std::vector<std::vector<WeightParam>>& residual_weights)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
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
    double weight_decay,
    std::optional<uint32_t> seed
  ) noexcept
  {
    if (residual_layer_number == 0)
    {
      return nullptr;
    }
    return new ResidualProjector(input_size, output_size, activation_method, weight_decay, seed);
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
  mutable std::mutex _cache_mutex;
};
} // namespace myoddweb::nn
