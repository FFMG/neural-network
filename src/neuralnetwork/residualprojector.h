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
    _output_size(output_size)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    _weight_params.reserve(output_size);
    for (unsigned out = 0; out < _output_size; ++out)
    {
      std::vector<WeightParam> weights;
      weights.reserve(_input_size);
      auto values = activation_method.weight_initialization(input_size, 1);  // row: 1 x input_size
      for (auto& value : values)
      {
        weights.emplace_back(WeightParam(value, 0.0, 0.0, weight_decay));
      }
      _weight_params.emplace_back(weights);
    }
  }

  ResidualProjector(const std::vector<std::vector<WeightParam>>& weight_params) :
    _input_size(0),
    _output_size(0),
    _weight_params(weight_params)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    _output_size = static_cast<unsigned>(weight_params.size());
    _input_size = _output_size > 0 ? static_cast<unsigned>(weight_params.back().size()) : 0;
  }

  ResidualProjector(const ResidualProjector& rp) :
    _input_size(rp._input_size),
    _output_size(rp._output_size),
    _weight_params(rp._weight_params)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
  }

  ResidualProjector(ResidualProjector&& rp) noexcept :
    _input_size(rp._input_size),
    _output_size(rp._output_size),
    _weight_params(std::move(rp._weight_params))
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
    for (size_t out = 0; out < _output_size; ++out)
    {
      for (size_t in = 0; in < _input_size; ++in)
      {
        auto value = _weight_params[out][in].get_value();
        projected[out] += value * residual_layer_outputs[in];
      }
    }
    return projected;
  }

  void apply_weight_gradient(const double gradient, const double learning_rate, WeightParam& weight_param, double clipping_scale, double gradient_clip_threshold)
  {
    MYODDWEB_PROFILE_FUNCTION("FFLayer");
    auto clipped_gradient = clipping_scale <= 0.0 ? clip_gradient(gradient, gradient_clip_threshold) : gradient * clipping_scale;

    double final_gradient = clipped_gradient;
    if (weight_param.get_weight_decay() > 0.0)
    {
      final_gradient += weight_param.get_weight_decay() * weight_param.get_value();
    }

    double new_weight = weight_param.get_value() - learning_rate * final_gradient;
    weight_param.set_raw_gradient(clipped_gradient);
    weight_param.set_value(new_weight);
  }

  const std::vector<std::vector<WeightParam>>& get_weights() const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _weight_params;
  }
  inline const std::vector<std::vector<WeightParam>>& get_weight_params() const
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    return _weight_params;
  }
  inline WeightParam& get_weight_params(unsigned out, unsigned in)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    assert(out < _weight_params.size());
    assert(in < _weight_params[out].size());
    return _weight_params[out][in];
  }
  void update_weight(size_t out, size_t in, double delta)
  {
    MYODDWEB_PROFILE_FUNCTION("ResidualProjector");
    assert(out < _output_size && in < _input_size);
    auto value = _weight_params[out][in].get_value();
    _weight_params[out][in].set_value(value + delta);
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
  static double clip_gradient(double gradient, double gradient_clip_threshold)
  {
    MYODDWEB_PROFILE_FUNCTION("Layer");
    if (!std::isfinite(gradient))
    {
      Logger::panic("Gradient is not finite.");
    }

    if (gradient > gradient_clip_threshold)
    {
      return gradient_clip_threshold;
    }
    if (gradient < -gradient_clip_threshold)
    {
      return -gradient_clip_threshold;
    }
    return gradient;
  }

  unsigned _input_size;
  unsigned _output_size;
  std::vector<std::vector<WeightParam>> _weight_params;  // shape: [output][input]
};