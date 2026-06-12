#pragma once
#include "../libraries/instrumentor.h"
#include "layersandneuronscontainer.h"
#include "logger.h"

#include <map>
#include <span>
#include <vector>


namespace myoddweb::nn
{
class GradientsAndOutputs
{
public:
  GradientsAndOutputs() = delete;

  GradientsAndOutputs(const std::vector<unsigned>& topology) noexcept: 
    _outputs(topology),
    _gradients(topology)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
  }

  GradientsAndOutputs(const GradientsAndOutputs& src) noexcept:
    _outputs(src._outputs),
    _gradients(src._gradients),
    _rnn_outputs(src._rnn_outputs),
    _rnn_gradients(src._rnn_gradients)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
  }

  GradientsAndOutputs(GradientsAndOutputs&& src) noexcept: 
    _outputs(std::move(src._outputs)),
    _gradients(std::move(src._gradients)),
    _rnn_outputs(std::move(src._rnn_outputs)),
    _rnn_gradients(std::move(src._rnn_gradients))
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
  }

  GradientsAndOutputs& operator=(const GradientsAndOutputs& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    if( &src != this)
    {
      _outputs = src._outputs;
      _gradients = src._gradients;
      _rnn_outputs = src._rnn_outputs;
      _rnn_gradients = src._rnn_gradients;
      _rnn_gate_gradients = src._rnn_gate_gradients;
    }
    return *this;
  }

  GradientsAndOutputs& operator=(GradientsAndOutputs&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    if( &src != this)
    {
      _outputs = std::move(src._outputs);
      _gradients = std::move(src._gradients);
      _rnn_outputs = std::move(src._rnn_outputs);
      _rnn_gradients = std::move(src._rnn_gradients);
      _rnn_gate_gradients = std::move(src._rnn_gate_gradients);
    }
    return *this;
  }
  virtual ~GradientsAndOutputs()
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
  }

  inline void zero()
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    _outputs.zero();
    _gradients.zero();
    for (auto& vec : _rnn_outputs) vec.clear();
    for (auto& vec : _rnn_gradients) vec.clear();
    for (auto& vec : _rnn_gate_gradients) vec.clear();
  }

  [[nodiscard]] inline std::span<const double> get_gradients(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return _gradients.get_span(layer);
  }

  [[nodiscard]] inline double* get_gradients_raw(unsigned layer)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return _gradients.get_raw_ptr(layer);
  }

  [[nodiscard]] inline const double* get_gradients_raw(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return _gradients.get_raw_ptr(layer);
  }

  inline void set_gradients(unsigned layer, const std::vector<double>& gradients)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    _gradients.set(layer, gradients);
  }

  [[nodiscard]] inline std::span<const double> get_outputs(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return _outputs.get_span(layer);
  }

  [[nodiscard]] inline double* get_outputs_raw(unsigned layer)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return _outputs.get_raw_ptr(layer);
  }

  [[nodiscard]] inline const double* get_outputs_raw(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return _outputs.get_raw_ptr(layer);
  }

  [[nodiscard]] inline double get_output(unsigned layer, unsigned neuron) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    if (_outputs.number_neurons(layer) == neuron)
    {
      return 1.0; // bias
    }
    return _outputs.get(layer, neuron);
  }

  inline void set_outputs(unsigned layer, const std::vector<double>& outputs)
  {
    _outputs.set(layer, outputs);
  }
  
  [[nodiscard]] inline std::vector<double> output_back() const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    const auto size = _outputs.number_layers();
    if (size == 0)
    {
      Logger::panic("No layers in container");
    }
    const auto s = _outputs.get_span(static_cast<unsigned>(size - 1));
    return std::vector<double>(s.begin(), s.end());
  }

  inline void set_rnn_outputs(unsigned layer, const std::vector<double>& outputs)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    if (layer >= _rnn_outputs.size())
    {
      _rnn_outputs.resize(layer + 1);
    }
    _rnn_outputs[layer] = outputs;
  }

  [[nodiscard]] inline const std::vector<double>& get_rnn_outputs(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    if (layer >= _rnn_outputs.size())
    {
      static const std::vector<double> empty = {};
      return empty;
    }
    return _rnn_outputs[layer];
  }

  inline void set_rnn_gradients(unsigned layer, const std::vector<double>& gradients)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    if (layer >= _rnn_gradients.size())
    {
      _rnn_gradients.resize(layer + 1);
    }
    _rnn_gradients[layer] = gradients;
  }

  [[nodiscard]] inline const std::vector<double>& get_rnn_gradients(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    if (layer >= _rnn_gradients.size())
    {
      static const std::vector<double> empty = {};
      return empty;
    }
    return _rnn_gradients[layer];
  }

  [[nodiscard]] inline double* get_rnn_gradients_raw(unsigned layer, size_t size)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    if (layer >= _rnn_gradients.size())
    {
      _rnn_gradients.resize(layer + 1);
    }
    auto& vec = _rnn_gradients[layer];
    if (vec.size() != size)
    {
      vec.resize(size);
    }
    return vec.data();
  }

  inline void set_rnn_gate_gradients(unsigned layer, const std::vector<double>& gradients)
  {
    if (layer >= _rnn_gate_gradients.size()) _rnn_gate_gradients.resize(layer + 1);
    _rnn_gate_gradients[layer] = gradients;
  }

  [[nodiscard]] inline const std::vector<double>& get_rnn_gate_gradients(unsigned layer) const
  {
    if(layer >= _rnn_gate_gradients.size())
    {
      static const std::vector<double> empty = {};
      return empty;
    }
    return _rnn_gate_gradients[layer];
  }

  [[nodiscard]] inline double* get_rnn_gate_gradients_raw(unsigned layer, size_t size)
  {
    if (layer >= _rnn_gate_gradients.size()) _rnn_gate_gradients.resize(layer + 1);
    auto& vec = _rnn_gate_gradients[layer];
    if (vec.size() != size) vec.resize(size);
    return vec.data();
  }

private:
  LayersAndNeuronsContainer _outputs;
  LayersAndNeuronsContainer _gradients;
  std::vector<std::vector<double>> _rnn_outputs;
  std::vector<std::vector<double>> _rnn_gradients;
  std::vector<std::vector<double>> _rnn_gate_gradients;
};

} // namespace myoddweb::nn
