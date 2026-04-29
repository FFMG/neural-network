#pragma once
#include "./libraries/instrumentor.h"
#include "layersandneuronscontainer.h"
#include "logger.h"
#include <map>
#include <span>
#include <vector>

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
    _rnn_outputs.clear();
    _rnn_gradients.clear();
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

  [[nodiscard]] inline std::span<const double> get_outputs(unsigned layer) const noexcept
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

  [[nodiscard]] inline double get_output(unsigned layer, unsigned neuron) const noexcept
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
    _rnn_outputs[layer] = outputs;
  }

  [[nodiscard]] inline const std::vector<double>& get_rnn_outputs(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    const auto it = _rnn_outputs.find(layer);
    if(it == _rnn_outputs.end())
    {
      static const std::vector<double> empty = {};
      return empty;
    }
    return it->second;
  }

  inline void set_rnn_gradients(unsigned layer, const std::vector<double>& gradients)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    _rnn_gradients[layer] = gradients;
  }

  [[nodiscard]] inline const std::vector<double>& get_rnn_gradients(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    const auto it = _rnn_gradients.find(layer);
    if(it == _rnn_gradients.end())
    {
      static const std::vector<double> empty = {};
      return empty;
    }
    return it->second;
  }

  [[nodiscard]] inline double* get_rnn_gradients_raw(unsigned layer, size_t size)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    auto& vec = _rnn_gradients[layer];
    if (vec.size() != size)
    {
      vec.resize(size);
    }
    return vec.data();
  }

private:
  LayersAndNeuronsContainer _outputs;
  LayersAndNeuronsContainer _gradients;
  std::map<unsigned, std::vector<double>> _rnn_outputs;
  std::map<unsigned, std::vector<double>> _rnn_gradients;
};
