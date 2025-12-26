#pragma once

#include <map>
#include <vector>

#include "./libraries/instrumentor.h"
#include "layersandneuronscontainer.h"

class GradientsAndOutputs
{
public:
  GradientsAndOutputs() = delete;

  GradientsAndOutputs(const std::vector<unsigned>& topology) noexcept: 
    _outputs(topology),
    _gradients(topology),
    _rnn_outputs({}),
    _rnn_gradients({})
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
  virtual ~GradientsAndOutputs() = default;

  void zero()
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    _outputs.zero();
    _gradients.zero();
    _rnn_outputs.clear();
    _rnn_gradients.clear();
  }

  [[nodiscard]] inline unsigned num_output_layers() const  noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return static_cast<unsigned>(this->_outputs.number_layers());
  }

  [[nodiscard]] inline unsigned num_output_neurons(unsigned layer) const noexcept
  { 
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return static_cast<unsigned>(this->_outputs.number_neurons(layer));
  }

  double get_gradient(unsigned layer, unsigned neuron) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return this->_gradients.get(layer, neuron);
  }

  void set_gradient(unsigned layer, unsigned neuron, double gradient)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    this->_gradients.set(layer, neuron, gradient);
  }

  [[nodiscard]] inline const std::vector<double>& get_gradients(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return this->_gradients.get_neurons(layer);
  }

  [[nodiscard]] inline double* get_gradients_raw(unsigned layer)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return this->_gradients.get_raw_ptr(layer);
  }

  [[nodiscard]] inline const double* get_gradients_raw(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return this->_gradients.get_raw_ptr(layer);
  }

  void set_gradients(unsigned layer, const std::vector<double>& gradients)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    this->_gradients.set(layer, gradients);
  }

  void set_gradients(const LayersAndNeuronsContainer& gradients)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    this->_gradients = gradients;
  }

  void set_gradients(const std::vector<std::vector<double>>& gradients)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    this->_gradients = gradients;
  }

  unsigned num_gradient_layers() const 
  { 
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return static_cast<unsigned>(this->_gradients.number_layers());
  }

  unsigned num_gradient_neurons(unsigned layer) const 
  { 
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return static_cast<unsigned>(this->_gradients.number_neurons(layer));
  }

  [[nodiscard]] inline const std::vector<double>& get_outputs(unsigned layer) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return this->_outputs.get_neurons(layer);
  }

  [[nodiscard]] inline double* get_outputs_raw(unsigned layer)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return this->_outputs.get_raw_ptr(layer);
  }

  [[nodiscard]] inline const double* get_outputs_raw(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return this->_outputs.get_raw_ptr(layer);
  }

  [[nodiscard]] inline double get_output(unsigned layer, unsigned neuron) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
#ifdef _MSC_VER      
    if (this->_outputs.number_neurons(layer) == neuron)
#else
    if(__builtin_expect(this->_outputs.number_neurons(layer) == neuron, 0))
#endif
    {
      return 1.0; //  bias
    }      
    return this->_outputs.get(layer, neuron);
  }

  unsigned num_outputs(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    return static_cast<unsigned>(this->_outputs.number_neurons(layer));
  }
  
  void set_outputs(unsigned layer, const std::vector<double>& outputs)
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    this->_outputs.set(layer, outputs);
  }
  
  [[nodiscard]] const std::vector<double>& output_back() const
  {
    MYODDWEB_PROFILE_FUNCTION("GradientsAndOutputs");
    const size_t size = this->_outputs.number_layers();
    if(size == 0)
    {
      std::cerr << "Trying to get the last output but none available!" << std::endl;
      throw std::invalid_argument("Trying to get the last output but none available!");
    }
    return this->_outputs.get_neurons(static_cast<unsigned>(size -1));
  }

private:
  LayersAndNeuronsContainer _outputs;
  LayersAndNeuronsContainer _gradients;
  std::map<unsigned, std::vector<double>> _rnn_outputs;
  std::map<unsigned, std::vector<double>> _rnn_gradients;

public:
  void set_rnn_outputs(unsigned layer, const std::vector<double>& outputs)
  {
    _rnn_outputs[layer] = outputs;
  }

  const std::vector<double>& get_rnn_outputs(unsigned layer) const
  {
    const auto it = _rnn_outputs.find(layer);
    if(it == _rnn_outputs.end())
    {
      static const std::vector<double> empty = {};
      return empty;
    }
    return it->second;
  }

  void set_rnn_gradients(unsigned layer, const std::vector<double>& gradients)
  {
    _rnn_gradients[layer] = gradients;
  }

  const std::vector<double>& get_rnn_gradients(unsigned layer) const
  {
    const auto it = _rnn_gradients.find(layer);
    if(it == _rnn_gradients.end())
    {
      static const std::vector<double> empty = {};
      return empty;
    }
    return it->second;
  }
};
