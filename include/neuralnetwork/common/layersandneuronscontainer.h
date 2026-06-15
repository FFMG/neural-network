#include "../libraries/instrumentor.h"
#include "aligned_allocator.h"
#include "logger.h"
#include <algorithm>
#include <span>
#include <vector>


namespace myoddweb::nn
{
class LayersAndNeuronsContainer
{
public:
  LayersAndNeuronsContainer() = default;

  LayersAndNeuronsContainer(const std::vector<unsigned>& topology) noexcept :
    _topology(topology)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    const size_t size = _topology.size();
    _offsets.resize(size);
    _total_size = 0;
    for(size_t layer = 0; layer < size; ++layer)
    {
      _offsets[layer] = _total_size;
      _total_size += _topology[layer];
    }
    _data.assign(_total_size, 0.0);
  }

  LayersAndNeuronsContainer(const LayersAndNeuronsContainer& src) noexcept :
    _offsets(src._offsets),
    _total_size(src._total_size),
    _data(src._data),
    _topology(src._topology)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
  }

  LayersAndNeuronsContainer(LayersAndNeuronsContainer&& src) noexcept :
    _offsets(std::move(src._offsets)),
    _total_size(src._total_size),
    _data(std::move(src._data)),
    _topology(std::move(src._topology))
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    src._total_size = 0;
  }

  LayersAndNeuronsContainer& operator=(const LayersAndNeuronsContainer& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    if(this != &src)
    {
      _offsets = src._offsets;
      _total_size = src._total_size;
      _data = src._data;
      _topology = src._topology;
    }
    return *this;
  }
 
  LayersAndNeuronsContainer& operator=(LayersAndNeuronsContainer&& src) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    if(this != &src)
    {
      _offsets = std::move(src._offsets);
      _total_size = src._total_size;
      _data = std::move(src._data);
      _topology = std::move(src._topology);
      src._total_size = 0;
    }
    return *this;
  }

  inline void zero()
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    std::fill(_data.begin(), _data.end(), 0.0);
  }

  inline void set(unsigned layer, unsigned neuron, const double& data)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
#if VALIDATE_DATA == 1
    if (layer >= _offsets.size())
    {
      Logger::panic("trying to set value past the layer size: ", layer);
    }
    if (neuron >= _topology[layer])
    {
      Logger::panic("trying to set value past the neuron size: ", neuron);
    }
#endif
    _data[_offsets[layer] + neuron] = data;
  }
  
  void set(unsigned layer, const std::vector<double>& data)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
#if VALIDATE_DATA == 1
    if (number_neurons(layer) != data.size())
    {
      Logger::panic("The number of neurons in the layer does not match the data size: ", layer);
    }
#endif
    std::copy(data.begin(), data.end(), _data.begin() + _offsets[layer]);
  }

  inline void set(unsigned layer, const double* data, size_t size)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
#if VALIDATE_DATA == 1
    if (number_neurons(layer) != size)
    {
      Logger::panic("The number of neurons in the layer does not match the data size: ", layer);
    }
#endif
    std::copy(data, data + size, _data.begin() + _offsets[layer]);
  }
  
  [[nodiscard]] inline const double& get(unsigned layer, unsigned neuron) const
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
#if VALIDATE_DATA == 1
    if (layer >= _offsets.size())
    {
      Logger::panic("trying to get value past the layer size: ", layer);
    }
    if (neuron >= _topology[layer])
    {
      Logger::panic("trying to get value past the neuron size: ", neuron);
    }
#endif
    return _data[_offsets[layer] + neuron];
  }

  [[nodiscard]] inline const double* get_raw_ptr(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
#if VALIDATE_DATA == 1
    if (layer >= _offsets.size())
    {
      Logger::panic("trying to get raw value past the layer size: ", layer);
    }
#endif
    return _data.data() + _offsets[layer];
  }

  [[nodiscard]] inline double* get_raw_ptr(unsigned layer)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
#if VALIDATE_DATA == 1
    if (layer >= _offsets.size())
    {
      Logger::panic("trying to get raw value past the layer size: ", layer);
    }
#endif
    return _data.data() + _offsets[layer];
  }

  [[nodiscard]] inline std::span<const double> get_span(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
#if VALIDATE_DATA == 1
    if (layer >= _offsets.size())
    {
      Logger::panic("trying to neurons past the layer size: ", layer);
    }
#endif
    return std::span<const double>(_data.data() + _offsets[layer], _topology[layer]);
  }

  [[nodiscard]] inline std::vector<double> get_neurons(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    auto s = get_span(layer);
    return std::vector<double>(s.begin(), s.end());
  }

  [[nodiscard]] inline size_t number_layers() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    return _topology.size();
  }

  [[nodiscard]] inline size_t number_neurons(size_t layer) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    if (layer >= _topology.size())
    {
      Logger::warning("Trying to get the number of neurons past the number of layers: ", layer);
      return 0;
    }
    return _topology[layer];
  }

private:
  std::vector<size_t> _offsets;
  size_t _total_size = 0;
  std::vector<double, AlignedAllocator<double, 32>> _data;
  std::vector<unsigned> _topology;
};

} // namespace myoddweb::nn
