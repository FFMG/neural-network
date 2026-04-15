#include "./libraries/instrumentor.h"
#include "aligned_allocator.h"
#include <cassert>
#include <vector>
#include <algorithm>

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
    _data[_offsets[layer] + neuron] = data;
  }
  
  void set(unsigned layer, const std::vector<double>& data)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    assert(number_neurons(layer) == data.size());
    std::copy(data.begin(), data.end(), _data.begin() + _offsets[layer]);
  }
  
  inline const double& get(unsigned layer, unsigned neuron) const noexcept
  {
    return _data[_offsets[layer] + neuron];
  }

  const double* get_raw_ptr(unsigned layer) const
  {
    return _data.data() + _offsets[layer];
  }

  double* get_raw_ptr(unsigned layer)
  {
    return _data.data() + _offsets[layer];
  }

  std::vector<double> get_neurons(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    const auto start = _data.begin() + _offsets[layer];
    return std::vector<double>(start, start + _topology[layer]);
  }

  size_t number_layers() const noexcept
  {
    return _topology.size();
  }

  size_t number_neurons(size_t layer) const noexcept
  {
    if(layer >= _topology.size()) return 0;
    return _topology[layer];
  }

private:
  std::vector<size_t> _offsets;
  size_t _total_size = 0;
  std::vector<double, AlignedAllocator<double, 32>> _data;
  std::vector<unsigned> _topology;
};
