#include "./libraries/instrumentor.h"

#include "aligned_allocator.h"

#include <cassert>

class LayersAndNeuronsContainer
{
public:
  LayersAndNeuronsContainer(const LayersAndNeuronsContainer& src) noexcept :
    _offsets(src._offsets),
    _total_size(src._total_size),
    _data(src._data),
    _topology(src._topology),
    _cached_layers(src._cached_layers),
    _cache_dirty(src._cache_dirty)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
  }

  LayersAndNeuronsContainer(LayersAndNeuronsContainer&& src) noexcept :
    _offsets(std::move(src._offsets)),
    _total_size(src._total_size),
    _data(std::move(src._data)),
    _topology(std::move(src._topology)),
    _cached_layers(std::move(src._cached_layers)),
    _cache_dirty(std::move(src._cache_dirty))
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
      _cached_layers = src._cached_layers;
      _cache_dirty = src._cache_dirty;
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
      _cached_layers = std::move(src._cached_layers);
      _cache_dirty = std::move(src._cache_dirty);
      src._total_size = 0;
    }
    return *this;
  }

  LayersAndNeuronsContainer(const std::vector<unsigned>& topology) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    const size_t topology_size = topology.size();
    _topology.reserve(topology_size);
    _offsets.reserve(topology_size);

    _topology.insert(_topology.begin(), topology.begin(), topology.end());

    // finaly populate the data
    const auto& size = _topology.size();
    _offsets.resize(size);
    _total_size = 0;
    _cached_layers.resize(size);
    _cache_dirty.assign(size, true);
    for(size_t layer = 0; layer < size; ++layer)
    {
      _offsets[layer] = _total_size;
      _total_size+= _topology[layer];
      _cached_layers[layer].resize(_topology[layer]);
    }
    _data.resize(_total_size);
  }

  LayersAndNeuronsContainer& operator=(const std::vector<std::vector<double>>& data)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    assert(_topology.size() == data.size());
    for(size_t layer = 0; layer < data.size(); ++layer)
    {
      set(static_cast<unsigned>(layer), data[layer]);
    }
    return *this;
  }

  inline void zero()
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    std::fill(_data.begin(), _data.end(), 0.0);
    std::fill(_cache_dirty.begin(), _cache_dirty.end(), true);
  }

  inline void set( unsigned layer, unsigned neuron, double&& data)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    ensure_size(layer, neuron);
    _data[_offsets[layer]+neuron] = std::move(data);
  }

  inline void set( unsigned layer, unsigned neuron, const double& data)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    ensure_size(layer, neuron);
    _data[_offsets[layer]+neuron] = data;
  }
  
  void set(unsigned layer, const std::vector<double>& data)
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    assert(number_neurons(layer) == data.size());
    const auto& layer_offset = _offsets[layer];
    std::copy(data.begin(), data.end(), _data.begin() + layer_offset);
    _cache_dirty[layer] = true;
  }
  
  inline const double& get(unsigned layer, unsigned neuron) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    ensure_size(layer, neuron);
    return _data[_offsets[layer]+neuron];
  }

  const std::vector<double>& get_neurons(unsigned layer) const
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    if (_cache_dirty[layer])
    {
      const auto layer_offset = _offsets[layer];
      const auto num_neurons = _topology[layer];
      _cached_layers[layer].assign(_data.begin() + layer_offset, _data.begin() + layer_offset + num_neurons);
      _cache_dirty[layer] = false;
    }
    return _cached_layers[layer];
  }

  size_t number_layers() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    return _topology.size();
  }

  size_t number_neurons(size_t layer) const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    if(layer >= _topology.size())
    {
      return 0;
    }
    return _topology[layer];
  }
private:
  #ifdef NDEBUG
  void ensure_size(size_t, size_t) const
  {
  }
  #else
  void ensure_size(size_t layer, size_t neuron) const
  {
    MYODDWEB_PROFILE_FUNCTION("LayersAndNeuronsContainer");
    if (layer >= _topology.size() || neuron >= _topology[layer])
    {
      std::cerr << "The layer/neuron is out of bound!" << std::endl;
      throw std::invalid_argument("The layer/neuron is out of bound!");
    }
  }
  #endif

  std::vector<size_t> _offsets;
  size_t _total_size;
  std::vector<double, AlignedAllocator<double, 32>> _data;
  std::vector<unsigned short> _topology;

  mutable std::vector<std::vector<double>> _cached_layers;
  mutable std::vector<bool> _cache_dirty;
};
