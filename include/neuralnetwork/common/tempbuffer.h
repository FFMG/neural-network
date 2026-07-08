#pragma once
#include <vector>
#include <algorithm>
#include <type_traits>
#include "../libraries/instrumentor.h"

namespace myoddweb::nn
{
struct ThreadBufferCache
{
  std::vector<std::vector<double>> caches;
};

inline ThreadBufferCache& get_thread_buffer_cache() noexcept
{
  MYODDWEB_PROFILE_FUNCTION("ThreadBufferCache");
  static thread_local ThreadBufferCache instance;
  return instance;
}

template <typename T, int Tag = 0>
class TempBuffer
{
  static_assert(std::is_same_v<T, double>, "TempBuffer only supports double type for thread-local caching.");

public:
  TempBuffer(size_t size, bool zero_init = false) :
    _size(size),
    _temp(),
    _ptr(nullptr)
  {
    MYODDWEB_PROFILE_FUNCTION("TempBuffer");
    // Capped at 1,048,576 elements (~8MB for double) to prevent TLS bloat
    if (size <= 1048576)
    {
      auto& thread_cache = get_thread_buffer_cache();
      if (static_cast<size_t>(Tag) >= thread_cache.caches.size())
      {
        thread_cache.caches.resize(Tag + 1);
      }
      std::vector<T>& cache = thread_cache.caches[Tag];
      if (cache.size() < size)
      {
        cache.resize(size);
      }
      _ptr = &cache;
      if (zero_init)
      {
        std::fill(cache.begin(), cache.begin() + size, static_cast<T>(0));
      }
    }
    else
    {
      if (zero_init)
      {
        _temp.assign(size, static_cast<T>(0));
      }
      else
      {
        _temp.resize(size);
      }
      _ptr = &_temp;
    }
  }

  inline void assign(size_t size, const T& val)
  {
    MYODDWEB_PROFILE_FUNCTION("TempBuffer");
    _size = size;
    // Capped at 1,048,576 elements (~8MB for double) to prevent TLS bloat
    if (size <= 1048576)
    {
      auto& thread_cache = get_thread_buffer_cache();
      if (static_cast<size_t>(Tag) >= thread_cache.caches.size())
      {
        thread_cache.caches.resize(Tag + 1);
      }
      std::vector<T>& cache = thread_cache.caches[Tag];
      if (cache.size() < size)
      {
        cache.resize(size);
      }
      _ptr = &cache;
      std::fill(cache.begin(), cache.begin() + size, val);
    }
    else
    {
      _temp.assign(size, val);
      _ptr = &_temp;
    }
  }

  inline T* data() noexcept
  {
    return _ptr->data();
  }

  inline const T* data() const noexcept
  {
    return _ptr->data();
  }

  inline size_t size() const noexcept
  {
    return _size;
  }

  inline bool empty() const noexcept
  {
    return _size == 0;
  }

  inline std::vector<T>& vec() noexcept
  {
    return *_ptr;
  }

  inline const std::vector<T>& vec() const noexcept
  {
    return *_ptr;
  }

private:
  size_t _size;
  std::vector<T> _temp;
  std::vector<T>* _ptr;
};
} // namespace myoddweb::nn
