#pragma once
#include <vector>
#include <algorithm>
#include "../libraries/instrumentor.h"

namespace myoddweb::nn
{
template <typename T, int Tag = 0>
class TempBuffer
{
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
      std::vector<T>& cache = get_cache();
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
      std::vector<T>& cache = get_cache();
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
  static std::vector<T>& get_cache() noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("TempBuffer");
    static thread_local std::vector<T> cache;
    return cache;
  }

  size_t _size;
  std::vector<T> _temp;
  std::vector<T>* _ptr;
};
} // namespace myoddweb::nn
