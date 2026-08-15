#pragma once
#include <vector>
#include <algorithm>
#include <type_traits>
#include "../libraries/instrumentor.h"

namespace myoddweb::nn
{
template <typename T, int Tag = 0>
class TempBuffer
{
  static_assert(std::is_same_v<T, double>, "TempBuffer only supports double type.");

public:
  TempBuffer(size_t size, bool zero_init = false) :
    _size(size),
    _buffer(zero_init ? size : 0)
  {
    MYODDWEB_PROFILE_FUNCTION("TempBuffer");
    if (!zero_init && size > 0)
    {
      _buffer.resize(size);
    }
  }

  inline void assign(size_t size, const T& val)
  {
    MYODDWEB_PROFILE_FUNCTION("TempBuffer");
    _size = size;
    _buffer.assign(size, val);
  }

  inline T* data() noexcept
  {
    return _buffer.data();
  }

  inline const T* data() const noexcept
  {
    return _buffer.data();
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
    return _buffer;
  }

  inline const std::vector<T>& vec() const noexcept
  {
    return _buffer;
  }

private:
  size_t _size;
  std::vector<T> _buffer;
};
} // namespace myoddweb::nn
