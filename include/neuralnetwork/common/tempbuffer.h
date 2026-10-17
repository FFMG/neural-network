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
    _active_buffer(nullptr),
    _in_use_flag(nullptr)
  {
    MYODDWEB_PROFILE_FUNCTION("TempBuffer");
    if (size == 0 && !zero_init)
    {
      _active_buffer = &_fallback_buffer;
      return;
    }

    acquire_buffer();
    if (_active_buffer->size() < size)
    {
      _active_buffer->resize(size);
    }
    else if (_active_buffer->capacity() > 262144 && size <= 4096)
    {
      _active_buffer->resize(size);
      _active_buffer->shrink_to_fit();
    }

    if (zero_init && size > 0)
    {
      std::fill_n(_active_buffer->data(), size, T{});
    }
  }

  ~TempBuffer() noexcept
  {
    if (_in_use_flag != nullptr)
    {
      *_in_use_flag = false;
    }
  }

  TempBuffer(const TempBuffer&) = delete;
  TempBuffer& operator=(const TempBuffer&) = delete;
  TempBuffer(TempBuffer&&) = delete;
  TempBuffer& operator=(TempBuffer&&) = delete;

  inline void assign(size_t size, const T& val)
  {
    MYODDWEB_PROFILE_FUNCTION("TempBuffer");
    _size = size;
    if (_in_use_flag == nullptr && size > 0)
    {
      acquire_buffer();
    }
    if (_active_buffer->size() < size)
    {
      _active_buffer->resize(size);
    }
    else if (_active_buffer->capacity() > 262144 && size <= 4096)
    {
      _active_buffer->resize(size);
      _active_buffer->shrink_to_fit();
    }
    if (size > 0)
    {
      std::fill_n(_active_buffer->data(), size, val);
    }
  }

  inline T* data() noexcept
  {
    return _active_buffer->data();
  }

  inline const T* data() const noexcept
  {
    return _active_buffer->data();
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
    if (_active_buffer->size() != _size)
    {
      _active_buffer->resize(_size);
    }
    return *_active_buffer;
  }

  inline const std::vector<T>& vec() const noexcept
  {
    if (_active_buffer->size() != _size)
    {
      _active_buffer->resize(_size);
    }
    return *_active_buffer;
  }

private:
  struct tls_slot
  {
    std::vector<T> buffer;
    bool in_use = false;
  };

  static inline tls_slot& get_tls_slot() noexcept
  {
    thread_local tls_slot slot;
    return slot;
  }

  inline void acquire_buffer() noexcept
  {
    auto& slot = get_tls_slot();
    if (!slot.in_use)
    {
      slot.in_use = true;
      _in_use_flag = &slot.in_use;
      _active_buffer = &slot.buffer;
    }
    else
    {
      _in_use_flag = nullptr;
      _active_buffer = &_fallback_buffer;
    }
  }

  size_t _size;
  std::vector<T>* _active_buffer;
  bool* _in_use_flag;
  std::vector<T> _fallback_buffer;
};
} // namespace myoddweb::nn
