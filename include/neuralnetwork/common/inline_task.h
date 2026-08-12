#pragma once

#include "../libraries/instrumentor.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace myoddweb::nn
{
template <typename R = void>
class inline_task;

template <typename R>
class inline_task
{
public:
  static constexpr size_t BufferSize = 96;

  static size_t heap_fallback_count() noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    return _heap_fallback_counter.load(std::memory_order_relaxed);
  }

  static void reset_heap_fallback_count() noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    _heap_fallback_counter.store(0, std::memory_order_relaxed);
  }

  inline_task() noexcept
    : _invoke(nullptr), _move(nullptr), _destroy(nullptr)
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
  }

  template <typename F, typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, inline_task>>>
  inline_task(F&& f)
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    using DecayedF = std::decay_t<F>;
    constexpr bool fits_inline = (sizeof(DecayedF) <= BufferSize) && (alignof(DecayedF) <= alignof(std::max_align_t));

    if constexpr (fits_inline)
    {
      new (&_buffer) DecayedF(std::forward<F>(f));
      _invoke = [](void* buf) -> R
      {
        return (*reinterpret_cast<DecayedF*>(buf))();
      };
      _move = [](void* src, void* dest)
      {
        new (dest) DecayedF(std::move(*reinterpret_cast<DecayedF*>(src)));
        reinterpret_cast<DecayedF*>(src)->~DecayedF();
      };
      _destroy = [](void* buf)
      {
        reinterpret_cast<DecayedF*>(buf)->~DecayedF();
      };
    }
    else
    {
      _heap_fallback_counter.fetch_add(1, std::memory_order_relaxed);
      DecayedF* heap_ptr = new DecayedF(std::forward<F>(f));
      *reinterpret_cast<DecayedF**>(&_buffer) = heap_ptr;

      _invoke = [](void* buf) -> R
      {
        DecayedF* ptr = *reinterpret_cast<DecayedF**>(buf);
        return (*ptr)();
      };
      _move = [](void* src, void* dest)
      {
        *reinterpret_cast<DecayedF**>(dest) = *reinterpret_cast<DecayedF**>(src);
        *reinterpret_cast<DecayedF**>(src) = nullptr;
      };
      _destroy = [](void* buf)
      {
        DecayedF* ptr = *reinterpret_cast<DecayedF**>(buf);
        delete ptr;
      };
    }
  }

  ~inline_task()
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    reset();
  }

  inline_task(const inline_task&) = delete;
  inline_task& operator=(const inline_task&) = delete;

  inline_task(inline_task&& other) noexcept
    : _invoke(other._invoke), _move(other._move), _destroy(other._destroy)
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    if (other._move != nullptr)
    {
      other._move(&other._buffer, &_buffer);
      other._invoke = nullptr;
      other._move = nullptr;
      other._destroy = nullptr;
    }
  }

  inline_task& operator=(inline_task&& other) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    if (this != &other)
    {
      reset();
      _invoke = other._invoke;
      _move = other._move;
      _destroy = other._destroy;
      if (other._move != nullptr)
      {
        other._move(&other._buffer, &_buffer);
        other._invoke = nullptr;
        other._move = nullptr;
        other._destroy = nullptr;
      }
    }
    return *this;
  }

  explicit operator bool() const noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    return _invoke != nullptr;
  }

  R operator()()
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    return _invoke(&_buffer);
  }

private:
  void reset() noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("inline_task");
    if (_destroy != nullptr)
    {
      _destroy(&_buffer);
      _invoke = nullptr;
      _move = nullptr;
      _destroy = nullptr;
    }
  }

  using InvokeFn = R (*)(void*);
  using MoveFn = void (*)(void*, void*);
  using DestroyFn = void (*)(void*);

  InvokeFn _invoke{ nullptr };
  MoveFn _move{ nullptr };
  DestroyFn _destroy{ nullptr };

  alignas(std::max_align_t) std::byte _buffer[BufferSize];
  static inline std::atomic<size_t> _heap_fallback_counter{ 0 };
};
} // namespace myoddweb::nn
