#pragma once


#include "libraries/instrumentor.h"

#include <cstddef>
#include <memory>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

template <typename T, size_t Alignment>
class AlignedAllocator 
{
public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  template <typename U>
  struct rebind {
      using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() noexcept {}

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  pointer allocate(size_type n) 
  {
    MYODDWEB_PROFILE_FUNCTION("AlignedAllocator");
    pointer p;
#ifdef _WIN32
    p = static_cast<pointer>(_aligned_malloc(n * sizeof(T), Alignment));
    if (!p) 
    {
      throw std::bad_alloc();
    }
#else
    if (posix_memalign(reinterpret_cast<void**>(&p), Alignment, n * sizeof(T)) != 0) 
    {
      throw std::bad_alloc();
    }
#endif
    return p;
  }

  void deallocate(pointer p, size_type) 
  {
    MYODDWEB_PROFILE_FUNCTION("AlignedAllocator");
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
  }

  size_type max_size() const noexcept 
  {
    MYODDWEB_PROFILE_FUNCTION("AlignedAllocator");
    return (size_type)(-1) / sizeof(T);
  }

  void construct(pointer p, const_reference val) 
  {
    MYODDWEB_PROFILE_FUNCTION("AlignedAllocator");
    new (p) T(val);
  }

  void destroy(pointer p) 
  {
    MYODDWEB_PROFILE_FUNCTION("AlignedAllocator");
    p->~T();
  }
};

template <typename T1, size_t A1, typename T2, size_t A2>
bool operator==(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept 
{
  MYODDWEB_PROFILE_FUNCTION("AlignedAllocator");
  return A1 == A2;
}

template <typename T1, size_t A1, typename T2, size_t A2>
bool operator!=(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept 
{
  MYODDWEB_PROFILE_FUNCTION("AlignedAllocator");
  return A1 != A2;
}
