#pragma once

#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

namespace myoddweb::nn
{
class DenormalDisabler
{
public:
  DenormalDisabler() noexcept
  {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    _old_mxcsr = _mm_getcsr();
    _needs_restore = ((_old_mxcsr & 0x8040) != 0x8040);
    if (_needs_restore)
    {
      _mm_setcsr(_old_mxcsr | 0x8000 | 0x0040);
    }
#endif
  }

  ~DenormalDisabler() noexcept
  {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    if (_needs_restore)
    {
      _mm_setcsr(_old_mxcsr);
    }
#endif
  }

  DenormalDisabler(const DenormalDisabler&) = delete;
  DenormalDisabler& operator=(const DenormalDisabler&) = delete;
  DenormalDisabler(DenormalDisabler&&) = delete;
  DenormalDisabler& operator=(DenormalDisabler&&) = delete;

private:
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
  unsigned int _old_mxcsr{ 0 };
  bool _needs_restore{ false };
#endif
};
} // namespace myoddweb::nn
