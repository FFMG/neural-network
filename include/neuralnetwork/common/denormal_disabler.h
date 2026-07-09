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
    // Enable FTZ (Flush to Zero) and DAZ (Denormals are Zero)
    // Bit 15 is FTZ (Flush to Zero)
    // Bit 6 is DAZ (Denormals are Zero)
    _mm_setcsr(_old_mxcsr | 0x8000 | 0x0040);
#endif
  }

  ~DenormalDisabler() noexcept
  {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(_old_mxcsr);
#endif
  }

  DenormalDisabler(const DenormalDisabler&) = delete;
  DenormalDisabler& operator=(const DenormalDisabler&) = delete;
  DenormalDisabler(DenormalDisabler&&) = delete;
  DenormalDisabler& operator=(DenormalDisabler&&) = delete;

private:
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
  unsigned int _old_mxcsr;
#endif
};
} // namespace myoddweb::nn
