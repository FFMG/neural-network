#pragma once

#include "../libraries/instrumentor.h"
#include <algorithm>
#include <cmath>
#include <functional>

// Check if AVX2 is available on x86/x64 architectures
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__AVX2__)
#include <immintrin.h>
#define SIMD_AVX2_ENABLED
#define SIMD_FMA_ENABLED
#endif

namespace myoddweb::nn
{
class simd
{
public:
#ifdef SIMD_AVX2_ENABLED
  inline static double horizontal_sum(__m256d v) noexcept
  {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum128 = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(sum128, sum128));
  }
#endif

  // Scalar fallback for mul_add
  inline static void scalar_mul_add(const double x, const double* w, double* y, size_t n, size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      y[j] += x * w[j];
    }
  }

  // A simple vectorized GEMM block (y += x * w)
  // Computes: y[j] += x * w[j] for j = 0..N
  inline static void mul_add(const double x, const double* w, double* y, size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;

#ifdef SIMD_AVX2_ENABLED
    // Broadcast x into a 4-double vector
    __m256d vec_x = _mm256_set1_pd(x);

    // Process 4 doubles at a time
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_w = _mm256_loadu_pd(&w[j]);
      __m256d vec_y = _mm256_loadu_pd(&y[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y = _mm256_fmadd_pd(vec_x, vec_w, vec_y);
#else
      vec_y = _mm256_add_pd(vec_y, _mm256_mul_pd(vec_x, vec_w));
#endif
      _mm256_storeu_pd(&y[j], vec_y);
    }
#endif
    scalar_mul_add(x, w, y, n, j);
  }

  // Scalar fallback for mul_add_two
  inline static void scalar_mul_add_two(
    const double x,
    const double* w0,
    const double* w1,
    double* y0,
    double* y1,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      y0[j] += x * w0[j];
      y1[j] += x * w1[j];
    }
  }

  // A vectorized mul_add for two targets (y0 += x * w0, y1 += x * w1)
  inline static void mul_add_two(
    const double x,
    const double* w0,
    const double* w1,
    double* y0,
    double* y1,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;

#ifdef SIMD_AVX2_ENABLED
    // Broadcast x into a 4-double vector
    __m256d vec_x = _mm256_set1_pd(x);

    // Process 4 doubles at a time
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_w0 = _mm256_loadu_pd(&w0[j]);
      __m256d vec_y0 = _mm256_loadu_pd(&y0[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y0 = _mm256_fmadd_pd(vec_x, vec_w0, vec_y0);
#else
      vec_y0 = _mm256_add_pd(vec_y0, _mm256_mul_pd(vec_x, vec_w0));
#endif
      _mm256_storeu_pd(&y0[j], vec_y0);

      __m256d vec_w1 = _mm256_loadu_pd(&w1[j]);
      __m256d vec_y1 = _mm256_loadu_pd(&y1[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y1 = _mm256_fmadd_pd(vec_x, vec_w1, vec_y1);
#else
      vec_y1 = _mm256_add_pd(vec_y1, _mm256_mul_pd(vec_x, vec_w1));
#endif
      _mm256_storeu_pd(&y1[j], vec_y1);
    }
#endif
    scalar_mul_add_two(x, w0, w1, y0, y1, n, j);
  }

  // Scalar fallback for mul_add_three
  inline static void scalar_mul_add_three(
    const double x,
    const double* w0,
    const double* w1,
    const double* w2,
    double* y0,
    double* y1,
    double* y2,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      y0[j] += x * w0[j];
      y1[j] += x * w1[j];
      y2[j] += x * w2[j];
    }
  }

  // A vectorized mul_add for three targets (y0 += x * w0, y1 += x * w1, y2 += x * w2)
  inline static void mul_add_three(
    const double x,
    const double* w0,
    const double* w1,
    const double* w2,
    double* y0,
    double* y1,
    double* y2,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;

#ifdef SIMD_AVX2_ENABLED
    // Broadcast x into a 4-double vector
    __m256d vec_x = _mm256_set1_pd(x);

    // Process 4 doubles at a time
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_w0 = _mm256_loadu_pd(&w0[j]);
      __m256d vec_y0 = _mm256_loadu_pd(&y0[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y0 = _mm256_fmadd_pd(vec_x, vec_w0, vec_y0);
#else
      vec_y0 = _mm256_add_pd(vec_y0, _mm256_mul_pd(vec_x, vec_w0));
#endif
      _mm256_storeu_pd(&y0[j], vec_y0);

      __m256d vec_w1 = _mm256_loadu_pd(&w1[j]);
      __m256d vec_y1 = _mm256_loadu_pd(&y1[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y1 = _mm256_fmadd_pd(vec_x, vec_w1, vec_y1);
#else
      vec_y1 = _mm256_add_pd(vec_y1, _mm256_mul_pd(vec_x, vec_w1));
#endif
      _mm256_storeu_pd(&y1[j], vec_y1);

      __m256d vec_w2 = _mm256_loadu_pd(&w2[j]);
      __m256d vec_y2 = _mm256_loadu_pd(&y2[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y2 = _mm256_fmadd_pd(vec_x, vec_w2, vec_y2);
#else
      vec_y2 = _mm256_add_pd(vec_y2, _mm256_mul_pd(vec_x, vec_w2));
#endif
      _mm256_storeu_pd(&y2[j], vec_y2);
    }
#endif
    scalar_mul_add_three(x, w0, w1, w2, y0, y1, y2, n, j);
  }

  // Scalar fallback for mul_add_four
  inline static void scalar_mul_add_four(
    const double x,
    const double* w0,
    const double* w1,
    const double* w2,
    const double* w3,
    double* y0,
    double* y1,
    double* y2,
    double* y3,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      y0[j] += x * w0[j];
      y1[j] += x * w1[j];
      y2[j] += x * w2[j];
      y3[j] += x * w3[j];
    }
  }

  // A vectorized mul_add for four targets (y0 += x * w0, y1 += x * w1, y2 += x * w2, y3 += x * w3)
  inline static void mul_add_four(
    const double x,
    const double* w0,
    const double* w1,
    const double* w2,
    const double* w3,
    double* y0,
    double* y1,
    double* y2,
    double* y3,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;

#ifdef SIMD_AVX2_ENABLED
    // Broadcast x into a 4-double vector
    __m256d vec_x = _mm256_set1_pd(x);

    // Process 4 doubles at a time
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_w0 = _mm256_loadu_pd(&w0[j]);
      __m256d vec_y0 = _mm256_loadu_pd(&y0[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y0 = _mm256_fmadd_pd(vec_x, vec_w0, vec_y0);
#else
      vec_y0 = _mm256_add_pd(vec_y0, _mm256_mul_pd(vec_x, vec_w0));
#endif
      _mm256_storeu_pd(&y0[j], vec_y0);

      __m256d vec_w1 = _mm256_loadu_pd(&w1[j]);
      __m256d vec_y1 = _mm256_loadu_pd(&y1[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y1 = _mm256_fmadd_pd(vec_x, vec_w1, vec_y1);
#else
      vec_y1 = _mm256_add_pd(vec_y1, _mm256_mul_pd(vec_x, vec_w1));
#endif
      _mm256_storeu_pd(&y1[j], vec_y1);

      __m256d vec_w2 = _mm256_loadu_pd(&w2[j]);
      __m256d vec_y2 = _mm256_loadu_pd(&y2[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y2 = _mm256_fmadd_pd(vec_x, vec_w2, vec_y2);
#else
      vec_y2 = _mm256_add_pd(vec_y2, _mm256_mul_pd(vec_x, vec_w2));
#endif
      _mm256_storeu_pd(&y2[j], vec_y2);

      __m256d vec_w3 = _mm256_loadu_pd(&w3[j]);
      __m256d vec_y3 = _mm256_loadu_pd(&y3[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y3 = _mm256_fmadd_pd(vec_x, vec_w3, vec_y3);
#else
      vec_y3 = _mm256_add_pd(vec_y3, _mm256_mul_pd(vec_x, vec_w3));
#endif
      _mm256_storeu_pd(&y3[j], vec_y3);
    }
#endif
    scalar_mul_add_four(x, w0, w1, w2, w3, y0, y1, y2, y3, n, j);
  }

  // Scalar fallback for mul_add_three_scalars
  inline static void scalar_mul_add_three_scalars(
    const double x0, const double x1, const double x2,
    const double* w0, const double* w1, const double* w2,
    double* y0, double* y1, double* y2,
    size_t n, size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      y0[j] += x0 * w0[j];
      y1[j] += x1 * w1[j];
      y2[j] += x2 * w2[j];
    }
  }

  // Scalar fallback for mul_add_four_scalars
  inline static void scalar_mul_add_four_scalars(
    const double x0, const double x1, const double x2, const double x3,
    const double* w,
    double* y0, double* y1, double* y2, double* y3,
    size_t n, size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      double w_val = w[j];
      y0[j] += x0 * w_val;
      y1[j] += x1 * w_val;
      y2[j] += x2 * w_val;
      y3[j] += x3 * w_val;
    }
  }

  // Scalar fallback for mul_add_two_scalars
  inline static void scalar_mul_add_two_scalars(
    const double x0, const double x1,
    const double* w,
    double* y0, double* y1,
    size_t n, size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      double w_val = w[j];
      y0[j] += x0 * w_val;
      y1[j] += x1 * w_val;
    }
  }

  // A vectorized mul_add for three targets with three scalars
  inline static void mul_add_three_scalars(
    const double x0, const double x1, const double x2,
    const double* w0, const double* w1, const double* w2,
    double* y0, double* y1, double* y2,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    __m256d vec_x0 = _mm256_set1_pd(x0);
    __m256d vec_x1 = _mm256_set1_pd(x1);
    __m256d vec_x2 = _mm256_set1_pd(x2);

    for (; j + 3 < n; j += 4)
    {
      __m256d vec_w0 = _mm256_loadu_pd(&w0[j]);
      __m256d vec_y0 = _mm256_loadu_pd(&y0[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y0 = _mm256_fmadd_pd(vec_x0, vec_w0, vec_y0);
#else
      vec_y0 = _mm256_add_pd(vec_y0, _mm256_mul_pd(vec_x0, vec_w0));
#endif
      _mm256_storeu_pd(&y0[j], vec_y0);

      __m256d vec_w1 = _mm256_loadu_pd(&w1[j]);
      __m256d vec_y1 = _mm256_loadu_pd(&y1[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y1 = _mm256_fmadd_pd(vec_x1, vec_w1, vec_y1);
#else
      vec_y1 = _mm256_add_pd(vec_y1, _mm256_mul_pd(vec_x1, vec_w1));
#endif
      _mm256_storeu_pd(&y1[j], vec_y1);

      __m256d vec_w2 = _mm256_loadu_pd(&w2[j]);
      __m256d vec_y2 = _mm256_loadu_pd(&y2[j]);
#ifdef SIMD_FMA_ENABLED
      vec_y2 = _mm256_fmadd_pd(vec_x2, vec_w2, vec_y2);
#else
      vec_y2 = _mm256_add_pd(vec_y2, _mm256_mul_pd(vec_x2, vec_w2));
#endif
      _mm256_storeu_pd(&y2[j], vec_y2);
    }
#endif
    scalar_mul_add_three_scalars(x0, x1, x2, w0, w1, w2, y0, y1, y2, n, j);
  }

  // A vectorized mul_add for four targets with four scalars
  inline static void mul_add_four_scalars(
    const double x0, const double x1, const double x2, const double x3,
    const double* w,
    double* y0, double* y1, double* y2, double* y3,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    __m256d vec_x0 = _mm256_set1_pd(x0);
    __m256d vec_x1 = _mm256_set1_pd(x1);
    __m256d vec_x2 = _mm256_set1_pd(x2);
    __m256d vec_x3 = _mm256_set1_pd(x3);

    for (; j + 3 < n; j += 4)
    {
      __m256d vec_w = _mm256_loadu_pd(&w[j]);

      __m256d vec_y0 = _mm256_loadu_pd(&y0[j]);
      __m256d vec_y1 = _mm256_loadu_pd(&y1[j]);
      __m256d vec_y2 = _mm256_loadu_pd(&y2[j]);
      __m256d vec_y3 = _mm256_loadu_pd(&y3[j]);

#ifdef SIMD_FMA_ENABLED
      vec_y0 = _mm256_fmadd_pd(vec_w, vec_x0, vec_y0);
      vec_y1 = _mm256_fmadd_pd(vec_w, vec_x1, vec_y1);
      vec_y2 = _mm256_fmadd_pd(vec_w, vec_x2, vec_y2);
      vec_y3 = _mm256_fmadd_pd(vec_w, vec_x3, vec_y3);
#else
      vec_y0 = _mm256_add_pd(vec_y0, _mm256_mul_pd(vec_w, vec_x0));
      vec_y1 = _mm256_add_pd(vec_y1, _mm256_mul_pd(vec_w, vec_x1));
      vec_y2 = _mm256_add_pd(vec_y2, _mm256_mul_pd(vec_w, vec_x2));
      vec_y3 = _mm256_add_pd(vec_y3, _mm256_mul_pd(vec_w, vec_x3));
#endif

      _mm256_storeu_pd(&y0[j], vec_y0);
      _mm256_storeu_pd(&y1[j], vec_y1);
      _mm256_storeu_pd(&y2[j], vec_y2);
      _mm256_storeu_pd(&y3[j], vec_y3);
    }
#endif
    scalar_mul_add_four_scalars(x0, x1, x2, x3, w, y0, y1, y2, y3, n, j);
  }

  // A vectorized mul_add for two targets with two scalars
  inline static void mul_add_two_scalars(
    const double x0, const double x1,
    const double* w,
    double* y0, double* y1,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    __m256d vec_x0 = _mm256_set1_pd(x0);
    __m256d vec_x1 = _mm256_set1_pd(x1);

    for (; j + 3 < n; j += 4)
    {
      __m256d vec_w = _mm256_loadu_pd(&w[j]);

      __m256d vec_y0 = _mm256_loadu_pd(&y0[j]);
      __m256d vec_y1 = _mm256_loadu_pd(&y1[j]);

#ifdef SIMD_FMA_ENABLED
      vec_y0 = _mm256_fmadd_pd(vec_w, vec_x0, vec_y0);
      vec_y1 = _mm256_fmadd_pd(vec_w, vec_x1, vec_y1);
#else
      vec_y0 = _mm256_add_pd(vec_y0, _mm256_mul_pd(vec_w, vec_x0));
      vec_y1 = _mm256_add_pd(vec_y1, _mm256_mul_pd(vec_w, vec_x1));
#endif

      _mm256_storeu_pd(&y0[j], vec_y0);
      _mm256_storeu_pd(&y1[j], vec_y1);
    }
#endif
    scalar_mul_add_two_scalars(x0, x1, w, y0, y1, n, j);
  }


  // A vectorized GEMM for four batches (y0 += x0 * W, y1 += x1 * W, y2 += x2 * W, y3 += x3 * W)
  // This interchanged loop minimises memory loads/stores of y0..y3.
  inline static void gemm_four_batches(
    const double* x0, const double* x1, const double* x2, const double* x3,
    const double* W,
    double* y0, double* y1, double* y2, double* y3,
    size_t N_prev, size_t N_this) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    for (; j + 3 < N_this; j += 4)
    {
      __m256d vec_y0 = _mm256_loadu_pd(&y0[j]);
      __m256d vec_y1 = _mm256_loadu_pd(&y1[j]);
      __m256d vec_y2 = _mm256_loadu_pd(&y2[j]);
      __m256d vec_y3 = _mm256_loadu_pd(&y3[j]);

      for (size_t i = 0; i < N_prev; ++i)
      {
        __m256d vec_w = _mm256_loadu_pd(&W[i * N_this + j]);
        __m256d vec_x0 = _mm256_set1_pd(x0[i]);
        __m256d vec_x1 = _mm256_set1_pd(x1[i]);
        __m256d vec_x2 = _mm256_set1_pd(x2[i]);
        __m256d vec_x3 = _mm256_set1_pd(x3[i]);

#ifdef SIMD_FMA_ENABLED
        vec_y0 = _mm256_fmadd_pd(vec_w, vec_x0, vec_y0);
        vec_y1 = _mm256_fmadd_pd(vec_w, vec_x1, vec_y1);
        vec_y2 = _mm256_fmadd_pd(vec_w, vec_x2, vec_y2);
        vec_y3 = _mm256_fmadd_pd(vec_w, vec_x3, vec_y3);
#else
        vec_y0 = _mm256_add_pd(vec_y0, _mm256_mul_pd(vec_w, vec_x0));
        vec_y1 = _mm256_add_pd(vec_y1, _mm256_mul_pd(vec_w, vec_x1));
        vec_y2 = _mm256_add_pd(vec_y2, _mm256_mul_pd(vec_w, vec_x2));
        vec_y3 = _mm256_add_pd(vec_y3, _mm256_mul_pd(vec_w, vec_x3));
#endif
      }

      _mm256_storeu_pd(&y0[j], vec_y0);
      _mm256_storeu_pd(&y1[j], vec_y1);
      _mm256_storeu_pd(&y2[j], vec_y2);
      _mm256_storeu_pd(&y3[j], vec_y3);
    }
#endif
    // Scalar cleanup
    if (j < N_this)
    {
      for (size_t i = 0; i < N_prev; ++i)
      {
        double val0 = x0[i];
        double val1 = x1[i];
        double val2 = x2[i];
        double val3 = x3[i];
        const double* w_row = W + i * N_this;
        for (size_t col = j; col < N_this; ++col)
        {
          y0[col] += val0 * w_row[col];
          y1[col] += val1 * w_row[col];
          y2[col] += val2 * w_row[col];
          y3[col] += val3 * w_row[col];
        }
      }
    }
  }

  // A vectorized GEMM for two batches (y0 += x0 * W, y1 += x1 * W)
  inline static void gemm_two_batches(
    const double* x0, const double* x1,
    const double* W,
    double* y0, double* y1,
    size_t N_prev, size_t N_this) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    for (; j + 3 < N_this; j += 4)
    {
      __m256d vec_y0 = _mm256_loadu_pd(&y0[j]);
      __m256d vec_y1 = _mm256_loadu_pd(&y1[j]);

      for (size_t i = 0; i < N_prev; ++i)
      {
        __m256d vec_w = _mm256_loadu_pd(&W[i * N_this + j]);
        __m256d vec_x0 = _mm256_set1_pd(x0[i]);
        __m256d vec_x1 = _mm256_set1_pd(x1[i]);

#ifdef SIMD_FMA_ENABLED
        vec_y0 = _mm256_fmadd_pd(vec_w, vec_x0, vec_y0);
        vec_y1 = _mm256_fmadd_pd(vec_w, vec_x1, vec_y1);
#else
        vec_y0 = _mm256_add_pd(vec_y0, _mm256_mul_pd(vec_w, vec_x0));
        vec_y1 = _mm256_add_pd(vec_y1, _mm256_mul_pd(vec_w, vec_x1));
#endif
      }

      _mm256_storeu_pd(&y0[j], vec_y0);
      _mm256_storeu_pd(&y1[j], vec_y1);
    }
#endif
    // Scalar cleanup
    if (j < N_this)
    {
      for (size_t i = 0; i < N_prev; ++i)
      {
        double val0 = x0[i];
        double val1 = x1[i];
        const double* w_row = W + i * N_this;
        for (size_t col = j; col < N_this; ++col)
        {
          y0[col] += val0 * w_row[col];
          y1[col] += val1 * w_row[col];
        }
      }
    }
  }

  // A vectorized GEMM for one batch (y += x * W)
  inline static void gemm_one_batch(
    const double* x,
    const double* W,
    double* y,
    size_t N_prev, size_t N_this) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    for (; j + 3 < N_this; j += 4)
    {
      __m256d vec_y = _mm256_loadu_pd(&y[j]);

      for (size_t i = 0; i < N_prev; ++i)
      {
        __m256d vec_w = _mm256_loadu_pd(&W[i * N_this + j]);
        __m256d vec_x = _mm256_set1_pd(x[i]);

#ifdef SIMD_FMA_ENABLED
        vec_y = _mm256_fmadd_pd(vec_w, vec_x, vec_y);
#else
        vec_y = _mm256_add_pd(vec_y, _mm256_mul_pd(vec_w, vec_x));
#endif
      }

      _mm256_storeu_pd(&y[j], vec_y);
    }
#endif
    // Scalar cleanup
    if (j < N_this)
    {
      for (size_t i = 0; i < N_prev; ++i)
      {
        double val = x[i];
        const double* w_row = W + i * N_this;
        for (size_t col = j; col < N_this; ++col)
        {
          y[col] += val * w_row[col];
        }
      }
    }
  }


  // Scalar fallback for dot_product
  [[nodiscard]] inline static double scalar_dot_product(const double* a, const double* b, size_t n, size_t start = 0) noexcept
  {
    double total_sum = 0.0;
    for (size_t j = start; j < n; ++j)
    {
      total_sum += a[j] * b[j];
    }
    return total_sum;
  }

  // Vectorized dot product (returns sum(a[j] * b[j]))
  [[nodiscard]] inline static double dot_product(const double* a, const double* b, size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
    double total_sum = 0.0;

#ifdef SIMD_AVX2_ENABLED
    __m256d vec_sum = _mm256_setzero_pd();
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_a = _mm256_loadu_pd(&a[j]);
      __m256d vec_b = _mm256_loadu_pd(&b[j]);
#ifdef SIMD_FMA_ENABLED
      vec_sum = _mm256_fmadd_pd(vec_a, vec_b, vec_sum);
#else
      vec_sum = _mm256_add_pd(vec_sum, _mm256_mul_pd(vec_a, vec_b));
#endif
    }
    total_sum = horizontal_sum(vec_sum);
#endif
    total_sum += scalar_dot_product(a, b, n, j);
    return total_sum;
  }

  // Scalar fallback for adam_step
  inline static void scalar_adam_step(
    double* values,
    const double* grads,
    double* m1,
    double* m2,
    double b1,
    double b2,
    double p1,
    double p2,
    double lr,
    double epsilon,
    size_t n,
    const double* decays = nullptr,
    size_t start = 0) noexcept
  {
    const double inv_p1 = (p1 > 1e-15) ? 1.0 / p1 : 1.0;
    const double inv_p2 = (p2 > 1e-15) ? 1.0 / p2 : 1.0;
    for (size_t j = start; j < n; ++j)
    {
      m1[j] = b1 * m1[j] + (1.0 - b1) * grads[j];
      m2[j] = b2 * m2[j] + (1.0 - b2) * (grads[j] * grads[j]);
      double m_hat = m1[j] * inv_p1;
      double v_hat = m2[j] * inv_p2;
      double update = m_hat / (std::sqrt(v_hat) + epsilon);
      double w = values[j];
      if (decays != nullptr)
      {
        w *= (1.0 - lr * decays[j]);
      }
      values[j] = std::clamp(w - lr * update, -100000.0, 100000.0);
    }
  }

  // Full Adam Update Step
  inline static void adam_step(
    double* values,
    const double* grads,
    double* m1,
    double* m2,
    double b1,
    double b2,
    double p1,
    double p2,
    double lr,
    double epsilon,
    size_t n,
    const double* decays = nullptr) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    const double inv_p1 = (p1 > 1e-15) ? 1.0 / p1 : 1.0;
    const double inv_p2 = (p2 > 1e-15) ? 1.0 / p2 : 1.0;

    __m256d vec_b1 = _mm256_set1_pd(b1);
    __m256d vec_one_minus_b1 = _mm256_set1_pd(1.0 - b1);
    __m256d vec_b2 = _mm256_set1_pd(b2);
    __m256d vec_one_minus_b2 = _mm256_set1_pd(1.0 - b2);
    __m256d vec_inv_p1 = _mm256_set1_pd(inv_p1);
    __m256d vec_inv_p2 = _mm256_set1_pd(inv_p2);
    __m256d vec_lr = _mm256_set1_pd(lr);
    __m256d vec_eps = _mm256_set1_pd(epsilon);
    __m256d vec_one = _mm256_set1_pd(1.0);
    __m256d vec_clamp_max = _mm256_set1_pd(100000.0);
    __m256d vec_clamp_min = _mm256_set1_pd(-100000.0);

    for (; j + 3 < n; j += 4) 
    {
      __m256d g = _mm256_loadu_pd(&grads[j]);
      __m256d cur_m1 = _mm256_loadu_pd(&m1[j]);
      __m256d cur_m2 = _mm256_loadu_pd(&m2[j]);
      __m256d cur_w = _mm256_loadu_pd(&values[j]);

      // Moments update
#ifdef SIMD_FMA_ENABLED
      __m256d next_m1 = _mm256_fmadd_pd(vec_one_minus_b1, g, _mm256_mul_pd(vec_b1, cur_m1));
      __m256d g_sq = _mm256_mul_pd(g, g);
      __m256d next_m2 = _mm256_fmadd_pd(vec_one_minus_b2, g_sq, _mm256_mul_pd(vec_b2, cur_m2));
#else
      __m256d next_m1 = _mm256_add_pd(_mm256_mul_pd(vec_b1, cur_m1), _mm256_mul_pd(vec_one_minus_b1, g));
      __m256d next_m2 = _mm256_add_pd(_mm256_mul_pd(vec_b2, cur_m2), _mm256_mul_pd(vec_one_minus_b2, _mm256_mul_pd(g, g)));
#endif
      _mm256_storeu_pd(&m1[j], next_m1);
      _mm256_storeu_pd(&m2[j], next_m2);

      // Adam scaling
      __m256d m_hat = _mm256_mul_pd(next_m1, vec_inv_p1);
      __m256d v_hat = _mm256_mul_pd(next_m2, vec_inv_p2);

      __m256d update = _mm256_div_pd(m_hat, _mm256_add_pd(_mm256_sqrt_pd(v_hat), vec_eps));

      // Optional AdamW weight decay
      if (decays != nullptr)
      {
        __m256d d = _mm256_loadu_pd(&decays[j]);
#ifdef SIMD_FMA_ENABLED
        cur_w = _mm256_mul_pd(cur_w, _mm256_fnmadd_pd(vec_lr, d, vec_one));
#else
        cur_w = _mm256_mul_pd(cur_w, _mm256_sub_pd(vec_one, _mm256_mul_pd(vec_lr, d)));
#endif
      }

#ifdef SIMD_FMA_ENABLED
      __m256d next_w_raw = _mm256_fnmadd_pd(vec_lr, update, cur_w);
#else
      __m256d next_w_raw = _mm256_sub_pd(cur_w, _mm256_mul_pd(vec_lr, update));
#endif

      // Hard clamp weights to prevent catastrophic numerical explosion (+/- 1 million)
      __m256d next_w = _mm256_max_pd(_mm256_min_pd(next_w_raw, vec_clamp_max), vec_clamp_min);
      _mm256_storeu_pd(&values[j], next_w);
    }
#endif
    scalar_adam_step(values, grads, m1, m2, b1, b2, p1, p2, lr, epsilon, n, decays, j);
  }
  // Scalar fallback for nadam_step
  inline static void scalar_nadam_step(
    double* values,
    const double* grads,
    double* m1,
    double* m2,
    double b1,
    double b2,
    double p1,
    double p2,
    double lr,
    double epsilon,
    size_t n,
    const double* decays = nullptr,
    size_t start = 0) noexcept
  {
    const double inv_p1 = (p1 > 1e-15) ? 1.0 / p1 : 1.0;
    const double inv_p2 = (p2 > 1e-15) ? 1.0 / p2 : 1.0;
    for (size_t j = start; j < n; ++j)
    {
      m1[j] = b1 * m1[j] + (1.0 - b1) * grads[j];
      m2[j] = b2 * m2[j] + (1.0 - b2) * (grads[j] * grads[j]);
      double m_hat = m1[j] * inv_p1;
      double v_hat = m2[j] * inv_p2;
      double m_nadam = b1 * m_hat + ((1.0 - b1) * grads[j]) * inv_p1;
      double update = m_nadam / (std::sqrt(v_hat) + epsilon);
      double w = values[j];
      if (decays != nullptr)
      {
        w *= (1.0 - lr * decays[j]);
      }
      values[j] = std::clamp(w - lr * update, -100000.0, 100000.0);
    }
  }

  // Full Nadam Update Step
  inline static void nadam_step(
    double* values,
    const double* grads,
    double* m1,
    double* m2,
    double b1,
    double b2,
    double p1,
    double p2,
    double lr,
    double epsilon,
    size_t n,
    const double* decays = nullptr) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    const double inv_p1 = (p1 > 1e-15) ? 1.0 / p1 : 1.0;
    const double inv_p2 = (p2 > 1e-15) ? 1.0 / p2 : 1.0;

    __m256d vec_b1 = _mm256_set1_pd(b1);
    __m256d vec_one_minus_b1 = _mm256_set1_pd(1.0 - b1);
    __m256d vec_b2 = _mm256_set1_pd(b2);
    __m256d vec_one_minus_b2 = _mm256_set1_pd(1.0 - b2);
    __m256d vec_inv_p1 = _mm256_set1_pd(inv_p1);
    __m256d vec_inv_p2 = _mm256_set1_pd(inv_p2);
    __m256d vec_lr = _mm256_set1_pd(lr);
    __m256d vec_eps = _mm256_set1_pd(epsilon);
    __m256d vec_one = _mm256_set1_pd(1.0);
    __m256d vec_clamp_max = _mm256_set1_pd(100000.0);
    __m256d vec_clamp_min = _mm256_set1_pd(-100000.0);

#ifdef SIMD_FMA_ENABLED
    // Precomputed constant term for Nadam update
    __m256d vec_one_minus_b1_inv_p1 = _mm256_set1_pd((1.0 - b1) * inv_p1);
#endif

    for (; j + 3 < n; j += 4)
    {
      __m256d g = _mm256_loadu_pd(&grads[j]);
      __m256d cur_m1 = _mm256_loadu_pd(&m1[j]);
      __m256d cur_m2 = _mm256_loadu_pd(&m2[j]);
      __m256d cur_w = _mm256_loadu_pd(&values[j]);

      // Moments update
#ifdef SIMD_FMA_ENABLED
      __m256d next_m1 = _mm256_fmadd_pd(vec_one_minus_b1, g, _mm256_mul_pd(vec_b1, cur_m1));
      __m256d g_sq = _mm256_mul_pd(g, g);
      __m256d next_m2 = _mm256_fmadd_pd(vec_one_minus_b2, g_sq, _mm256_mul_pd(vec_b2, cur_m2));
#else
      __m256d next_m1 = _mm256_add_pd(_mm256_mul_pd(vec_b1, cur_m1), _mm256_mul_pd(vec_one_minus_b1, g));
      __m256d next_m2 = _mm256_add_pd(_mm256_mul_pd(vec_b2, cur_m2), _mm256_mul_pd(vec_one_minus_b2, _mm256_mul_pd(g, g)));
#endif
      _mm256_storeu_pd(&m1[j], next_m1);
      _mm256_storeu_pd(&m2[j], next_m2);

      // Nadam scaling
      __m256d m_hat = _mm256_mul_pd(next_m1, vec_inv_p1);
      __m256d v_hat = _mm256_mul_pd(next_m2, vec_inv_p2);

      // m_nadam = beta1 * m_hat + ((1-beta1)*g)/p1
#ifdef SIMD_FMA_ENABLED
      __m256d term2 = _mm256_mul_pd(vec_one_minus_b1_inv_p1, g);
      __m256d m_nadam = _mm256_fmadd_pd(vec_b1, m_hat, term2);
#else
      __m256d m_nadam = _mm256_add_pd(_mm256_mul_pd(vec_b1, m_hat), _mm256_mul_pd(_mm256_mul_pd(vec_one_minus_b1, g), vec_inv_p1));
#endif
      __m256d update = _mm256_div_pd(m_nadam, _mm256_add_pd(_mm256_sqrt_pd(v_hat), vec_eps));

      // Optional NadamW weight decay
      if (decays != nullptr) 
      {
        __m256d d = _mm256_loadu_pd(&decays[j]);
#ifdef SIMD_FMA_ENABLED
        cur_w = _mm256_mul_pd(cur_w, _mm256_fnmadd_pd(vec_lr, d, vec_one));
#else
        cur_w = _mm256_mul_pd(cur_w, _mm256_sub_pd(vec_one, _mm256_mul_pd(vec_lr, d)));
#endif
      }

#ifdef SIMD_FMA_ENABLED
      __m256d next_w_raw = _mm256_fnmadd_pd(vec_lr, update, cur_w);
#else
      __m256d next_w_raw = _mm256_sub_pd(cur_w, _mm256_mul_pd(vec_lr, update));
#endif

      // Hard clamp weights to prevent catastrophic numerical explosion
      __m256d next_w = _mm256_max_pd(_mm256_min_pd(next_w_raw, vec_clamp_max), vec_clamp_min);
      _mm256_storeu_pd(&values[j], next_w);
    }
#endif
    scalar_nadam_step(values, grads, m1, m2, b1, b2, p1, p2, lr, epsilon, n, decays, j);
  }

  // Scalar fallback for gru_bptt_gate_step
  inline static void scalar_gru_bptt_gate_step(
    size_t n,
    const double* grad_next,
    const double* d_next_h,
    const double* z_vals,
    const double* h_hat_vals,
    const double* h_prev_vals,
    const double* h_hat_pre_vals,
    const double* mask_vals,
    double* dz_out,
    double* dh_hat_out,
    double* dh_prev_accum_out,
    const double* h_hat_pre_deriv_vals,
    size_t start = 0) noexcept
  {
    (void)h_hat_pre_vals;
    for (size_t j = start; j < n; ++j)
    {
      double dh = std::clamp(grad_next[j] + d_next_h[j], -50.0, 50.0);
      double z = z_vals[j];
      double h_hat = h_hat_vals[j];
      double mask = mask_vals[j];
      double h_prev = (h_prev_vals) ? h_prev_vals[j] : 0.0;
      double h_hat_final = h_hat * mask;

      double d_z_pre = dh * (h_hat_final - h_prev) * z * (1.0 - z);
      double d_h_hat_pre = dh * z * h_hat_pre_deriv_vals[j] * mask;

      dz_out[j] = d_z_pre;
      dh_hat_out[j] = d_h_hat_pre;
      dh_prev_accum_out[j] = dh * (1.0 - z);
    }
  }

  // GRU BPTT Gate Step
  inline static void gru_bptt_gate_step(
    size_t n,
    const double* grad_next,
    const double* d_next_h,
    const double* z_vals,
    const double* h_hat_vals,      // Activated but UNMASKED
    const double* h_prev_vals,
    const double* h_hat_pre_vals,
    const double* mask_vals,       // Dropout mask
    double* dz_out,
    double* dh_hat_out,
    double* dh_prev_accum_out,
    const double* h_hat_pre_deriv_vals) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d clip_limit = _mm256_set1_pd(50.0);
    const __m256d neg_clip_limit = _mm256_set1_pd(-50.0);

    for (; j + 3 < n; j += 4)
    {
      __m256d dh_raw = _mm256_add_pd(_mm256_loadu_pd(&grad_next[j]), _mm256_loadu_pd(&d_next_h[j]));
      __m256d dh = _mm256_max_pd(_mm256_min_pd(dh_raw, clip_limit), neg_clip_limit);

      __m256d z = _mm256_loadu_pd(&z_vals[j]);
      __m256d h_hat = _mm256_loadu_pd(&h_hat_vals[j]); // Unmasked
      __m256d mask = _mm256_loadu_pd(&mask_vals[j]);
      __m256d h_prev = h_prev_vals ? _mm256_loadu_pd(&h_prev_vals[j]) : _mm256_setzero_pd();
      __m256d deriv = _mm256_loadu_pd(&h_hat_pre_deriv_vals[j]);

      // h_hat_final = h_hat * mask
      __m256d h_hat_final = _mm256_mul_pd(h_hat, mask);

      // dz_pre = dh * (h_hat_final - h_prev) * z * (1 - z)
      __m256d d_z_pre = _mm256_mul_pd(_mm256_mul_pd(dh, _mm256_sub_pd(h_hat_final, h_prev)), _mm256_mul_pd(z, _mm256_sub_pd(one, z)));

      // dh_hat_pre = dh * z * activation_derivative(h_hat_pre) * mask
      __m256d d_h_hat_pre = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(dh, z), deriv), mask);
      
#ifdef SIMD_FMA_ENABLED
      __m256d d_h_prev_direct = _mm256_fnmadd_pd(dh, z, dh);
#else
      __m256d d_h_prev_direct = _mm256_mul_pd(dh, _mm256_sub_pd(one, z));
#endif

      _mm256_storeu_pd(&dz_out[j], d_z_pre);
      _mm256_storeu_pd(&dh_hat_out[j], d_h_hat_pre);
      _mm256_storeu_pd(&dh_prev_accum_out[j], d_h_prev_direct);
    }
#endif
    scalar_gru_bptt_gate_step(n, grad_next, d_next_h, z_vals, h_hat_vals, h_prev_vals, h_hat_pre_vals, mask_vals, dz_out, dh_hat_out, dh_prev_accum_out, h_hat_pre_deriv_vals, j);
  }

  // Scalar fallback for gru_bptt_reset_step
  inline static void scalar_gru_bptt_reset_step(
    size_t n,
    const double* temp_Uh,
    const double* h_prev_vals,
    const double* r_vals,
    const double* dh_prev_accum,
    double* dr_out,
    double* dh_next_out,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      double grad_rh = temp_Uh[j];
      double h_prev = (h_prev_vals != nullptr) ? h_prev_vals[j] : 0.0;
      double r = r_vals[j];
      dr_out[j] = grad_rh * h_prev * r * (1.0 - r);
      dh_next_out[j] = dh_prev_accum[j] + grad_rh * r;
    }
  }

  // GRU BPTT Reset Gate Step
  inline static void gru_bptt_reset_step(
    size_t n,
    const double* temp_Uh,
    const double* h_prev_vals,
    const double* r_vals,
    const double* dh_prev_accum,
    double* dr_out,
    double* dh_next_out) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    const __m256d one = _mm256_set1_pd(1.0);
    for (; j + 3 < n; j += 4)
    {
      __m256d grad_rh = _mm256_loadu_pd(&temp_Uh[j]);
      __m256d h_prev = (h_prev_vals != nullptr) ? _mm256_loadu_pd(&h_prev_vals[j]) : _mm256_setzero_pd();
      __m256d r = _mm256_loadu_pd(&r_vals[j]);
      __m256d dh_prev = _mm256_loadu_pd(&dh_prev_accum[j]);

      // dr = grad_rh * h_prev * r * (1.0 - r)
      __m256d dr = _mm256_mul_pd(_mm256_mul_pd(grad_rh, h_prev), _mm256_mul_pd(r, _mm256_sub_pd(one, r)));
      
      // dh_next = dh_prev + grad_rh * r
#ifdef SIMD_FMA_ENABLED
      __m256d dh_next = _mm256_fmadd_pd(grad_rh, r, dh_prev);
#else
      __m256d dh_next = _mm256_add_pd(dh_prev, _mm256_mul_pd(grad_rh, r));
#endif

      _mm256_storeu_pd(&dr_out[j], dr);
      _mm256_storeu_pd(&dh_next_out[j], dh_next);
    }
#endif
    scalar_gru_bptt_reset_step(n, temp_Uh, h_prev_vals, r_vals, dh_prev_accum, dr_out, dh_next_out, j);
  }

  // Scalar fallback for lstm_bptt_gate_step
  inline static void scalar_lstm_bptt_gate_step(
    size_t n,
    const double* dh_curr,
    const double* dc_next_in,
    const double* f,
    const double* i,
    const double* o,
    const double* g_pre_vals,
    const double* activated_g_vals,
    const double* activated_c_vals,
    const double* c_prev,
    bool has_prev,
    double* df_out,
    double* di_out,
    double* do_out,
    double* dg_out,
    double* dc_next_out,
    const double* dc_act_deriv_vals,
    const double* dg_act_deriv_vals,
    size_t start = 0) noexcept
  {
    (void)g_pre_vals;
    for (size_t j = start; j < n; ++j)
    {
      double dh = std::clamp(dh_curr[j], -50.0, 50.0);
      double act_c = activated_c_vals[j];
      double do_gate_s = dh * act_c * o[j] * (1.0 - o[j]);

      double dc = dh * o[j] * dc_act_deriv_vals[j] + dc_next_in[j];

      double g_act = activated_g_vals[j];

      df_out[j] = dc * (has_prev ? c_prev[j] : 0.0) * f[j] * (1.0 - f[j]);
      di_out[j] = dc * g_act * i[j] * (1.0 - i[j]);
      do_out[j] = do_gate_s;
      dg_out[j] = dc * i[j] * dg_act_deriv_vals[j];
      dc_next_out[j] = dc * f[j];
    }
  }

  // LSTM BPTT Gate Step
  inline static void lstm_bptt_gate_step(
    size_t n,
    const double* dh_curr,
    const double* dc_next_in,
    const double* f,
    const double* i,
    const double* o,
    const double* g_pre_vals,
    const double* activated_g_vals,
    const double* activated_c_vals,
    const double* c_prev,
    bool has_prev,
    double* df_out,
    double* di_out,
    double* do_out,
    double* dg_out,
    double* dc_next_out,
    const double* dc_act_deriv_vals,
    const double* dg_act_deriv_vals) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d clip_limit = _mm256_set1_pd(50.0);
    const __m256d neg_clip_limit = _mm256_set1_pd(-50.0);

    for (; j + 3 < n; j += 4)
    {
      __m256d dh_raw = _mm256_loadu_pd(&dh_curr[j]);
      __m256d dh = _mm256_max_pd(_mm256_min_pd(dh_raw, clip_limit), neg_clip_limit);
      __m256d o_gate = _mm256_loadu_pd(&o[j]);
      __m256d dc_nxt = _mm256_loadu_pd(&dc_next_in[j]);

      __m256d act_c = _mm256_loadu_pd(&activated_c_vals[j]);
      __m256d do_gate_v = _mm256_mul_pd(_mm256_mul_pd(dh, act_c), _mm256_mul_pd(o_gate, _mm256_sub_pd(one, o_gate)));
      
      __m256d dc_deriv = _mm256_loadu_pd(&dc_act_deriv_vals[j]);
#ifdef SIMD_FMA_ENABLED
      __m256d dc = _mm256_fmadd_pd(_mm256_mul_pd(dh, o_gate), dc_deriv, dc_nxt);
#else
      __m256d dc = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(dh, o_gate), dc_deriv), dc_nxt);
#endif

      __m256d f_gate = _mm256_loadu_pd(&f[j]);
      __m256d i_gate = _mm256_loadu_pd(&i[j]);
      __m256d cp = has_prev ? _mm256_loadu_pd(&c_prev[j]) : _mm256_setzero_pd();
      __m256d g_act = _mm256_loadu_pd(&activated_g_vals[j]);
      __m256d dg_deriv = _mm256_loadu_pd(&dg_act_deriv_vals[j]);

      __m256d df = _mm256_mul_pd(_mm256_mul_pd(dc, cp), _mm256_mul_pd(f_gate, _mm256_sub_pd(one, f_gate)));
      __m256d di = _mm256_mul_pd(_mm256_mul_pd(dc, g_act), _mm256_mul_pd(i_gate, _mm256_sub_pd(one, i_gate)));
      __m256d dg = _mm256_mul_pd(_mm256_mul_pd(dc, i_gate), dg_deriv);

      _mm256_storeu_pd(&df_out[j], df);
      _mm256_storeu_pd(&di_out[j], di);
      _mm256_storeu_pd(&do_out[j], do_gate_v);
      _mm256_storeu_pd(&dg_out[j], dg);
      _mm256_storeu_pd(&dc_next_out[j], _mm256_mul_pd(dc, f_gate));
    }
#endif
    scalar_lstm_bptt_gate_step(
      n, 
      dh_curr, 
      dc_next_in, 
      f, 
      i, 
      o, 
      g_pre_vals, 
      activated_g_vals, 
      activated_c_vals, 
      c_prev, 
      has_prev, 
      df_out, 
      di_out, 
      do_out, 
      dg_out, 
      dc_next_out, 
      dc_act_deriv_vals, 
      dg_act_deriv_vals, 
      j
    );
  }

  // Calculate sum of squares (sum(x_i^2))
  inline static double sum_sq(const double* x, size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t i = 0;
    double total = 0.0;
#ifdef SIMD_AVX2_ENABLED
    __m256d vec_total = _mm256_setzero_pd();
    for (; i + 3 < n; i += 4)
    {
      __m256d vec_x = _mm256_loadu_pd(x + i);
#ifdef SIMD_FMA_ENABLED
      vec_total = _mm256_fmadd_pd(vec_x, vec_x, vec_total);
#else
      vec_total = _mm256_add_pd(vec_total, _mm256_mul_pd(vec_x, vec_x));
#endif
    }
    total = horizontal_sum(vec_total);
#endif
    for (; i < n; ++i)
    {
      total += x[i] * x[i];
    }
    return total;
  }

  // Scalar fallback for gemv_add
  inline static void scalar_gemv_add(const double* A, const double* x, double* y, size_t rows, size_t cols) noexcept
  {
    for (size_t i = 0; i < rows; ++i)
    {
      const double* row_ptr = A + i * cols;
      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum += row_ptr[j] * x[j];
      }
      y[i] += sum;
    }
  }

  // Row-major matrix-vector multiplication (y += A * x)
  inline static void gemv_add(const double* A, const double* x, double* y, size_t rows, size_t cols) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
#ifdef SIMD_AVX2_ENABLED
    size_t i = 0;
    for (; i + 3 < rows; i += 4)
    {
      const double* row0 = A + i * cols;
      const double* row1 = A + (i + 1) * cols;
      const double* row2 = A + (i + 2) * cols;
      const double* row3 = A + (i + 3) * cols;

      __m256d vec_sum0 = _mm256_setzero_pd();
      __m256d vec_sum1 = _mm256_setzero_pd();
      __m256d vec_sum2 = _mm256_setzero_pd();
      __m256d vec_sum3 = _mm256_setzero_pd();

      size_t j = 0;
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_x = _mm256_loadu_pd(x + j);
        __m256d vec_a0 = _mm256_loadu_pd(row0 + j);
        __m256d vec_a1 = _mm256_loadu_pd(row1 + j);
        __m256d vec_a2 = _mm256_loadu_pd(row2 + j);
        __m256d vec_a3 = _mm256_loadu_pd(row3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a0, vec_x, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a1, vec_x, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a2, vec_x, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a3, vec_x, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a0, vec_x));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a1, vec_x));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a2, vec_x));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a3, vec_x));
#endif
      }

      double sum0 = horizontal_sum(vec_sum0);
      double sum1 = horizontal_sum(vec_sum1);
      double sum2 = horizontal_sum(vec_sum2);
      double sum3 = horizontal_sum(vec_sum3);

      for (; j < cols; ++j)
      {
        sum0 += row0[j] * x[j];
        sum1 += row1[j] * x[j];
        sum2 += row2[j] * x[j];
        sum3 += row3[j] * x[j];
      }

      y[i] += sum0;
      y[i + 1] += sum1;
      y[i + 2] += sum2;
      y[i + 3] += sum3;
    }

    for (; i < rows; ++i)
    {
      const double* row_ptr = A + i * cols;
      double sum = 0.0;
      size_t j = 0;
      __m256d vec_sum = _mm256_setzero_pd();
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_a = _mm256_loadu_pd(row_ptr + j);
        __m256d vec_b = _mm256_loadu_pd(x + j);
#ifdef SIMD_FMA_ENABLED
        vec_sum = _mm256_fmadd_pd(vec_a, vec_b, vec_sum);
#else
        vec_sum = _mm256_add_pd(vec_sum, _mm256_mul_pd(vec_a, vec_b));
#endif
      }
      sum = horizontal_sum(vec_sum);
      for (; j < cols; ++j)
      {
        sum += row_ptr[j] * x[j];
      }
      y[i] += sum;
    }
#else
    scalar_gemv_add(A, x, y, rows, cols);
#endif
  }

  // Row-major matrix-vector multiplication with accumulation of two matrix-vector products (y += A0*x0 + A1*x1)
  inline static void gemv_accumulate_two(
    const double* A0, const double* A1,
    const double* x0, const double* x1,
    double* y, size_t rows, size_t cols) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
#ifdef SIMD_AVX2_ENABLED
    size_t i = 0;
    for (; i + 3 < rows; i += 4)
    {
      const double* row0_0 = A0 + i * cols;
      const double* row0_1 = A0 + (i + 1) * cols;
      const double* row0_2 = A0 + (i + 2) * cols;
      const double* row0_3 = A0 + (i + 3) * cols;

      const double* row1_0 = A1 + i * cols;
      const double* row1_1 = A1 + (i + 1) * cols;
      const double* row1_2 = A1 + (i + 2) * cols;
      const double* row1_3 = A1 + (i + 3) * cols;

      __m256d vec_sum0 = _mm256_setzero_pd();
      __m256d vec_sum1 = _mm256_setzero_pd();
      __m256d vec_sum2 = _mm256_setzero_pd();
      __m256d vec_sum3 = _mm256_setzero_pd();

      size_t j = 0;
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_x0 = _mm256_loadu_pd(x0 + j);
        __m256d vec_x1 = _mm256_loadu_pd(x1 + j);

        // A0 * x0
        __m256d vec_a0_0 = _mm256_loadu_pd(row0_0 + j);
        __m256d vec_a0_1 = _mm256_loadu_pd(row0_1 + j);
        __m256d vec_a0_2 = _mm256_loadu_pd(row0_2 + j);
        __m256d vec_a0_3 = _mm256_loadu_pd(row0_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a0_0, vec_x0, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a0_1, vec_x0, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a0_2, vec_x0, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a0_3, vec_x0, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a0_0, vec_x0));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a0_1, vec_x0));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a0_2, vec_x0));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a0_3, vec_x0));
#endif

        // A1 * x1
        __m256d vec_a1_0 = _mm256_loadu_pd(row1_0 + j);
        __m256d vec_a1_1 = _mm256_loadu_pd(row1_1 + j);
        __m256d vec_a1_2 = _mm256_loadu_pd(row1_2 + j);
        __m256d vec_a1_3 = _mm256_loadu_pd(row1_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a1_0, vec_x1, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a1_1, vec_x1, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a1_2, vec_x1, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a1_3, vec_x1, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a1_0, vec_x1));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a1_1, vec_x1));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a1_2, vec_x1));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a1_3, vec_x1));
#endif
      }

      double sum0 = horizontal_sum(vec_sum0);
      double sum1 = horizontal_sum(vec_sum1);
      double sum2 = horizontal_sum(vec_sum2);
      double sum3 = horizontal_sum(vec_sum3);

      for (; j < cols; ++j)
      {
        double x0_val = x0[j];
        double x1_val = x1[j];

        sum0 += row0_0[j] * x0_val + row1_0[j] * x1_val;
        sum1 += row0_1[j] * x0_val + row1_1[j] * x1_val;
        sum2 += row0_2[j] * x0_val + row1_2[j] * x1_val;
        sum3 += row0_3[j] * x0_val + row1_3[j] * x1_val;
      }

      y[i] += sum0;
      y[i + 1] += sum1;
      y[i + 2] += sum2;
      y[i + 3] += sum3;
    }

    for (; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;

      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum += row0[j] * x0[j] + row1[j] * x1[j];
      }
      y[i] += sum;
    }
#else
    for (size_t i = 0; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;

      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum += row0[j] * x0[j] + row1[j] * x1[j];
      }
      y[i] += sum;
    }
#endif
  }

  // Row-major matrix-vector multiplication with accumulation of three matrix-vector products (y += A0*x0 + A1*x1 + A2*x2)
  inline static void gemv_accumulate_three(
    const double* A0, const double* A1, const double* A2,
    const double* x0, const double* x1, const double* x2,
    double* y, size_t rows, size_t cols) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
#ifdef SIMD_AVX2_ENABLED
    size_t i = 0;
    for (; i + 3 < rows; i += 4)
    {
      const double* row0_0 = A0 + i * cols;
      const double* row0_1 = A0 + (i + 1) * cols;
      const double* row0_2 = A0 + (i + 2) * cols;
      const double* row0_3 = A0 + (i + 3) * cols;

      const double* row1_0 = A1 + i * cols;
      const double* row1_1 = A1 + (i + 1) * cols;
      const double* row1_2 = A1 + (i + 2) * cols;
      const double* row1_3 = A1 + (i + 3) * cols;

      const double* row2_0 = A2 + i * cols;
      const double* row2_1 = A2 + (i + 1) * cols;
      const double* row2_2 = A2 + (i + 2) * cols;
      const double* row2_3 = A2 + (i + 3) * cols;

      __m256d vec_sum0 = _mm256_setzero_pd();
      __m256d vec_sum1 = _mm256_setzero_pd();
      __m256d vec_sum2 = _mm256_setzero_pd();
      __m256d vec_sum3 = _mm256_setzero_pd();

      size_t j = 0;
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_x0 = _mm256_loadu_pd(x0 + j);
        __m256d vec_x1 = _mm256_loadu_pd(x1 + j);
        __m256d vec_x2 = _mm256_loadu_pd(x2 + j);

        // A0 * x0
        __m256d vec_a0_0 = _mm256_loadu_pd(row0_0 + j);
        __m256d vec_a0_1 = _mm256_loadu_pd(row0_1 + j);
        __m256d vec_a0_2 = _mm256_loadu_pd(row0_2 + j);
        __m256d vec_a0_3 = _mm256_loadu_pd(row0_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a0_0, vec_x0, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a0_1, vec_x0, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a0_2, vec_x0, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a0_3, vec_x0, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a0_0, vec_x0));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a0_1, vec_x0));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a0_2, vec_x0));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a0_3, vec_x0));
#endif

        // A1 * x1
        __m256d vec_a1_0 = _mm256_loadu_pd(row1_0 + j);
        __m256d vec_a1_1 = _mm256_loadu_pd(row1_1 + j);
        __m256d vec_a1_2 = _mm256_loadu_pd(row1_2 + j);
        __m256d vec_a1_3 = _mm256_loadu_pd(row1_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a1_0, vec_x1, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a1_1, vec_x1, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a1_2, vec_x1, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a1_3, vec_x1, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a1_0, vec_x1));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a1_1, vec_x1));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a1_2, vec_x1));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a1_3, vec_x1));
#endif

        // A2 * x2
        __m256d vec_a2_0 = _mm256_loadu_pd(row2_0 + j);
        __m256d vec_a2_1 = _mm256_loadu_pd(row2_1 + j);
        __m256d vec_a2_2 = _mm256_loadu_pd(row2_2 + j);
        __m256d vec_a2_3 = _mm256_loadu_pd(row2_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a2_0, vec_x2, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a2_1, vec_x2, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a2_2, vec_x2, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a2_3, vec_x2, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a2_0, vec_x2));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a2_1, vec_x2));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a2_2, vec_x2));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a2_3, vec_x2));
#endif
      }

      double sum0 = horizontal_sum(vec_sum0);
      double sum1 = horizontal_sum(vec_sum1);
      double sum2 = horizontal_sum(vec_sum2);
      double sum3 = horizontal_sum(vec_sum3);

      for (; j < cols; ++j)
      {
        double x0_val = x0[j];
        double x1_val = x1[j];
        double x2_val = x2[j];

        sum0 += row0_0[j] * x0_val + row1_0[j] * x1_val + row2_0[j] * x2_val;
        sum1 += row0_1[j] * x0_val + row1_1[j] * x1_val + row2_1[j] * x2_val;
        sum2 += row0_2[j] * x0_val + row1_2[j] * x1_val + row2_2[j] * x2_val;
        sum3 += row0_3[j] * x0_val + row1_3[j] * x1_val + row2_3[j] * x2_val;
      }

      y[i] += sum0;
      y[i + 1] += sum1;
      y[i + 2] += sum2;
      y[i + 3] += sum3;
    }

    for (; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;
      const double* row2 = A2 + i * cols;

      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum += row0[j] * x0[j] + row1[j] * x1[j] + row2[j] * x2[j];
      }
      y[i] += sum;
    }
#else
    for (size_t i = 0; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;
      const double* row2 = A2 + i * cols;

      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum += row0[j] * x0[j] + row1[j] * x1[j] + row2[j] * x2[j];
      }
      y[i] += sum;
    }
#endif
  }

  // Row-major matrix-vector multiplication with accumulation of four matrix-vector products (y += A0*x0 + A1*x1 + A2*x2 + A3*x3)
  inline static void gemv_accumulate_four(
    const double* A0, const double* A1, const double* A2, const double* A3,
    const double* x0, const double* x1, const double* x2, const double* x3,
    double* y, size_t rows, size_t cols) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
#ifdef SIMD_AVX2_ENABLED
    size_t i = 0;
    for (; i + 3 < rows; i += 4)
    {
      const double* row0_0 = A0 + i * cols;
      const double* row0_1 = A0 + (i + 1) * cols;
      const double* row0_2 = A0 + (i + 2) * cols;
      const double* row0_3 = A0 + (i + 3) * cols;

      const double* row1_0 = A1 + i * cols;
      const double* row1_1 = A1 + (i + 1) * cols;
      const double* row1_2 = A1 + (i + 2) * cols;
      const double* row1_3 = A1 + (i + 3) * cols;

      const double* row2_0 = A2 + i * cols;
      const double* row2_1 = A2 + (i + 1) * cols;
      const double* row2_2 = A2 + (i + 2) * cols;
      const double* row2_3 = A2 + (i + 3) * cols;

      const double* row3_0 = A3 + i * cols;
      const double* row3_1 = A3 + (i + 1) * cols;
      const double* row3_2 = A3 + (i + 2) * cols;
      const double* row3_3 = A3 + (i + 3) * cols;

      __m256d vec_sum0 = _mm256_setzero_pd();
      __m256d vec_sum1 = _mm256_setzero_pd();
      __m256d vec_sum2 = _mm256_setzero_pd();
      __m256d vec_sum3 = _mm256_setzero_pd();

      size_t j = 0;
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_x0 = _mm256_loadu_pd(x0 + j);
        __m256d vec_x1 = _mm256_loadu_pd(x1 + j);
        __m256d vec_x2 = _mm256_loadu_pd(x2 + j);
        __m256d vec_x3 = _mm256_loadu_pd(x3 + j);

        // A0 * x0
        __m256d vec_a0_0 = _mm256_loadu_pd(row0_0 + j);
        __m256d vec_a0_1 = _mm256_loadu_pd(row0_1 + j);
        __m256d vec_a0_2 = _mm256_loadu_pd(row0_2 + j);
        __m256d vec_a0_3 = _mm256_loadu_pd(row0_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a0_0, vec_x0, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a0_1, vec_x0, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a0_2, vec_x0, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a0_3, vec_x0, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a0_0, vec_x0));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a0_1, vec_x0));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a0_2, vec_x0));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a0_3, vec_x0));
#endif

        // A1 * x1
        __m256d vec_a1_0 = _mm256_loadu_pd(row1_0 + j);
        __m256d vec_a1_1 = _mm256_loadu_pd(row1_1 + j);
        __m256d vec_a1_2 = _mm256_loadu_pd(row1_2 + j);
        __m256d vec_a1_3 = _mm256_loadu_pd(row1_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a1_0, vec_x1, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a1_1, vec_x1, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a1_2, vec_x1, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a1_3, vec_x1, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a1_0, vec_x1));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a1_1, vec_x1));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a1_2, vec_x1));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a1_3, vec_x1));
#endif

        // A2 * x2
        __m256d vec_a2_0 = _mm256_loadu_pd(row2_0 + j);
        __m256d vec_a2_1 = _mm256_loadu_pd(row2_1 + j);
        __m256d vec_a2_2 = _mm256_loadu_pd(row2_2 + j);
        __m256d vec_a2_3 = _mm256_loadu_pd(row2_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a2_0, vec_x2, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a2_1, vec_x2, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a2_2, vec_x2, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a2_3, vec_x2, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a2_0, vec_x2));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a2_1, vec_x2));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a2_2, vec_x2));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a2_3, vec_x2));
#endif

        // A3 * x3
        __m256d vec_a3_0 = _mm256_loadu_pd(row3_0 + j);
        __m256d vec_a3_1 = _mm256_loadu_pd(row3_1 + j);
        __m256d vec_a3_2 = _mm256_loadu_pd(row3_2 + j);
        __m256d vec_a3_3 = _mm256_loadu_pd(row3_3 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a3_0, vec_x3, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a3_1, vec_x3, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a3_2, vec_x3, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a3_3, vec_x3, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a3_0, vec_x3));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a3_1, vec_x3));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a3_2, vec_x3));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a3_3, vec_x3));
#endif
      }

      double sum0 = horizontal_sum(vec_sum0);
      double sum1 = horizontal_sum(vec_sum1);
      double sum2 = horizontal_sum(vec_sum2);
      double sum3 = horizontal_sum(vec_sum3);

      for (; j < cols; ++j)
      {
        double x0_val = x0[j];
        double x1_val = x1[j];
        double x2_val = x2[j];
        double x3_val = x3[j];

        sum0 += row0_0[j] * x0_val + row1_0[j] * x1_val + row2_0[j] * x2_val + row3_0[j] * x3_val;
        sum1 += row0_1[j] * x0_val + row1_1[j] * x1_val + row2_1[j] * x2_val + row3_1[j] * x3_val;
        sum2 += row0_2[j] * x0_val + row1_2[j] * x1_val + row2_2[j] * x2_val + row3_2[j] * x3_val;
        sum3 += row0_3[j] * x0_val + row1_3[j] * x1_val + row2_3[j] * x2_val + row3_3[j] * x3_val;
      }

      y[i] += sum0;
      y[i + 1] += sum1;
      y[i + 2] += sum2;
      y[i + 3] += sum3;
    }

    for (; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;
      const double* row2 = A2 + i * cols;
      const double* row3 = A3 + i * cols;

      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum += row0[j] * x0[j] + row1[j] * x1[j] + row2[j] * x2[j] + row3[j] * x3[j];
      }
      y[i] += sum;
    }
#else
    for (size_t i = 0; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;
      const double* row2 = A2 + i * cols;
      const double* row3 = A3 + i * cols;

      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum += row0[j] * x0[j] + row1[j] * x1[j] + row2[j] * x2[j] + row3[j] * x3[j];
      }
      y[i] += sum;
    }
#endif
  }

  // Scalar fallback for gemv_add_two
  inline static void scalar_gemv_add_two(
    const double* A0, const double* A1,
    const double* x,
    double* y0, double* y1,
    size_t rows, size_t cols) noexcept
  {
    for (size_t i = 0; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;
      double sum0 = 0.0;
      double sum1 = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum0 += row0[j] * x[j];
        sum1 += row1[j] * x[j];
      }
      y0[i] += sum0;
      y1[i] += sum1;
    }
  }

  // Row-major matrix-vector multiplication for two gates (y0 += A0 * x, y1 += A1 * x)
  inline static void gemv_add_two(
    const double* A0, const double* A1,
    const double* x,
    double* y0, double* y1,
    size_t rows, size_t cols) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
#ifdef SIMD_AVX2_ENABLED
    size_t i = 0;
    for (; i + 1 < rows; i += 2)
    {
      const double* row0_0 = A0 + i * cols;
      const double* row0_1 = A0 + (i + 1) * cols;
      const double* row1_0 = A1 + i * cols;
      const double* row1_1 = A1 + (i + 1) * cols;

      __m256d vec_sum0_0 = _mm256_setzero_pd();
      __m256d vec_sum0_1 = _mm256_setzero_pd();
      __m256d vec_sum1_0 = _mm256_setzero_pd();
      __m256d vec_sum1_1 = _mm256_setzero_pd();

      size_t j = 0;
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_x = _mm256_loadu_pd(x + j);

        __m256d vec_a0_0 = _mm256_loadu_pd(row0_0 + j);
        __m256d vec_a0_1 = _mm256_loadu_pd(row0_1 + j);

        __m256d vec_a1_0 = _mm256_loadu_pd(row1_0 + j);
        __m256d vec_a1_1 = _mm256_loadu_pd(row1_1 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0_0 = _mm256_fmadd_pd(vec_a0_0, vec_x, vec_sum0_0);
        vec_sum0_1 = _mm256_fmadd_pd(vec_a0_1, vec_x, vec_sum0_1);
        vec_sum1_0 = _mm256_fmadd_pd(vec_a1_0, vec_x, vec_sum1_0);
        vec_sum1_1 = _mm256_fmadd_pd(vec_a1_1, vec_x, vec_sum1_1);
#else
        vec_sum0_0 = _mm256_add_pd(vec_sum0_0, _mm256_mul_pd(vec_a0_0, vec_x));
        vec_sum0_1 = _mm256_add_pd(vec_sum0_1, _mm256_mul_pd(vec_a0_1, vec_x));
        vec_sum1_0 = _mm256_add_pd(vec_sum1_0, _mm256_mul_pd(vec_a1_0, vec_x));
        vec_sum1_1 = _mm256_add_pd(vec_sum1_1, _mm256_mul_pd(vec_a1_1, vec_x));
#endif
      }

      double sum0_0 = horizontal_sum(vec_sum0_0);
      double sum0_1 = horizontal_sum(vec_sum0_1);
      double sum1_0 = horizontal_sum(vec_sum1_0);
      double sum1_1 = horizontal_sum(vec_sum1_1);

      for (; j < cols; ++j)
      {
        sum0_0 += row0_0[j] * x[j];
        sum0_1 += row0_1[j] * x[j];
        sum1_0 += row1_0[j] * x[j];
        sum1_1 += row1_1[j] * x[j];
      }

      y0[i] += sum0_0;
      y0[i + 1] += sum0_1;
      y1[i] += sum1_0;
      y1[i + 1] += sum1_1;
    }

    for (; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;
      double sum0 = 0.0;
      double sum1 = 0.0;
      size_t j = 0;
      __m256d vec_sum0 = _mm256_setzero_pd();
      __m256d vec_sum1 = _mm256_setzero_pd();
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_x = _mm256_loadu_pd(x + j);
        __m256d vec_a0 = _mm256_loadu_pd(row0 + j);
        __m256d vec_a1 = _mm256_loadu_pd(row1 + j);
#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a0, vec_x, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a1, vec_x, vec_sum1);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a0, vec_x));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a1, vec_x));
#endif
      }
      sum0 = horizontal_sum(vec_sum0);
      sum1 = horizontal_sum(vec_sum1);
      for (; j < cols; ++j)
      {
        sum0 += row0[j] * x[j];
        sum1 += row1[j] * x[j];
      }
      y0[i] += sum0;
      y1[i] += sum1;
    }
#else
    scalar_gemv_add_two(A0, A1, x, y0, y1, rows, cols);
#endif
  }

  // Scalar fallback for gemv_add_four
  inline static void scalar_gemv_add_four(
    const double* A0, const double* A1, const double* A2, const double* A3,
    const double* x,
    double* y0, double* y1, double* y2, double* y3,
    size_t rows, size_t cols) noexcept
  {
    for (size_t i = 0; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;
      const double* row2 = A2 + i * cols;
      const double* row3 = A3 + i * cols;
      double sum0 = 0.0;
      double sum1 = 0.0;
      double sum2 = 0.0;
      double sum3 = 0.0;
      for (size_t j = 0; j < cols; ++j)
      {
        sum0 += row0[j] * x[j];
        sum1 += row1[j] * x[j];
        sum2 += row2[j] * x[j];
        sum3 += row3[j] * x[j];
      }
      y0[i] += sum0;
      y1[i] += sum1;
      y2[i] += sum2;
      y3[i] += sum3;
    }
  }

  // Row-major matrix-vector multiplication for four gates
  inline static void gemv_add_four(
    const double* A0, const double* A1, const double* A2, const double* A3,
    const double* x,
    double* y0, double* y1, double* y2, double* y3,
    size_t rows, size_t cols) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
#ifdef SIMD_AVX2_ENABLED
    size_t i = 0;
    for (; i + 1 < rows; i += 2)
    {
      const double* row0_0 = A0 + i * cols;
      const double* row0_1 = A0 + (i + 1) * cols;
      const double* row1_0 = A1 + i * cols;
      const double* row1_1 = A1 + (i + 1) * cols;
      const double* row2_0 = A2 + i * cols;
      const double* row2_1 = A2 + (i + 1) * cols;
      const double* row3_0 = A3 + i * cols;
      const double* row3_1 = A3 + (i + 1) * cols;

      __m256d vec_sum0_0 = _mm256_setzero_pd();
      __m256d vec_sum0_1 = _mm256_setzero_pd();
      __m256d vec_sum1_0 = _mm256_setzero_pd();
      __m256d vec_sum1_1 = _mm256_setzero_pd();
      __m256d vec_sum2_0 = _mm256_setzero_pd();
      __m256d vec_sum2_1 = _mm256_setzero_pd();
      __m256d vec_sum3_0 = _mm256_setzero_pd();
      __m256d vec_sum3_1 = _mm256_setzero_pd();

      size_t j = 0;
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_x = _mm256_loadu_pd(x + j);

        __m256d vec_a0_0 = _mm256_loadu_pd(row0_0 + j);
        __m256d vec_a0_1 = _mm256_loadu_pd(row0_1 + j);

        __m256d vec_a1_0 = _mm256_loadu_pd(row1_0 + j);
        __m256d vec_a1_1 = _mm256_loadu_pd(row1_1 + j);

        __m256d vec_a2_0 = _mm256_loadu_pd(row2_0 + j);
        __m256d vec_a2_1 = _mm256_loadu_pd(row2_1 + j);

        __m256d vec_a3_0 = _mm256_loadu_pd(row3_0 + j);
        __m256d vec_a3_1 = _mm256_loadu_pd(row3_1 + j);

#ifdef SIMD_FMA_ENABLED
        vec_sum0_0 = _mm256_fmadd_pd(vec_a0_0, vec_x, vec_sum0_0);
        vec_sum0_1 = _mm256_fmadd_pd(vec_a0_1, vec_x, vec_sum0_1);
        vec_sum1_0 = _mm256_fmadd_pd(vec_a1_0, vec_x, vec_sum1_0);
        vec_sum1_1 = _mm256_fmadd_pd(vec_a1_1, vec_x, vec_sum1_1);
        vec_sum2_0 = _mm256_fmadd_pd(vec_a2_0, vec_x, vec_sum2_0);
        vec_sum2_1 = _mm256_fmadd_pd(vec_a2_1, vec_x, vec_sum2_1);
        vec_sum3_0 = _mm256_fmadd_pd(vec_a3_0, vec_x, vec_sum3_0);
        vec_sum3_1 = _mm256_fmadd_pd(vec_a3_1, vec_x, vec_sum3_1);
#else
        vec_sum0_0 = _mm256_add_pd(vec_sum0_0, _mm256_mul_pd(vec_a0_0, vec_x));
        vec_sum0_1 = _mm256_add_pd(vec_sum0_1, _mm256_mul_pd(vec_a0_1, vec_x));
        vec_sum1_0 = _mm256_add_pd(vec_sum1_0, _mm256_mul_pd(vec_a1_0, vec_x));
        vec_sum1_1 = _mm256_add_pd(vec_sum1_1, _mm256_mul_pd(vec_a1_1, vec_x));
        vec_sum2_0 = _mm256_add_pd(vec_sum2_0, _mm256_mul_pd(vec_a2_0, vec_x));
        vec_sum2_1 = _mm256_add_pd(vec_sum2_1, _mm256_mul_pd(vec_a2_1, vec_x));
        vec_sum3_0 = _mm256_add_pd(vec_sum3_0, _mm256_mul_pd(vec_a3_0, vec_x));
        vec_sum3_1 = _mm256_add_pd(vec_sum3_1, _mm256_mul_pd(vec_a3_1, vec_x));
#endif
      }

      double sum0_0 = horizontal_sum(vec_sum0_0);
      double sum0_1 = horizontal_sum(vec_sum0_1);
      double sum1_0 = horizontal_sum(vec_sum1_0);
      double sum1_1 = horizontal_sum(vec_sum1_1);
      double sum2_0 = horizontal_sum(vec_sum2_0);
      double sum2_1 = horizontal_sum(vec_sum2_1);
      double sum3_0 = horizontal_sum(vec_sum3_0);
      double sum3_1 = horizontal_sum(vec_sum3_1);

      for (; j < cols; ++j)
      {
        sum0_0 += row0_0[j] * x[j];
        sum0_1 += row0_1[j] * x[j];
        sum1_0 += row1_0[j] * x[j];
        sum1_1 += row1_1[j] * x[j];
        sum2_0 += row2_0[j] * x[j];
        sum2_1 += row2_1[j] * x[j];
        sum3_0 += row3_0[j] * x[j];
        sum3_1 += row3_1[j] * x[j];
      }

      y0[i] += sum0_0;
      y0[i + 1] += sum0_1;
      y1[i] += sum1_0;
      y1[i + 1] += sum1_1;
      y2[i] += sum2_0;
      y2[i + 1] += sum2_1;
      y3[i] += sum3_0;
      y3[i + 1] += sum3_1;
    }

    for (; i < rows; ++i)
    {
      const double* row0 = A0 + i * cols;
      const double* row1 = A1 + i * cols;
      const double* row2 = A2 + i * cols;
      const double* row3 = A3 + i * cols;
      double sum0 = 0.0;
      double sum1 = 0.0;
      double sum2 = 0.0;
      double sum3 = 0.0;
      size_t j = 0;
      __m256d vec_sum0 = _mm256_setzero_pd();
      __m256d vec_sum1 = _mm256_setzero_pd();
      __m256d vec_sum2 = _mm256_setzero_pd();
      __m256d vec_sum3 = _mm256_setzero_pd();
      for (; j + 3 < cols; j += 4)
      {
        __m256d vec_x = _mm256_loadu_pd(x + j);
        __m256d vec_a0 = _mm256_loadu_pd(row0 + j);
        __m256d vec_a1 = _mm256_loadu_pd(row1 + j);
        __m256d vec_a2 = _mm256_loadu_pd(row2 + j);
        __m256d vec_a3 = _mm256_loadu_pd(row3 + j);
#ifdef SIMD_FMA_ENABLED
        vec_sum0 = _mm256_fmadd_pd(vec_a0, vec_x, vec_sum0);
        vec_sum1 = _mm256_fmadd_pd(vec_a1, vec_x, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(vec_a2, vec_x, vec_sum2);
        vec_sum3 = _mm256_fmadd_pd(vec_a3, vec_x, vec_sum3);
#else
        vec_sum0 = _mm256_add_pd(vec_sum0, _mm256_mul_pd(vec_a0, vec_x));
        vec_sum1 = _mm256_add_pd(vec_sum1, _mm256_mul_pd(vec_a1, vec_x));
        vec_sum2 = _mm256_add_pd(vec_sum2, _mm256_mul_pd(vec_a2, vec_x));
        vec_sum3 = _mm256_add_pd(vec_sum3, _mm256_mul_pd(vec_a3, vec_x));
#endif
      }
      sum0 = horizontal_sum(vec_sum0);
      sum1 = horizontal_sum(vec_sum1);
      sum2 = horizontal_sum(vec_sum2);
      sum3 = horizontal_sum(vec_sum3);
      for (; j < cols; ++j)
      {
        sum0 += row0[j] * x[j];
        sum1 += row1[j] * x[j];
        sum2 += row2[j] * x[j];
        sum3 += row3[j] * x[j];
      }
      y0[i] += sum0;
      y1[i] += sum1;
      y2[i] += sum2;
      y3[i] += sum3;
    }
#else
    scalar_gemv_add_four(A0, A1, A2, A3, x, y0, y1, y2, y3, rows, cols);
#endif
  }

  // Scalar fallback for add_vectors
  inline static void scalar_add_vectors(const double* x, double* y, size_t n, size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      y[j] += x[j];
    }
  }

  // Vector-vector addition (y += x)
  inline static void add_vectors(const double* x, double* y, size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_x = _mm256_loadu_pd(x + j);
      __m256d vec_y = _mm256_loadu_pd(y + j);
      vec_y = _mm256_add_pd(vec_y, vec_x);
      _mm256_storeu_pd(y + j, vec_y);
    }
#endif
    scalar_add_vectors(x, y, n, j);
  }

  // Scalar fallback for scale_vector
  inline static void scalar_scale_vector(double* y, const double scale, size_t n, size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      y[j] *= scale;
    }
  }

  // Vector-scalar multiplication (y *= scale)
  inline static void scale_vector(double* y, const double scale, size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    __m256d vec_scale = _mm256_set1_pd(scale);
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_y = _mm256_loadu_pd(y + j);
      vec_y = _mm256_mul_pd(vec_y, vec_scale);
      _mm256_storeu_pd(y + j, vec_y);
    }
#endif
    scalar_scale_vector(y, scale, n, j);
  }

  // Scalar fallback for mul_vectors
  inline static void scalar_mul_vectors(const double* x, const double* y, double* z, size_t n, size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      z[j] = x[j] * y[j];
    }
  }

  // Vector-vector multiplication (z = x * y)
  inline static void mul_vectors(const double* x, const double* y, double* z, size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_x = _mm256_loadu_pd(x + j);
      __m256d vec_y = _mm256_loadu_pd(y + j);
      __m256d vec_z = _mm256_mul_pd(vec_x, vec_y);
      _mm256_storeu_pd(z + j, vec_z);
    }
#endif
    scalar_mul_vectors(x, y, z, n, j);
  }

  // Scalar fallback for mul_three_vectors
  inline static void scalar_mul_three_vectors(
    const double* x,
    const double* y,
    const double* z,
    double* w,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      w[j] = x[j] * y[j] * z[j];
    }
  }

  // Vectorised multiplication of three vectors (w = x * y * z)
  inline static void mul_three_vectors(
    const double* x,
    const double* y,
    const double* z,
    double* w,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_x = _mm256_loadu_pd(x + j);
      __m256d vec_y = _mm256_loadu_pd(y + j);
      __m256d vec_z = _mm256_loadu_pd(z + j);
      __m256d vec_w = _mm256_mul_pd(_mm256_mul_pd(vec_x, vec_y), vec_z);
      _mm256_storeu_pd(w + j, vec_w);
    }
#endif
    scalar_mul_three_vectors(x, y, z, w, n, j);
  }

  // Scalar fallback for lstm_bptt_upstream_step
  inline static void scalar_lstm_bptt_upstream_step(
    const double* upstream,
    const double* dh_next,
    const double* mask,
    double* dh_curr,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      dh_curr[j] = std::clamp((upstream[j] + dh_next[j]) * mask[j], -50.0, 50.0);
    }
  }

  // Vectorised LSTM BPTT upstream step: dh_curr[j] = clamp((upstream[j] + dh_next[j]) * mask[j], -50.0, 50.0)
  inline static void lstm_bptt_upstream_step(
    const double* upstream,
    const double* dh_next,
    const double* mask,
    double* dh_curr,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    const __m256d clip_limit = _mm256_set1_pd(50.0);
    const __m256d neg_clip_limit = _mm256_set1_pd(-50.0);
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_up = _mm256_loadu_pd(upstream + j);
      __m256d vec_next = _mm256_loadu_pd(dh_next + j);
      __m256d vec_mask = _mm256_loadu_pd(mask + j);

#ifdef SIMD_FMA_ENABLED
      __m256d val = _mm256_fmadd_pd(vec_up, vec_mask, _mm256_mul_pd(vec_next, vec_mask));
#else
      __m256d val = _mm256_mul_pd(_mm256_add_pd(vec_up, vec_next), vec_mask);
#endif
      __m256d clamped = _mm256_max_pd(_mm256_min_pd(val, clip_limit), neg_clip_limit);
      _mm256_storeu_pd(dh_curr + j, clamped);
    }
#endif
    scalar_lstm_bptt_upstream_step(upstream, dh_next, mask, dh_curr, n, j);
  }

  // Scalar fallback for elman_bptt_gate_step
  inline static void scalar_elman_bptt_gate_step(
    const double* upstream,
    const double* dh_next,
    const double* deriv,
    const double* mask,
    double* g_this_tick,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      double dh = std::clamp(upstream[j] + dh_next[j], -50.0, 50.0);
      g_this_tick[j] = dh * deriv[j] * mask[j];
    }
  }

  // Vectorised Elman RNN BPTT gate step: g_this_tick[j] = clamp(upstream[j] + dh_next[j], -50.0, 50.0) * deriv[j] * mask[j]
  inline static void elman_bptt_gate_step(
    const double* upstream,
    const double* dh_next,
    const double* deriv,
    const double* mask,
    double* g_this_tick,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    const __m256d clip_limit = _mm256_set1_pd(50.0);
    const __m256d neg_clip_limit = _mm256_set1_pd(-50.0);
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_up = _mm256_loadu_pd(upstream + j);
      __m256d vec_next = _mm256_loadu_pd(dh_next + j);
      __m256d vec_deriv = _mm256_loadu_pd(deriv + j);
      __m256d vec_mask = _mm256_loadu_pd(mask + j);

      __m256d dh_raw = _mm256_add_pd(vec_up, vec_next);
      __m256d dh = _mm256_max_pd(_mm256_min_pd(dh_raw, clip_limit), neg_clip_limit);

      __m256d res = _mm256_mul_pd(_mm256_mul_pd(dh, vec_deriv), vec_mask);
      _mm256_storeu_pd(g_this_tick + j, res);
    }
#endif
    scalar_elman_bptt_gate_step(upstream, dh_next, deriv, mask, g_this_tick, n, j);
  }

  // Scalar fallback for gru_output_step
  inline static void scalar_gru_output_step(
    const double* z,
    const double* prev_h,
    const double* h_hat,
    double* current_h,
    double* batch_output_seq,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      double val = (1.0 - z[j]) * prev_h[j] + z[j] * h_hat[j];
      current_h[j] = val;
      batch_output_seq[j] = val;
    }
  }

  // Vectorized GRU output step (current_h = (1 - z) * prev_h + z * h_hat)
  inline static void gru_output_step(
    const double* z,
    const double* prev_h,
    const double* h_hat,
    double* current_h,
    double* batch_output_seq,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    const __m256d one = _mm256_set1_pd(1.0);
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_z = _mm256_loadu_pd(z + j);
      __m256d vec_prev = _mm256_loadu_pd(prev_h + j);
      __m256d vec_h_hat = _mm256_loadu_pd(h_hat + j);

      __m256d vec_one_minus_z = _mm256_sub_pd(one, vec_z);
#ifdef SIMD_FMA_ENABLED
      __m256d vec_res = _mm256_fmadd_pd(vec_z, vec_h_hat, _mm256_mul_pd(vec_one_minus_z, vec_prev));
#else
      __m256d vec_res = _mm256_add_pd(_mm256_mul_pd(vec_one_minus_z, vec_prev), _mm256_mul_pd(vec_z, vec_h_hat));
#endif
      _mm256_storeu_pd(current_h + j, vec_res);
      _mm256_storeu_pd(batch_output_seq + j, vec_res);
    }
#endif
    scalar_gru_output_step(z, prev_h, h_hat, current_h, batch_output_seq, n, j);
  }

  // Scalar fallback for lstm_cell_step
  inline static void scalar_lstm_cell_step(
    const double* f,
    const double* i,
    const double* g_act,
    double* current_c,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t j = start; j < n; ++j)
    {
      current_c[j] = f[j] * current_c[j] + i[j] * g_act[j];
    }
  }

  // Vectorized LSTM cell step (current_c = f * current_c + i * g_act)
  inline static void lstm_cell_step(
    const double* f,
    const double* i,
    const double* g_act,
    double* current_c,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    for (; j + 3 < n; j += 4)
    {
      __m256d vec_f = _mm256_loadu_pd(f + j);
      __m256d vec_i = _mm256_loadu_pd(i + j);
      __m256d vec_g = _mm256_loadu_pd(g_act + j);
      __m256d vec_c = _mm256_loadu_pd(current_c + j);

#ifdef SIMD_FMA_ENABLED
      __m256d vec_res = _mm256_fmadd_pd(vec_f, vec_c, _mm256_mul_pd(vec_i, vec_g));
#else
      __m256d vec_res = _mm256_add_pd(_mm256_mul_pd(vec_f, vec_c), _mm256_mul_pd(vec_i, vec_g));
#endif
      _mm256_storeu_pd(current_c + j, vec_res);
    }
#endif
    scalar_lstm_cell_step(f, i, g_act, current_c, n, j);
  }

  // Scalar fallback for sgd_step
  inline static void scalar_sgd_step(
    double* values,
    double* grads,
    double* velocities,
    const double* decays,
    double momentum,
    double lr,
    double clipping_scale,
    bool is_bias,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t i = start; i < n; ++i)
    {
      double grad = grads[i] * clipping_scale;
      if (!is_bias && decays != nullptr && decays[i] > 0.0)
      {
        grad += decays[i] * values[i];
      }
      double v = momentum * velocities[i] + grad;
      values[i] -= lr * v;
      velocities[i] = v;
      grads[i] = grad;
    }
  }

  // Vectorized SGD step
  inline static void sgd_step(
    double* values,
    double* grads,
    double* velocities,
    const double* decays,
    double momentum,
    double lr,
    double clipping_scale,
    bool is_bias,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    __m256d vec_clip = _mm256_set1_pd(clipping_scale);
    __m256d vec_momentum = _mm256_set1_pd(momentum);
    __m256d vec_lr = _mm256_set1_pd(lr);

    for (; j + 3 < n; j += 4)
    {
      __m256d g = _mm256_loadu_pd(&grads[j]);
      __m256d cur_w = _mm256_loadu_pd(&values[j]);
      __m256d cur_v = _mm256_loadu_pd(&velocities[j]);

      __m256d grad = _mm256_mul_pd(g, vec_clip);

      if (!is_bias && decays != nullptr)
      {
        __m256d d = _mm256_loadu_pd(&decays[j]);
#ifdef SIMD_FMA_ENABLED
        grad = _mm256_fmadd_pd(d, cur_w, grad);
#else
        grad = _mm256_add_pd(grad, _mm256_mul_pd(d, cur_w));
#endif
      }

#ifdef SIMD_FMA_ENABLED
      __m256d next_v = _mm256_fmadd_pd(vec_momentum, cur_v, grad);
#else
      __m256d next_v = _mm256_add_pd(_mm256_mul_pd(vec_momentum, cur_v), grad);
#endif

      __m256d next_w = _mm256_sub_pd(cur_w, _mm256_mul_pd(vec_lr, next_v));

      _mm256_storeu_pd(&velocities[j], next_v);
      _mm256_storeu_pd(&values[j], next_w);
      _mm256_storeu_pd(&grads[j], grad);
    }
#endif
    scalar_sgd_step(values, grads, velocities, decays, momentum, lr, clipping_scale, is_bias, n, j);
  }

  // Scalar fallback for none_step
  inline static void scalar_none_step(
    double* values,
    double* grads,
    double lr,
    double clipping_scale,
    size_t n,
    size_t start = 0) noexcept
  {
    for (size_t i = start; i < n; ++i)
    {
      double grad = grads[i] * clipping_scale;
      values[i] -= lr * grad;
      grads[i] = grad;
    }
  }

  // Vectorised increment values for timesteps
  inline static void increment_values(long long* values, size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t i = 0;
#ifdef SIMD_AVX2_ENABLED
    // Process 4 long longs at a time using AVX2
    __m256i vec_one = _mm256_set1_epi64x(1);
    for (; i + 3 < n; i += 4)
    {
      __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&values[i]));
      v = _mm256_add_epi64(v, vec_one);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&values[i]), v);
    }
#endif
    for (; i < n; ++i)
    {
      ++values[i];
    }
  }

  // Vectorized None step (plain SGD without momentum)
  inline static void none_step(
    double* values,
    double* grads,
    double lr,
    double clipping_scale,
    size_t n) noexcept
  {
    MYODDWEB_PROFILE_FUNCTION("simd");
    size_t j = 0;
#ifdef SIMD_AVX2_ENABLED
    __m256d vec_clip = _mm256_set1_pd(clipping_scale);
    __m256d vec_lr = _mm256_set1_pd(lr);

    for (; j + 3 < n; j += 4)
    {
      __m256d g = _mm256_loadu_pd(&grads[j]);
      __m256d cur_w = _mm256_loadu_pd(&values[j]);

      __m256d grad = _mm256_mul_pd(g, vec_clip);
      __m256d next_w = _mm256_sub_pd(cur_w, _mm256_mul_pd(vec_lr, grad));

      _mm256_storeu_pd(&values[j], next_w);
      _mm256_storeu_pd(&grads[j], grad);
    }
#endif
    scalar_none_step(values, grads, lr, clipping_scale, n, j);
  }
};
} // namespace myoddweb::nn
