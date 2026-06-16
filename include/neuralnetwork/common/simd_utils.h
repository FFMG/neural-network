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
    double sums[4];
    _mm256_storeu_pd(sums, vec_sum);
    total_sum = sums[0] + sums[1] + sums[2] + sums[3];
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
    for (size_t j = start; j < n; ++j)
    {
      m1[j] = b1 * m1[j] + (1.0 - b1) * grads[j];
      m2[j] = b2 * m2[j] + (1.0 - b2) * (grads[j] * grads[j]);
      double m_hat = (p1 > 1e-15) ? m1[j] / p1 : m1[j];
      double v_hat = (p2 > 1e-15) ? m2[j] / p2 : m2[j];
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
    __m256d vec_b1 = _mm256_set1_pd(b1);
    __m256d vec_one_minus_b1 = _mm256_set1_pd(1.0 - b1);
    __m256d vec_b2 = _mm256_set1_pd(b2);
    __m256d vec_one_minus_b2 = _mm256_set1_pd(1.0 - b2);
    __m256d vec_p1 = _mm256_set1_pd(p1);
    __m256d vec_p2 = _mm256_set1_pd(p2);
    __m256d vec_lr = _mm256_set1_pd(lr);
    __m256d vec_eps = _mm256_set1_pd(epsilon);

    for (; j + 3 < n; j += 4) 
    {
      __m256d g = _mm256_loadu_pd(&grads[j]);
      __m256d cur_m1 = _mm256_loadu_pd(&m1[j]);
      __m256d cur_m2 = _mm256_loadu_pd(&m2[j]);
      __m256d cur_w = _mm256_loadu_pd(&values[j]);

      // Moments update
      __m256d next_m1 = _mm256_add_pd(_mm256_mul_pd(vec_b1, cur_m1), _mm256_mul_pd(vec_one_minus_b1, g));
      __m256d next_m2 = _mm256_add_pd(_mm256_mul_pd(vec_b2, cur_m2), _mm256_mul_pd(vec_one_minus_b2, _mm256_mul_pd(g, g)));
      _mm256_storeu_pd(&m1[j], next_m1);
      _mm256_storeu_pd(&m2[j], next_m2);

      // Adam scaling
      // Safety: p1 and p2 should be > 0. If not (e.g. t=0), skip scaling.
      __m256d m_hat = (p1 > 1e-15) ? _mm256_div_pd(next_m1, vec_p1) : next_m1;
      __m256d v_hat = (p2 > 1e-15) ? _mm256_div_pd(next_m2, vec_p2) : next_m2;

      __m256d update = _mm256_div_pd(m_hat, _mm256_add_pd(_mm256_sqrt_pd(v_hat), vec_eps));

      // Optional AdamW weight decay
      if (decays != nullptr) {
        __m256d d = _mm256_loadu_pd(&decays[j]);
        cur_w = _mm256_mul_pd(cur_w, _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(vec_lr, d)));
      }

      __m256d next_w_raw = _mm256_sub_pd(cur_w, _mm256_mul_pd(vec_lr, update));

      // Hard clamp weights to prevent catastrophic numerical explosion (+/- 1 million)
      __m256d next_w = _mm256_max_pd(_mm256_min_pd(next_w_raw, _mm256_set1_pd(100000.0)), _mm256_set1_pd(-100000.0));
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
    for (size_t j = start; j < n; ++j)
    {
      m1[j] = b1 * m1[j] + (1.0 - b1) * grads[j];
      m2[j] = b2 * m2[j] + (1.0 - b2) * (grads[j] * grads[j]);
      double m_hat = (p1 > 1e-15) ? m1[j] / p1 : m1[j];
      double v_hat = (p2 > 1e-15) ? m2[j] / p2 : m2[j];
      double m_nadam = b1 * m_hat + ((1.0 - b1) * grads[j]) / p1;
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
    __m256d vec_b1 = _mm256_set1_pd(b1);
    __m256d vec_one_minus_b1 = _mm256_set1_pd(1.0 - b1);
    __m256d vec_b2 = _mm256_set1_pd(b2);
    __m256d vec_one_minus_b2 = _mm256_set1_pd(1.0 - b2);
    __m256d vec_p1 = _mm256_set1_pd(p1);
    __m256d vec_p2 = _mm256_set1_pd(p2);
    __m256d vec_lr = _mm256_set1_pd(lr);
    __m256d vec_eps = _mm256_set1_pd(epsilon);

    for (; j + 3 < n; j += 4)
    {
      __m256d g = _mm256_loadu_pd(&grads[j]);
      __m256d cur_m1 = _mm256_loadu_pd(&m1[j]);
      __m256d cur_m2 = _mm256_loadu_pd(&m2[j]);
      __m256d cur_w = _mm256_loadu_pd(&values[j]);

      // Moments update
      __m256d next_m1 = _mm256_add_pd(_mm256_mul_pd(vec_b1, cur_m1), _mm256_mul_pd(vec_one_minus_b1, g));
      __m256d next_m2 = _mm256_add_pd(_mm256_mul_pd(vec_b2, cur_m2), _mm256_mul_pd(vec_one_minus_b2, _mm256_mul_pd(g, g)));
      _mm256_storeu_pd(&m1[j], next_m1);
      _mm256_storeu_pd(&m2[j], next_m2);

      // Nadam scaling
      __m256d m_hat = (p1 > 1e-15) ? _mm256_div_pd(next_m1, vec_p1) : next_m1;
      __m256d v_hat = (p2 > 1e-15) ? _mm256_div_pd(next_m2, vec_p2) : next_m2;

      // m_nadam = beta1 * m_hat + ((1-beta1)*g)/p1
      __m256d m_nadam = _mm256_add_pd(_mm256_mul_pd(vec_b1, m_hat), _mm256_div_pd(_mm256_mul_pd(vec_one_minus_b1, g), vec_p1));
      __m256d update = _mm256_div_pd(m_nadam, _mm256_add_pd(_mm256_sqrt_pd(v_hat), vec_eps));

      // Optional NadamW weight decay
      if (decays != nullptr) 
      {
        __m256d d = _mm256_loadu_pd(&decays[j]);
        cur_w = _mm256_mul_pd(cur_w, _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(vec_lr, d)));
      }

      __m256d next_w_raw = _mm256_sub_pd(cur_w, _mm256_mul_pd(vec_lr, update));

      // Hard clamp weights to prevent catastrophic numerical explosion
      __m256d next_w = _mm256_max_pd(_mm256_min_pd(next_w_raw, _mm256_set1_pd(100000.0)), _mm256_set1_pd(-100000.0));
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
      
      __m256d d_h_prev_direct = _mm256_mul_pd(dh, _mm256_sub_pd(one, z));

      _mm256_storeu_pd(&dz_out[j], d_z_pre);
      _mm256_storeu_pd(&dh_hat_out[j], d_h_hat_pre);
      _mm256_storeu_pd(&dh_prev_accum_out[j], d_h_prev_direct);
    }
#endif
    scalar_gru_bptt_gate_step(n, grad_next, d_next_h, z_vals, h_hat_vals, h_prev_vals, h_hat_pre_vals, mask_vals, dz_out, dh_hat_out, dh_prev_accum_out, h_hat_pre_deriv_vals, j);
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
      __m256d dc = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(dh, o_gate), dc_deriv), dc_nxt);

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

      double sums0[4], sums1[4], sums2[4], sums3[4];
      _mm256_storeu_pd(sums0, vec_sum0);
      _mm256_storeu_pd(sums1, vec_sum1);
      _mm256_storeu_pd(sums2, vec_sum2);
      _mm256_storeu_pd(sums3, vec_sum3);

      double sum0 = sums0[0] + sums0[1] + sums0[2] + sums0[3];
      double sum1 = sums1[0] + sums1[1] + sums1[2] + sums1[3];
      double sum2 = sums2[0] + sums2[1] + sums2[2] + sums2[3];
      double sum3 = sums3[0] + sums3[1] + sums3[2] + sums3[3];

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
      double sums[4];
      _mm256_storeu_pd(sums, vec_sum);
      sum = sums[0] + sums[1] + sums[2] + sums[3];
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

      double sums0_0[4], sums0_1[4], sums1_0[4], sums1_1[4];
      _mm256_storeu_pd(sums0_0, vec_sum0_0);
      _mm256_storeu_pd(sums0_1, vec_sum0_1);
      _mm256_storeu_pd(sums1_0, vec_sum1_0);
      _mm256_storeu_pd(sums1_1, vec_sum1_1);

      double sum0_0 = sums0_0[0] + sums0_0[1] + sums0_0[2] + sums0_0[3];
      double sum0_1 = sums0_1[0] + sums0_1[1] + sums0_1[2] + sums0_1[3];
      double sum1_0 = sums1_0[0] + sums1_0[1] + sums1_0[2] + sums1_0[3];
      double sum1_1 = sums1_1[0] + sums1_1[1] + sums1_1[2] + sums1_1[3];

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
      double sums0[4];
      double sums1[4];
      _mm256_storeu_pd(sums0, vec_sum0);
      _mm256_storeu_pd(sums1, vec_sum1);
      sum0 = sums0[0] + sums0[1] + sums0[2] + sums0[3];
      sum1 = sums1[0] + sums1[1] + sums1[2] + sums1[3];
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

      double sums0_0[4], sums0_1[4], sums1_0[4], sums1_1[4];
      double sums2_0[4], sums2_1[4], sums3_0[4], sums3_1[4];
      _mm256_storeu_pd(sums0_0, vec_sum0_0);
      _mm256_storeu_pd(sums0_1, vec_sum0_1);
      _mm256_storeu_pd(sums1_0, vec_sum1_0);
      _mm256_storeu_pd(sums1_1, vec_sum1_1);
      _mm256_storeu_pd(sums2_0, vec_sum2_0);
      _mm256_storeu_pd(sums2_1, vec_sum2_1);
      _mm256_storeu_pd(sums3_0, vec_sum3_0);
      _mm256_storeu_pd(sums3_1, vec_sum3_1);

      double sum0_0 = sums0_0[0] + sums0_0[1] + sums0_0[2] + sums0_0[3];
      double sum0_1 = sums0_1[0] + sums0_1[1] + sums0_1[2] + sums0_1[3];
      double sum1_0 = sums1_0[0] + sums1_0[1] + sums1_0[2] + sums1_0[3];
      double sum1_1 = sums1_1[0] + sums1_1[1] + sums1_1[2] + sums1_1[3];
      double sum2_0 = sums2_0[0] + sums2_0[1] + sums2_0[2] + sums2_0[3];
      double sum2_1 = sums2_1[0] + sums2_1[1] + sums2_1[2] + sums2_1[3];
      double sum3_0 = sums3_0[0] + sums3_0[1] + sums3_0[2] + sums3_0[3];
      double sum3_1 = sums3_1[0] + sums3_1[1] + sums3_1[2] + sums3_1[3];

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
      double sums0[4];
      double sums1[4];
      double sums2[4];
      double sums3[4];
      _mm256_storeu_pd(sums0, vec_sum0);
      _mm256_storeu_pd(sums1, vec_sum1);
      _mm256_storeu_pd(sums2, vec_sum2);
      _mm256_storeu_pd(sums3, vec_sum3);
      sum0 = sums0[0] + sums0[1] + sums0[2] + sums0[3];
      sum1 = sums1[0] + sums1[1] + sums1[2] + sums1[3];
      sum2 = sums2[0] + sums2[1] + sums2[2] + sums2[3];
      sum3 = sums3[0] + sums3[1] + sums3[2] + sums3[3];
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
};
} // namespace myoddweb::nn
