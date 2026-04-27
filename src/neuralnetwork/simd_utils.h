#pragma once

#include "./libraries/instrumentor.h"
#include <cmath>

// Check if AVX2 is available (MSVC, GCC, Clang)
#if defined(__AVX2__)
#include <immintrin.h>
#define SIMD_AVX2_ENABLED
#endif

class simd
{
public:
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
#ifdef __FMA__
      vec_y = _mm256_fmadd_pd(vec_x, vec_w, vec_y);
#else
      vec_y = _mm256_add_pd(vec_y, _mm256_mul_pd(vec_x, vec_w));
#endif
      _mm256_storeu_pd(&y[j], vec_y);
    }
#endif
    for (; j < n; ++j)
    {
      y[j] += x * w[j];
    }
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
#ifdef __FMA__
      vec_sum = _mm256_fmadd_pd(vec_a, vec_b, vec_sum);
#else
      vec_sum = _mm256_add_pd(vec_sum, _mm256_mul_pd(vec_a, vec_b));
#endif
    }
    double sums[4];
    _mm256_storeu_pd(sums, vec_sum);
    total_sum = sums[0] + sums[1] + sums[2] + sums[3];
#endif
    for (; j < n; ++j)
    {
      total_sum += a[j] * b[j];
    }
    return total_sum;
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
    for (; j < n; ++j) 
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
    for (; j < n; ++j)
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
};