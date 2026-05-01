#pragma once

#include "./libraries/instrumentor.h"
#include <algorithm>
#include <cmath>
#include <functional>

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
    for (; j < n; ++j)
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
    for (; j < n; ++j)
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
};