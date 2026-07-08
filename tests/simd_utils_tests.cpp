#include <gtest/gtest.h>
#include "common/simd_utils.h"
#include "layers/residualprojector.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
namespace {
  constexpr double EPSILON = 1e-9;

  // Helper to compare vectors
  void expect_vec_near(const std::vector<double>& actual, const std::vector<double>& expected, double tol = EPSILON) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], tol) << "at index " << i;
    }
  }
}

TEST(SimdUtilsTest, ScalarMulAdd) {
  const double x = 2.5;
  std::vector<double> w = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> y = { 10.0, 20.0, 30.0, 40.0, 50.0 };
  std::vector<double> expected_y(y.size());

  for (size_t i = 0; i < y.size(); ++i) {
    expected_y[i] = y[i] + x * w[i];
  }

  simd::scalar_mul_add(x, w.data(), y.data(), y.size());

  expect_vec_near(y, expected_y);
}

TEST(SimdUtilsTest, ScalarDotProduct) {
  std::vector<double> a = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> b = { 0.5, 1.5, 2.5, 3.5, 4.5 };
  double expected_dot = 0.0;

  for (size_t i = 0; i < a.size(); ++i) {
    expected_dot += a[i] * b[i];
  }

  double actual_dot = simd::scalar_dot_product(a.data(), b.data(), a.size());

  EXPECT_NEAR(actual_dot, expected_dot, EPSILON);
}

TEST(SimdUtilsTest, ScalarAdamStep) {
  const size_t n = 5;
  std::vector<double> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> grads = { 0.1, -0.2, 0.3, -0.4, 0.5 };
  std::vector<double> m1 = { 0.01, 0.02, 0.03, 0.04, 0.05 };
  std::vector<double> m2 = { 0.001, 0.002, 0.003, 0.004, 0.005 };
  std::vector<double> decays = { 0.01, 0.01, 0.01, 0.01, 0.01 };

  double b1 = 0.9;
  double b2 = 0.999;
  double p1 = 0.8;
  double p2 = 0.7;
  double lr = 0.001;
  double eps = 1e-8;

  std::vector<double> expected_values = values;
  std::vector<double> expected_m1 = m1;
  std::vector<double> expected_m2 = m2;

  for (size_t i = 0; i < n; ++i) {
    expected_m1[i] = b1 * expected_m1[i] + (1.0 - b1) * grads[i];
    expected_m2[i] = b2 * expected_m2[i] + (1.0 - b2) * (grads[i] * grads[i]);
    double m_hat = (p1 > 1e-15) ? expected_m1[i] / p1 : expected_m1[i];
    double v_hat = (p2 > 1e-15) ? expected_m2[i] / p2 : expected_m2[i];
    double update = m_hat / (std::sqrt(v_hat) + eps);
    double w = expected_values[i];
    w *= (1.0 - lr * decays[i]);
    expected_values[i] = std::clamp(w - lr * update, -100000.0, 100000.0);
  }

  simd::scalar_adam_step(values.data(), grads.data(), m1.data(), m2.data(), b1, b2, p1, p2, lr, eps, n, decays.data());

  expect_vec_near(m1, expected_m1);
  expect_vec_near(m2, expected_m2);
  expect_vec_near(values, expected_values);
}

TEST(SimdUtilsTest, ScalarNadamStep) {
  const size_t n = 5;
  std::vector<double> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> grads = { 0.1, -0.2, 0.3, -0.4, 0.5 };
  std::vector<double> m1 = { 0.01, 0.02, 0.03, 0.04, 0.05 };
  std::vector<double> m2 = { 0.001, 0.002, 0.003, 0.004, 0.005 };
  std::vector<double> decays = { 0.01, 0.01, 0.01, 0.01, 0.01 };

  double b1 = 0.9;
  double b2 = 0.999;
  double p1 = 0.8;
  double p2 = 0.7;
  double lr = 0.001;
  double eps = 1e-8;

  std::vector<double> expected_values = values;
  std::vector<double> expected_m1 = m1;
  std::vector<double> expected_m2 = m2;

  for (size_t i = 0; i < n; ++i) {
    expected_m1[i] = b1 * expected_m1[i] + (1.0 - b1) * grads[i];
    expected_m2[i] = b2 * expected_m2[i] + (1.0 - b2) * (grads[i] * grads[i]);
    double m_hat = (p1 > 1e-15) ? expected_m1[i] / p1 : expected_m1[i];
    double v_hat = (p2 > 1e-15) ? expected_m2[i] / p2 : expected_m2[i];
    double m_nadam = b1 * m_hat + ((1.0 - b1) * grads[i]) / p1;
    double update = m_nadam / (std::sqrt(v_hat) + eps);
    double w = expected_values[i];
    w *= (1.0 - lr * decays[i]);
    expected_values[i] = std::clamp(w - lr * update, -100000.0, 100000.0);
  }

  simd::scalar_nadam_step(values.data(), grads.data(), m1.data(), m2.data(), b1, b2, p1, p2, lr, eps, n, decays.data());

  expect_vec_near(m1, expected_m1);
  expect_vec_near(m2, expected_m2);
  expect_vec_near(values, expected_values);
}

TEST(SimdUtilsTest, ScalarGruBpttGateStep) {
  const size_t n = 5;
  std::vector<double> grad_next = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> d_next_h = { 0.05, 0.06, 0.07, 0.08, 0.09 };
  std::vector<double> z_vals = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> h_hat_vals = { 0.6, 0.7, 0.8, 0.9, 1.0 };
  std::vector<double> h_prev_vals = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> h_hat_pre_vals = { 0.5, 0.4, 0.3, 0.2, 0.1 };
  std::vector<double> mask_vals = { 1.0, 2.0, 0.0, 1.0, 1.0 };

  std::vector<double> dz_out(n);
  std::vector<double> dh_hat_out(n);
  std::vector<double> dh_prev_accum_out(n);

  auto activate_derivative = [](double x) { return 1.0 - std::tanh(x) * std::tanh(x); };

  std::vector<double> expected_dz(n);
  std::vector<double> expected_dh_hat(n);
  std::vector<double> expected_dh_prev(n);

  for (size_t i = 0; i < n; ++i) {
    double dh = std::clamp(grad_next[i] + d_next_h[i], -50.0, 50.0);
    double h_hat_final = h_hat_vals[i] * mask_vals[i];
    expected_dz[i] = dh * (h_hat_final - h_prev_vals[i]) * z_vals[i] * (1.0 - z_vals[i]);
    expected_dh_hat[i] = dh * z_vals[i] * activate_derivative(h_hat_pre_vals[i]) * mask_vals[i];
    expected_dh_prev[i] = dh * (1.0 - z_vals[i]);
  }

  std::vector<double> dh_hat_pre_deriv(n);
  for (size_t i = 0; i < n; ++i) {
    dh_hat_pre_deriv[i] = activate_derivative(h_hat_pre_vals[i]);
  }

  simd::scalar_gru_bptt_gate_step(n, grad_next.data(), d_next_h.data(), z_vals.data(), h_hat_vals.data(), h_prev_vals.data(), h_hat_pre_vals.data(), mask_vals.data(), dz_out.data(), dh_hat_out.data(), dh_prev_accum_out.data(), dh_hat_pre_deriv.data());

  expect_vec_near(dz_out, expected_dz);
  expect_vec_near(dh_hat_out, expected_dh_hat);
  expect_vec_near(dh_prev_accum_out, expected_dh_prev);
}

TEST(SimdUtilsTest, ScalarGruBpttResetStep) {
  const size_t n = 5;
  std::vector<double> temp_Uh = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> h_prev_vals = { 0.15, 0.25, 0.35, 0.45, 0.55 };
  std::vector<double> r_vals = { 0.6, 0.7, 0.8, 0.9, 1.0 };
  std::vector<double> dh_prev_accum = { 0.05, 0.06, 0.07, 0.08, 0.09 };

  std::vector<double> dr_out(n);
  std::vector<double> dh_next_out(n);

  std::vector<double> expected_dr(n);
  std::vector<double> expected_dh_next(n);

  for (size_t i = 0; i < n; ++i) {
    expected_dr[i] = temp_Uh[i] * h_prev_vals[i] * r_vals[i] * (1.0 - r_vals[i]);
    expected_dh_next[i] = dh_prev_accum[i] + temp_Uh[i] * r_vals[i];
  }

  simd::scalar_gru_bptt_reset_step(n, temp_Uh.data(), h_prev_vals.data(), r_vals.data(), dh_prev_accum.data(), dr_out.data(), dh_next_out.data());

  expect_vec_near(dr_out, expected_dr);
  expect_vec_near(dh_next_out, expected_dh_next);
}

TEST(SimdUtilsTest, ScalarLstmBpttGateStep) {
  const size_t n = 5;
  std::vector<double> dh_curr = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> dc_next = { 0.05, 0.06, 0.07, 0.08, 0.09 };
  std::vector<double> f = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> i_gate = { 0.2, 0.3, 0.4, 0.5, 0.6 };
  std::vector<double> o = { 0.3, 0.4, 0.5, 0.6, 0.7 };
  std::vector<double> g_pre = { 0.4, 0.5, 0.6, 0.7, 0.8 };
  std::vector<double> c_curr = { 0.5, 0.6, 0.7, 0.8, 0.9 };
  std::vector<double> c_prev = { 0.6, 0.7, 0.8, 0.9, 1.0 };

  std::vector<double> df_out(n);
  std::vector<double> di_out(n);
  std::vector<double> do_out(n);
  std::vector<double> dg_out(n);
  std::vector<double> dc_next_out(n);

  std::vector<double> expected_df(n);
  std::vector<double> expected_di(n);
  std::vector<double> expected_do(n);
  std::vector<double> expected_dg(n);
  std::vector<double> expected_dc_next(n);

  std::vector<double> tanh_c_vals(n);
  std::vector<double> g_act_vals(n);

  for (size_t j = 0; j < n; ++j) {
    double dh = dh_curr[j];
    double tanh_c = std::tanh(c_curr[j]);
    tanh_c_vals[j] = tanh_c;
    double do_gate = dh * tanh_c * o[j] * (1.0 - o[j]);
    double dc = dh * o[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
    double g = std::tanh(g_pre[j]);
    g_act_vals[j] = g;
    expected_df[j] = dc * (c_prev[j]) * f[j] * (1.0 - f[j]);
    expected_di[j] = dc * g * i_gate[j] * (1.0 - i_gate[j]);
    expected_do[j] = do_gate;
    expected_dg[j] = dc * i_gate[j] * (1.0 - g * g);
    expected_dc_next[j] = dc * f[j];
  }

  std::vector<double> dc_act_deriv(n);
  std::vector<double> dg_act_deriv(n);
  for (size_t j = 0; j < n; ++j) {
    dc_act_deriv[j] = 1.0 - tanh_c_vals[j] * tanh_c_vals[j];
    dg_act_deriv[j] = 1.0 - g_act_vals[j] * g_act_vals[j];
  }

  simd::scalar_lstm_bptt_gate_step(
    n,
    dh_curr.data(),
    dc_next.data(),
    f.data(),
    i_gate.data(),
    o.data(),
    g_pre.data(),
    g_act_vals.data(),
    tanh_c_vals.data(),
    c_prev.data(),
    true,
    df_out.data(),
    di_out.data(),
    do_out.data(),
    dg_out.data(),
    dc_next_out.data(),
    dc_act_deriv.data(),
    dg_act_deriv.data()
  );

  expect_vec_near(df_out, expected_df);
  expect_vec_near(di_out, expected_di);
  expect_vec_near(do_out, expected_do);
  expect_vec_near(dg_out, expected_dg);
  expect_vec_near(dc_next_out, expected_dc_next);
}

TEST(SimdUtilsTest, MulAdd) {
  const double x = 2.5;
  std::vector<double> w = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> y = { 10.0, 20.0, 30.0, 40.0, 50.0 };
  std::vector<double> expected_y(y.size());

  for (size_t i = 0; i < y.size(); ++i) {
    expected_y[i] = y[i] + x * w[i];
  }

  simd::mul_add(x, w.data(), y.data(), y.size());

  expect_vec_near(y, expected_y);
}

TEST(SimdUtilsTest, DotProduct) {
  std::vector<double> a = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> b = { 0.5, 1.5, 2.5, 3.5, 4.5 };
  double expected_dot = 0.0;

  for (size_t i = 0; i < a.size(); ++i) {
    expected_dot += a[i] * b[i];
  }

  double actual_dot = simd::dot_product(a.data(), b.data(), a.size());

  EXPECT_NEAR(actual_dot, expected_dot, EPSILON);
}

TEST(SimdUtilsTest, AdamStep) {
  const size_t n = 5;
  std::vector<double> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> grads = { 0.1, -0.2, 0.3, -0.4, 0.5 };
  std::vector<double> m1 = { 0.01, 0.02, 0.03, 0.04, 0.05 };
  std::vector<double> m2 = { 0.001, 0.002, 0.003, 0.004, 0.005 };
  std::vector<double> decays = { 0.01, 0.01, 0.01, 0.01, 0.01 };
  
  double b1 = 0.9;
  double b2 = 0.999;
  double p1 = 0.8;
  double p2 = 0.7;
  double lr = 0.001;
  double eps = 1e-8;

  std::vector<double> expected_values = values;
  std::vector<double> expected_m1 = m1;
  std::vector<double> expected_m2 = m2;

  for (size_t i = 0; i < n; ++i) {
    expected_m1[i] = b1 * expected_m1[i] + (1.0 - b1) * grads[i];
    expected_m2[i] = b2 * expected_m2[i] + (1.0 - b2) * (grads[i] * grads[i]);
    double m_hat = (p1 > 1e-15) ? expected_m1[i] / p1 : expected_m1[i];
    double v_hat = (p2 > 1e-15) ? expected_m2[i] / p2 : expected_m2[i];
    double update = m_hat / (std::sqrt(v_hat) + eps);
    double w = expected_values[i];
    w *= (1.0 - lr * decays[i]);
    expected_values[i] = std::clamp(w - lr * update, -100000.0, 100000.0);
  }

  simd::adam_step(values.data(), grads.data(), m1.data(), m2.data(), b1, b2, p1, p2, lr, eps, n, decays.data());

  expect_vec_near(m1, expected_m1);
  expect_vec_near(m2, expected_m2);
  expect_vec_near(values, expected_values);
}

TEST(SimdUtilsTest, NadamStep) {
  const size_t n = 5;
  std::vector<double> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> grads = { 0.1, -0.2, 0.3, -0.4, 0.5 };
  std::vector<double> m1 = { 0.01, 0.02, 0.03, 0.04, 0.05 };
  std::vector<double> m2 = { 0.001, 0.002, 0.003, 0.004, 0.005 };
  std::vector<double> decays = { 0.01, 0.01, 0.01, 0.01, 0.01 };

  double b1 = 0.9;
  double b2 = 0.999;
  double p1 = 0.8;
  double p2 = 0.7;
  double lr = 0.001;
  double eps = 1e-8;

  std::vector<double> expected_values = values;
  std::vector<double> expected_m1 = m1;
  std::vector<double> expected_m2 = m2;

  for (size_t i = 0; i < n; ++i) {
    expected_m1[i] = b1 * expected_m1[i] + (1.0 - b1) * grads[i];
    expected_m2[i] = b2 * expected_m2[i] + (1.0 - b2) * (grads[i] * grads[i]);
    double m_hat = (p1 > 1e-15) ? expected_m1[i] / p1 : expected_m1[i];
    double v_hat = (p2 > 1e-15) ? expected_m2[i] / p2 : expected_m2[i];
    double m_nadam = b1 * m_hat + ((1.0 - b1) * grads[i]) / p1;
    double update = m_nadam / (std::sqrt(v_hat) + eps);
    double w = expected_values[i];
    w *= (1.0 - lr * decays[i]);
    expected_values[i] = std::clamp(w - lr * update, -100000.0, 100000.0);
  }

  simd::nadam_step(values.data(), grads.data(), m1.data(), m2.data(), b1, b2, p1, p2, lr, eps, n, decays.data());

  expect_vec_near(m1, expected_m1);
  expect_vec_near(m2, expected_m2);
  expect_vec_near(values, expected_values);
}

TEST(SimdUtilsTest, AdamStepWithClipping)
{
  const size_t n = 5;
  std::vector<double> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> grads = { 0.1, -0.2, 0.3, -0.4, 0.5 };
  std::vector<double> m1 = { 0.01, 0.02, 0.03, 0.04, 0.05 };
  std::vector<double> m2 = { 0.001, 0.002, 0.003, 0.004, 0.005 };
  std::vector<double> decays = { 0.01, 0.01, 0.01, 0.01, 0.01 };
  
  double b1 = 0.9;
  double b2 = 0.999;
  double p1 = 0.8;
  double p2 = 0.7;
  double lr = 0.001;
  double eps = 1e-8;
  double clipping_scale = 0.5;

  std::vector<double> expected_values = values;
  std::vector<double> expected_m1 = m1;
  std::vector<double> expected_m2 = m2;

  for (size_t i = 0; i < n; ++i)
  {
    double clipped_grad = grads[i] * clipping_scale;
    expected_m1[i] = b1 * expected_m1[i] + (1.0 - b1) * clipped_grad;
    expected_m2[i] = b2 * expected_m2[i] + (1.0 - b2) * (clipped_grad * clipped_grad);
    double m_hat = (p1 > 1e-15) ? expected_m1[i] / p1 : expected_m1[i];
    double v_hat = (p2 > 1e-15) ? expected_m2[i] / p2 : expected_m2[i];
    double update = m_hat / (std::sqrt(v_hat) + eps);
    double w = expected_values[i];
    w *= (1.0 - lr * decays[i]);
    expected_values[i] = std::clamp(w - lr * update, -100000.0, 100000.0);
  }

  simd::adam_step(values.data(), grads.data(), m1.data(), m2.data(), b1, b2, p1, p2, lr, eps, n, decays.data(), clipping_scale);

  expect_vec_near(m1, expected_m1);
  expect_vec_near(m2, expected_m2);
  expect_vec_near(values, expected_values);
}

TEST(SimdUtilsTest, NadamStepWithClipping)
{
  const size_t n = 5;
  std::vector<double> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> grads = { 0.1, -0.2, 0.3, -0.4, 0.5 };
  std::vector<double> m1 = { 0.01, 0.02, 0.03, 0.04, 0.05 };
  std::vector<double> m2 = { 0.001, 0.002, 0.003, 0.004, 0.005 };
  std::vector<double> decays = { 0.01, 0.01, 0.01, 0.01, 0.01 };

  double b1 = 0.9;
  double b2 = 0.999;
  double p1 = 0.8;
  double p2 = 0.7;
  double lr = 0.001;
  double eps = 1e-8;
  double clipping_scale = 0.5;

  std::vector<double> expected_values = values;
  std::vector<double> expected_m1 = m1;
  std::vector<double> expected_m2 = m2;

  for (size_t i = 0; i < n; ++i)
  {
    double clipped_grad = grads[i] * clipping_scale;
    expected_m1[i] = b1 * expected_m1[i] + (1.0 - b1) * clipped_grad;
    expected_m2[i] = b2 * expected_m2[i] + (1.0 - b2) * (clipped_grad * clipped_grad);
    double m_hat = (p1 > 1e-15) ? expected_m1[i] / p1 : expected_m1[i];
    double v_hat = (p2 > 1e-15) ? expected_m2[i] / p2 : expected_m2[i];
    double m_nadam = b1 * m_hat + ((1.0 - b1) * clipped_grad) / p1;
    double update = m_nadam / (std::sqrt(v_hat) + eps);
    double w = expected_values[i];
    w *= (1.0 - lr * decays[i]);
    expected_values[i] = std::clamp(w - lr * update, -100000.0, 100000.0);
  }

  simd::nadam_step(values.data(), grads.data(), m1.data(), m2.data(), b1, b2, p1, p2, lr, eps, n, decays.data(), clipping_scale);

  expect_vec_near(m1, expected_m1);
  expect_vec_near(m2, expected_m2);
  expect_vec_near(values, expected_values);
}

TEST(SimdUtilsTest, GruBpttGateStep) {
  const size_t n = 5;
  std::vector<double> grad_next = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> d_next_h = { 0.05, 0.06, 0.07, 0.08, 0.09 };
  std::vector<double> z_vals = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> h_hat_vals = { 0.6, 0.7, 0.8, 0.9, 1.0 };
  std::vector<double> h_prev_vals = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> h_hat_pre_vals = { 0.5, 0.4, 0.3, 0.2, 0.1 };

  std::vector<double> mask_vals = { 1.0, 2.0, 0.0, 1.0, 1.0 }; // Test various masks

  std::vector<double> dz_out(n);
  std::vector<double> dh_hat_out(n);
  std::vector<double> dh_prev_accum_out(n);

  auto activate_derivative = [](double x) { return 1.0 - std::tanh(x) * std::tanh(x); };

  std::vector<double> expected_dz(n);
  std::vector<double> expected_dh_hat(n);
  std::vector<double> expected_dh_prev(n);

  for (size_t i = 0; i < n; ++i) {
    double dh = std::clamp(grad_next[i] + d_next_h[i], -50.0, 50.0);
    double h_hat_final = h_hat_vals[i] * mask_vals[i];
    expected_dz[i] = dh * (h_hat_final - h_prev_vals[i]) * z_vals[i] * (1.0 - z_vals[i]);
    expected_dh_hat[i] = dh * z_vals[i] * activate_derivative(h_hat_pre_vals[i]) * mask_vals[i];
    expected_dh_prev[i] = dh * (1.0 - z_vals[i]);
  }

  std::vector<double> dh_hat_pre_deriv(n);
  for (size_t i = 0; i < n; ++i) {
    dh_hat_pre_deriv[i] = activate_derivative(h_hat_pre_vals[i]);
  }

  simd::gru_bptt_gate_step(n, grad_next.data(), d_next_h.data(), z_vals.data(), h_hat_vals.data(), h_prev_vals.data(), h_hat_pre_vals.data(), mask_vals.data(), dz_out.data(), dh_hat_out.data(), dh_prev_accum_out.data(), dh_hat_pre_deriv.data());

  expect_vec_near(dz_out, expected_dz);
  expect_vec_near(dh_hat_out, expected_dh_hat);
  expect_vec_near(dh_prev_accum_out, expected_dh_prev);
}

TEST(SimdUtilsTest, GruBpttResetStep) {
  const size_t n = 5;
  std::vector<double> temp_Uh = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> h_prev_vals = { 0.15, 0.25, 0.35, 0.45, 0.55 };
  std::vector<double> r_vals = { 0.6, 0.7, 0.8, 0.9, 1.0 };
  std::vector<double> dh_prev_accum = { 0.05, 0.06, 0.07, 0.08, 0.09 };

  std::vector<double> dr_out(n);
  std::vector<double> dh_next_out(n);

  std::vector<double> expected_dr(n);
  std::vector<double> expected_dh_next(n);

  for (size_t i = 0; i < n; ++i) {
    expected_dr[i] = temp_Uh[i] * h_prev_vals[i] * r_vals[i] * (1.0 - r_vals[i]);
    expected_dh_next[i] = dh_prev_accum[i] + temp_Uh[i] * r_vals[i];
  }

  simd::gru_bptt_reset_step(n, temp_Uh.data(), h_prev_vals.data(), r_vals.data(), dh_prev_accum.data(), dr_out.data(), dh_next_out.data());

  expect_vec_near(dr_out, expected_dr);
  expect_vec_near(dh_next_out, expected_dh_next);
}

TEST(SimdUtilsTest, GruBpttResetStepNullPrev) {
  const size_t n = 5;
  std::vector<double> temp_Uh = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> r_vals = { 0.6, 0.7, 0.8, 0.9, 1.0 };
  std::vector<double> dh_prev_accum = { 0.05, 0.06, 0.07, 0.08, 0.09 };

  std::vector<double> dr_out(n);
  std::vector<double> dh_next_out(n);

  std::vector<double> expected_dr(n, 0.0);
  std::vector<double> expected_dh_next(n);

  for (size_t i = 0; i < n; ++i) {
    expected_dh_next[i] = dh_prev_accum[i] + temp_Uh[i] * r_vals[i];
  }

  simd::gru_bptt_reset_step(n, temp_Uh.data(), nullptr, r_vals.data(), dh_prev_accum.data(), dr_out.data(), dh_next_out.data());

  expect_vec_near(dr_out, expected_dr);
  expect_vec_near(dh_next_out, expected_dh_next);
}

TEST(SimdUtilsTest, LstmBpttGateStep) {
  const size_t n = 5;
  std::vector<double> dh_curr = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> dc_next = { 0.05, 0.06, 0.07, 0.08, 0.09 };
  std::vector<double> f = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> i_gate = { 0.2, 0.3, 0.4, 0.5, 0.6 };
  std::vector<double> o = { 0.3, 0.4, 0.5, 0.6, 0.7 };
  std::vector<double> g_pre = { 0.4, 0.5, 0.6, 0.7, 0.8 };
  std::vector<double> c_curr = { 0.5, 0.6, 0.7, 0.8, 0.9 };
  std::vector<double> c_prev = { 0.6, 0.7, 0.8, 0.9, 1.0 };

  std::vector<double> df_out(n);
  std::vector<double> di_out(n);
  std::vector<double> do_out(n);
  std::vector<double> dg_out(n);
  std::vector<double> dc_next_out(n);

  std::vector<double> expected_df(n);
  std::vector<double> expected_di(n);
  std::vector<double> expected_do(n);
  std::vector<double> expected_dg(n);
  std::vector<double> expected_dc_next(n);

  std::vector<double> tanh_c_vals(n);
  std::vector<double> g_act_vals(n);

  for (size_t j = 0; j < n; ++j) {
    double dh = dh_curr[j];
    double tanh_c = std::tanh(c_curr[j]);
    tanh_c_vals[j] = tanh_c;
    double do_gate = dh * tanh_c * o[j] * (1.0 - o[j]);
    double dc = dh * o[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
    double g = std::tanh(g_pre[j]);
    g_act_vals[j] = g;
    expected_df[j] = dc * c_prev[j] * f[j] * (1.0 - f[j]);
    expected_di[j] = dc * g * i_gate[j] * (1.0 - i_gate[j]);
    expected_do[j] = do_gate;
    expected_dg[j] = dc * i_gate[j] * (1.0 - g * g);
    expected_dc_next[j] = dc * f[j];
  }

  std::vector<double> dc_act_deriv(n);
  std::vector<double> dg_act_deriv(n);
  for (size_t j = 0; j < n; ++j) {
    dc_act_deriv[j] = 1.0 - tanh_c_vals[j] * tanh_c_vals[j];
    dg_act_deriv[j] = 1.0 - g_act_vals[j] * g_act_vals[j];
  }

  simd::lstm_bptt_gate_step(
    n, 
    dh_curr.data(), 
    dc_next.data(), 
    f.data(), 
    i_gate.data(), 
    o.data(), 
    g_pre.data(), // Pass pre-activation g
    g_act_vals.data(), // Pass activated g
    tanh_c_vals.data(), 
    c_prev.data(), 
    true, 
    df_out.data(), 
    di_out.data(), 
    do_out.data(), 
    dg_out.data(), 
    dc_next_out.data(),
    dc_act_deriv.data(),
    dg_act_deriv.data()
  );

  expect_vec_near(df_out, expected_df);
  expect_vec_near(di_out, expected_di);
  expect_vec_near(do_out, expected_do);
  expect_vec_near(dg_out, expected_dg);
  expect_vec_near(dc_next_out, expected_dc_next);
}

TEST(SimdUtilsTest, VariousSizes) {
  // Test sizes: 0, 1, 3, 4, 7, 8, 15, 16
  std::vector<size_t> sizes = { 0, 1, 3, 4, 7, 8, 15, 16 };
  for (size_t n : sizes) {
    std::vector<double> a(n, 1.1);
    std::vector<double> b(n, 2.2);
    std::vector<double> y(n, 3.3);
    double x = 0.5;

    std::vector<double> expected_y(n);
    double expected_dot = 0.0;
    for (size_t i = 0; i < n; ++i) {
      expected_y[i] = y[i] + x * a[i];
      expected_dot += a[i] * b[i];
    }

    simd::mul_add(x, a.data(), y.data(), n);
    expect_vec_near(y, expected_y);

    double actual_dot = simd::dot_product(a.data(), b.data(), n);
    EXPECT_NEAR(actual_dot, expected_dot, EPSILON) << "at size " << n;
  }
}

TEST(SimdUtilsTest, GemvAdd)
{
  std::vector<size_t> rows_list = { 1, 2, 3, 4, 7, 8 };
  std::vector<size_t> cols_list = { 1, 2, 3, 4, 7, 8, 15, 16 };
  for (size_t rows : rows_list)
  {
    for (size_t cols : cols_list)
    {
      std::vector<double> A(rows * cols);
      for (size_t i = 0; i < A.size(); ++i)
      {
        A[i] = static_cast<double>(i) * 0.1;
      }
      std::vector<double> x(cols);
      for (size_t i = 0; i < cols; ++i)
      {
        x[i] = static_cast<double>(i) * 0.5 + 1.0;
      }
      std::vector<double> y(rows, 2.0);
      std::vector<double> expected_y = y;

      // Calculate expected
      for (size_t i = 0; i < rows; ++i)
      {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j)
        {
          sum += A[i * cols + j] * x[j];
        }
        expected_y[i] += sum;
      }

      simd::gemv_add(A.data(), x.data(), y.data(), rows, cols);
      expect_vec_near(y, expected_y);
    }
  }
}

TEST(SimdUtilsTest, GemvAccumulateTwo)
{
  std::vector<size_t> rows_list = { 1, 2, 3, 4, 7, 8 };
  std::vector<size_t> cols_list = { 1, 2, 3, 4, 7, 8, 15, 16 };
  for (size_t rows : rows_list)
  {
    for (size_t cols : cols_list)
    {
      std::vector<double> A0(rows * cols);
      std::vector<double> A1(rows * cols);
      for (size_t i = 0; i < A0.size(); ++i)
      {
        A0[i] = static_cast<double>(i) * 0.1;
        A1[i] = static_cast<double>(i) * -0.05 + 0.2;
      }
      std::vector<double> x0(cols);
      std::vector<double> x1(cols);
      for (size_t i = 0; i < cols; ++i)
      {
        x0[i] = static_cast<double>(i) * 0.5 + 1.0;
        x1[i] = static_cast<double>(i) * -0.3 + 0.8;
      }
      std::vector<double> y(rows, 2.0);
      std::vector<double> expected_y = y;

      // Calculate expected
      for (size_t i = 0; i < rows; ++i)
      {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j)
        {
          sum += A0[i * cols + j] * x0[j] + A1[i * cols + j] * x1[j];
        }
        expected_y[i] += sum;
      }

      simd::gemv_accumulate_two(A0.data(), A1.data(), x0.data(), x1.data(), y.data(), rows, cols);
      expect_vec_near(y, expected_y);
    }
  }
}

TEST(SimdUtilsTest, GemvAccumulateThree)
{
  std::vector<size_t> rows_list = { 1, 2, 3, 4, 7, 8 };
  std::vector<size_t> cols_list = { 1, 2, 3, 4, 7, 8, 15, 16 };
  for (size_t rows : rows_list)
  {
    for (size_t cols : cols_list)
    {
      std::vector<double> A0(rows * cols);
      std::vector<double> A1(rows * cols);
      std::vector<double> A2(rows * cols);
      for (size_t i = 0; i < A0.size(); ++i)
      {
        A0[i] = static_cast<double>(i) * 0.1;
        A1[i] = static_cast<double>(i) * -0.05 + 0.2;
        A2[i] = static_cast<double>(i) * 0.08 - 0.1;
      }
      std::vector<double> x0(cols);
      std::vector<double> x1(cols);
      std::vector<double> x2(cols);
      for (size_t i = 0; i < cols; ++i)
      {
        x0[i] = static_cast<double>(i) * 0.5 + 1.0;
        x1[i] = static_cast<double>(i) * -0.3 + 0.8;
        x2[i] = static_cast<double>(i) * 0.25 - 0.4;
      }
      std::vector<double> y(rows, 2.0);
      std::vector<double> expected_y = y;

      // Calculate expected
      for (size_t i = 0; i < rows; ++i)
      {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j)
        {
          sum += A0[i * cols + j] * x0[j] + A1[i * cols + j] * x1[j] + A2[i * cols + j] * x2[j];
        }
        expected_y[i] += sum;
      }

      simd::gemv_accumulate_three(A0.data(), A1.data(), A2.data(), x0.data(), x1.data(), x2.data(), y.data(), rows, cols);
      expect_vec_near(y, expected_y);
    }
  }
}

TEST(SimdUtilsTest, GemvAccumulateFour)
{
  std::vector<size_t> rows_list = { 1, 2, 3, 4, 7, 8 };
  std::vector<size_t> cols_list = { 1, 2, 3, 4, 7, 8, 15, 16 };
  for (size_t rows : rows_list)
  {
    for (size_t cols : cols_list)
    {
      std::vector<double> A0(rows * cols);
      std::vector<double> A1(rows * cols);
      std::vector<double> A2(rows * cols);
      std::vector<double> A3(rows * cols);
      for (size_t i = 0; i < A0.size(); ++i)
      {
        A0[i] = static_cast<double>(i) * 0.1;
        A1[i] = static_cast<double>(i) * -0.05 + 0.2;
        A2[i] = static_cast<double>(i) * 0.08 - 0.1;
        A3[i] = static_cast<double>(i) * -0.12 + 0.15;
      }
      std::vector<double> x0(cols);
      std::vector<double> x1(cols);
      std::vector<double> x2(cols);
      std::vector<double> x3(cols);
      for (size_t i = 0; i < cols; ++i)
      {
        x0[i] = static_cast<double>(i) * 0.5 + 1.0;
        x1[i] = static_cast<double>(i) * -0.3 + 0.8;
        x2[i] = static_cast<double>(i) * 0.25 - 0.4;
        x3[i] = static_cast<double>(i) * -0.15 + 0.5;
      }
      std::vector<double> y(rows, 2.0);
      std::vector<double> expected_y = y;

      // Calculate expected
      for (size_t i = 0; i < rows; ++i)
      {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j)
        {
          sum += A0[i * cols + j] * x0[j] + A1[i * cols + j] * x1[j] + A2[i * cols + j] * x2[j] + A3[i * cols + j] * x3[j];
        }
        expected_y[i] += sum;
      }

      simd::gemv_accumulate_four(A0.data(), A1.data(), A2.data(), A3.data(), x0.data(), x1.data(), x2.data(), x3.data(), y.data(), rows, cols);
      expect_vec_near(y, expected_y);
    }
  }
}

TEST(SimdUtilsTest, AddVectors)
{
  std::vector<size_t> sizes = { 0, 1, 3, 4, 7, 8, 15, 16 };
  for (size_t n : sizes)
  {
    std::vector<double> x(n);
    std::vector<double> y(n, 5.0);
    for (size_t i = 0; i < n; ++i)
    {
      x[i] = static_cast<double>(i) * 1.5;
    }
    std::vector<double> expected_y = y;
    for (size_t i = 0; i < n; ++i)
    {
      expected_y[i] += x[i];
    }
    simd::add_vectors(x.data(), y.data(), n);
    expect_vec_near(y, expected_y);
  }
}

TEST(SimdUtilsTest, MulAddTwoThreeFourVariousSizes)
{
  std::vector<size_t> sizes = { 0, 1, 3, 4, 7, 8, 15, 16 };
  for (size_t n : sizes)
  {
    std::vector<double> w0(n, 1.1);
    std::vector<double> w1(n, 2.2);
    std::vector<double> w2(n, 3.3);
    std::vector<double> w3(n, 4.4);

    std::vector<double> y0(n, 10.0);
    std::vector<double> y1(n, 20.0);
    std::vector<double> y2(n, 30.0);
    std::vector<double> y3(n, 40.0);

    double x = 2.5;

    std::vector<double> expected_y0(n);
    std::vector<double> expected_y1(n);
    std::vector<double> expected_y2(n);
    std::vector<double> expected_y3(n);

    for (size_t i = 0; i < n; ++i)
    {
      expected_y0[i] = y0[i] + x * w0[i];
      expected_y1[i] = y1[i] + x * w1[i];
      expected_y2[i] = y2[i] + x * w2[i];
      expected_y3[i] = y3[i] + x * w3[i];
    }

    // Test mul_add_two
    std::vector<double> test_y0 = y0;
    std::vector<double> test_y1 = y1;
    simd::mul_add_two(x, w0.data(), w1.data(), test_y0.data(), test_y1.data(), n);
    expect_vec_near(test_y0, expected_y0);
    expect_vec_near(test_y1, expected_y1);

    // Test scalar_mul_add_two
    test_y0 = y0;
    test_y1 = y1;
    simd::scalar_mul_add_two(x, w0.data(), w1.data(), test_y0.data(), test_y1.data(), n);
    expect_vec_near(test_y0, expected_y0);
    expect_vec_near(test_y1, expected_y1);

    // Test mul_add_three
    test_y0 = y0;
    test_y1 = y1;
    std::vector<double> test_y2 = y2;
    simd::mul_add_three(x, w0.data(), w1.data(), w2.data(), test_y0.data(), test_y1.data(), test_y2.data(), n);
    expect_vec_near(test_y0, expected_y0);
    expect_vec_near(test_y1, expected_y1);
    expect_vec_near(test_y2, expected_y2);

    // Test scalar_mul_add_three
    test_y0 = y0;
    test_y1 = y1;
    test_y2 = y2;
    simd::scalar_mul_add_three(x, w0.data(), w1.data(), w2.data(), test_y0.data(), test_y1.data(), test_y2.data(), n);
    expect_vec_near(test_y0, expected_y0);
    expect_vec_near(test_y1, expected_y1);
    expect_vec_near(test_y2, expected_y2);

    // Test mul_add_four
    test_y0 = y0;
    test_y1 = y1;
    test_y2 = y2;
    std::vector<double> test_y3 = y3;
    simd::mul_add_four(x, w0.data(), w1.data(), w2.data(), w3.data(), test_y0.data(), test_y1.data(), test_y2.data(), test_y3.data(), n);
    expect_vec_near(test_y0, expected_y0);
    expect_vec_near(test_y1, expected_y1);
    expect_vec_near(test_y2, expected_y2);
    expect_vec_near(test_y3, expected_y3);

    // Test scalar_mul_add_four
    test_y0 = y0;
    test_y1 = y1;
    test_y2 = y2;
    test_y3 = y3;
    simd::scalar_mul_add_four(x, w0.data(), w1.data(), w2.data(), w3.data(), test_y0.data(), test_y1.data(), test_y2.data(), test_y3.data(), n);
    expect_vec_near(test_y0, expected_y0);
    expect_vec_near(test_y1, expected_y1);
    expect_vec_near(test_y2, expected_y2);
    expect_vec_near(test_y3, expected_y3);
  }
}

TEST(SimdUtilsTest, GemvAddTwoFour)
{
  std::vector<size_t> rows_list = { 1, 2, 3, 4, 7, 8 };
  std::vector<size_t> cols_list = { 1, 2, 3, 4, 7, 8, 15, 16 };
  for (size_t rows : rows_list)
  {
    for (size_t cols : cols_list)
    {
      std::vector<double> A0(rows * cols);
      std::vector<double> A1(rows * cols);
      std::vector<double> A2(rows * cols);
      std::vector<double> A3(rows * cols);
      for (size_t i = 0; i < A0.size(); ++i)
      {
        A0[i] = static_cast<double>(i) * 0.1;
        A1[i] = static_cast<double>(i) * 0.2 + 0.5;
        A2[i] = static_cast<double>(i) * 0.3 - 0.2;
        A3[i] = static_cast<double>(i) * 0.05 + 1.2;
      }
      std::vector<double> x(cols);
      for (size_t i = 0; i < cols; ++i)
      {
        x[i] = static_cast<double>(i) * 0.5 + 1.0;
      }

      std::vector<double> y0(rows, 2.0);
      std::vector<double> y1(rows, 3.0);
      std::vector<double> y2(rows, 4.0);
      std::vector<double> y3(rows, 5.0);

      std::vector<double> expected_y0 = y0;
      std::vector<double> expected_y1 = y1;
      std::vector<double> expected_y2 = y2;
      std::vector<double> expected_y3 = y3;

      for (size_t i = 0; i < rows; ++i)
      {
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        for (size_t j = 0; j < cols; ++j)
        {
          sum0 += A0[i * cols + j] * x[j];
          sum1 += A1[i * cols + j] * x[j];
          sum2 += A2[i * cols + j] * x[j];
          sum3 += A3[i * cols + j] * x[j];
        }
        expected_y0[i] += sum0;
        expected_y1[i] += sum1;
        expected_y2[i] += sum2;
        expected_y3[i] += sum3;
      }

      // Test gemv_add_two
      std::vector<double> test_y0 = y0;
      std::vector<double> test_y1 = y1;
      simd::gemv_add_two(A0.data(), A1.data(), x.data(), test_y0.data(), test_y1.data(), rows, cols);
      expect_vec_near(test_y0, expected_y0);
      expect_vec_near(test_y1, expected_y1);

      // Test scalar_gemv_add_two
      test_y0 = y0;
      test_y1 = y1;
      simd::scalar_gemv_add_two(A0.data(), A1.data(), x.data(), test_y0.data(), test_y1.data(), rows, cols);
      expect_vec_near(test_y0, expected_y0);
      expect_vec_near(test_y1, expected_y1);

      // Test gemv_add_four
      test_y0 = y0;
      test_y1 = y1;
      std::vector<double> test_y2 = y2;
      std::vector<double> test_y3 = y3;
      simd::gemv_add_four(A0.data(), A1.data(), A2.data(), A3.data(), x.data(), test_y0.data(), test_y1.data(), test_y2.data(), test_y3.data(), rows, cols);
      expect_vec_near(test_y0, expected_y0);
      expect_vec_near(test_y1, expected_y1);
      expect_vec_near(test_y2, expected_y2);
      expect_vec_near(test_y3, expected_y3);

      // Test scalar_gemv_add_four
      test_y0 = y0;
      test_y1 = y1;
      test_y2 = y2;
      test_y3 = y3;
      simd::scalar_gemv_add_four(A0.data(), A1.data(), A2.data(), A3.data(), x.data(), test_y0.data(), test_y1.data(), test_y2.data(), test_y3.data(), rows, cols);
      expect_vec_near(test_y0, expected_y0);
      expect_vec_near(test_y1, expected_y1);
      expect_vec_near(test_y2, expected_y2);
      expect_vec_near(test_y3, expected_y3);
    }
  }
}

TEST(SimdUtilsTest, MulAddScalarsVariousSizes)
{
  std::vector<size_t> sizes = { 0, 1, 3, 4, 7, 8, 15, 16, 31, 32 };
  for (size_t n : sizes)
  {
    std::vector<double> w0(n, 1.1);
    std::vector<double> w1(n, 2.2);
    std::vector<double> w2(n, 3.3);
    std::vector<double> w3(n, 4.4);

    std::vector<double> y0(n, 10.0);
    std::vector<double> y1(n, 20.0);
    std::vector<double> y2(n, 30.0);
    std::vector<double> y3(n, 40.0);

    double x0 = 1.5;
    double x1 = 2.5;
    double x2 = 3.5;
    double x3 = 4.5;

    std::vector<double> expected_y0_three(n);
    std::vector<double> expected_y1_three(n);
    std::vector<double> expected_y2_three(n);

    std::vector<double> expected_y0_four(n);
    std::vector<double> expected_y1_four(n);
    std::vector<double> expected_y2_four(n);
    std::vector<double> expected_y3_four(n);

    std::vector<double> expected_y0_two(n);
    std::vector<double> expected_y1_two(n);

    for (size_t i = 0; i < n; ++i)
    {
      expected_y0_three[i] = y0[i] + x0 * w0[i];
      expected_y1_three[i] = y1[i] + x1 * w1[i];
      expected_y2_three[i] = y2[i] + x2 * w2[i];

      double w0_val = w0[i];
      expected_y0_four[i] = y0[i] + x0 * w0_val;
      expected_y1_four[i] = y1[i] + x1 * w0_val;
      expected_y2_four[i] = y2[i] + x2 * w0_val;
      expected_y3_four[i] = y3[i] + x3 * w0_val;

      expected_y0_two[i] = y0[i] + x0 * w0_val;
      expected_y1_two[i] = y1[i] + x1 * w0_val;
    }

    // Test mul_add_three_scalars
    std::vector<double> test_y0 = y0;
    std::vector<double> test_y1 = y1;
    std::vector<double> test_y2 = y2;
    simd::mul_add_three_scalars(x0, x1, x2, w0.data(), w1.data(), w2.data(), test_y0.data(), test_y1.data(), test_y2.data(), n);
    expect_vec_near(test_y0, expected_y0_three);
    expect_vec_near(test_y1, expected_y1_three);
    expect_vec_near(test_y2, expected_y2_three);

    // Test scalar_mul_add_three_scalars
    test_y0 = y0;
    test_y1 = y1;
    test_y2 = y2;
    simd::scalar_mul_add_three_scalars(x0, x1, x2, w0.data(), w1.data(), w2.data(), test_y0.data(), test_y1.data(), test_y2.data(), n);
    expect_vec_near(test_y0, expected_y0_three);
    expect_vec_near(test_y1, expected_y1_three);
    expect_vec_near(test_y2, expected_y2_three);

    // Test mul_add_four_scalars
    test_y0 = y0;
    test_y1 = y1;
    test_y2 = y2;
    std::vector<double> test_y3 = y3;
    simd::mul_add_four_scalars(x0, x1, x2, x3, w0.data(), test_y0.data(), test_y1.data(), test_y2.data(), test_y3.data(), n);
    expect_vec_near(test_y0, expected_y0_four);
    expect_vec_near(test_y1, expected_y1_four);
    expect_vec_near(test_y2, expected_y2_four);
    expect_vec_near(test_y3, expected_y3_four);

    // Test scalar_mul_add_four_scalars
    test_y0 = y0;
    test_y1 = y1;
    test_y2 = y2;
    test_y3 = y3;
    simd::scalar_mul_add_four_scalars(x0, x1, x2, x3, w0.data(), test_y0.data(), test_y1.data(), test_y2.data(), test_y3.data(), n);
    expect_vec_near(test_y0, expected_y0_four);
    expect_vec_near(test_y1, expected_y1_four);
    expect_vec_near(test_y2, expected_y2_four);
    expect_vec_near(test_y3, expected_y3_four);

    // Test mul_add_two_scalars
    test_y0 = y0;
    test_y1 = y1;
    simd::mul_add_two_scalars(x0, x1, w0.data(), test_y0.data(), test_y1.data(), n);
    expect_vec_near(test_y0, expected_y0_two);
    expect_vec_near(test_y1, expected_y1_two);

    // Test scalar_mul_add_two_scalars
    test_y0 = y0;
    test_y1 = y1;
    simd::scalar_mul_add_two_scalars(x0, x1, w0.data(), test_y0.data(), test_y1.data(), n);
    expect_vec_near(test_y0, expected_y0_two);
    expect_vec_near(test_y1, expected_y1_two);
  }
}

TEST(SimdUtilsTest, SumSq)
{
  std::vector<size_t> sizes = { 0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 127, 128 };
  for (size_t n : sizes)
  {
    std::vector<double> x(n);
    double expected = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
      x[i] = static_cast<double>(i) * 0.1;
      expected += x[i] * x[i];
    }
    double actual = simd::sum_sq(x.data(), n);
    EXPECT_NEAR(expected, actual, 1e-12) << "sum_sq failed for size " << n;
  }
}

TEST(SimdUtilsTest, AdamNadamStepBoundaryConditions)
{
  const size_t n = 8;
  std::vector<double> values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
  std::vector<double> grads = { 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8 };
  std::vector<double> m1_init = { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 };
  std::vector<double> m2_init = { 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008 };
  std::vector<double> decays = { 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 };

  double b1 = 0.9;
  double b2 = 0.999;
  double lr = 0.001;
  double eps = 1e-8;

  // Case 1: p1 = 0.0, p2 = 0.0 (below safety threshold)
  {
    double p1 = 0.0;
    double p2 = 0.0;

    std::vector<double> val_simd = values;
    std::vector<double> m1_simd = m1_init;
    std::vector<double> m2_simd = m2_init;

    std::vector<double> val_scalar = values;
    std::vector<double> m1_scalar = m1_init;
    std::vector<double> m2_scalar = m2_init;

    simd::adam_step(val_simd.data(), grads.data(), m1_simd.data(), m2_simd.data(), b1, b2, p1, p2, lr, eps, n, decays.data());
    simd::scalar_adam_step(val_scalar.data(), grads.data(), m1_scalar.data(), m2_scalar.data(), b1, b2, p1, p2, lr, eps, n, decays.data());

    expect_vec_near(m1_simd, m1_scalar);
    expect_vec_near(m2_simd, m2_scalar);
    expect_vec_near(val_simd, val_scalar);
  }

  // Case 2: p1 = 1e-16, p2 = 1e-16 (extremely small, also below threshold)
  {
    double p1 = 1e-16;
    double p2 = 1e-16;

    std::vector<double> val_simd = values;
    std::vector<double> m1_simd = m1_init;
    std::vector<double> m2_simd = m2_init;

    std::vector<double> val_scalar = values;
    std::vector<double> m1_scalar = m1_init;
    std::vector<double> m2_scalar = m2_init;

    simd::nadam_step(val_simd.data(), grads.data(), m1_simd.data(), m2_simd.data(), b1, b2, p1, p2, lr, eps, n, decays.data());
    simd::scalar_nadam_step(val_scalar.data(), grads.data(), m1_scalar.data(), m2_scalar.data(), b1, b2, p1, p2, lr, eps, n, decays.data());

    expect_vec_near(m1_simd, m1_scalar);
    expect_vec_near(m2_simd, m2_scalar);
    expect_vec_near(val_simd, val_scalar);
  }
}

#ifdef SIMD_AVX2_ENABLED
TEST(SimdUtilsTest, HorizontalSum)
{
  __m256d vec = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
  double expected = 1.0 + 2.0 + 3.0 + 4.0;
  double actual = simd::horizontal_sum(vec);
  EXPECT_DOUBLE_EQ(expected, actual);
}
#endif

TEST(ResidualProjectorTest, Correctness)
{
  const unsigned input_size = 4;
  const unsigned output_size = 3;
  std::vector<double> w_values = {
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6,
    0.7, 0.8, 0.9,
    1.0, 1.1, 1.2
  };
  std::vector<double> w_grads(12, 0.0);
  std::vector<double> w_velocities(12, 0.0);
  std::vector<double> w_m1(12, 0.0);
  std::vector<double> w_m2(12, 0.0);
  std::vector<long long> w_timesteps(12, 0);
  std::vector<double> w_decays(12, 0.0);

  ResidualProjector proj(
    input_size,
    output_size,
    w_values,
    w_grads,
    w_velocities,
    w_m1,
    w_m2,
    w_timesteps,
    w_decays
  );

  // Test project
  std::vector<double> inputs = { 0.5, 1.5, 0.0, -1.0 };
  std::vector<double> expected(output_size, 0.0);
  for (size_t in = 0; in < input_size; ++in)
  {
    for (size_t out = 0; out < output_size; ++out)
    {
      expected[out] += w_values[in * output_size + out] * inputs[in];
    }
  }

  std::vector<double> actual = proj.project(inputs);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i)
  {
    EXPECT_NEAR(actual[i], expected[i], 1e-12);
  }

  // Test project_batch
  std::vector<std::vector<double>> batch_inputs = {
    { 0.5, 1.5, 0.0, -1.0 },
    { 1.0, -2.0, 0.5, 0.0 }
  };
  std::vector<std::vector<double>> expected_batch(batch_inputs.size(), std::vector<double>(output_size, 0.0));
  for (size_t b = 0; b < batch_inputs.size(); ++b)
  {
    for (size_t in = 0; in < input_size; ++in)
    {
      for (size_t out = 0; out < output_size; ++out)
      {
        expected_batch[b][out] += w_values[in * output_size + out] * batch_inputs[b][in];
      }
    }
  }

  std::vector<std::vector<double>> actual_batch = proj.project_batch(batch_inputs);
  ASSERT_EQ(actual_batch.size(), expected_batch.size());
  for (size_t b = 0; b < actual_batch.size(); ++b)
  {
    ASSERT_EQ(actual_batch[b].size(), expected_batch[b].size());
    for (size_t i = 0; i < actual_batch[b].size(); ++i)
    {
      EXPECT_NEAR(actual_batch[b][i], expected_batch[b][i], 1e-12);
    }
  }

  // Test project_batch with const double*
  std::vector<const double*> pointer_batch = {
    batch_inputs[0].data(),
    batch_inputs[1].data()
  };
  std::vector<std::vector<double>> actual_ptr_batch = proj.project_batch(pointer_batch);
  ASSERT_EQ(actual_ptr_batch.size(), expected_batch.size());
  for (size_t b = 0; b < actual_ptr_batch.size(); ++b)
  {
    ASSERT_EQ(actual_ptr_batch[b].size(), expected_batch[b].size());
    for (size_t i = 0; i < actual_ptr_batch[b].size(); ++i)
    {
      EXPECT_NEAR(actual_ptr_batch[b][i], expected_batch[b][i], 1e-12);
    }
  }
}

TEST(SimdUtilsTest, MulVectors) {
  std::vector<double> x = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> y = { 0.5, 1.5, 2.5, 3.5, 4.5 };
  std::vector<double> expected(x.size());
  for (size_t i = 0; i < x.size(); ++i)
  {
    expected[i] = x[i] * y[i];
  }

  std::vector<double> actual(x.size(), 0.0);
  simd::mul_vectors(x.data(), y.data(), actual.data(), x.size());

  expect_vec_near(actual, expected);
}

TEST(SimdUtilsTest, GruOutputStep) {
  std::vector<double> z = { 0.1, 0.5, 0.9, 0.0, 1.0 };
  std::vector<double> prev_h = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> h_hat = { -1.0, -2.0, -3.0, -4.0, -5.0 };
  std::vector<double> expected(z.size());
  for (size_t i = 0; i < z.size(); ++i)
  {
    expected[i] = (1.0 - z[i]) * prev_h[i] + z[i] * h_hat[i];
  }

  std::vector<double> actual_h(z.size(), 0.0);
  std::vector<double> actual_seq(z.size(), 0.0);
  simd::gru_output_step(z.data(), prev_h.data(), h_hat.data(), actual_h.data(), actual_seq.data(), z.size());

  expect_vec_near(actual_h, expected);
  expect_vec_near(actual_seq, expected);
}

TEST(SimdUtilsTest, LstmCellStep) {
  std::vector<double> f = { 0.9, 0.8, 0.7, 0.6, 0.5 };
  std::vector<double> i = { 0.1, 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> g_act = { 1.5, -1.5, 2.5, -2.5, 3.5 };
  std::vector<double> current_c = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  std::vector<double> expected(f.size());
  for (size_t j = 0; j < f.size(); ++j)
  {
    expected[j] = f[j] * current_c[j] + i[j] * g_act[j];
  }

  simd::lstm_cell_step(f.data(), i.data(), g_act.data(), current_c.data(), f.size());

  expect_vec_near(current_c, expected);
}

TEST(SimdUtilsTest, MulThreeVectors) {
  std::vector<double> x = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
  std::vector<double> y = { 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 };
  std::vector<double> z = { 2.0, 0.5, 4.0, 0.25, 8.0, 0.125 };
  std::vector<double> expected(x.size());
  for (size_t i = 0; i < x.size(); ++i)
  {
    expected[i] = x[i] * y[i] * z[i];
  }

  std::vector<double> actual(x.size(), 0.0);
  simd::mul_three_vectors(x.data(), y.data(), z.data(), actual.data(), x.size());

  expect_vec_near(actual, expected);
}

TEST(SimdUtilsTest, LstmBpttUpstreamStep) {
  std::vector<double> upstream = { 1.0, -2.0, 3.0, -4.0, 5.0, -6.0 };
  std::vector<double> dh_next = { 0.5, 1.5, -2.5, 3.5, -4.5, 5.5 };
  std::vector<double> mask = { 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
  std::vector<double> expected(upstream.size());
  for (size_t i = 0; i < upstream.size(); ++i)
  {
    expected[i] = std::clamp((upstream[i] + dh_next[i]) * mask[i], -50.0, 50.0);
  }

  std::vector<double> actual(upstream.size(), 0.0);
  simd::lstm_bptt_upstream_step(upstream.data(), dh_next.data(), mask.data(), actual.data(), upstream.size());

  expect_vec_near(actual, expected);
}

TEST(SimdUtilsTest, ElmanBpttGateStep) {
  std::vector<double> upstream = { 1.0, 10.0, 100.0, -10.0, -100.0, 0.0 };
  std::vector<double> dh_next = { 2.0, 20.0, 200.0, -20.0, -200.0, 1.0 };
  std::vector<double> deriv = { 0.5, 0.25, 0.125, 0.2, 0.1, 0.5 };
  std::vector<double> mask = { 1.0, 0.0, 1.0, 0.0, 1.0, 1.0 };
  std::vector<double> expected(upstream.size());
  for (size_t i = 0; i < upstream.size(); ++i)
  {
    double dh = std::clamp(upstream[i] + dh_next[i], -50.0, 50.0);
    expected[i] = dh * deriv[i] * mask[i];
  }

  std::vector<double> actual(upstream.size(), 0.0);
  simd::elman_bptt_gate_step(upstream.data(), dh_next.data(), deriv.data(), mask.data(), actual.data(), upstream.size());

  expect_vec_near(actual, expected);
}

TEST(SimdUtilsTest, FmaEquivalenceVerify)
{
  // Test case to verify all the optimised FMA/AVX2 steps against their scalar fallbacks with larger arrays (n = 100)
  const size_t n = 100;
  
  // Initialize inputs
  std::vector<double> values(n);
  std::vector<double> grads(n);
  std::vector<double> m1(n);
  std::vector<double> m2(n);
  std::vector<double> decays(n);
  for (size_t i = 0; i < n; ++i)
  {
    values[i] = 1.0 + 0.1 * static_cast<double>(i);
    grads[i] = -0.5 + 0.01 * static_cast<double>(i);
    m1[i] = 0.01 * static_cast<double>(i);
    m2[i] = 0.002 * static_cast<double>(i);
    decays[i] = 0.005;
  }

  double b1 = 0.9;
  double b2 = 0.999;
  double p1 = 0.8;
  double p2 = 0.7;
  double lr = 0.001;
  double eps = 1e-8;

  // 1. Adam Step Verify
  {
    std::vector<double> val_simd = values;
    std::vector<double> m1_simd = m1;
    std::vector<double> m2_simd = m2;

    std::vector<double> val_scalar = values;
    std::vector<double> m1_scalar = m1;
    std::vector<double> m2_scalar = m2;

    simd::adam_step(val_simd.data(), grads.data(), m1_simd.data(), m2_simd.data(), b1, b2, p1, p2, lr, eps, n, decays.data());
    simd::scalar_adam_step(val_scalar.data(), grads.data(), m1_scalar.data(), m2_scalar.data(), b1, b2, p1, p2, lr, eps, n, decays.data());

    expect_vec_near(m1_simd, m1_scalar);
    expect_vec_near(m2_simd, m2_scalar);
    expect_vec_near(val_simd, val_scalar);
  }

  // 2. Nadam Step Verify
  {
    std::vector<double> val_simd = values;
    std::vector<double> m1_simd = m1;
    std::vector<double> m2_simd = m2;

    std::vector<double> val_scalar = values;
    std::vector<double> m1_scalar = m1;
    std::vector<double> m2_scalar = m2;

    simd::nadam_step(val_simd.data(), grads.data(), m1_simd.data(), m2_simd.data(), b1, b2, p1, p2, lr, eps, n, decays.data());
    simd::scalar_nadam_step(val_scalar.data(), grads.data(), m1_scalar.data(), m2_scalar.data(), b1, b2, p1, p2, lr, eps, n, decays.data());

    expect_vec_near(m1_simd, m1_scalar);
    expect_vec_near(m2_simd, m2_scalar);
    expect_vec_near(val_simd, val_scalar);
  }

  // 3. GRU BPTT Gate Step Verify
  {
    std::vector<double> grad_next(n);
    std::vector<double> d_next_h(n);
    std::vector<double> z_vals(n);
    std::vector<double> h_hat_vals(n);
    std::vector<double> h_prev_vals(n);
    std::vector<double> h_hat_pre_vals(n);
    std::vector<double> mask_vals(n);
    std::vector<double> h_hat_pre_deriv(n);
    for (size_t i = 0; i < n; ++i)
    {
      grad_next[i] = 0.1 * static_cast<double>(i);
      d_next_h[i] = -0.05 * static_cast<double>(i);
      z_vals[i] = 0.5;
      h_hat_vals[i] = 0.2;
      h_prev_vals[i] = 0.1;
      h_hat_pre_vals[i] = 0.3;
      mask_vals[i] = 1.0;
      h_hat_pre_deriv[i] = 0.4;
    }

    std::vector<double> dz_simd(n, 0.0);
    std::vector<double> dh_hat_simd(n, 0.0);
    std::vector<double> dh_prev_accum_simd(n, 0.0);

    std::vector<double> dz_scalar(n, 0.0);
    std::vector<double> dh_hat_scalar(n, 0.0);
    std::vector<double> dh_prev_accum_scalar(n, 0.0);

    simd::gru_bptt_gate_step(n, grad_next.data(), d_next_h.data(), z_vals.data(), h_hat_vals.data(), h_prev_vals.data(), h_hat_pre_vals.data(), mask_vals.data(), dz_simd.data(), dh_hat_simd.data(), dh_prev_accum_simd.data(), h_hat_pre_deriv.data());
    simd::scalar_gru_bptt_gate_step(n, grad_next.data(), d_next_h.data(), z_vals.data(), h_hat_vals.data(), h_prev_vals.data(), h_hat_pre_vals.data(), mask_vals.data(), dz_scalar.data(), dh_hat_scalar.data(), dh_prev_accum_scalar.data(), h_hat_pre_deriv.data());

    expect_vec_near(dz_simd, dz_scalar);
    expect_vec_near(dh_hat_simd, dh_hat_scalar);
    expect_vec_near(dh_prev_accum_simd, dh_prev_accum_scalar);
  }

  // 4. GRU BPTT Reset Step Verify
  {
    std::vector<double> temp_Uh(n);
    std::vector<double> h_prev_vals(n);
    std::vector<double> r_vals(n);
    std::vector<double> dh_prev_accum(n);
    for (size_t i = 0; i < n; ++i)
    {
      temp_Uh[i] = 0.25 * static_cast<double>(i);
      h_prev_vals[i] = 0.12 * static_cast<double>(i);
      r_vals[i] = 0.33 * static_cast<double>(i);
      dh_prev_accum[i] = 0.05 * static_cast<double>(i);
    }

    std::vector<double> dr_simd(n, 0.0);
    std::vector<double> dh_next_simd(n, 0.0);

    std::vector<double> dr_scalar(n, 0.0);
    std::vector<double> dh_next_scalar(n, 0.0);

    simd::gru_bptt_reset_step(n, temp_Uh.data(), h_prev_vals.data(), r_vals.data(), dh_prev_accum.data(), dr_simd.data(), dh_next_simd.data());
    simd::scalar_gru_bptt_reset_step(n, temp_Uh.data(), h_prev_vals.data(), r_vals.data(), dh_prev_accum.data(), dr_scalar.data(), dh_next_scalar.data());

    expect_vec_near(dr_simd, dr_scalar);
    expect_vec_near(dh_next_simd, dh_next_scalar);
  }

  // 5. LSTM BPTT Gate Step Verify
  {
    std::vector<double> dh_curr(n);
    std::vector<double> dc_next_in(n);
    std::vector<double> f(n);
    std::vector<double> i_gate(n);
    std::vector<double> o(n);
    std::vector<double> g_pre_vals(n);
    std::vector<double> activated_g_vals(n);
    std::vector<double> activated_c_vals(n);
    std::vector<double> c_prev(n);
    std::vector<double> dc_act_deriv_vals(n);
    std::vector<double> dg_act_deriv_vals(n);
    for (size_t idx = 0; idx < n; ++idx)
    {
      dh_curr[idx] = 1.25 * static_cast<double>(idx);
      dc_next_in[idx] = 0.5 * static_cast<double>(idx);
      f[idx] = 0.8;
      i_gate[idx] = 0.2;
      o[idx] = 0.7;
      g_pre_vals[idx] = 0.1 * static_cast<double>(idx);
      activated_g_vals[idx] = 0.9;
      activated_c_vals[idx] = 0.3;
      c_prev[idx] = 0.4;
      dc_act_deriv_vals[idx] = 0.6;
      dg_act_deriv_vals[idx] = 0.5;
    }

    std::vector<double> df_simd(n, 0.0);
    std::vector<double> di_simd(n, 0.0);
    std::vector<double> do_simd(n, 0.0);
    std::vector<double> dg_simd(n, 0.0);
    std::vector<double> dc_next_simd(n, 0.0);

    std::vector<double> df_scalar(n, 0.0);
    std::vector<double> di_scalar(n, 0.0);
    std::vector<double> do_scalar(n, 0.0);
    std::vector<double> dg_scalar(n, 0.0);
    std::vector<double> dc_next_scalar(n, 0.0);

    simd::lstm_bptt_gate_step(n, dh_curr.data(), dc_next_in.data(), f.data(), i_gate.data(), o.data(), g_pre_vals.data(), activated_g_vals.data(), activated_c_vals.data(), c_prev.data(), true, df_simd.data(), di_simd.data(), do_simd.data(), dg_simd.data(), dc_next_simd.data(), dc_act_deriv_vals.data(), dg_act_deriv_vals.data());
    simd::scalar_lstm_bptt_gate_step(n, dh_curr.data(), dc_next_in.data(), f.data(), i_gate.data(), o.data(), g_pre_vals.data(), activated_g_vals.data(), activated_c_vals.data(), c_prev.data(), true, df_scalar.data(), di_scalar.data(), do_scalar.data(), dg_scalar.data(), dc_next_scalar.data(), dc_act_deriv_vals.data(), dg_act_deriv_vals.data());

    expect_vec_near(df_simd, df_scalar);
    expect_vec_near(di_simd, di_scalar);
    expect_vec_near(do_simd, do_scalar);
    expect_vec_near(dg_simd, dg_scalar);
    expect_vec_near(dc_next_simd, dc_next_scalar);
  }

  // 6. LSTM BPTT Upstream Step Verify
  {
    std::vector<double> upstream(n);
    std::vector<double> dh_next(n);
    std::vector<double> mask(n);
    for (size_t i = 0; i < n; ++i)
    {
      upstream[i] = 1.2 * static_cast<double>(i);
      dh_next[i] = -0.8 * static_cast<double>(i);
      mask[i] = (i % 2 == 0) ? 1.0 : 0.0;
    }

    std::vector<double> dh_curr_simd(n, 0.0);
    std::vector<double> dh_curr_scalar(n, 0.0);

    simd::lstm_bptt_upstream_step(upstream.data(), dh_next.data(), mask.data(), dh_curr_simd.data(), n);
    simd::scalar_lstm_bptt_upstream_step(upstream.data(), dh_next.data(), mask.data(), dh_curr_scalar.data(), n);

    expect_vec_near(dh_curr_simd, dh_curr_scalar);
  }
}

TEST(SimdUtilsTest, GemmBatchesVerify)
{
  const size_t n_prev = 64;
  const size_t n_this = 35;

  // Initialize input vectors and matrix
  std::vector<double> x0(n_prev);
  std::vector<double> x1(n_prev);
  std::vector<double> x2(n_prev);
  std::vector<double> x3(n_prev);
  std::vector<double> W(n_prev * n_this);

  for (size_t i = 0; i < n_prev; ++i)
  {
    x0[i] = 0.1 * static_cast<double>(i);
    x1[i] = -0.05 * static_cast<double>(i);
    x2[i] = 0.02 * static_cast<double>(i);
    x3[i] = -0.01 * static_cast<double>(i);
    for (size_t j = 0; j < n_this; ++j)
    {
      W[i * n_this + j] = 0.001 * static_cast<double>(i * j);
    }
  }

  // 1. Verify gemm_four_batches
  {
    std::vector<double> y0_simd(n_this, 1.0);
    std::vector<double> y1_simd(n_this, 2.0);
    std::vector<double> y2_simd(n_this, 3.0);
    std::vector<double> y3_simd(n_this, 4.0);

    std::vector<double> y0_expected(n_this, 1.0);
    std::vector<double> y1_expected(n_this, 2.0);
    std::vector<double> y2_expected(n_this, 3.0);
    std::vector<double> y3_expected(n_this, 4.0);

    // Compute expected results using original scalar loop
    for (size_t i = 0; i < n_prev; ++i)
    {
      simd::scalar_mul_add_four_scalars(x0[i], x1[i], x2[i], x3[i], &W[i * n_this], y0_expected.data(), y1_expected.data(), y2_expected.data(), y3_expected.data(), n_this);
    }

    simd::gemm_four_batches(x0.data(), x1.data(), x2.data(), x3.data(), W.data(), y0_simd.data(), y1_simd.data(), y2_simd.data(), y3_simd.data(), n_prev, n_this);

    expect_vec_near(y0_simd, y0_expected);
    expect_vec_near(y1_simd, y1_expected);
    expect_vec_near(y2_simd, y2_expected);
    expect_vec_near(y3_simd, y3_expected);
  }

  // 2. Verify gemm_two_batches
  {
    std::vector<double> y0_simd(n_this, 1.0);
    std::vector<double> y1_simd(n_this, 2.0);

    std::vector<double> y0_expected(n_this, 1.0);
    std::vector<double> y1_expected(n_this, 2.0);

    for (size_t i = 0; i < n_prev; ++i)
    {
      simd::scalar_mul_add_two_scalars(x0[i], x1[i], &W[i * n_this], y0_expected.data(), y1_expected.data(), n_this);
    }

    simd::gemm_two_batches(x0.data(), x1.data(), W.data(), y0_simd.data(), y1_simd.data(), n_prev, n_this);

    expect_vec_near(y0_simd, y0_expected);
    expect_vec_near(y1_simd, y1_expected);
  }

  // 3. Verify gemm_one_batch
  {
    std::vector<double> y_simd(n_this, 1.0);
    std::vector<double> y_expected(n_this, 1.0);

    for (size_t i = 0; i < n_prev; ++i)
    {
      simd::scalar_mul_add(x0[i], &W[i * n_this], y_expected.data(), n_this);
    }

    simd::gemm_one_batch(x0.data(), W.data(), y_simd.data(), n_prev, n_this);

    expect_vec_near(y_simd, y_expected);
  }
}

TEST(SimdUtilsTest, GemmTransposedBatchesVerify)
{
  const size_t n_this = 64;
  const size_t n_next = 35;

  std::vector<double> x0(n_next);
  std::vector<double> x1(n_next);
  std::vector<double> x2(n_next);
  std::vector<double> x3(n_next);
  std::vector<double> W(n_this * n_next);

  for (size_t i = 0; i < n_next; ++i)
  {
    x0[i] = 0.1 * static_cast<double>(i);
    x1[i] = -0.05 * static_cast<double>(i);
    x2[i] = 0.02 * static_cast<double>(i);
    x3[i] = -0.01 * static_cast<double>(i);
  }

  for (size_t i = 0; i < n_this; ++i)
  {
    for (size_t j = 0; j < n_next; ++j)
    {
      W[i * n_next + j] = 0.001 * static_cast<double>(i * j);
    }
  }

  // 1. Verify gemm_transposed_four_batches
  {
    std::vector<double> y0_simd(n_this, 1.0);
    std::vector<double> y1_simd(n_this, 2.0);
    std::vector<double> y2_simd(n_this, 3.0);
    std::vector<double> y3_simd(n_this, 4.0);

    std::vector<double> y0_expected(n_this, 1.0);
    std::vector<double> y1_expected(n_this, 2.0);
    std::vector<double> y2_expected(n_this, 3.0);
    std::vector<double> y3_expected(n_this, 4.0);

    simd::scalar_gemv_add(W.data(), x0.data(), y0_expected.data(), n_this, n_next);
    simd::scalar_gemv_add(W.data(), x1.data(), y1_expected.data(), n_this, n_next);
    simd::scalar_gemv_add(W.data(), x2.data(), y2_expected.data(), n_this, n_next);
    simd::scalar_gemv_add(W.data(), x3.data(), y3_expected.data(), n_this, n_next);

    simd::gemm_transposed_four_batches(
      x0.data(), x1.data(), x2.data(), x3.data(),
      W.data(),
      y0_simd.data(), y1_simd.data(), y2_simd.data(), y3_simd.data(),
      n_this, n_next
    );

    expect_vec_near(y0_simd, y0_expected);
    expect_vec_near(y1_simd, y1_expected);
    expect_vec_near(y2_simd, y2_expected);
    expect_vec_near(y3_simd, y3_expected);
  }

  // 2. Verify gemm_transposed_two_batches
  {
    std::vector<double> y0_simd(n_this, 1.0);
    std::vector<double> y1_simd(n_this, 2.0);

    std::vector<double> y0_expected(n_this, 1.0);
    std::vector<double> y1_expected(n_this, 2.0);

    simd::scalar_gemv_add(W.data(), x0.data(), y0_expected.data(), n_this, n_next);
    simd::scalar_gemv_add(W.data(), x1.data(), y1_expected.data(), n_this, n_next);

    simd::gemm_transposed_two_batches(
      x0.data(), x1.data(),
      W.data(),
      y0_simd.data(), y1_simd.data(),
      n_this, n_next
    );

    expect_vec_near(y0_simd, y0_expected);
    expect_vec_near(y1_simd, y1_expected);
  }

  // 3. Verify gemm_transposed_one_batch
  {
    std::vector<double> y_simd(n_this, 1.0);
    std::vector<double> y_expected(n_this, 1.0);

    simd::scalar_gemv_add(W.data(), x0.data(), y_expected.data(), n_this, n_next);

    simd::gemm_transposed_one_batch(x0.data(), W.data(), y_simd.data(), n_this, n_next);

    expect_vec_near(y_simd, y_expected);
  }
}

TEST(SimdUtilsTest, IncrementValues)
{
  const size_t sizes[] = { 1, 3, 4, 7, 8, 15, 16, 100 };
  for (size_t n : sizes)
  {
    std::vector<long long> values(n);
    std::vector<long long> expected(n);
    for (size_t i = 0; i < n; ++i)
    {
      values[i] = static_cast<long long>(i) * 10;
      expected[i] = values[i] + 1;
    }

    simd::increment_values(values.data(), n);

    for (size_t i = 0; i < n; ++i)
    {
      EXPECT_EQ(values[i], expected[i]) << "Mismatch at index " << i << " for size " << n;
    }
  }
}


