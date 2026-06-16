#include <gtest/gtest.h>
#include "common/simd_utils.h"
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


