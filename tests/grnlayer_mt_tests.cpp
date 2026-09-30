#include <gtest/gtest.h>
#include "layers/grnlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>

using namespace myoddweb::nn;
using namespace test_helper;

namespace
{
GrnLayer make_mt_grn_layer(
  unsigned d_in,
  unsigned d_out,
  unsigned d_ff,
  bool use_layer_norm,
  int number_of_threads)
{
  return GrnLayer(
    1,
    d_in,
    d_out,
    d_ff,
    0.0,
    Layer::Role::Hidden,
    activation(activation::method::elu, 1.0),
    OptimiserType::SGD,
    -1,
    0.0,
    nullptr,
    number_of_threads,
    true,
    use_layer_norm,
    0.0,
    std::nullopt);
}

void expect_vectors_match(const std::vector<double>& single, const std::vector<double>& multi, const char* name)
{
  ASSERT_EQ(single.size(), multi.size()) << name;
  for (size_t i = 0; i < single.size(); ++i)
  {
    EXPECT_NEAR(single[i], multi[i], 1e-12) << name << " grad mismatch at " << i;
  }
}
} // namespace

class GrnLayerMTTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
  }
};

TEST_F(GrnLayerMTTest, ForwardFeedThreadCountInvariance)
{
  const unsigned d_in = 4;
  const unsigned d_out = 3;
  const unsigned d_ff = 6;
  const size_t time_steps = 4;
  const size_t batch_size = 8;
  const int num_threads = static_cast<int>(get_test_threads());

  GrnLayer layer_st = make_mt_grn_layer(d_in, d_out, d_ff, true, 1);
  GrnLayer layer_mt = make_mt_grn_layer(d_in, d_out, d_ff, true, num_threads);

  // Synchronise weights exactly between ST and MT instances
  layer_st.set_w1_values(layer_mt.get_w1_values());
  layer_st.set_b1_values(layer_mt.get_b1_values());
  layer_st.set_w2_values(layer_mt.get_w2_values());
  layer_st.set_b2_values(layer_mt.get_b2_values());
  layer_st.set_w_skip_values(layer_mt.get_w_skip_values());
  layer_st.set_b_skip_values(layer_mt.get_b_skip_values());
  layer_st.set_ln_gain_values(layer_mt.get_ln_gain_values());
  layer_st.set_ln_bias_values(layer_mt.get_ln_bias_values());

  std::vector<unsigned> topology = { d_in, d_out };
  auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
  auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);
  MockLayer previous_layer(0, d_in);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> x_seq(time_steps * d_in);
    for (size_t i = 0; i < x_seq.size(); ++i)
    {
      x_seq[i] = std::sin(static_cast<double>(b * 11 + i) * 0.17);
    }
    batch_go_st[b].set_rnn_outputs(0, x_seq.data(), x_seq.size());
    batch_go_mt[b].set_rnn_outputs(0, x_seq.data(), x_seq.size());
  }

  layer_st.calculate_forward_feed(batch_go_st, previous_layer, {}, batch_hs_st, batch_size, false);
  layer_mt.calculate_forward_feed(batch_go_mt, previous_layer, {}, batch_hs_mt, batch_size, false);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto out_st = batch_go_st[b].get_rnn_outputs(1);
    const auto out_mt = batch_go_mt[b].get_rnn_outputs(1);
    ASSERT_EQ(out_st.size(), out_mt.size());
    for (size_t i = 0; i < out_st.size(); ++i)
    {
      EXPECT_NEAR(out_st[i], out_mt[i], 1e-12)
        << "Mismatch at batch " << b << " output element " << i;
    }
  }
}

TEST_F(GrnLayerMTTest, GradientCalculationThreadCountInvariance)
{
  const unsigned d_in = 3;
  const unsigned d_out = 4;
  const unsigned d_ff = 5;
  const size_t time_steps = 3;
  const size_t batch_size = 8;
  const int num_threads = static_cast<int>(get_test_threads());

  GrnLayer layer_st = make_mt_grn_layer(d_in, d_out, d_ff, true, 1);
  GrnLayer layer_mt = make_mt_grn_layer(d_in, d_out, d_ff, true, num_threads);

  // Synchronise weights
  layer_st.set_w1_values(layer_mt.get_w1_values());
  layer_st.set_b1_values(layer_mt.get_b1_values());
  layer_st.set_w2_values(layer_mt.get_w2_values());
  layer_st.set_b2_values(layer_mt.get_b2_values());
  layer_st.set_w_skip_values(layer_mt.get_w_skip_values());
  layer_st.set_b_skip_values(layer_mt.get_b_skip_values());
  layer_st.set_ln_gain_values(layer_mt.get_ln_gain_values());
  layer_st.set_ln_bias_values(layer_mt.get_ln_bias_values());

  std::vector<unsigned> topology = { d_in, d_out };
  auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
  auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);
  MockLayer previous_layer(0, d_in);

  std::vector<std::vector<double>> deltas(batch_size);
  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> x_seq(time_steps * d_in);
    deltas[b].resize(time_steps * d_out);
    for (size_t i = 0; i < x_seq.size(); ++i)
    {
      x_seq[i] = std::cos(static_cast<double>(b * 7 + i) * 0.23);
    }
    for (size_t i = 0; i < deltas[b].size(); ++i)
    {
      deltas[b][i] = std::sin(static_cast<double>(b * 5 + i) * 0.31);
    }
    batch_go_st[b].set_rnn_outputs(0, x_seq.data(), x_seq.size());
    batch_go_mt[b].set_rnn_outputs(0, x_seq.data(), x_seq.size());
  }

  // Forward feed
  layer_st.calculate_forward_feed(batch_go_st, previous_layer, {}, batch_hs_st, batch_size, true);
  layer_mt.calculate_forward_feed(batch_go_mt, previous_layer, {}, batch_hs_mt, batch_size, true);

  // Hidden gradients (propagated to layer 1)
  layer_st.calculate_hidden_gradients_from_output_gradients(batch_go_st, deltas, batch_hs_st, batch_size, 0);
  layer_mt.calculate_hidden_gradients_from_output_gradients(batch_go_mt, deltas, batch_hs_mt, batch_size, 0);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto grad_st = batch_go_st[b].get_rnn_gradients(1);
    const auto grad_mt = batch_go_mt[b].get_rnn_gradients(1);
    ASSERT_EQ(grad_st.size(), grad_mt.size());
    for (size_t i = 0; i < grad_st.size(); ++i)
    {
      EXPECT_NEAR(grad_st[i], grad_mt[i], 1e-12)
        << "Hidden gradient mismatch at batch " << b << " element " << i;
    }
  }

  // Weight gradients
  layer_st.calculate_and_store_gradients(batch_go_st, batch_hs_st, previous_layer, batch_size, 0);
  layer_mt.calculate_and_store_gradients(batch_go_mt, batch_hs_mt, previous_layer, batch_size, 0);

  const auto& w1_st = layer_st.get_w1_grads();
  const auto& w1_mt = layer_mt.get_w1_grads();
  ASSERT_EQ(w1_st.size(), w1_mt.size());
  for (size_t i = 0; i < w1_st.size(); ++i)
  {
    EXPECT_NEAR(w1_st[i], w1_mt[i], 1e-12)
      << "w1 grad mismatch at " << i;
  }

  const auto& w_skip_st = layer_st.get_w_skip_grads();
  const auto& w_skip_mt = layer_mt.get_w_skip_grads();
  ASSERT_EQ(w_skip_st.size(), w_skip_mt.size());
  for (size_t i = 0; i < w_skip_st.size(); ++i)
  {
    EXPECT_NEAR(w_skip_st[i], w_skip_mt[i], 1e-12)
      << "w_skip grad mismatch at " << i;
  }

  expect_vectors_match(layer_st.get_b1_grads(), layer_mt.get_b1_grads(), "b1");
  expect_vectors_match(layer_st.get_w2_grads(), layer_mt.get_w2_grads(), "w2");
  expect_vectors_match(layer_st.get_b2_grads(), layer_mt.get_b2_grads(), "b2");
  expect_vectors_match(layer_st.get_b_skip_grads(), layer_mt.get_b_skip_grads(), "b_skip");
  expect_vectors_match(layer_st.get_ln_gain_grads(), layer_mt.get_ln_gain_grads(), "ln_gain");
  expect_vectors_match(layer_st.get_ln_bias_grads(), layer_mt.get_ln_bias_grads(), "ln_bias");
}
