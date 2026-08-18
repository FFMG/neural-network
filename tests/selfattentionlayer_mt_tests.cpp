#include <gtest/gtest.h>
#include "layers/selfattentionlayer.h"
#include "test_helper.h"
#include <vector>


using namespace myoddweb::nn;
using namespace test_helper;

namespace
{
SelfAttentionLayer make_mt_layer(unsigned d, unsigned number_of_heads, unsigned feed_forward_hidden_size, int number_of_threads)
{
  SelfAttentionLayer layer(
    1, d, d, number_of_heads, feed_forward_hidden_size,
    0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD,
    -1, 0.0, nullptr, number_of_threads, true, true, 0.0, std::nullopt);
  return layer;
}
} // namespace

// SelfAttentionLayer does not dispatch its own per-batch-item work across
// _task_queue_pool (unlike FFLayer/LSTMLayer/TcnLayer) - its math is
// single-threaded internally. This test therefore verifies that changing
// number_of_threads is inert (a regression guard), mirroring the same
// single-threaded-vs-"multi-threaded" numeric equivalence pattern every
// other layer's MT test suite already checks for its own threading.
class SelfAttentionLayerMTTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(SelfAttentionLayerMTTest, ForwardFeedThreadCountInvariance) {
    const unsigned d = 6, H = 2, d_ff = 5;
    const size_t T = 5;
    const size_t batch_size = 8;

    SelfAttentionLayer layer_st = make_mt_layer(d, H, d_ff, 1);
    SelfAttentionLayer layer_mt = make_mt_layer(d, H, d_ff, static_cast<int>(get_test_threads()));

    layer_st.set_wq_values(layer_mt.get_wq_values());
    layer_st.set_bq_values(layer_mt.get_bq_values());
    layer_st.set_wk_values(layer_mt.get_wk_values());
    layer_st.set_bk_values(layer_mt.get_bk_values());
    layer_st.set_wv_values(layer_mt.get_wv_values());
    layer_st.set_bv_values(layer_mt.get_bv_values());
    layer_st.set_wo_values(layer_mt.get_wo_values());
    layer_st.set_bo_values(layer_mt.get_bo_values());
    layer_st.set_ff1_w_values(layer_mt.get_ff1_w_values());
    layer_st.set_ff1_b_values(layer_mt.get_ff1_b_values());
    layer_st.set_ff2_w_values(layer_mt.get_ff2_w_values());
    layer_st.set_ff2_b_values(layer_mt.get_ff2_b_values());
    layer_st.set_ln1_gain_values(layer_mt.get_ln1_gain_values());
    layer_st.set_ln1_bias_values(layer_mt.get_ln1_bias_values());
    layer_st.set_ln2_gain_values(layer_mt.get_ln2_gain_values());
    layer_st.set_ln2_bias_values(layer_mt.get_ln2_bias_values());

    std::vector<unsigned> topology = { d, d };
    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);
    MockLayer previous_layer(0, d);

    std::vector<std::vector<double>> x_seqs(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
      x_seqs[b].resize(T * d);
      for (size_t i = 0; i < x_seqs[b].size(); ++i)
      {
        x_seqs[b][i] = std::sin(static_cast<double>(b * 13 + i) * 0.11);
      }
      batch_go_st[b].set_rnn_outputs(0, x_seqs[b].data(), x_seqs[b].size());
      batch_go_mt[b].set_rnn_outputs(0, x_seqs[b].data(), x_seqs[b].size());
    }

    layer_st.calculate_forward_feed(batch_go_st, previous_layer, {}, batch_hs_st, batch_size, true);
    layer_mt.calculate_forward_feed(batch_go_mt, previous_layer, {}, batch_hs_mt, batch_size, true);

    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto out_st = batch_go_st[b].get_rnn_outputs(1);
      const auto out_mt = batch_go_mt[b].get_rnn_outputs(1);
      ASSERT_EQ(out_st.size(), out_mt.size());
      for (size_t i = 0; i < out_st.size(); ++i)
      {
        EXPECT_NEAR(out_st[i], out_mt[i], 1e-12) << "batch " << b << " index " << i;
      }
    }
}

TEST_F(SelfAttentionLayerMTTest, BackwardFeedThreadCountInvariance) {
    const unsigned d = 6, H = 2, d_ff = 5;
    const size_t T = 5;
    const size_t batch_size = 8;

    SelfAttentionLayer layer_st = make_mt_layer(d, H, d_ff, 1);
    SelfAttentionLayer layer_mt = make_mt_layer(d, H, d_ff, static_cast<int>(get_test_threads()));

    layer_st.set_wq_values(layer_mt.get_wq_values());
    layer_st.set_bq_values(layer_mt.get_bq_values());
    layer_st.set_wk_values(layer_mt.get_wk_values());
    layer_st.set_bk_values(layer_mt.get_bk_values());
    layer_st.set_wv_values(layer_mt.get_wv_values());
    layer_st.set_bv_values(layer_mt.get_bv_values());
    layer_st.set_wo_values(layer_mt.get_wo_values());
    layer_st.set_bo_values(layer_mt.get_bo_values());
    layer_st.set_ff1_w_values(layer_mt.get_ff1_w_values());
    layer_st.set_ff1_b_values(layer_mt.get_ff1_b_values());
    layer_st.set_ff2_w_values(layer_mt.get_ff2_w_values());
    layer_st.set_ff2_b_values(layer_mt.get_ff2_b_values());
    layer_st.set_ln1_gain_values(layer_mt.get_ln1_gain_values());
    layer_st.set_ln1_bias_values(layer_mt.get_ln1_bias_values());
    layer_st.set_ln2_gain_values(layer_mt.get_ln2_gain_values());
    layer_st.set_ln2_bias_values(layer_mt.get_ln2_bias_values());

    std::vector<unsigned> topology = { d, d };
    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);
    MockLayer previous_layer(0, d);

    std::vector<std::vector<double>> x_seqs(batch_size);
    std::vector<std::vector<double>> deltas(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
      x_seqs[b].resize(T * d);
      for (size_t i = 0; i < x_seqs[b].size(); ++i)
      {
        x_seqs[b][i] = std::sin(static_cast<double>(b * 17 + i) * 0.13);
      }
      batch_go_st[b].set_rnn_outputs(0, x_seqs[b].data(), x_seqs[b].size());
      batch_go_mt[b].set_rnn_outputs(0, x_seqs[b].data(), x_seqs[b].size());

      deltas[b].resize(T * d);
      for (size_t i = 0; i < deltas[b].size(); ++i)
      {
        deltas[b][i] = std::cos(static_cast<double>(b * 7 + i) * 0.19) * 0.3;
      }
    }

    layer_st.calculate_forward_feed(batch_go_st, previous_layer, {}, batch_hs_st, batch_size, true);
    layer_mt.calculate_forward_feed(batch_go_mt, previous_layer, {}, batch_hs_mt, batch_size, true);

    layer_st.calculate_hidden_gradients_from_output_gradients(batch_go_st, deltas, batch_hs_st, batch_size, 0);
    layer_mt.calculate_hidden_gradients_from_output_gradients(batch_go_mt, deltas, batch_hs_mt, batch_size, 0);

    layer_st.calculate_and_store_gradients(batch_go_st, batch_hs_st, previous_layer, batch_size, 0);
    layer_mt.calculate_and_store_gradients(batch_go_mt, batch_hs_mt, previous_layer, batch_size, 0);

    const auto wq_grads_st = layer_st.get_wq_grads();
    const auto wq_grads_mt = layer_mt.get_wq_grads();
    ASSERT_EQ(wq_grads_st.size(), wq_grads_mt.size());
    for (size_t i = 0; i < wq_grads_st.size(); ++i)
    {
      EXPECT_NEAR(wq_grads_st[i], wq_grads_mt[i], 1e-12) << "wq index " << i;
    }

    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto dx_st = batch_go_st[b].get_rnn_gradients(1);
      const auto dx_mt = batch_go_mt[b].get_rnn_gradients(1);
      ASSERT_EQ(dx_st.size(), dx_mt.size());
      for (size_t i = 0; i < dx_st.size(); ++i)
      {
        EXPECT_NEAR(dx_st[i], dx_mt[i], 1e-12) << "batch " << b << " index " << i;
      }
    }
}
