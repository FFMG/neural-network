#include <gtest/gtest.h>
#include "layers/tcnlayer.h"
#include "test_helper.h"
#include <vector>


using namespace myoddweb::nn;
using namespace test_helper;

namespace
{
TcnLayer make_mt_layer(
  unsigned num_neurons_in_previous_layer,
  unsigned layer_size,
  unsigned kernel_size,
  unsigned dilation,
  int number_of_threads)
{
  TcnLayer layer(
    1,
    num_neurons_in_previous_layer,
    layer_size,
    kernel_size,
    dilation,
    0.0,
    Layer::Role::Hidden,
    activation(activation::method::tanh, 0.0),
    OptimiserType::SGD,
    -1,
    0.0,
    nullptr,
    number_of_threads,
    true,
    0.0,
    std::nullopt);
  return layer;
}
} // namespace

class TcnLayerMTTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(TcnLayerMTTest, ForwardFeedMTConsistency) {
    const unsigned N_in = 3, N_out = 4, K = 3, D = 2;
    const size_t T = 6;
    const size_t batch_size = 16;

    TcnLayer layer_st = make_mt_layer(N_in, N_out, K, D, 1);
    TcnLayer layer_mt = make_mt_layer(N_in, N_out, K, D, static_cast<int>(get_test_threads()));

    std::vector<double> w_values(static_cast<size_t>(K) * N_in * N_out);
    for (size_t i = 0; i < w_values.size(); ++i)
    {
      w_values[i] = std::sin(static_cast<double>(i) * 0.37) * 0.5;
    }
    std::vector<double> b_values(N_out);
    for (size_t i = 0; i < b_values.size(); ++i)
    {
      b_values[i] = std::cos(static_cast<double>(i) * 0.21) * 0.1;
    }
    layer_st.set_w_values(w_values);
    layer_st.set_b_values(b_values);
    layer_mt.set_w_values(w_values);
    layer_mt.set_b_values(b_values);

    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);
    MockLayer previous_layer(0, N_in);

    std::vector<std::vector<double>> x_seqs(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
      x_seqs[b].resize(T * N_in);
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

TEST_F(TcnLayerMTTest, BackwardFeedMTConsistency) {
    const unsigned N_in = 3, N_out = 4, K = 3, D = 2;
    const size_t T = 6;
    const size_t batch_size = 16;

    TcnLayer layer_st = make_mt_layer(N_in, N_out, K, D, 1);
    TcnLayer layer_mt = make_mt_layer(N_in, N_out, K, D, static_cast<int>(get_test_threads()));

    std::vector<double> w_values(static_cast<size_t>(K) * N_in * N_out);
    for (size_t i = 0; i < w_values.size(); ++i)
    {
      w_values[i] = std::sin(static_cast<double>(i) * 0.29) * 0.4;
    }
    std::vector<double> b_values(N_out);
    for (size_t i = 0; i < b_values.size(); ++i)
    {
      b_values[i] = std::cos(static_cast<double>(i) * 0.17) * 0.1;
    }
    layer_st.set_w_values(w_values);
    layer_st.set_b_values(b_values);
    layer_mt.set_w_values(w_values);
    layer_mt.set_b_values(b_values);

    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);
    MockLayer previous_layer(0, N_in);

    std::vector<std::vector<double>> x_seqs(batch_size);
    std::vector<std::vector<double>> deltas(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
      x_seqs[b].resize(T * N_in);
      for (size_t i = 0; i < x_seqs[b].size(); ++i)
      {
        x_seqs[b][i] = std::sin(static_cast<double>(b * 17 + i) * 0.13);
      }
      batch_go_st[b].set_rnn_outputs(0, x_seqs[b].data(), x_seqs[b].size());
      batch_go_mt[b].set_rnn_outputs(0, x_seqs[b].data(), x_seqs[b].size());

      deltas[b].resize(T * N_out);
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

    const auto w_grads_st = layer_st.get_w_grads();
    const auto w_grads_mt = layer_mt.get_w_grads();
    ASSERT_EQ(w_grads_st.size(), w_grads_mt.size());
    for (size_t i = 0; i < w_grads_st.size(); ++i)
    {
      EXPECT_NEAR(w_grads_st[i], w_grads_mt[i], 1e-12) << "w index " << i;
    }

    const auto b_grads_st = layer_st.get_b_grads();
    const auto b_grads_mt = layer_mt.get_b_grads();
    ASSERT_EQ(b_grads_st.size(), b_grads_mt.size());
    for (size_t i = 0; i < b_grads_st.size(); ++i)
    {
      EXPECT_NEAR(b_grads_st[i], b_grads_mt[i], 1e-12) << "b index " << i;
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

TEST_F(TcnLayerMTTest, OddBatchSizeAllGradsThreadCountInvariance) {
    const unsigned N_in = 4, N_out = 3, K = 3, D = 2;
    const size_t T = 5;
    const size_t batch_size = 7; // Odd batch size to test thread partitioning

    TcnLayer layer_st = make_mt_layer(N_in, N_out, K, D, 1);
    TcnLayer layer_mt = make_mt_layer(N_in, N_out, K, D, static_cast<int>(get_test_threads()));

    std::vector<double> w_values(static_cast<size_t>(K) * N_in * N_out);
    for (size_t i = 0; i < w_values.size(); ++i)
    {
      w_values[i] = std::sin(static_cast<double>(i) * 0.23) * 0.45;
    }
    std::vector<double> b_values(N_out);
    for (size_t i = 0; i < b_values.size(); ++i)
    {
      b_values[i] = std::cos(static_cast<double>(i) * 0.31) * 0.15;
    }
    layer_st.set_w_values(w_values);
    layer_st.set_b_values(b_values);
    layer_mt.set_w_values(w_values);
    layer_mt.set_b_values(b_values);

    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);
    MockLayer previous_layer(0, N_in);

    std::vector<std::vector<double>> x_seqs(batch_size);
    std::vector<std::vector<double>> deltas(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
      x_seqs[b].resize(T * N_in);
      for (size_t i = 0; i < x_seqs[b].size(); ++i)
      {
        x_seqs[b][i] = std::sin(static_cast<double>(b * 19 + i) * 0.17);
      }
      batch_go_st[b].set_rnn_outputs(0, x_seqs[b].data(), x_seqs[b].size());
      batch_go_mt[b].set_rnn_outputs(0, x_seqs[b].data(), x_seqs[b].size());

      deltas[b].resize(T * N_out);
      for (size_t i = 0; i < deltas[b].size(); ++i)
      {
        deltas[b][i] = std::cos(static_cast<double>(b * 11 + i) * 0.23) * 0.4;
      }
    }

    layer_st.calculate_forward_feed(batch_go_st, previous_layer, {}, batch_hs_st, batch_size, true);
    layer_mt.calculate_forward_feed(batch_go_mt, previous_layer, {}, batch_hs_mt, batch_size, true);

    layer_st.calculate_hidden_gradients_from_output_gradients(batch_go_st, deltas, batch_hs_st, batch_size, 0);
    layer_mt.calculate_hidden_gradients_from_output_gradients(batch_go_mt, deltas, batch_hs_mt, batch_size, 0);

    layer_st.calculate_and_store_gradients(batch_go_st, batch_hs_st, previous_layer, batch_size, 0);
    layer_mt.calculate_and_store_gradients(batch_go_mt, batch_hs_mt, previous_layer, batch_size, 0);

    const auto w_grads_st = layer_st.get_w_grads();
    const auto w_grads_mt = layer_mt.get_w_grads();
    ASSERT_EQ(w_grads_st.size(), w_grads_mt.size());
    for (size_t i = 0; i < w_grads_st.size(); ++i)
    {
      EXPECT_NEAR(w_grads_st[i], w_grads_mt[i], 1e-12) << "w_grads index " << i;
    }

    const auto b_grads_st = layer_st.get_b_grads();
    const auto b_grads_mt = layer_mt.get_b_grads();
    ASSERT_EQ(b_grads_st.size(), b_grads_mt.size());
    for (size_t i = 0; i < b_grads_st.size(); ++i)
    {
      EXPECT_NEAR(b_grads_st[i], b_grads_mt[i], 1e-12) << "b_grads index " << i;
    }

    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto out_st = batch_go_st[b].get_rnn_outputs(1);
      const auto out_mt = batch_go_mt[b].get_rnn_outputs(1);
      ASSERT_EQ(out_st.size(), out_mt.size());
      for (size_t i = 0; i < out_st.size(); ++i)
      {
        EXPECT_NEAR(out_st[i], out_mt[i], 1e-12) << "batch " << b << " out index " << i;
      }

      const auto dx_st = batch_go_st[b].get_rnn_gradients(1);
      const auto dx_mt = batch_go_mt[b].get_rnn_gradients(1);
      ASSERT_EQ(dx_st.size(), dx_mt.size());
      for (size_t i = 0; i < dx_st.size(); ++i)
      {
        EXPECT_NEAR(dx_st[i], dx_mt[i], 1e-12) << "batch " << b << " dx index " << i;
      }
    }
}
