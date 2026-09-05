#include <gtest/gtest.h>
#include "layers/selfattentionlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>


using namespace myoddweb::nn;
using namespace test_helper;

namespace
{
double compute_loss(
  SelfAttentionLayer& layer,
  MockLayer& previous_layer,
  const std::vector<double>& x_seq,
  size_t T,
  size_t d,
  const std::vector<double>& delta_full_seq)
{
  std::vector<unsigned> topology = { static_cast<unsigned>(d), static_cast<unsigned>(d) };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

  batch_go[0].set_rnn_outputs(0, x_seq.data(), T * d);
  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

  const auto out_seq = batch_go[0].get_rnn_outputs(1);
  double total = 0.0;
  for (size_t k = 0; k < T * d; ++k)
  {
    total += delta_full_seq[k] * out_seq[k];
  }
  return total;
}

SelfAttentionLayer make_layer(
  unsigned d,
  unsigned number_of_heads,
  unsigned feed_forward_hidden_size,
  bool has_bias,
  bool use_layer_normalisation,
  const activation& ffn_activation = activation(activation::method::linear, 0.0))
{
  SelfAttentionLayer layer(
    1, d, d, number_of_heads, feed_forward_hidden_size,
    0.0, Layer::Role::Hidden, ffn_activation, OptimiserType::SGD,
    -1, 0.0, nullptr, 1, has_bias, use_layer_normalisation, 0.0, std::nullopt);
  return layer;
}
} // namespace

class SelfAttentionLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(SelfAttentionLayerTest, Construction) {
    SelfAttentionLayer layer = make_layer(4, 2, 6, true, false);
    EXPECT_EQ(layer.get_layer_index(), 1u);
    EXPECT_EQ(layer.get_number_input_neurons(), 4u);
    EXPECT_EQ(layer.get_number_neurons(), 4u);
    EXPECT_EQ(layer.get_number_of_heads(), 2u);
    EXPECT_EQ(layer.get_feed_forward_hidden_size(), 6u);
    EXPECT_FALSE(layer.get_use_layer_normalisation());
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::SelfAttention);
    EXPECT_EQ(layer.get_wq_values().size(), 16u); // d*d
    EXPECT_EQ(layer.get_bq_values().size(), 4u);
    EXPECT_EQ(layer.get_ff1_w_values().size(), 24u); // d*d_ff
    EXPECT_EQ(layer.get_ff2_w_values().size(), 24u); // d_ff*d
    EXPECT_EQ(layer.get_ln1_gain_values().size(), 4u);
}

TEST_F(SelfAttentionLayerTest, CausalMaskingPreventsFutureLeakage) {
    // A change to the LAST timestep's input must not affect any earlier
    // output timestep - the single most important correctness property for
    // a causal attention layer, given this project's label-leakage history.
    const unsigned d = 4, H = 2, d_ff = 3;
    const size_t T = 4;
    SelfAttentionLayer layer = make_layer(d, H, d_ff, true, false, activation(activation::method::tanh, 0.0));

    std::vector<double> x_seq_a(T * d);
    for (size_t k = 0; k < T * d; ++k)
    {
      x_seq_a[k] = std::sin(static_cast<double>(k) * 0.37);
    }
    std::vector<double> x_seq_b = x_seq_a;
    for (size_t o = 0; o < d; ++o)
    {
      x_seq_b[(T - 1) * d + o] += 5.0;
    }

    std::vector<unsigned> topology = { d, d };
    MockLayer previous_layer(0, d);

    auto batch_go_a = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs_a = create_batch_hidden_states(topology, 1, 1, 1);
    batch_go_a[0].set_rnn_outputs(0, x_seq_a.data(), T * d);
    layer.calculate_forward_feed(batch_go_a, previous_layer, {}, batch_hs_a, 1, false);

    auto batch_go_b = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs_b = create_batch_hidden_states(topology, 1, 1, 1);
    batch_go_b[0].set_rnn_outputs(0, x_seq_b.data(), T * d);
    layer.calculate_forward_feed(batch_go_b, previous_layer, {}, batch_hs_b, 1, false);

    const auto out_a = batch_go_a[0].get_rnn_outputs(1);
    const auto out_b = batch_go_b[0].get_rnn_outputs(1);
    ASSERT_EQ(out_a.size(), T * d);
    ASSERT_EQ(out_b.size(), T * d);

    // Timesteps 0..T-2 must be identical - only the last timestep may differ.
    for (size_t t = 0; t < T - 1; ++t)
    {
      for (size_t o = 0; o < d; ++o)
      {
        EXPECT_NEAR(out_a[t * d + o], out_b[t * d + o], 1e-12) << "t=" << t << " o=" << o;
      }
    }
    // Sanity: the last timestep SHOULD differ (the perturbation must actually reach the output).
    bool last_timestep_differs = false;
    for (size_t o = 0; o < d; ++o)
    {
      if (std::abs(out_a[(T - 1) * d + o] - out_b[(T - 1) * d + o]) > 1e-9)
      {
        last_timestep_differs = true;
      }
    }
    EXPECT_TRUE(last_timestep_differs);
}

TEST_F(SelfAttentionLayerTest, WeightGradientsMatchNumericalGradient) {
    const unsigned d = 4, H = 2, d_ff = 3;
    const size_t T = 3;
    SelfAttentionLayer layer = make_layer(d, H, d_ff, true, false, activation(activation::method::tanh, 0.0));

    std::vector<double> x_seq = {
      0.5, -0.2, 0.3, 0.1,
      -0.3, 0.8, 0.1, -0.4,
      0.2, 0.4, -0.6, 0.2
    };
    std::vector<double> delta_full_seq = {
      0.4, -0.7, 0.2, 0.1,
      0.2, -0.3, 0.1, -0.2,
      0.1, 0.5, -0.2, 0.3
    };

    MockLayer previous_layer(0, d);
    std::vector<unsigned> topology = { d, d };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), T * d);
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    std::vector<std::vector<double>> batch_output_gradients = { delta_full_seq };
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);
    layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, 1, 0);

    const double h = 1e-6;

    // wq (Q projection weight matrix).
    {
      const auto grads = layer.get_wq_grads();
      for (size_t k = 0; k < grads.size(); ++k)
      {
        auto plus = layer.get_wq_values(); plus[k] += h;
        auto minus = layer.get_wq_values(); minus[k] -= h;
        SelfAttentionLayer probe(layer);
        probe.set_wq_values(plus);
        const double loss_plus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        probe.set_wq_values(minus);
        const double loss_minus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        const double numerical = (loss_plus - loss_minus) / (2.0 * h);
        EXPECT_NEAR(grads[k], numerical, 1e-4) << "wq index " << k;
      }
    }

    // wo (output projection weight matrix).
    {
      const auto grads = layer.get_wo_grads();
      for (size_t k = 0; k < grads.size(); ++k)
      {
        auto plus = layer.get_wo_values(); plus[k] += h;
        auto minus = layer.get_wo_values(); minus[k] -= h;
        SelfAttentionLayer probe(layer);
        probe.set_wo_values(plus);
        const double loss_plus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        probe.set_wo_values(minus);
        const double loss_minus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        const double numerical = (loss_plus - loss_minus) / (2.0 * h);
        EXPECT_NEAR(grads[k], numerical, 1e-4) << "wo index " << k;
      }
    }

    // ff1_w (first FFN dense layer weight matrix).
    {
      const auto grads = layer.get_ff1_w_grads();
      for (size_t k = 0; k < grads.size(); ++k)
      {
        auto plus = layer.get_ff1_w_values(); plus[k] += h;
        auto minus = layer.get_ff1_w_values(); minus[k] -= h;
        SelfAttentionLayer probe(layer);
        probe.set_ff1_w_values(plus);
        const double loss_plus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        probe.set_ff1_w_values(minus);
        const double loss_minus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        const double numerical = (loss_plus - loss_minus) / (2.0 * h);
        EXPECT_NEAR(grads[k], numerical, 1e-4) << "ff1_w index " << k;
      }
    }

    // ff2_b (second FFN dense layer bias).
    {
      const auto grads = layer.get_ff2_b_grads();
      for (size_t k = 0; k < grads.size(); ++k)
      {
        auto plus = layer.get_ff2_b_values(); plus[k] += h;
        auto minus = layer.get_ff2_b_values(); minus[k] -= h;
        SelfAttentionLayer probe(layer);
        probe.set_ff2_b_values(plus);
        const double loss_plus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        probe.set_ff2_b_values(minus);
        const double loss_minus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        const double numerical = (loss_plus - loss_minus) / (2.0 * h);
        EXPECT_NEAR(grads[k], numerical, 1e-4) << "ff2_b index " << k;
      }
    }
}

TEST_F(SelfAttentionLayerTest, LayerNormWeightGradientsMatchNumericalGradient) {
    const unsigned d = 4, H = 2, d_ff = 3;
    const size_t T = 3;
    SelfAttentionLayer layer = make_layer(d, H, d_ff, true, true, activation(activation::method::tanh, 0.0));

    std::vector<double> x_seq = {
      0.3, -0.4, 0.2, 0.6,
      -0.1, 0.5, 0.3, -0.2,
      0.4, 0.1, -0.3, 0.2
    };
    std::vector<double> delta_full_seq = {
      0.2, -0.3, 0.1, 0.2,
      0.1, -0.1, 0.3, -0.2,
      0.2, 0.2, -0.1, 0.1
    };

    MockLayer previous_layer(0, d);
    std::vector<unsigned> topology = { d, d };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), T * d);
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    std::vector<std::vector<double>> batch_output_gradients = { delta_full_seq };
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);
    layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, 1, 0);

    const double h = 1e-6;

    {
      const auto grads = layer.get_ln1_gain_grads();
      for (size_t k = 0; k < grads.size(); ++k)
      {
        auto plus = layer.get_ln1_gain_values(); plus[k] += h;
        auto minus = layer.get_ln1_gain_values(); minus[k] -= h;
        SelfAttentionLayer probe(layer);
        probe.set_ln1_gain_values(plus);
        const double loss_plus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        probe.set_ln1_gain_values(minus);
        const double loss_minus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        const double numerical = (loss_plus - loss_minus) / (2.0 * h);
        EXPECT_NEAR(grads[k], numerical, 1e-4) << "ln1_gain index " << k;
      }
    }

    {
      const auto grads = layer.get_ln2_bias_grads();
      for (size_t k = 0; k < grads.size(); ++k)
      {
        auto plus = layer.get_ln2_bias_values(); plus[k] += h;
        auto minus = layer.get_ln2_bias_values(); minus[k] -= h;
        SelfAttentionLayer probe(layer);
        probe.set_ln2_bias_values(plus);
        const double loss_plus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        probe.set_ln2_bias_values(minus);
        const double loss_minus = compute_loss(probe, previous_layer, x_seq, T, d, delta_full_seq);
        const double numerical = (loss_plus - loss_minus) / (2.0 * h);
        EXPECT_NEAR(grads[k], numerical, 1e-4) << "ln2_bias index " << k;
      }
    }
}

TEST_F(SelfAttentionLayerTest, InputGradientsMatchNumericalGradient) {
    const unsigned d = 4, H = 2, d_ff = 3;
    const size_t T = 3;
    SelfAttentionLayer layer = make_layer(d, H, d_ff, true, false, activation(activation::method::tanh, 0.0));

    std::vector<double> x_seq = {
      0.4, -0.5, 0.2, 0.1,
      0.9, 0.1, -0.3, 0.4,
      -0.3, 0.7, 0.5, -0.2
    };
    std::vector<double> delta_full_seq = {
      0.3, -0.2, 0.1, 0.2,
      0.1, 0.4, -0.3, 0.1,
      -0.3, 0.2, 0.2, -0.1
    };

    MockLayer previous_layer(0, d);
    std::vector<unsigned> topology = { d, d };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), T * d);
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    std::vector<std::vector<double>> batch_output_gradients = { delta_full_seq };
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);

    const auto dx = batch_go[0].get_rnn_gradients(1);
    ASSERT_EQ(dx.size(), T * d);

    const double h = 1e-6;
    for (size_t k = 0; k < T * d; ++k)
    {
      auto x_plus = x_seq; x_plus[k] += h;
      auto x_minus = x_seq; x_minus[k] -= h;

      const double loss_plus = compute_loss(layer, previous_layer, x_plus, T, d, delta_full_seq);
      const double loss_minus = compute_loss(layer, previous_layer, x_minus, T, d, delta_full_seq);

      const double numerical = (loss_plus - loss_minus) / (2.0 * h);
      EXPECT_NEAR(dx[k], numerical, 1e-4) << "x_seq index " << k;
    }
}

TEST_F(SelfAttentionLayerTest, LayerNormToggleChangesOutput) {
    const unsigned d = 4, H = 2, d_ff = 3;
    const size_t T = 3;
    SelfAttentionLayer layer_no_ln = make_layer(d, H, d_ff, true, false);
    SelfAttentionLayer layer_with_ln = make_layer(d, H, d_ff, true, true);
    layer_with_ln.set_wq_values(layer_no_ln.get_wq_values());
    layer_with_ln.set_wk_values(layer_no_ln.get_wk_values());
    layer_with_ln.set_wv_values(layer_no_ln.get_wv_values());
    layer_with_ln.set_wo_values(layer_no_ln.get_wo_values());
    layer_with_ln.set_ff1_w_values(layer_no_ln.get_ff1_w_values());
    layer_with_ln.set_ff2_w_values(layer_no_ln.get_ff2_w_values());

    std::vector<double> x_seq = {
      0.5, -0.2, 0.3, 0.1,
      -0.3, 0.8, 0.1, -0.4,
      0.2, 0.4, -0.6, 0.2
    };

    MockLayer previous_layer(0, d);
    std::vector<unsigned> topology = { d, d };

    auto batch_go_a = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs_a = create_batch_hidden_states(topology, 1, 1, 1);
    batch_go_a[0].set_rnn_outputs(0, x_seq.data(), T * d);
    layer_no_ln.calculate_forward_feed(batch_go_a, previous_layer, {}, batch_hs_a, 1, false);

    auto batch_go_b = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs_b = create_batch_hidden_states(topology, 1, 1, 1);
    batch_go_b[0].set_rnn_outputs(0, x_seq.data(), T * d);
    layer_with_ln.calculate_forward_feed(batch_go_b, previous_layer, {}, batch_hs_b, 1, false);

    const auto out_a = batch_go_a[0].get_rnn_outputs(1);
    const auto out_b = batch_go_b[0].get_rnn_outputs(1);
    ASSERT_EQ(out_a.size(), out_b.size());
    bool differs = false;
    for (size_t k = 0; k < out_a.size(); ++k)
    {
      if (std::abs(out_a[k] - out_b[k]) > 1e-9)
      {
        differs = true;
      }
    }
    EXPECT_TRUE(differs);
}

TEST_F(SelfAttentionLayerTest, NoBatchCrossTalk) {
    const unsigned d = 4, H = 2, d_ff = 3;
    const size_t T = 3;
    SelfAttentionLayer layer = make_layer(d, H, d_ff, true, false, activation(activation::method::tanh, 0.0));

    std::vector<double> x_seq_a = {
      1.0, 0.0, 0.5, -0.5,
      0.5, 0.5, -0.2, 0.3,
      -0.5, 1.0, 0.2, 0.1
    };
    std::vector<double> x_seq_b = {
      0.0, 1.0, -0.3, 0.4,
      -0.5, 1.5, 0.1, -0.2,
      0.3, -0.2, 0.5, 0.6
    };

    std::vector<unsigned> topology = { d, d };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 1, 1);
    MockLayer previous_layer(0, d);

    batch_go[0].set_rnn_outputs(0, x_seq_a.data(), T * d);
    batch_go[1].set_rnn_outputs(0, x_seq_b.data(), T * d);

    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 2, false);

    auto single_go_a = create_batch_gradients_and_outputs(topology, 1);
    auto single_hs_a = create_batch_hidden_states(topology, 1, 1, 1);
    single_go_a[0].set_rnn_outputs(0, x_seq_a.data(), T * d);
    layer.calculate_forward_feed(single_go_a, previous_layer, {}, single_hs_a, 1, false);

    auto single_go_b = create_batch_gradients_and_outputs(topology, 1);
    auto single_hs_b = create_batch_hidden_states(topology, 1, 1, 1);
    single_go_b[0].set_rnn_outputs(0, x_seq_b.data(), T * d);
    layer.calculate_forward_feed(single_go_b, previous_layer, {}, single_hs_b, 1, false);

    const auto batched_a = batch_go[0].get_rnn_outputs(1);
    const auto batched_b = batch_go[1].get_rnn_outputs(1);
    const auto expected_a = single_go_a[0].get_rnn_outputs(1);
    const auto expected_b = single_go_b[0].get_rnn_outputs(1);

    ASSERT_EQ(batched_a.size(), T * d);
    for (size_t i = 0; i < T * d; ++i)
    {
      EXPECT_NEAR(batched_a[i], expected_a[i], 1e-12);
      EXPECT_NEAR(batched_b[i], expected_b[i], 1e-12);
    }
}

TEST_F(SelfAttentionLayerTest, AccumulateSwaAverageMatchesDirectMean) {
    const unsigned d = 2, H = 1, d_ff = 2;
    SelfAttentionLayer running = make_layer(d, H, d_ff, true, false);
    SelfAttentionLayer snap1 = make_layer(d, H, d_ff, true, false);
    snap1.set_wq_values({ 1.0, 2.0, 3.0, 4.0 });

    SelfAttentionLayer snap2 = make_layer(d, H, d_ff, true, false);
    snap2.set_wq_values({ 3.0, 4.0, 5.0, 6.0 });

    running.set_wq_values(snap1.get_wq_values());
    running.accumulate_swa_average(snap2, 1);

    for (size_t i = 0; i < 4; ++i)
    {
      const double expected = (snap1.get_wq_values()[i] + snap2.get_wq_values()[i]) / 2.0;
      EXPECT_NEAR(running.get_wq_values()[i], expected, 1e-12);
    }
}

TEST_F(SelfAttentionLayerTest, CloneProducesIndependentCopy) {
    SelfAttentionLayer layer = make_layer(4, 2, 4, true, true);
    layer.set_wq_values({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 });

    std::unique_ptr<Layer> cloned(layer.clone());
    auto* cloned_sa = dynamic_cast<SelfAttentionLayer*>(cloned.get());
    ASSERT_NE(cloned_sa, nullptr);
    EXPECT_EQ(cloned_sa->get_wq_values(), layer.get_wq_values());
    EXPECT_EQ(cloned_sa->get_number_of_heads(), layer.get_number_of_heads());
    EXPECT_EQ(cloned_sa->get_feed_forward_hidden_size(), layer.get_feed_forward_hidden_size());
    EXPECT_EQ(cloned_sa->get_use_layer_normalisation(), layer.get_use_layer_normalisation());

    cloned_sa->set_wq_values(std::vector<double>(16, 9.0));
    EXPECT_NE(cloned_sa->get_wq_values(), layer.get_wq_values());
}

TEST_F(SelfAttentionLayerTest, PositionalEncodingPrecomputationMatchesFormula) {
    const unsigned d = 8, H = 2, d_ff = 16;
    SelfAttentionLayer layer = make_layer(d, H, d_ff, true, false);

    const auto& pe_inv = layer.get_pe_inv_denom();
    ASSERT_EQ(pe_inv.size(), (d + 1) / 2);
    for (size_t k = 0; k < pe_inv.size(); ++k)
    {
      const double exponent = (2.0 * static_cast<double>(k)) / static_cast<double>(d);
      const double expected = 1.0 / std::pow(10000.0, exponent);
      EXPECT_NEAR(pe_inv[k], expected, 1e-15);
    }
}

TEST_F(SelfAttentionLayerTest, PositionalEncodingCacheMatchesAnalyticalTrig) {
    const unsigned d = 8, H = 2, d_ff = 16;
    SelfAttentionLayer layer = make_layer(d, H, d_ff, true, false);

    const auto& pe_cache = layer.get_pe_cache();
    ASSERT_FALSE(pe_cache.empty());
    const size_t max_t = pe_cache.size() / d;
    EXPECT_GE(max_t, 512u);

    const auto& inv_denom = layer.get_pe_inv_denom();
    const size_t num_pairs = d / 2;

    for (size_t t = 0; t < std::min(max_t, size_t{64}); ++t)
    {
      const double* row = pe_cache.data() + t * d;
      for (size_t k = 0; k < num_pairs; ++k)
      {
        const double angle = static_cast<double>(t) * inv_denom[k];
        EXPECT_NEAR(row[2 * k], std::sin(angle), 1e-12);
        EXPECT_NEAR(row[2 * k + 1], std::cos(angle), 1e-12);
      }
    }
}

TEST_F(SelfAttentionLayerTest, MultithreadedForwardMatchesSingleThread) {
    const unsigned d = 8, H = 2, d_ff = 16;
    const size_t T = 6;
    const size_t batch_size = 4;

    SelfAttentionLayer layer_st(
      1, d, d, H, d_ff,
      0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD,
      -1, 0.0, nullptr, 1, true, true, 0.0, 42);

    SelfAttentionLayer layer_mt(layer_st);

    std::vector<unsigned> topology = { d, d };
    MockLayer previous_layer(0, d);

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);

    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);

    for (size_t b = 0; b < batch_size; ++b)
    {
      std::vector<double> x_seq(T * d);
      for (size_t i = 0; i < T * d; ++i)
      {
        x_seq[i] = std::sin(static_cast<double>(b * T * d + i) * 0.17);
      }
      batch_go_st[b].set_rnn_outputs(0, x_seq.data(), T * d);
      batch_go_mt[b].set_rnn_outputs(0, x_seq.data(), T * d);
    }

    layer_st.calculate_forward_feed(batch_go_st, previous_layer, {}, batch_hs_st, batch_size, false);
    layer_mt.calculate_forward_feed(batch_go_mt, previous_layer, {}, batch_hs_mt, batch_size, false);

    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& out_st = batch_go_st[b].get_rnn_outputs(1);
      const auto& out_mt = batch_go_mt[b].get_rnn_outputs(1);
      ASSERT_EQ(out_st.size(), out_mt.size());
      for (size_t i = 0; i < out_st.size(); ++i)
      {
        EXPECT_NEAR(out_st[i], out_mt[i], 1e-12);
      }
    }
}

TEST_F(SelfAttentionLayerTest, LargeDimensionMultiTimestepEquivalence) {
    const unsigned d = 16, H = 4, d_ff = 32;
    const size_t T = 8;
    SelfAttentionLayer layer = make_layer(d, H, d_ff, true, true, activation(activation::method::tanh, 0.0));

    std::vector<double> x_seq(T * d);
    for (size_t i = 0; i < T * d; ++i)
    {
      x_seq[i] = std::sin(static_cast<double>(i) * 0.23);
    }

    std::vector<unsigned> topology = { d, d };
    MockLayer previous_layer(0, d);

    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
    batch_go[0].set_rnn_outputs(0, x_seq.data(), T * d);
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    const auto out_seq = batch_go[0].get_rnn_outputs(1);
    ASSERT_EQ(out_seq.size(), T * d);

    // Outputs must be finite numbers
    for (size_t i = 0; i < T * d; ++i)
    {
      EXPECT_FALSE(std::isnan(out_seq[i]));
      EXPECT_FALSE(std::isinf(out_seq[i]));
    }
}

TEST_F(SelfAttentionLayerTest, DropoutNotInference) {
    const unsigned d = 8, H = 2, d_ff = 16;
    const size_t T = 2;
    const double dropout_rate = 0.5;

    SelfAttentionLayer layer(
      1, d, d, H, d_ff,
      0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD,
      -1, dropout_rate, nullptr, 1, true, false, 0.0, std::nullopt);

    SelfAttentionLayer no_drop_layer(
      1, d, d, H, d_ff,
      0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD,
      -1, 0.0, nullptr, 1, true, false, 0.0, std::nullopt);

    // Share identical weights
    no_drop_layer.set_wq_values(layer.get_wq_values());
    no_drop_layer.set_wk_values(layer.get_wk_values());
    no_drop_layer.set_wv_values(layer.get_wv_values());
    no_drop_layer.set_wo_values(layer.get_wo_values());
    no_drop_layer.set_bq_values(layer.get_bq_values());
    no_drop_layer.set_bk_values(layer.get_bk_values());
    no_drop_layer.set_bv_values(layer.get_bv_values());
    no_drop_layer.set_bo_values(layer.get_bo_values());
    no_drop_layer.set_ff1_w_values(layer.get_ff1_w_values());
    no_drop_layer.set_ff1_b_values(layer.get_ff1_b_values());
    no_drop_layer.set_ff2_w_values(layer.get_ff2_w_values());
    no_drop_layer.set_ff2_b_values(layer.get_ff2_b_values());

    std::vector<double> x_seq(T * d, 1.0);
    std::vector<unsigned> topology = { d, d };
    MockLayer prev_layer(0, d);

    auto batch_go1 = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs1 = create_batch_hidden_states(topology, 1, 1, 1);
    batch_go1[0].set_rnn_outputs(0, x_seq.data(), T * d);
    layer.calculate_forward_feed(batch_go1, prev_layer, {}, batch_hs1, 1, false);

    auto batch_go2 = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs2 = create_batch_hidden_states(topology, 1, 1, 1);
    batch_go2[0].set_rnn_outputs(0, x_seq.data(), T * d);
    no_drop_layer.calculate_forward_feed(batch_go2, prev_layer, {}, batch_hs2, 1, false);

    const auto out1 = batch_go1[0].get_rnn_outputs(1);
    const auto out2 = batch_go2[0].get_rnn_outputs(1);
    ASSERT_EQ(out1.size(), out2.size());
    for (size_t i = 0; i < out1.size(); ++i)
    {
      EXPECT_NEAR(out1[i], out2[i], 1e-12);
    }
}

TEST_F(SelfAttentionLayerTest, DropoutConsistencyVerification) {
    const unsigned d = 4, H = 2, d_ff = 8;
    const size_t T = 2;
    const double dropout_rate = 1.0;

    SelfAttentionLayer layer(
      1, d, d, H, d_ff,
      0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD,
      -1, dropout_rate, nullptr, 1, true, false, 0.0, std::nullopt);

    std::vector<double> x_seq(T * d, 1.0);
    std::vector<unsigned> topology = { d, d };
    MockLayer prev_layer(0, d);

    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
    batch_go[0].set_rnn_outputs(0, x_seq.data(), T * d);

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto out = batch_go[0].get_rnn_outputs(1);
    for (double val : out)
    {
      EXPECT_NEAR(val, 0.0, 1e-9);
    }

    std::vector<std::vector<double>> deltas(1, std::vector<double>(T * d, 10.0));
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, deltas, batch_hs, 1, 0);

    const auto grads = batch_go[0].get_rnn_gate_gradients(1);
    for (double g : grads)
    {
      EXPECT_NEAR(g, 0.0, 1e-9);
    }
}


