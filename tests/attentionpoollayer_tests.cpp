#include <gtest/gtest.h>
#include "layers/attentionpoollayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <numeric>


using namespace myoddweb::nn;
using namespace test_helper;

namespace
{
// Reruns the real forward pass with the given wa/ba/v weights and the given
// h_seq (T*N, previous - recurrent - layer's full hidden-state sequence),
// then returns dot(delta, context) where context is this layer's raw
// (pre-activation, since the layer under test always uses a linear
// activation with no dropout) pooled output. Used as the scalar loss
// function for central-finite-difference gradient checks: since delta is
// held fixed, d(loss)/d(weight) or d(loss)/d(h_seq) is exactly what the
// layer's backward pass should produce.
double compute_loss(
  AttentionPoolLayer& layer,
  MockLayer& previous_layer,
  const std::vector<double>& h_seq,
  size_t T,
  size_t N,
  const std::vector<double>& delta)
{
  std::vector<unsigned> topology = { static_cast<unsigned>(N), static_cast<unsigned>(N) };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

  batch_go[0].set_rnn_outputs(0, h_seq.data(), T * N);
  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

  const auto context = batch_go[0].get_outputs(1);
  double total = 0.0;
  for (size_t i = 0; i < N; ++i)
  {
    total += delta[i] * context[i];
  }
  return total;
}

AttentionPoolLayer make_layer(unsigned N, unsigned d_a, bool has_bias)
{
  AttentionPoolLayer layer(
    1, N, d_a, 0.0, Layer::Role::Hidden,
    activation(activation::method::linear, 0.0), OptimiserType::SGD, 0.0, 1, has_bias, 0.0);
  return layer;
}
} // namespace

class AttentionPoolLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(AttentionPoolLayerTest, Construction) {
    AttentionPoolLayer layer = make_layer(3, 2, true);
    EXPECT_EQ(layer.get_layer_index(), 1u);
    EXPECT_EQ(layer.get_number_input_neurons(), 3u);
    EXPECT_EQ(layer.get_number_neurons(), 3u);
    EXPECT_EQ(layer.get_attention_hidden_size(), 2u);
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::AttentionPool);
    EXPECT_EQ(layer.get_wa_values().size(), 6u); // N * d_a
    EXPECT_EQ(layer.get_ba_values().size(), 2u); // d_a
    EXPECT_EQ(layer.get_v_values().size(), 2u);  // d_a
}

TEST_F(AttentionPoolLayerTest, ForwardHandComputedExample) {
    // N=2, d_a=1: score_t = v * tanh(Wa . h_t + ba), a single scalar per
    // timestep, small enough to hand-compute exactly.
    const unsigned N = 2, d_a = 1;
    AttentionPoolLayer layer = make_layer(N, d_a, true);
    layer.set_wa_values({ 0.5, -0.5 }); // Wa shape N x d_a = [0.5; -0.5]
    layer.set_ba_values({ 0.1 });
    layer.set_v_values({ 1.0 });

    // T=2 timesteps, h_0 = (1, 0), h_1 = (0, 1)
    std::vector<double> h_seq = { 1.0, 0.0, 0.0, 1.0 };

    const double e0 = 0.5 * 1.0 + -0.5 * 0.0 + 0.1; // 0.6
    const double e1 = 0.5 * 0.0 + -0.5 * 1.0 + 0.1; // -0.4
    const double u0 = std::tanh(e0);
    const double u1 = std::tanh(e1);
    const double score0 = 1.0 * u0;
    const double score1 = 1.0 * u1;
    const double max_s = std::max(score0, score1);
    const double exp0 = std::exp(score0 - max_s);
    const double exp1 = std::exp(score1 - max_s);
    const double alpha0 = exp0 / (exp0 + exp1);
    const double alpha1 = exp1 / (exp0 + exp1);

    const std::vector<double> expected_context = {
      alpha0 * 1.0 + alpha1 * 0.0,
      alpha0 * 0.0 + alpha1 * 1.0
    };

    // Attention weights must form a valid convex combination.
    EXPECT_NEAR(alpha0 + alpha1, 1.0, 1e-9);
    EXPECT_GE(alpha0, 0.0);
    EXPECT_GE(alpha1, 0.0);

    std::vector<unsigned> topology = { N, N };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
    MockLayer previous_layer(0, N);

    batch_go[0].set_rnn_outputs(0, h_seq.data(), h_seq.size());
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    const auto outputs = batch_go[0].get_outputs(1);
    ASSERT_EQ(outputs.size(), N);
    EXPECT_NEAR(outputs[0], expected_context[0], 1e-9);
    EXPECT_NEAR(outputs[1], expected_context[1], 1e-9);
}

TEST_F(AttentionPoolLayerTest, WeightGradientsMatchNumericalGradient) {
    const unsigned N = 3, d_a = 2;
    const size_t T = 3;
    AttentionPoolLayer layer = make_layer(N, d_a, true);
    layer.set_wa_values({ 0.3, -0.2, 0.1, 0.4, -0.5, 0.2 });
    layer.set_ba_values({ 0.05, -0.1 });
    layer.set_v_values({ 0.7, -0.3 });

    std::vector<double> h_seq = {
      0.5, -0.2, 1.0,
      -0.3, 0.8, 0.1,
      0.2, 0.4, -0.6
    };
    std::vector<double> delta = { 0.4, -0.7, 0.2 };

    MockLayer previous_layer(0, N);
    std::vector<unsigned> topology = { N, N };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_rnn_outputs(0, h_seq.data(), T * N);
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    std::vector<std::vector<double>> batch_output_gradients = { delta };
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);
    layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, 1, 0);

    const auto wa_grads = layer.get_wa_grads();
    const auto ba_grads = layer.get_ba_grads();
    const auto v_grads = layer.get_v_grads();

    const double h = 1e-6;

    for (size_t k = 0; k < N * d_a; ++k)
    {
      auto wa_plus = layer.get_wa_values(); wa_plus[k] += h;
      auto wa_minus = layer.get_wa_values(); wa_minus[k] -= h;

      AttentionPoolLayer probe = make_layer(N, d_a, true);
      probe.set_ba_values(layer.get_ba_values());
      probe.set_v_values(layer.get_v_values());

      probe.set_wa_values(wa_plus);
      const double loss_plus = compute_loss(probe, previous_layer, h_seq, T, N, delta);
      probe.set_wa_values(wa_minus);
      const double loss_minus = compute_loss(probe, previous_layer, h_seq, T, N, delta);

      const double numerical = (loss_plus - loss_minus) / (2.0 * h);
      EXPECT_NEAR(wa_grads[k], numerical, 1e-5) << "wa index " << k;
    }

    for (size_t k = 0; k < d_a; ++k)
    {
      AttentionPoolLayer probe = make_layer(N, d_a, true);
      probe.set_wa_values(layer.get_wa_values());
      probe.set_v_values(layer.get_v_values());

      auto ba_plus = layer.get_ba_values(); ba_plus[k] += h;
      auto ba_minus = layer.get_ba_values(); ba_minus[k] -= h;

      probe.set_ba_values(ba_plus);
      const double loss_plus = compute_loss(probe, previous_layer, h_seq, T, N, delta);
      probe.set_ba_values(ba_minus);
      const double loss_minus = compute_loss(probe, previous_layer, h_seq, T, N, delta);

      const double numerical = (loss_plus - loss_minus) / (2.0 * h);
      EXPECT_NEAR(ba_grads[k], numerical, 1e-5) << "ba index " << k;
    }

    for (size_t k = 0; k < d_a; ++k)
    {
      AttentionPoolLayer probe = make_layer(N, d_a, true);
      probe.set_wa_values(layer.get_wa_values());
      probe.set_ba_values(layer.get_ba_values());

      auto v_plus = layer.get_v_values(); v_plus[k] += h;
      auto v_minus = layer.get_v_values(); v_minus[k] -= h;

      probe.set_v_values(v_plus);
      const double loss_plus = compute_loss(probe, previous_layer, h_seq, T, N, delta);
      probe.set_v_values(v_minus);
      const double loss_minus = compute_loss(probe, previous_layer, h_seq, T, N, delta);

      const double numerical = (loss_plus - loss_minus) / (2.0 * h);
      EXPECT_NEAR(v_grads[k], numerical, 1e-5) << "v index " << k;
    }
}

TEST_F(AttentionPoolLayerTest, InputGradientsMatchNumericalGradient) {
    // Verifies the T*N gradient handed to the preceding recurrent layer
    // (via set_rnn_gradients) matches d(loss)/d(h_seq).
    const unsigned N = 2, d_a = 2;
    const size_t T = 3;
    AttentionPoolLayer layer = make_layer(N, d_a, true);
    layer.set_wa_values({ 0.6, -0.4, 0.2, 0.3 });
    layer.set_ba_values({ 0.0, 0.1 });
    layer.set_v_values({ 0.5, -0.6 });

    std::vector<double> h_seq = {
      0.4, -0.5,
      0.9, 0.1,
      -0.3, 0.7
    };
    std::vector<double> delta = { 0.3, -0.2 };

    MockLayer previous_layer(0, N);
    std::vector<unsigned> topology = { N, N };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_rnn_outputs(0, h_seq.data(), T * N);
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    std::vector<std::vector<double>> batch_output_gradients = { delta };
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);

    const auto dh = batch_go[0].get_rnn_gradients(1);
    ASSERT_EQ(dh.size(), T * N);

    const double h = 1e-6;
    for (size_t k = 0; k < T * N; ++k)
    {
      auto h_plus = h_seq; h_plus[k] += h;
      auto h_minus = h_seq; h_minus[k] -= h;

      const double loss_plus = compute_loss(layer, previous_layer, h_plus, T, N, delta);
      const double loss_minus = compute_loss(layer, previous_layer, h_minus, T, N, delta);

      const double numerical = (loss_plus - loss_minus) / (2.0 * h);
      EXPECT_NEAR(dh[k], numerical, 1e-5) << "h_seq index " << k;
    }
}

TEST_F(AttentionPoolLayerTest, CalculateHiddenGradientsMatchesDirectInjectionThroughIdentity) {
    // calculate_hidden_gradients (the generic "matrix multiply through
    // next_layer's weights" path) with an identity next_layer weight matrix
    // must produce exactly the same delta as calculate_hidden_gradients_from_output_gradients
    // given the same raw gradient.
    const unsigned N = 3, d_a = 2;
    const size_t T = 2;
    AttentionPoolLayer layer_a = make_layer(N, d_a, true);
    AttentionPoolLayer layer_b = make_layer(N, d_a, true);
    layer_a.set_wa_values({ 0.1, 0.2, 0.3, -0.1, 0.4, -0.2 });
    layer_b.set_wa_values(layer_a.get_wa_values());
    layer_a.set_ba_values({ 0.0, 0.05 });
    layer_b.set_ba_values(layer_a.get_ba_values());
    layer_a.set_v_values({ 0.2, -0.4 });
    layer_b.set_v_values(layer_a.get_v_values());

    std::vector<double> h_seq = { 0.3, -0.1, 0.2, 0.5, 0.4, -0.3 };
    std::vector<double> next_grad = { 0.2, -0.3, 0.1 };

    std::vector<unsigned> topology = { N, N, N };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
    MockLayer previous_layer(0, N);

    batch_go[0].set_rnn_outputs(0, h_seq.data(), T * N);
    layer_a.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);
    layer_b.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    // Path A: identity next_layer, standard calculate_hidden_gradients.
    MockLayer identity_next(2, N, N);
    std::vector<double> identity(static_cast<size_t>(N) * N, 0.0);
    for (unsigned i = 0; i < N; ++i) identity[i * N + i] = 1.0;
    identity_next.set_w_values(identity);
    batch_go[0].set_gradients(2, next_grad.data(), N);
    layer_a.calculate_hidden_gradients(batch_go, identity_next, {}, batch_hs, 1, 0);
    const auto dh_a_span = batch_go[0].get_rnn_gradients(1);
    const std::vector<double> dh_a(dh_a_span.begin(), dh_a_span.end());

    // Path B: direct injection with the same raw gradient.
    std::vector<std::vector<double>> batch_output_gradients = { next_grad };
    layer_b.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);
    const auto dh_b_span = batch_go[0].get_rnn_gradients(1);
    const std::vector<double> dh_b(dh_b_span.begin(), dh_b_span.end());

    ASSERT_EQ(dh_a.size(), dh_b.size());
    for (size_t k = 0; k < dh_a.size(); ++k)
    {
      EXPECT_NEAR(dh_a[k], dh_b[k], 1e-9) << "index " << k;
    }
}

TEST_F(AttentionPoolLayerTest, NoBatchCrossTalk) {
    const unsigned N = 2, d_a = 2;
    const size_t T = 2;
    AttentionPoolLayer layer = make_layer(N, d_a, true);
    layer.set_wa_values({ 0.4, -0.3, 0.2, 0.5 });
    layer.set_ba_values({ 0.1, -0.1 });
    layer.set_v_values({ 0.6, -0.2 });

    std::vector<double> h_seq_a = { 1.0, 0.0, 0.5, 0.5 };
    std::vector<double> h_seq_b = { 0.0, 1.0, -0.5, 1.5 };

    std::vector<unsigned> topology = { N, N };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 1, 1);
    MockLayer previous_layer(0, N);

    batch_go[0].set_rnn_outputs(0, h_seq_a.data(), T * N);
    batch_go[1].set_rnn_outputs(0, h_seq_b.data(), T * N);

    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 2, false);

    // Compute each batch item's expected output independently (batch_size=1)
    // and verify the batched run matches - i.e. no cross-contamination.
    std::vector<unsigned> single_topology = { N, N };
    auto single_go_a = create_batch_gradients_and_outputs(single_topology, 1);
    auto single_hs_a = create_batch_hidden_states(single_topology, 1, 1, 1);
    single_go_a[0].set_rnn_outputs(0, h_seq_a.data(), T * N);
    layer.calculate_forward_feed(single_go_a, previous_layer, {}, single_hs_a, 1, false);

    auto single_go_b = create_batch_gradients_and_outputs(single_topology, 1);
    auto single_hs_b = create_batch_hidden_states(single_topology, 1, 1, 1);
    single_go_b[0].set_rnn_outputs(0, h_seq_b.data(), T * N);
    layer.calculate_forward_feed(single_go_b, previous_layer, {}, single_hs_b, 1, false);

    const auto batched_a = batch_go[0].get_outputs(1);
    const auto batched_b = batch_go[1].get_outputs(1);
    const auto expected_a = single_go_a[0].get_outputs(1);
    const auto expected_b = single_go_b[0].get_outputs(1);

    for (size_t i = 0; i < N; ++i)
    {
      EXPECT_NEAR(batched_a[i], expected_a[i], 1e-12);
      EXPECT_NEAR(batched_b[i], expected_b[i], 1e-12);
    }
}

TEST_F(AttentionPoolLayerTest, AccumulateSwaAverageMatchesDirectMean) {
    const unsigned N = 2, d_a = 2;
    AttentionPoolLayer running = make_layer(N, d_a, true);
    running.set_wa_values({ 0.0, 0.0, 0.0, 0.0 });
    running.set_ba_values({ 0.0, 0.0 });
    running.set_v_values({ 0.0, 0.0 });

    AttentionPoolLayer snap1 = make_layer(N, d_a, true);
    snap1.set_wa_values({ 1.0, 2.0, 3.0, 4.0 });
    snap1.set_ba_values({ 0.1, 0.2 });
    snap1.set_v_values({ 0.5, -0.5 });

    AttentionPoolLayer snap2 = make_layer(N, d_a, true);
    snap2.set_wa_values({ 3.0, 4.0, 5.0, 6.0 });
    snap2.set_ba_values({ 0.3, 0.4 });
    snap2.set_v_values({ -0.5, 1.5 });

    // running starts as an exact copy of snap1 (matching the codebase's
    // "first snapshot becomes the initial running average" convention).
    running.set_wa_values(snap1.get_wa_values());
    running.set_ba_values(snap1.get_ba_values());
    running.set_v_values(snap1.get_v_values());

    running.accumulate_swa_average(snap2, 1);

    for (size_t i = 0; i < 4; ++i)
    {
      const double expected = (snap1.get_wa_values()[i] + snap2.get_wa_values()[i]) / 2.0;
      EXPECT_NEAR(running.get_wa_values()[i], expected, 1e-12);
    }
    for (size_t i = 0; i < 2; ++i)
    {
      const double expected_ba = (snap1.get_ba_values()[i] + snap2.get_ba_values()[i]) / 2.0;
      const double expected_v = (snap1.get_v_values()[i] + snap2.get_v_values()[i]) / 2.0;
      EXPECT_NEAR(running.get_ba_values()[i], expected_ba, 1e-12);
      EXPECT_NEAR(running.get_v_values()[i], expected_v, 1e-12);
    }
}

TEST_F(AttentionPoolLayerTest, CloneProducesIndependentCopy) {
    AttentionPoolLayer layer = make_layer(2, 2, true);
    layer.set_wa_values({ 1.0, 2.0, 3.0, 4.0 });

    std::unique_ptr<Layer> cloned(layer.clone());
    auto* cloned_attn = dynamic_cast<AttentionPoolLayer*>(cloned.get());
    ASSERT_NE(cloned_attn, nullptr);
    EXPECT_EQ(cloned_attn->get_wa_values(), layer.get_wa_values());

    cloned_attn->set_wa_values({ 9.0, 9.0, 9.0, 9.0 });
    EXPECT_NE(cloned_attn->get_wa_values(), layer.get_wa_values());
}
