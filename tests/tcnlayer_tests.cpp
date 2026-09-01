#include <gtest/gtest.h>
#include "layers/tcnlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <numeric>


using namespace myoddweb::nn;
using namespace test_helper;

namespace
{
// Reruns the real forward pass with the given weights and the given x_seq
// (T*N_in, previous layer's full raw output sequence), then returns
// dot(delta, out_seq) where out_seq is this layer's full T*N_out activated
// output sequence. Used as the scalar loss function for central-finite-
// difference gradient checks: since delta is held fixed, d(loss)/d(weight)
// or d(loss)/d(x_seq) is exactly what the layer's backward pass should
// produce (the layer under test always uses dropout=0, so the returned
// output equals the un-masked activated output).
double compute_loss(
  TcnLayer& layer,
  MockLayer& previous_layer,
  const std::vector<double>& x_seq,
  size_t T,
  size_t N_in,
  size_t N_out,
  const std::vector<double>& delta_full_seq)
{
  std::vector<unsigned> topology = { static_cast<unsigned>(N_in), static_cast<unsigned>(N_out) };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

  batch_go[0].set_rnn_outputs(0, x_seq.data(), T * N_in);
  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

  const auto out_seq = batch_go[0].get_rnn_outputs(1);
  double total = 0.0;
  for (size_t k = 0; k < T * N_out; ++k)
  {
    total += delta_full_seq[k] * out_seq[k];
  }
  return total;
}

TcnLayer make_layer(
  unsigned num_neurons_in_previous_layer,
  unsigned layer_size,
  unsigned kernel_size,
  unsigned dilation,
  bool has_bias,
  const activation& activation_method = activation(activation::method::linear, 0.0),
  int residual_layer_number = -1,
  ResidualProjector* residual_projector = nullptr)
{
  TcnLayer layer(
    1,
    num_neurons_in_previous_layer,
    layer_size,
    kernel_size,
    dilation,
    0.0,
    Layer::Role::Hidden,
    activation_method,
    OptimiserType::SGD,
    residual_layer_number,
    0.0,
    residual_projector,
    1,
    has_bias,
    0.0,
    std::nullopt);
  return layer;
}
} // namespace

class TcnLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(TcnLayerTest, Construction) {
    TcnLayer layer = make_layer(3, 4, 2, 1, true);
    EXPECT_EQ(layer.get_layer_index(), 1u);
    EXPECT_EQ(layer.get_number_input_neurons(), 6u); // kernel_size * N_in
    EXPECT_EQ(layer.get_number_neurons(), 4u);
    EXPECT_EQ(layer.get_kernel_size(), 2u);
    EXPECT_EQ(layer.get_dilation(), 1u);
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Tcn);
    EXPECT_EQ(layer.get_w_values().size(), 24u); // (kernel_size * N_in) * layer_size
    EXPECT_EQ(layer.get_b_values().size(), 4u);
}

TEST_F(TcnLayerTest, ForwardHandComputedExample) {
    // N_in=1, N_out=1, kernel_size=2, dilation=1: pre_act_t = w0*X[t] + w1*X[t-1] + b,
    // with X[t-1] zero-padded for t=0 - small enough to hand-compute exactly.
    const unsigned N_in = 1, N_out = 1, K = 2, D = 1;
    TcnLayer layer = make_layer(N_in, N_out, K, D, true);
    layer.set_w_values({ 0.5, -0.3 }); // tap j=0 (current) -> 0.5, tap j=1 (one step back) -> -0.3
    layer.set_b_values({ 0.2 });

    const size_t T = 3;
    std::vector<double> x_seq = { 1.0, 2.0, 3.0 };

    const double expected_t0 = 0.5 * 1.0 + -0.3 * 0.0 + 0.2; // left-edge zero padding
    const double expected_t1 = 0.5 * 2.0 + -0.3 * 1.0 + 0.2;
    const double expected_t2 = 0.5 * 3.0 + -0.3 * 2.0 + 0.2;

    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
    MockLayer previous_layer(0, N_in);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), x_seq.size());
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    const auto out_seq = batch_go[0].get_rnn_outputs(1);
    ASSERT_EQ(out_seq.size(), T * N_out);
    EXPECT_NEAR(out_seq[0], expected_t0, 1e-9);
    EXPECT_NEAR(out_seq[1], expected_t1, 1e-9);
    EXPECT_NEAR(out_seq[2], expected_t2, 1e-9);

    const auto last_output = batch_go[0].get_outputs(1);
    ASSERT_EQ(last_output.size(), N_out);
    EXPECT_NEAR(last_output[0], expected_t2, 1e-9);
}

TEST_F(TcnLayerTest, WeightGradientsMatchNumericalGradient) {
    const unsigned N_in = 2, N_out = 2, K = 2, D = 2;
    const size_t T = 4;
    TcnLayer layer = make_layer(N_in, N_out, K, D, true, activation(activation::method::tanh, 0.0));
    layer.set_w_values({ 0.3, -0.2, 0.1, 0.4, -0.5, 0.2, 0.25, -0.15 });
    layer.set_b_values({ 0.05, -0.1 });

    std::vector<double> x_seq = {
      0.5, -0.2,
      -0.3, 0.8,
      0.2, 0.4,
      -0.6, 0.1
    };
    std::vector<double> delta_full_seq = {
      0.4, -0.7,
      0.2, -0.3,
      0.1, 0.5,
      -0.2, 0.3
    };

    MockLayer previous_layer(0, N_in);
    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), T * N_in);
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    std::vector<std::vector<double>> batch_output_gradients = { delta_full_seq };
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);
    layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, 1, 0);

    const auto w_grads = layer.get_w_grads();
    const auto b_grads = layer.get_b_grads();

    const double h = 1e-6;

    for (size_t k = 0; k < w_grads.size(); ++k)
    {
      auto w_plus = layer.get_w_values(); w_plus[k] += h;
      auto w_minus = layer.get_w_values(); w_minus[k] -= h;

      TcnLayer probe = make_layer(N_in, N_out, K, D, true, activation(activation::method::tanh, 0.0));
      probe.set_b_values(layer.get_b_values());

      probe.set_w_values(w_plus);
      const double loss_plus = compute_loss(probe, previous_layer, x_seq, T, N_in, N_out, delta_full_seq);
      probe.set_w_values(w_minus);
      const double loss_minus = compute_loss(probe, previous_layer, x_seq, T, N_in, N_out, delta_full_seq);

      const double numerical = (loss_plus - loss_minus) / (2.0 * h);
      EXPECT_NEAR(w_grads[k], numerical, 1e-5) << "w index " << k;
    }

    for (size_t k = 0; k < b_grads.size(); ++k)
    {
      TcnLayer probe = make_layer(N_in, N_out, K, D, true, activation(activation::method::tanh, 0.0));
      probe.set_w_values(layer.get_w_values());

      auto b_plus = layer.get_b_values(); b_plus[k] += h;
      auto b_minus = layer.get_b_values(); b_minus[k] -= h;

      probe.set_b_values(b_plus);
      const double loss_plus = compute_loss(probe, previous_layer, x_seq, T, N_in, N_out, delta_full_seq);
      probe.set_b_values(b_minus);
      const double loss_minus = compute_loss(probe, previous_layer, x_seq, T, N_in, N_out, delta_full_seq);

      const double numerical = (loss_plus - loss_minus) / (2.0 * h);
      EXPECT_NEAR(b_grads[k], numerical, 1e-5) << "b index " << k;
    }
}

TEST_F(TcnLayerTest, InputGradientsMatchNumericalGradient) {
    // Verifies the T*N_in gradient handed to the preceding layer (via
    // set_rnn_gradients) matches d(loss)/d(x_seq), exercising the scatter-add
    // across dilated taps (kernel_size=2, dilation=2 means one source
    // timestep can feed two different output timesteps).
    const unsigned N_in = 1, N_out = 2, K = 2, D = 2;
    const size_t T = 5;
    TcnLayer layer = make_layer(N_in, N_out, K, D, true, activation(activation::method::tanh, 0.0));
    layer.set_w_values({ 0.6, -0.4, 0.2, 0.3 });
    layer.set_b_values({ 0.0, 0.1 });

    std::vector<double> x_seq = { 0.4, -0.5, 0.9, 0.1, -0.3 };
    std::vector<double> delta_full_seq = {
      0.3, -0.2,
      0.1, 0.4,
      -0.3, 0.2,
      0.2, -0.1,
      0.05, 0.15
    };

    MockLayer previous_layer(0, N_in);
    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), T * N_in);
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    std::vector<std::vector<double>> batch_output_gradients = { delta_full_seq };
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);

    const auto dx = batch_go[0].get_rnn_gradients(1);
    ASSERT_EQ(dx.size(), T * N_in);

    const double h = 1e-6;
    for (size_t k = 0; k < T * N_in; ++k)
    {
      auto x_plus = x_seq; x_plus[k] += h;
      auto x_minus = x_seq; x_minus[k] -= h;

      const double loss_plus = compute_loss(layer, previous_layer, x_plus, T, N_in, N_out, delta_full_seq);
      const double loss_minus = compute_loss(layer, previous_layer, x_minus, T, N_in, N_out, delta_full_seq);

      const double numerical = (loss_plus - loss_minus) / (2.0 * h);
      EXPECT_NEAR(dx[k], numerical, 1e-5) << "x_seq index " << k;
    }
}

TEST_F(TcnLayerTest, CalculateHiddenGradientsMatchesDirectInjectionThroughIdentity) {
    // calculate_hidden_gradients (dense path, matmul through next_layer's
    // weights) with an identity next_layer weight matrix must produce the
    // same delta as calculate_hidden_gradients_from_output_gradients given
    // the same raw (broadcast, single-vector) gradient.
    const unsigned N_in = 2, N_out = 2, K = 2, D = 1;
    const size_t T = 3;
    TcnLayer layer_a = make_layer(N_in, N_out, K, D, true);
    TcnLayer layer_b = make_layer(N_in, N_out, K, D, true);
    layer_a.set_w_values({ 0.1, 0.2, 0.3, -0.1, 0.4, -0.2, 0.05, 0.15 });
    layer_b.set_w_values(layer_a.get_w_values());
    layer_a.set_b_values({ 0.0, 0.05 });
    layer_b.set_b_values(layer_a.get_b_values());

    std::vector<double> x_seq = { 0.3, -0.1, 0.2, 0.5, 0.4, -0.3 };
    std::vector<double> next_grad = { 0.2, -0.3 };

    std::vector<unsigned> topology = { N_in, N_out, N_out };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
    MockLayer previous_layer(0, N_in);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), T * N_in);
    layer_a.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);
    layer_b.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    // Path A: identity next_layer, standard calculate_hidden_gradients.
    MockLayer identity_next(2, N_out, N_out);
    std::vector<double> identity(static_cast<size_t>(N_out) * N_out, 0.0);
    for (unsigned i = 0; i < N_out; ++i) identity[i * N_out + i] = 1.0;
    identity_next.set_w_values(identity);
    batch_go[0].set_gradients(2, next_grad.data(), N_out);
    layer_a.calculate_hidden_gradients(batch_go, identity_next, {}, batch_hs, 1, 0);
    const auto dx_a_span = batch_go[0].get_rnn_gradients(1);
    const std::vector<double> dx_a(dx_a_span.begin(), dx_a_span.end());

    // Path B: direct injection with the same raw gradient.
    std::vector<std::vector<double>> batch_output_gradients = { next_grad };
    layer_b.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);
    const auto dx_b_span = batch_go[0].get_rnn_gradients(1);
    const std::vector<double> dx_b(dx_b_span.begin(), dx_b_span.end());

    ASSERT_EQ(dx_a.size(), dx_b.size());
    for (size_t k = 0; k < dx_a.size(); ++k)
    {
      EXPECT_NEAR(dx_a[k], dx_b[k], 1e-9) << "index " << k;
    }
}

TEST_F(TcnLayerTest, ResidualAddedOnlyAtLastTimestep) {
    const unsigned N_in = 1, N_out = 2, K = 1, D = 1;
    const size_t T = 3;
    TcnLayer layer = make_layer(N_in, N_out, K, D, true);
    layer.set_w_values({ 1.0, 1.0 });
    layer.set_b_values({ 0.0, 0.0 });

    std::vector<double> x_seq = { 1.0, 2.0, 3.0 };
    std::vector<std::vector<double>> batch_residual_output_values = { { 10.0, 20.0 } };

    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
    MockLayer previous_layer(0, N_in);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), x_seq.size());
    layer.calculate_forward_feed(batch_go, previous_layer, batch_residual_output_values, batch_hs, 1, false);

    const auto out_seq = batch_go[0].get_rnn_outputs(1);
    ASSERT_EQ(out_seq.size(), T * N_out);
    // Timesteps 0 and 1: no residual added.
    EXPECT_NEAR(out_seq[0], 1.0, 1e-9);
    EXPECT_NEAR(out_seq[1], 1.0, 1e-9);
    EXPECT_NEAR(out_seq[2], 2.0, 1e-9);
    EXPECT_NEAR(out_seq[3], 2.0, 1e-9);
    // Timestep 2 (last): residual added.
    EXPECT_NEAR(out_seq[4], 3.0 + 10.0, 1e-9);
    EXPECT_NEAR(out_seq[5], 3.0 + 20.0, 1e-9);
}

TEST_F(TcnLayerTest, NoBatchCrossTalk) {
    const unsigned N_in = 2, N_out = 2, K = 2, D = 1;
    const size_t T = 3;
    TcnLayer layer = make_layer(N_in, N_out, K, D, true, activation(activation::method::tanh, 0.0));
    layer.set_w_values({ 0.4, -0.3, 0.2, 0.5, -0.1, 0.6, 0.3, -0.2 });
    layer.set_b_values({ 0.1, -0.1 });

    std::vector<double> x_seq_a = { 1.0, 0.0, 0.5, 0.5, -0.5, 1.0 };
    std::vector<double> x_seq_b = { 0.0, 1.0, -0.5, 1.5, 0.3, -0.2 };

    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 1, 1);
    MockLayer previous_layer(0, N_in);

    batch_go[0].set_rnn_outputs(0, x_seq_a.data(), T * N_in);
    batch_go[1].set_rnn_outputs(0, x_seq_b.data(), T * N_in);

    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 2, false);

    std::vector<unsigned> single_topology = { N_in, N_out };
    auto single_go_a = create_batch_gradients_and_outputs(single_topology, 1);
    auto single_hs_a = create_batch_hidden_states(single_topology, 1, 1, 1);
    single_go_a[0].set_rnn_outputs(0, x_seq_a.data(), T * N_in);
    layer.calculate_forward_feed(single_go_a, previous_layer, {}, single_hs_a, 1, false);

    auto single_go_b = create_batch_gradients_and_outputs(single_topology, 1);
    auto single_hs_b = create_batch_hidden_states(single_topology, 1, 1, 1);
    single_go_b[0].set_rnn_outputs(0, x_seq_b.data(), T * N_in);
    layer.calculate_forward_feed(single_go_b, previous_layer, {}, single_hs_b, 1, false);

    const auto batched_a = batch_go[0].get_rnn_outputs(1);
    const auto batched_b = batch_go[1].get_rnn_outputs(1);
    const auto expected_a = single_go_a[0].get_rnn_outputs(1);
    const auto expected_b = single_go_b[0].get_rnn_outputs(1);

    ASSERT_EQ(batched_a.size(), T * N_out);
    for (size_t i = 0; i < T * N_out; ++i)
    {
      EXPECT_NEAR(batched_a[i], expected_a[i], 1e-12);
      EXPECT_NEAR(batched_b[i], expected_b[i], 1e-12);
    }
}

TEST_F(TcnLayerTest, AccumulateSwaAverageMatchesDirectMean) {
    const unsigned N_in = 2, N_out = 2, K = 2, D = 1;
    TcnLayer running = make_layer(N_in, N_out, K, D, true);
    TcnLayer snap1 = make_layer(N_in, N_out, K, D, true);
    snap1.set_w_values({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
    snap1.set_b_values({ 0.1, 0.2 });

    TcnLayer snap2 = make_layer(N_in, N_out, K, D, true);
    snap2.set_w_values({ 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });
    snap2.set_b_values({ 0.3, 0.4 });

    // running starts as an exact copy of snap1 (matching the codebase's
    // "first snapshot becomes the initial running average" convention).
    running.set_w_values(snap1.get_w_values());
    running.set_b_values(snap1.get_b_values());

    running.accumulate_swa_average(snap2, 1);

    for (size_t i = 0; i < 8; ++i)
    {
      const double expected = (snap1.get_w_values()[i] + snap2.get_w_values()[i]) / 2.0;
      EXPECT_NEAR(running.get_w_values()[i], expected, 1e-12);
    }
    for (size_t i = 0; i < 2; ++i)
    {
      const double expected_b = (snap1.get_b_values()[i] + snap2.get_b_values()[i]) / 2.0;
      EXPECT_NEAR(running.get_b_values()[i], expected_b, 1e-12);
    }
}

TEST_F(TcnLayerTest, CloneProducesIndependentCopy) {
    TcnLayer layer = make_layer(2, 2, 2, 1, true);
    layer.set_w_values({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

    std::unique_ptr<Layer> cloned(layer.clone());
    auto* cloned_tcn = dynamic_cast<TcnLayer*>(cloned.get());
    ASSERT_NE(cloned_tcn, nullptr);
    EXPECT_EQ(cloned_tcn->get_w_values(), layer.get_w_values());
    EXPECT_EQ(cloned_tcn->get_kernel_size(), layer.get_kernel_size());
    EXPECT_EQ(cloned_tcn->get_dilation(), layer.get_dilation());

    cloned_tcn->set_w_values({ 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0 });
    EXPECT_NE(cloned_tcn->get_w_values(), layer.get_w_values());
}

TEST_F(TcnLayerTest, DilationLongerThanSequencePaddingBehavior) {
    // K=3, D=4, receptive field = 1 + 2*4 = 9. T = 3 (shorter than receptive field).
    // All timesteps must execute without bounds violations and correctly zero-pad missing taps.
    const unsigned N_in = 2, N_out = 2, K = 3, D = 4;
    const size_t T = 3;
    TcnLayer layer = make_layer(N_in, N_out, K, D, true);

    std::vector<double> w(static_cast<size_t>(K) * N_in * N_out, 0.5);
    std::vector<double> b(N_out, 0.1);
    layer.set_w_values(w);
    layer.set_b_values(b);

    std::vector<double> x_seq = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
    MockLayer previous_layer(0, N_in);

    batch_go[0].set_rnn_outputs(0, x_seq.data(), x_seq.size());
    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

    const auto out_seq = batch_go[0].get_rnn_outputs(1);
    ASSERT_EQ(out_seq.size(), T * N_out);

    // For T=3 and D=4:
    // At t=0: tap 0 (s=0) is valid, taps 1,2 are zero-padded.
    // At t=1: tap 0 (s=1) is valid, taps 1,2 are zero-padded.
    // At t=2: tap 0 (s=2) is valid, taps 1,2 are zero-padded.
    for (size_t t = 0; t < T; ++t)
    {
      for (size_t o = 0; o < N_out; ++o)
      {
        EXPECT_FALSE(std::isnan(out_seq[t * N_out + o]));
        EXPECT_FALSE(std::isinf(out_seq[t * N_out + o]));
      }
    }
}

TEST_F(TcnLayerTest, MultiBatchForwardAndBackwardNumericalSoundness) {
    const unsigned N_in = 2, N_out = 3, K = 2, D = 1;
    const size_t T = 3;
    const size_t batch_size = 4;
    TcnLayer layer = make_layer(N_in, N_out, K, D, true, activation(activation::method::tanh, 0.0));

    std::vector<double> w_values(static_cast<size_t>(K) * N_in * N_out);
    for (size_t i = 0; i < w_values.size(); ++i)
    {
      w_values[i] = std::sin(static_cast<double>(i + 1) * 0.3) * 0.4;
    }
    std::vector<double> b_values(N_out);
    for (size_t i = 0; i < b_values.size(); ++i)
    {
      b_values[i] = std::cos(static_cast<double>(i + 1) * 0.2) * 0.1;
    }
    layer.set_w_values(w_values);
    layer.set_b_values(b_values);

    std::vector<unsigned> topology = { N_in, N_out };
    auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs = create_batch_hidden_states(topology, batch_size, 1, 1);
    MockLayer previous_layer(0, N_in);

    std::vector<std::vector<double>> deltas(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
      std::vector<double> x(T * N_in);
      for (size_t i = 0; i < x.size(); ++i)
      {
        x[i] = std::sin(static_cast<double>(b * 5 + i) * 0.4);
      }
      batch_go[b].set_rnn_outputs(0, x.data(), x.size());

      deltas[b].resize(T * N_out);
      for (size_t i = 0; i < deltas[b].size(); ++i)
      {
        deltas[b][i] = std::cos(static_cast<double>(b * 3 + i) * 0.5) * 0.25;
      }
    }

    layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, batch_size, false);
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, deltas, batch_hs, batch_size, 0);
    layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, batch_size, 0);

    const auto w_grads = layer.get_w_grads();
    const auto b_grads = layer.get_b_grads();
    ASSERT_EQ(w_grads.size(), w_values.size());
    ASSERT_EQ(b_grads.size(), b_values.size());

    for (size_t i = 0; i < w_grads.size(); ++i)
    {
      EXPECT_FALSE(std::isnan(w_grads[i]));
      EXPECT_FALSE(std::isinf(w_grads[i]));
    }
    for (size_t i = 0; i < b_grads.size(); ++i)
    {
      EXPECT_FALSE(std::isnan(b_grads[i]));
      EXPECT_FALSE(std::isinf(b_grads[i]));
    }
}
