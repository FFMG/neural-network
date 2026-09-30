#include <gtest/gtest.h>
#include "layers/grnlayer.h"
#include "layers/layerdetails.h"
#include "neuralnetworkoptions.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <string>

using namespace myoddweb::nn;
using namespace test_helper;

namespace
{
GrnLayer make_grn_layer(
  unsigned d_in,
  unsigned d_out,
  unsigned d_ff,
  bool use_layer_norm,
  const activation& act = activation(activation::method::elu, 1.0),
  int num_threads = 1,
  bool has_bias = true,
  unsigned layer_index = 1)
{
  return GrnLayer(
    layer_index,
    d_in,
    d_out,
    d_ff,
    0.0,
    Layer::Role::Hidden,
    act,
    OptimiserType::SGD,
    -1,
    0.0,
    nullptr,
    num_threads,
    has_bias,
    use_layer_norm,
    0.0,
    std::nullopt);
}

double compute_grn_loss(
  GrnLayer& layer,
  MockLayer& previous_layer,
  const std::vector<double>& input_seq,
  size_t time_steps,
  size_t d_in,
  size_t d_out,
  const std::vector<double>& upstream_grad,
  const std::vector<double>& residual = {})
{
  std::vector<unsigned> topology = { static_cast<unsigned>(d_in), static_cast<unsigned>(d_out) };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

  batch_go[0].set_rnn_outputs(0, input_seq.data(), time_steps * d_in);
  const std::vector<std::vector<double>> batch_residual = residual.empty() ? std::vector<std::vector<double>>() : std::vector<std::vector<double>>{ residual };
  layer.calculate_forward_feed(batch_go, previous_layer, batch_residual, batch_hs, 1, false);

  const auto out_seq = batch_go[0].get_rnn_outputs(1);
  double loss = 0.0;
  for (size_t k = 0; k < time_steps * d_out; ++k)
  {
    loss += upstream_grad[k] * out_seq[k];
  }
  return loss;
}

struct family_access
{
  const char* name;
  const std::vector<double>& (GrnLayer::*get_values)() const noexcept;
  const std::vector<double>& (GrnLayer::*get_grads)() const noexcept;
  void (GrnLayer::*set_values)(const std::vector<double>&);
};

#define GRN_FAMILY(family) { #family, &GrnLayer::get_##family##_values, &GrnLayer::get_##family##_grads, &GrnLayer::set_##family##_values }
const family_access all_family_access[] =
{
  GRN_FAMILY(w1), GRN_FAMILY(b1), GRN_FAMILY(w2), GRN_FAMILY(b2),
  GRN_FAMILY(w_skip), GRN_FAMILY(b_skip), GRN_FAMILY(ln_gain), GRN_FAMILY(ln_bias)
};
#undef GRN_FAMILY

void set_test_weights(GrnLayer& layer)
{
  double salt = 0.0;
  for (const auto& fam : all_family_access)
  {
    auto values = (layer.*fam.get_values)();
    const bool is_gain = std::string(fam.name) == "ln_gain";
    for (size_t i = 0; i < values.size(); ++i)
    {
      const double wave = std::sin(0.7 * static_cast<double>(i) + salt);
      values[i] = is_gain ? (1.0 + 0.3 * wave) : (0.35 * wave);
    }
    (layer.*fam.set_values)(values);
    salt += 1.3;
  }
}

std::vector<double> make_wave(size_t size, double frequency, double phase, double amplitude)
{
  std::vector<double> values(size);
  for (size_t i = 0; i < size; ++i)
  {
    values[i] = amplitude * std::sin(frequency * static_cast<double>(i) + phase);
  }
  return values;
}

void run_full_gradient_check(unsigned d_in, unsigned d_out, unsigned d_ff, bool use_layer_norm, bool has_bias, bool use_residual)
{
  const size_t time_steps = 3;
  GrnLayer layer = make_grn_layer(d_in, d_out, d_ff, use_layer_norm, activation(activation::method::elu, 1.0), 1, has_bias);
  set_test_weights(layer);

  const auto input_seq = make_wave(time_steps * d_in, 0.9, 0.2, 0.8);
  const auto upstream_grad = make_wave(time_steps * d_out, 1.1, 0.5, 1.0);
  const std::vector<double> residual = use_residual ? make_wave(d_out, 2.0, 0.1, 0.4) : std::vector<double>();

  std::vector<unsigned> topology = { d_in, d_out };
  MockLayer previous_layer(0, d_in);
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

  batch_go[0].set_rnn_outputs(0, input_seq.data(), input_seq.size());
  const std::vector<std::vector<double>> batch_residual = use_residual ? std::vector<std::vector<double>>{ residual } : std::vector<std::vector<double>>();
  layer.calculate_forward_feed(batch_go, previous_layer, batch_residual, batch_hs, 1, true);

  layer.calculate_hidden_gradients_from_output_gradients(batch_go, { upstream_grad }, batch_hs, 1, 0);
  layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, 1, 0);

  const double eps = 1e-6;
  const double tolerance = 1e-6;

  for (const auto& fam : all_family_access)
  {
    const auto base_values = (layer.*fam.get_values)();
    const auto analytical = (layer.*fam.get_grads)();
    ASSERT_EQ(base_values.size(), analytical.size()) << fam.name;
    for (size_t idx = 0; idx < base_values.size(); ++idx)
    {
      auto plus = base_values;
      plus[idx] += eps;
      (layer.*fam.set_values)(plus);
      const double loss_plus = compute_grn_loss(layer, previous_layer, input_seq, time_steps, d_in, d_out, upstream_grad, residual);

      auto minus = base_values;
      minus[idx] -= eps;
      (layer.*fam.set_values)(minus);
      const double loss_minus = compute_grn_loss(layer, previous_layer, input_seq, time_steps, d_in, d_out, upstream_grad, residual);

      (layer.*fam.set_values)(base_values);
      const double numerical = (loss_plus - loss_minus) / (2.0 * eps);
      EXPECT_NEAR(analytical[idx], numerical, tolerance + 1e-5 * std::abs(numerical))
        << fam.name << "[" << idx << "] (ln=" << use_layer_norm << ", bias=" << has_bias << ", residual=" << use_residual << ")";
    }
  }

  const auto dx = batch_go[0].get_rnn_gradients(1);
  ASSERT_EQ(dx.size(), input_seq.size());
  for (size_t k = 0; k < input_seq.size(); ++k)
  {
    auto plus = input_seq;
    plus[k] += eps;
    auto minus = input_seq;
    minus[k] -= eps;
    const double loss_plus = compute_grn_loss(layer, previous_layer, plus, time_steps, d_in, d_out, upstream_grad, residual);
    const double loss_minus = compute_grn_loss(layer, previous_layer, minus, time_steps, d_in, d_out, upstream_grad, residual);
    const double numerical = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(dx[k], numerical, tolerance + 1e-5 * std::abs(numerical))
      << "dx[" << k << "] (ln=" << use_layer_norm << ", bias=" << has_bias << ", residual=" << use_residual << ")";
  }
}

double compute_stack_loss(
  GrnLayer& first,
  GrnLayer& second,
  MockLayer& input_layer,
  const std::vector<unsigned>& topology,
  const std::vector<double>& input_seq,
  const std::vector<double>& upstream_grad)
{
  auto go = create_batch_gradients_and_outputs(topology, 1);
  auto hs = create_batch_hidden_states(topology, 1, 1, 1);
  go[0].set_rnn_outputs(0, input_seq.data(), input_seq.size());
  first.calculate_forward_feed(go, input_layer, {}, hs, 1, false);
  second.calculate_forward_feed(go, first, {}, hs, 1, false);
  const auto out = go[0].get_rnn_outputs(2);
  double loss = 0.0;
  for (size_t k = 0; k < out.size(); ++k)
  {
    loss += upstream_grad[k] * out[k];
  }
  return loss;
}
} // namespace

class GrnLayerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
  }
};

TEST_F(GrnLayerTest, ConstructionAndArchitecturalInvariants)
{
  // Dimension matching: d_in == d_out
  {
    GrnLayer layer = make_grn_layer(4, 4, 8, true);
    EXPECT_EQ(layer.get_layer_index(), 1u);
    EXPECT_EQ(layer.get_number_input_neurons(), 4u);
    EXPECT_EQ(layer.get_number_neurons(), 4u);
    EXPECT_EQ(layer.get_feed_forward_hidden_size(), 8u);
    EXPECT_TRUE(layer.get_use_layer_normalisation());
    EXPECT_FALSE(layer.has_skip_projection());
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Grn);
    EXPECT_TRUE(layer.is_recurrent());
    EXPECT_TRUE(layer.get_w_skip_values().empty());
    EXPECT_TRUE(layer.get_b_skip_values().empty());
    EXPECT_EQ(layer.get_w1_values().size(), 4u * 8u);
    EXPECT_EQ(layer.get_b1_values().size(), 8u);
    EXPECT_EQ(layer.get_w2_values().size(), 8u * 8u); // 2 * d_out = 8
    EXPECT_EQ(layer.get_b2_values().size(), 8u);
    EXPECT_EQ(layer.get_ln_gain_values().size(), 4u);
    EXPECT_EQ(layer.get_ln_bias_values().size(), 4u);
  }

  // Dimension change: d_in != d_out (skip projection required)
  {
    GrnLayer layer = make_grn_layer(3, 5, 8, false);
    EXPECT_EQ(layer.get_number_input_neurons(), 3u);
    EXPECT_EQ(layer.get_number_neurons(), 5u);
    EXPECT_TRUE(layer.has_skip_projection());
    EXPECT_FALSE(layer.get_use_layer_normalisation());
    EXPECT_EQ(layer.get_w_skip_values().size(), 3u * 5u);
    EXPECT_EQ(layer.get_b_skip_values().size(), 5u);
    EXPECT_EQ(layer.get_w1_values().size(), 3u * 8u);
    EXPECT_EQ(layer.get_b1_values().size(), 8u);
    EXPECT_EQ(layer.get_w2_values().size(), 8u * 10u); // 2 * d_out = 10
    EXPECT_EQ(layer.get_b2_values().size(), 10u);
    EXPECT_TRUE(layer.get_ln_gain_values().empty());
    EXPECT_TRUE(layer.get_ln_bias_values().empty());
  }
}

TEST_F(GrnLayerTest, ZeroGatingBypassesIntermediateTransformation)
{
  const unsigned d = 3;
  const unsigned d_ff = 6;
  GrnLayer layer = make_grn_layer(d, d, d_ff, false); // No LayerNorm

  // Force gate biases in b2 to a large negative value so sigma(gate) -> 0
  // b2 layout: [value_biases (d), gate_biases (d)]
  std::vector<double> b2(2 * d, 0.0);
  for (unsigned i = 0; i < d; ++i)
  {
    b2[d + i] = -100.0; // gate bias = -100
  }
  layer.set_b2_values(b2);

  std::vector<unsigned> topology = { d, d };
  MockLayer previous_layer(0, d);
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

  std::vector<double> input_val = { 1.5, -2.0, 0.75 };
  batch_go[0].set_rnn_outputs(0, input_val.data(), d);

  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

  const auto output_val = batch_go[0].get_rnn_outputs(1);
  for (unsigned i = 0; i < d; ++i)
  {
    // With zero gating and identity skip, output should equal input exactly
    EXPECT_NEAR(output_val[i], input_val[i], 1e-6);
  }
}

TEST_F(GrnLayerTest, ForwardFeedWithLayerNormalisation)
{
  const unsigned d = 4;
  const unsigned d_ff = 8;
  GrnLayer layer = make_grn_layer(d, d, d_ff, true); // With LayerNorm

  std::vector<unsigned> topology = { d, d };
  MockLayer previous_layer(0, d);
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

  std::vector<double> input_val = { 2.0, 4.0, -1.0, 3.0 };
  batch_go[0].set_rnn_outputs(0, input_val.data(), d);

  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

  const auto output_val = batch_go[0].get_rnn_outputs(1);
  double mean = 0.0;
  for (unsigned i = 0; i < d; ++i)
  {
    mean += output_val[i];
  }
  mean /= static_cast<double>(d);

  // Default LayerNorm gain=1, bias=0 implies zero mean
  EXPECT_NEAR(mean, 0.0, 1e-5);
}

TEST_F(GrnLayerTest, FiniteDifferenceNumericalGradientCheckWeights)
{
  const unsigned d_in = 3;
  const unsigned d_out = 2;
  const unsigned d_ff = 4;
  const size_t time_steps = 2;

  GrnLayer layer = make_grn_layer(d_in, d_out, d_ff, true, activation(activation::method::elu, 1.0));

  // Initialize reproducible deterministic weights
  std::vector<double> w1(d_in * d_ff, 0.2);
  std::vector<double> b1(d_ff, 0.1);
  std::vector<double> w2(d_ff * 2 * d_out, 0.15);
  std::vector<double> b2(2 * d_out, 0.05);
  std::vector<double> w_skip(d_in * d_out, 0.3);
  std::vector<double> b_skip(d_out, 0.02);
  std::vector<double> ln_gain(d_out, 1.0);
  std::vector<double> ln_bias(d_out, 0.0);

  layer.set_w1_values(w1);
  layer.set_b1_values(b1);
  layer.set_w2_values(w2);
  layer.set_b2_values(b2);
  layer.set_w_skip_values(w_skip);
  layer.set_b_skip_values(b_skip);
  layer.set_ln_gain_values(ln_gain);
  layer.set_ln_bias_values(ln_bias);

  std::vector<double> input_seq = { 0.5, -0.3, 0.8, -0.2, 0.7, 0.1 };
  std::vector<double> upstream_grad = { 1.0, -0.5, 0.3, 0.8 };

  std::vector<unsigned> topology = { d_in, d_out };
  MockLayer previous_layer(0, d_in);
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

  // 1. Forward feed
  batch_go[0].set_rnn_outputs(0, input_seq.data(), time_steps * d_in);
  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, true);

  // 2. Backward pass
  std::vector<std::vector<double>> batch_output_gradients = { upstream_grad };
  layer.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_gradients, batch_hs, 1, 0);
  layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, 1, 0);

  // Check w1 analytical gradients against numerical gradients
  const double eps = 1e-5;
  const auto& analytical_w1_grads = layer.get_w1_grads();
  for (size_t idx = 0; idx < w1.size(); ++idx)
  {
    auto w1_pos = w1;
    w1_pos[idx] += eps;
    layer.set_w1_values(w1_pos);
    double loss_pos = compute_grn_loss(layer, previous_layer, input_seq, time_steps, d_in, d_out, upstream_grad);

    auto w1_neg = w1;
    w1_neg[idx] -= eps;
    layer.set_w1_values(w1_neg);
    double loss_neg = compute_grn_loss(layer, previous_layer, input_seq, time_steps, d_in, d_out, upstream_grad);

    double num_grad = (loss_pos - loss_neg) / (2.0 * eps);
    layer.set_w1_values(w1);

    EXPECT_NEAR(analytical_w1_grads[idx], num_grad, 1e-3);
  }

  // Check w_skip analytical gradients against numerical gradients
  const auto& analytical_skip_grads = layer.get_w_skip_grads();
  for (size_t idx = 0; idx < w_skip.size(); ++idx)
  {
    auto skip_pos = w_skip;
    skip_pos[idx] += eps;
    layer.set_w_skip_values(skip_pos);
    double loss_pos = compute_grn_loss(layer, previous_layer, input_seq, time_steps, d_in, d_out, upstream_grad);

    auto skip_neg = w_skip;
    skip_neg[idx] -= eps;
    layer.set_w_skip_values(skip_neg);
    double loss_neg = compute_grn_loss(layer, previous_layer, input_seq, time_steps, d_in, d_out, upstream_grad);

    double num_grad = (loss_pos - loss_neg) / (2.0 * eps);
    layer.set_w_skip_values(w_skip);

    EXPECT_NEAR(analytical_skip_grads[idx], num_grad, 1e-3);
  }
}

TEST_F(GrnLayerTest, AllFamiliesGradientCheckProjectedSkipWithLayerNorm)
{
  run_full_gradient_check(3, 4, 5, true, true, false);
}

TEST_F(GrnLayerTest, AllFamiliesGradientCheckIdentitySkipWithLayerNorm)
{
  run_full_gradient_check(4, 4, 6, true, true, false);
}

TEST_F(GrnLayerTest, AllFamiliesGradientCheckWithoutLayerNorm)
{
  run_full_gradient_check(3, 4, 5, false, true, false);
  run_full_gradient_check(4, 4, 5, false, true, false);
}

TEST_F(GrnLayerTest, AllFamiliesGradientCheckWithoutBias)
{
  run_full_gradient_check(3, 4, 5, true, false, false);
  run_full_gradient_check(4, 4, 5, false, false, false);
}

TEST_F(GrnLayerTest, AllFamiliesGradientCheckWithExternalResidual)
{
  run_full_gradient_check(3, 4, 5, true, true, true);
  run_full_gradient_check(4, 4, 5, true, true, true);
}

TEST_F(GrnLayerTest, StackedGrnLayersPropagateGradientsToEarlierLayer)
{
  const unsigned d_in = 3;
  const unsigned d_mid = 4;
  const unsigned d_out = 2;
  const unsigned d_ff = 5;
  const size_t time_steps = 3;

  GrnLayer first = make_grn_layer(d_in, d_mid, d_ff, true, activation(activation::method::elu, 1.0), 1, true, 1);
  GrnLayer second = make_grn_layer(d_mid, d_out, d_ff, true, activation(activation::method::elu, 1.0), 1, true, 2);
  set_test_weights(first);
  set_test_weights(second);

  const auto input_seq = make_wave(time_steps * d_in, 0.9, 0.2, 0.8);
  const auto upstream_grad = make_wave(time_steps * d_out, 1.1, 0.5, 1.0);

  MockLayer input_layer(0, d_in);
  const std::vector<unsigned> topology = { d_in, d_mid, d_out };

  auto go = create_batch_gradients_and_outputs(topology, 1);
  auto hs = create_batch_hidden_states(topology, 1, 1, 1);
  go[0].set_rnn_outputs(0, input_seq.data(), input_seq.size());
  first.calculate_forward_feed(go, input_layer, {}, hs, 1, true);
  second.calculate_forward_feed(go, first, {}, hs, 1, true);

  second.calculate_hidden_gradients_from_output_gradients(go, { upstream_grad }, hs, 1, 0);
  const std::vector<std::vector<double>> no_gradients;
  first.calculate_hidden_gradients_from_output_gradients(go, no_gradients, hs, 1, 0);
  second.calculate_and_store_gradients(go, hs, first, 1, 0);
  first.calculate_and_store_gradients(go, hs, input_layer, 1, 0);

  const double eps = 1e-6;
  const auto base_w1 = first.get_w1_values();
  const auto analytical_w1 = first.get_w1_grads();
  for (size_t idx = 0; idx < base_w1.size(); ++idx)
  {
    auto plus = base_w1;
    plus[idx] += eps;
    first.set_w1_values(plus);
    const double loss_plus = compute_stack_loss(first, second, input_layer, topology, input_seq, upstream_grad);
    auto minus = base_w1;
    minus[idx] -= eps;
    first.set_w1_values(minus);
    const double loss_minus = compute_stack_loss(first, second, input_layer, topology, input_seq, upstream_grad);
    first.set_w1_values(base_w1);

    const double numerical = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(analytical_w1[idx], numerical, 1e-6 + 1e-5 * std::abs(numerical)) << "first.w1[" << idx << "]";
  }

  const auto base_ln_gain = first.get_ln_gain_values();
  const auto analytical_ln_gain = first.get_ln_gain_grads();
  for (size_t idx = 0; idx < base_ln_gain.size(); ++idx)
  {
    auto plus = base_ln_gain;
    plus[idx] += eps;
    first.set_ln_gain_values(plus);
    const double loss_plus = compute_stack_loss(first, second, input_layer, topology, input_seq, upstream_grad);
    auto minus = base_ln_gain;
    minus[idx] -= eps;
    first.set_ln_gain_values(minus);
    const double loss_minus = compute_stack_loss(first, second, input_layer, topology, input_seq, upstream_grad);
    first.set_ln_gain_values(base_ln_gain);

    const double numerical = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(analytical_ln_gain[idx], numerical, 1e-6 + 1e-5 * std::abs(numerical)) << "first.ln_gain[" << idx << "]";
  }
}

TEST_F(GrnLayerTest, DropoutMaskIsAppliedToGluGradient)
{
  const unsigned d = 3;
  GrnLayer layer(
    1, d, d, 4, 0.0, Layer::Role::Hidden, activation(activation::method::elu, 1.0), OptimiserType::SGD,
    -1, 0.9, nullptr, 1, true, false, 0.0, std::optional<uint32_t>(42u));
  set_test_weights(layer);

  const size_t time_steps = 8;
  const auto input_seq = make_wave(time_steps * d, 0.9, 0.2, 0.8);
  const auto upstream_grad = make_wave(time_steps * d, 1.1, 0.5, 1.0);

  std::vector<unsigned> topology = { d, d };
  MockLayer previous_layer(0, d);
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
  batch_go[0].set_rnn_outputs(0, input_seq.data(), input_seq.size());
  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, true);

  const auto out = batch_go[0].get_rnn_outputs(1);
  const auto& hs_row = batch_hs[0].at(1);
  ASSERT_EQ(hs_row.size(), time_steps);

  layer.calculate_hidden_gradients_from_output_gradients(batch_go, { upstream_grad }, batch_hs, 1, 0);
  const auto dx = batch_go[0].get_rnn_gradients(1);
  ASSERT_EQ(dx.size(), input_seq.size());

  size_t dropped_units = 0;
  size_t fully_dropped_steps = 0;
  for (size_t t = 0; t < time_steps; ++t)
  {
    const auto mask = hs_row[t].get_cell_state_values();
    bool all_dropped = true;
    for (size_t k = 0; k < d; ++k)
    {
      if (mask[k] == 0.0)
      {
        ++dropped_units;
        EXPECT_DOUBLE_EQ(out[t * d + k], input_seq[t * d + k]);
      }
      else
      {
        all_dropped = false;
      }
    }
    if (all_dropped)
    {
      ++fully_dropped_steps;
      for (size_t k = 0; k < d; ++k)
      {
        EXPECT_NEAR(dx[t * d + k], upstream_grad[t * d + k], 1e-12);
      }
    }
  }
  EXPECT_GT(dropped_units, 0u) << "with a 0.9 dropout rate some units must be dropped";
  RecordProperty("fully_dropped_steps", static_cast<int>(fully_dropped_steps));
}

TEST_F(GrnLayerTest, NoBiasLeavesBiasGradientsAtZero)
{
  GrnLayer layer = make_grn_layer(3, 4, 5, true, activation(activation::method::elu, 1.0), 1, false);
  set_test_weights(layer);

  std::vector<double> input_seq = { 0.5, -0.3, 0.8, -0.2, 0.7, 0.1 };
  std::vector<double> upstream_grad = { 1.0, -0.5, 0.3, 0.8, 0.1, 0.2, -0.4, 0.6 };
  std::vector<unsigned> topology = { 3, 4 };
  MockLayer previous_layer(0, 3);
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
  batch_go[0].set_rnn_outputs(0, input_seq.data(), input_seq.size());
  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, true);
  layer.calculate_hidden_gradients_from_output_gradients(batch_go, { upstream_grad }, batch_hs, 1, 0);
  layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, 1, 0);

  for (const double g : layer.get_b1_grads())
  {
    EXPECT_EQ(g, 0.0);
  }
  for (const double g : layer.get_b2_grads())
  {
    EXPECT_EQ(g, 0.0);
  }
  for (const double g : layer.get_b_skip_grads())
  {
    EXPECT_EQ(g, 0.0);
  }
}

TEST_F(GrnLayerTest, SwaAveragesTowardsSnapshot)
{
  const unsigned d = 3;
  const unsigned d_ff = 6;
  GrnLayer running = make_grn_layer(d, d, d_ff, true);
  GrnLayer snapshot = make_grn_layer(d, d, d_ff, true);
  running.set_w1_values(std::vector<double>(d * d_ff, 1.0));
  snapshot.set_w1_values(std::vector<double>(d * d_ff, 3.0));
  running.set_ln_gain_values(std::vector<double>(d, 1.0));
  snapshot.set_ln_gain_values(std::vector<double>(d, 2.0));

  running.accumulate_swa_average_impl(snapshot, 1);
  for (const double v : running.get_w1_values())
  {
    EXPECT_NEAR(v, 2.0, 1e-12);
  }
  for (const double v : running.get_ln_gain_values())
  {
    EXPECT_NEAR(v, 1.5, 1e-12);
  }
}

TEST_F(GrnLayerTest, LookaheadMovesSlowTowardsFastAndResetsFast)
{
  const unsigned d = 3;
  const unsigned d_ff = 6;
  GrnLayer slow = make_grn_layer(d, d, d_ff, true);
  GrnLayer fast = make_grn_layer(d, d, d_ff, true);
  slow.set_w1_values(std::vector<double>(d * d_ff, 0.5));
  fast.set_w1_values(std::vector<double>(d * d_ff, 1.0));

  slow.update_lookahead_slow_weights_impl(fast, 0.5);
  for (const double v : slow.get_w1_values())
  {
    EXPECT_NEAR(v, 0.75, 1e-12);
  }
  for (const double v : fast.get_w1_values())
  {
    EXPECT_NEAR(v, 0.75, 1e-12);
  }
}
