#include <gtest/gtest.h>
#include "../src/neuralnetwork/elmanrnnlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>

using namespace test_helper;

class ElmanRNNLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
  }
};

TEST_F(ElmanRNNLayerTest, ConstructionAndTopology) {
  ElmanRNNLayer layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

  EXPECT_EQ(layer.get_layer_index(), 1);
  EXPECT_EQ(layer.get_number_input_neurons(), 2);
  EXPECT_EQ(layer.get_number_output_neurons(), 3);
  EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Elman);
  EXPECT_TRUE(layer.use_bptt());
  EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);
}

TEST_F(ElmanRNNLayerTest, ForwardFeedMathematicalVerification) {
  // 2 inputs, 2 hidden neurons
  ElmanRNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

  // W = [[0.1, 0.2], [0.3, 0.4]]
  layer.set_w_values({ 0.1, 0.2, 0.3, 0.4 });
  // U = [[0.5, 0.6], [0.7, 0.8]]
  layer.set_rw_values({ 0.5, 0.6, 0.7, 0.8 });
  // B = [0.1, -0.1]
  layer.set_b_values({ 0.1, -0.1 });

  MockLayer prev_layer(0, 2);
  std::vector<unsigned> topology = { 2, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 2); // 2 time steps

  // Input sequence: x_t0=[1.0, 0.5], x_t1=[0.0, 1.0]
  batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, 0.0, 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  // Math check:
  // t=0:
  // z_0 = W * x_0 + B + U * h_{-1} (h_{-1} = 0)
  // z_0[0] = 1.0*0.1 + 0.5*0.3 + 0.1 = 0.1 + 0.15 + 0.1 = 0.35
  // z_0[1] = 1.0*0.2 + 0.5*0.4 - 0.1 = 0.2 + 0.20 - 0.1 = 0.30
  // h_0 = [0.35, 0.30]

  // t=1:
  // z_1 = W * x_1 + B + U * h_0
  // z_1[0] = 0.0*0.1 + 1.0*0.3 + 0.1 + (0.35*0.5 + 0.30*0.7)
  //        = 0.3 + 0.1 + (0.175 + 0.21) = 0.4 + 0.385 = 0.785
  // z_1[1] = 0.0*0.2 + 1.0*0.4 - 0.1 + (0.35*0.6 + 0.30*0.8)
  //        = 0.4 - 0.1 + (0.21 + 0.24) = 0.3 + 0.45 = 0.75
  // h_1 = [0.785, 0.75]

  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 0.35, 1e-9);
  EXPECT_NEAR(rnn_out[1], 0.30, 1e-9);
  EXPECT_NEAR(rnn_out[2], 0.785, 1e-9);
  EXPECT_NEAR(rnn_out[3], 0.75, 1e-9);
}

TEST_F(ElmanRNNLayerTest, BPTTMathematicalVerification) {
  // 1 input, 1 neuron for extreme simplicity
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0);

  layer.set_w_values({ 0.5 });
  layer.set_rw_values({ 0.8 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 2); // 2 time steps

  // Forward pass
  batch_go[0].set_rnn_outputs(0, { 1.0, 2.0 }); // x_0=1, x_1=2
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
  // h_0 = 1 * 0.5 = 0.5
  // h_1 = 2 * 0.5 + 0.5 * 0.8 = 1.0 + 0.4 = 1.4

  // Assume next layer is also 1 neuron, W_next = 2.0
  MockLayer next_layer(2, 1);
  next_layer.set_w_values({ 2.0 });

  // Next layer gradients for 2 time steps: [10.0, 10.0]
  std::vector<std::vector<double>> batch_next_grads = { { 10.0, 10.0 } };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

  // BPTT Math:
  // t=1:
  // upstream_1 = g_next_1 * W_next = 10.0 * 2.0 = 20.0
  // dh_1 = upstream_1 + dh_next (0) = 20.0
  // g_tick_1 = dh_1 * deriv(1.4) = 20.0 * 1.0 = 20.0
  // dx_1 = g_tick_1 * W = 20.0 * 0.5 = 10.0
  // dh_next_for_0 = g_tick_1 * U = 20.0 * 0.8 = 16.0

  // t=0:
  // upstream_0 = g_next_0 * W_next = 10.0 * 2.0 = 20.0
  // dh_0 = upstream_0 + dh_next_for_0 = 20.0 + 16.0 = 36.0
  // g_tick_0 = dh_0 * deriv(0.5) = 36.0 * 1.0 = 36.0
  // dx_0 = g_tick_0 * W = 36.0 * 0.5 = 18.0

  const auto rnn_grads = batch_go[0].get_rnn_gradients(1);
  EXPECT_NEAR(rnn_grads[0], 18.0, 1e-9);
  EXPECT_NEAR(rnn_grads[1], 10.0, 1e-9);

  const auto gate_grads = batch_go[0].get_rnn_gate_gradients(1);
  EXPECT_NEAR(gate_grads[0], 36.0, 1e-9);
  EXPECT_NEAR(gate_grads[1], 20.0, 1e-9);
}

TEST_F(ElmanRNNLayerTest, GradientStorageVerification) {
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0);
  layer.set_w_values({ 0.5 });
  layer.set_rw_values({ 0.8 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 2);

  batch_go[0].set_rnn_outputs(0, { 1.0, 2.0 });
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
  // h_0 = 0.5, h_1 = 1.4

  // Set gate gradients manually
  batch_go[0].set_rnn_gate_gradients(1, { 52.0, 40.0 });

  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 0);

  // W_grad = (g_0 * x_0 + g_1 * x_1) / batch_size
  //        = (52.0 * 1.0 + 40.0 * 2.0) / 1 = 52 + 80 = 132.0
  // RW_grad = (g_0 * h_{-1} + g_1 * h_0) / batch_size
  //         = (52.0 * 0.0 + 40.0 * 0.5) / 1 = 20.0

  EXPECT_NEAR(layer.get_w_grads()[0], 132.0, 1e-9);
  EXPECT_NEAR(layer.get_rw_grads()[0], 20.0, 1e-9);
}

TEST_F(ElmanRNNLayerTest, DropoutConsistencyVerification) {
  // 1 neuron with 100% dropout (always drops)
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 1.0, nullptr, 1, false, 0.0);
  layer.set_w_values({ 1.0 });
  layer.set_rw_values({ 1.0 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1);

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  // Forward pass: should drop the neuron (output 0.0)
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
  EXPECT_NEAR(batch_go[0].get_rnn_outputs(1)[0], 0.0, 1e-9);

  // Backward pass: gradient should also be 0.0 because of dropout mask
  MockLayer next_layer(2, 1);
  next_layer.set_w_values({ 1.0 });
  std::vector<std::vector<double>> batch_next_grads = { { 10.0 } };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

  // Grad should be 0.0
  EXPECT_NEAR(batch_go[0].get_rnn_gate_gradients(1)[0], 0.0, 1e-9);
}
