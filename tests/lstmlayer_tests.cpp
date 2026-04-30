#include <gtest/gtest.h>
#include "../src/neuralnetwork/lstmlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>

using namespace test_helper;

class LSTMLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
  }
};

TEST_F(LSTMLayerTest, ConstructionAndTopology) {
  LSTMLayer layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

  EXPECT_EQ(layer.get_layer_index(), 1);
  EXPECT_EQ(layer.get_number_input_neurons(), 2);
  EXPECT_EQ(layer.get_number_output_neurons(), 3);
  EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Lstm);
  EXPECT_TRUE(layer.use_bptt());
  EXPECT_EQ(layer.get_pre_activation_multiplier(), 5); 
}

TEST_F(LSTMLayerTest, ForwardFeedMathematicalVerification) {
  // 1 input, 1 hidden neuron
  LSTMLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0);

  // Set weights to simple values
  // Candidate weights (W, RW, B)
  layer.set_w_values({ 0.5 });
  layer.set_rw_values({ 0.1 });
  // Forget gate
  layer.set_f_w_values({ 0.2 });
  layer.set_f_rw_values({ 0.3 });
  // Input gate
  layer.set_i_w_values({ 0.4 });
  layer.set_i_rw_values({ 0.5 });
  // Output gate
  layer.set_o_w_values({ 0.6 });
  layer.set_o_rw_values({ 0.7 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5); // 1 time step, multiplier 5

  // x_0 = 1.0
  batch_go[0].set_rnn_outputs(0, { 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  // Math check:
  // x_0 = 1.0, h_{-1} = 0, c_{-1} = 0
  // f_pre = 1.0 * 0.2 + 0.0 * 0.3 = 0.2 => f = 1 / (1 + exp(-0.2)) = 0.549833997
  // i_pre = 1.0 * 0.4 + 0.0 * 0.5 = 0.4 => i = 1 / (1 + exp(-0.4)) = 0.598687660
  // o_pre = 1.0 * 0.6 + 0.0 * 0.7 = 0.6 => o = 1 / (1 + exp(-0.6)) = 0.645656306
  // g_pre = 1.0 * 0.5 + 0.0 * 0.1 = 0.5 => g = tanh(0.5) = 0.462117157
  
  // c_0 = f * c_{-1} + i * g = 0.549833997 * 0 + 0.598687660 * 0.462117157 = 0.276664219
  // h_0 = o * tanh(c_0) = 0.645656306 * tanh(0.276664219) = 0.645656306 * 0.26982046 = 0.174207488

  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 0.174207488, 1e-6);
}

TEST_F(LSTMLayerTest, DropoutConsistencyVerification) {
  // 1 neuron with 100% dropout
  LSTMLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 1.0, nullptr, 1, false, 0.0);
  layer.set_w_values({ 1.0 });
  layer.set_rw_values({ 1.0 });
  layer.set_f_w_values({ 1.0 });
  layer.set_f_rw_values({ 1.0 });
  layer.set_i_w_values({ 1.0 });
  layer.set_i_rw_values({ 1.0 });
  layer.set_o_w_values({ 1.0 });
  layer.set_o_rw_values({ 1.0 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5); // multiplier 5

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  // Forward pass: should drop (output 0.0)
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
  EXPECT_NEAR(batch_go[0].get_rnn_outputs(1)[0], 0.0, 1e-9);

  // Backward pass: gradient should also be 0.0 if mask is reused
  MockLayer next_layer(2, 1);
  next_layer.set_w_values({ 1.0 });
  std::vector<std::vector<double>> batch_next_grads = { { 10.0 } };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

  // Reusing the mask means dh_curr becomes (upstream + dh_next) * 0.0 = 0.0
  EXPECT_NEAR(batch_go[0].get_rnn_gate_gradients(1)[0], 0.0, 1e-9);
}
