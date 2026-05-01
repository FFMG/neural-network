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

TEST_F(LSTMLayerTest, LearningRateRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    std::vector<double> learning_rates = { 0.0, 0.0001, 0.01, 0.5, 1.0, 2.0 };
    
    for (double lr : learning_rates) {
        layer.set_w_values({ 1.0 });
        layer.set_rw_values({ 1.0 });
        layer.set_b_values({ 0.5 });
        layer.set_f_w_values({ 1.0 });
        layer.set_f_rw_values({ 1.0 });
        layer.set_f_b_values({ 0.5 });
        layer.set_i_w_values({ 1.0 });
        layer.set_i_rw_values({ 1.0 });
        layer.set_i_b_values({ 0.5 });
        layer.set_o_w_values({ 1.0 });
        layer.set_o_rw_values({ 1.0 });
        layer.set_o_b_values({ 0.5 });
        
        layer.set_w_grads({ 0.1 });
        layer.set_rw_grads({ 0.1 });
        layer.set_b_grads({ 0.05 });
        layer.set_f_w_grads({ 0.1 });
        layer.set_f_rw_grads({ 0.1 });
        layer.set_f_b_grads({ 0.05 });
        layer.set_i_w_grads({ 0.1 });
        layer.set_i_rw_grads({ 0.1 });
        layer.set_i_b_grads({ 0.05 });
        layer.set_o_w_grads({ 0.1 });
        layer.set_o_rw_grads({ 0.1 });
        layer.set_o_b_grads({ 0.05 });

        layer.apply_stored_gradients(lr, 1.0);

        EXPECT_NEAR(layer.get_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_b_values()[0], 0.5 - lr * 0.05, 1e-9);
        EXPECT_NEAR(layer.get_f_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_f_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_f_b_values()[0], 0.5 - lr * 0.05, 1e-9);
        EXPECT_NEAR(layer.get_i_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_i_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_i_b_values()[0], 0.5 - lr * 0.05, 1e-9);
        EXPECT_NEAR(layer.get_o_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_o_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_o_b_values()[0], 0.5 - lr * 0.05, 1e-9);
    }
}

TEST_F(LSTMLayerTest, BPTTRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_f_w_values({ 0.2 }); layer.set_f_rw_values({ 0.3 }); layer.set_f_b_values({ 0.1 });
    layer.set_i_w_values({ 0.4 }); layer.set_i_rw_values({ 0.5 }); layer.set_i_b_values({ 0.1 });
    layer.set_o_w_values({ 0.6 }); layer.set_o_rw_values({ 0.7 }); layer.set_o_b_values({ 0.1 });
    layer.set_w_values({ 0.5 });   layer.set_rw_values({ 0.1 });   layer.set_b_values({ 0.1 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2, 5); // 2 steps, multiplier 5

    // Forward pass x_0=1, x_1=1
    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0 });
    std::vector<std::vector<double>> batch_next_grads = { { 0.0, 1.0 } }; // dL/dh_1 = 1.0

    // Test BPTT=1
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 1);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 1);

    // Expected values from manual math (BPTT=1):
    EXPECT_NEAR(layer.get_w_grads()[0],   0.23526629, 1e-6);
    EXPECT_NEAR(layer.get_rw_grads()[0],  0.05066721, 1e-6);
    EXPECT_NEAR(layer.get_b_grads()[0],   0.23526629, 1e-6);
    EXPECT_NEAR(layer.get_f_w_grads()[0], 0.04226857, 1e-6);
    EXPECT_NEAR(layer.get_f_rw_grads()[0],0.00910300, 1e-6);
    EXPECT_NEAR(layer.get_i_w_grads()[0], 0.06588742, 1e-6);
    EXPECT_NEAR(layer.get_i_rw_grads()[0],0.01418959, 1e-6);
    EXPECT_NEAR(layer.get_o_w_grads()[0], 0.10568376, 1e-6);
    EXPECT_NEAR(layer.get_o_rw_grads()[0],0.02275997, 1e-6);

    // Test BPTT=2 (Full sequence)
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 2);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 2);

    // Expected values from manual math (BPTT=2):
    EXPECT_NEAR(layer.get_w_grads()[0],   0.23526629 + 0.17460737, 1e-6);
    EXPECT_NEAR(layer.get_rw_grads()[0],  0.05066721, 1e-6);
    EXPECT_NEAR(layer.get_b_grads()[0],   0.23526629 + 0.17460737, 1e-6);
    EXPECT_NEAR(layer.get_f_w_grads()[0], 0.04226857, 1e-6);
    EXPECT_NEAR(layer.get_f_rw_grads()[0],0.00910300, 1e-6);
    EXPECT_NEAR(layer.get_i_w_grads()[0], 0.06588742 + 0.04975254, 1e-6);
    EXPECT_NEAR(layer.get_i_rw_grads()[0],0.01418959, 1e-6);
    EXPECT_NEAR(layer.get_o_w_grads()[0], 0.10568376 + 0.01022788, 1e-6);
    EXPECT_NEAR(layer.get_o_rw_grads()[0],0.02275997, 1e-6);
}
