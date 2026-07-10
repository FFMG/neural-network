#include <gtest/gtest.h>
#include "layers/lstmlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
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
  LSTMLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0);

  // Set weights to simple values
  layer.set_w_values({ 0.5 });
  layer.set_rw_values({ 0.1 });
  layer.set_f_w_values({ 0.2 });
  layer.set_f_rw_values({ 0.3 });
  layer.set_i_w_values({ 0.4 });
  layer.set_i_rw_values({ 0.5 });
  layer.set_o_w_values({ 0.6 });
  layer.set_o_rw_values({ 0.7 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5); // 1 time step, multiplier 5

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 0.174207488, 1e-6);
}

TEST_F(LSTMLayerTest, DropoutConsistencyVerification) {
  // 1 neuron with 100% dropout
  LSTMLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 1.0, nullptr, 1, false, 0.0);
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

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
  EXPECT_NEAR(batch_go[0].get_rnn_outputs(1)[0], 0.0, 1e-9);

  MockLayer next_layer(2, 1);
  next_layer.set_w_values({ 1.0 });
  std::vector<std::vector<double>> batch_next_grads = { { 10.0 } };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

  EXPECT_NEAR(batch_go[0].get_rnn_gate_gradients(1)[0], 0.0, 1e-9);
}

TEST_F(LSTMLayerTest, DropoutStatisticalVerification) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0);

    layer.set_w_values(std::vector<double>(num_outputs * 4, 1.0));
    layer.set_rw_values(std::vector<double>(num_outputs * num_outputs * 4, 0.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));
    
    layer.set_i_b_values(std::vector<double>(num_outputs, 10.0)); // input gate open
    layer.set_f_b_values(std::vector<double>(num_outputs, -10.0)); // forget gate closed
    layer.set_o_b_values(std::vector<double>(num_outputs, 10.0)); // output gate open

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_outputs(1);
    int dropped_count = 0;
    int kept_count = 0;
    const double expected_kept = 1.0 / (1.0 - dropout_rate);
    for (size_t i = 0; i < outputs.size(); ++i) {
        double out = outputs[i];
        if (out == 0.0) {
            dropped_count++;
        } else if (std::abs(out - expected_kept) < 0.05) {
            kept_count++;
        } else {
            Logger::error("Neuron ", i, " output unexpected value: ", out, " (expected 0.0 or ~", expected_kept, ")");
        }
    }

    EXPECT_EQ(dropped_count + kept_count, (int)num_outputs);
    EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.05);
}

TEST_F(LSTMLayerTest, DropoutNotInference) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0);

    layer.set_w_values(std::vector<double>(num_outputs * 4, 1.0));
    layer.set_rw_values(std::vector<double>(num_outputs * num_outputs * 4, 0.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));
    
    layer.set_i_b_values(std::vector<double>(num_outputs, 10.0));
    layer.set_f_b_values(std::vector<double>(num_outputs, -10.0));
    layer.set_o_b_values(std::vector<double>(num_outputs, 10.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& outputs = batch_go[0].get_outputs(1);
    for (double out : outputs) {
        EXPECT_NEAR(out, 1.0, 1e-2);
    }
}

TEST_F(LSTMLayerTest, LearningRateRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

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
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_f_w_values({ 0.2 }); layer.set_f_rw_values({ 0.3 }); layer.set_f_b_values({ 0.1 });
    layer.set_i_w_values({ 0.4 }); layer.set_i_rw_values({ 0.5 }); layer.set_i_b_values({ 0.1 });
    layer.set_o_w_values({ 0.6 }); layer.set_o_rw_values({ 0.7 }); layer.set_o_b_values({ 0.1 });
    layer.set_w_values({ 0.5 });   layer.set_rw_values({ 0.1 });   layer.set_b_values({ 0.1 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2, LSTMLayer::Multiplier); // 2 steps, correct multiplier

    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0 });
    std::vector<std::vector<double>> batch_next_grads = { { 0.0, 1.0 } }; // dL/dh_1 = 1.0

    // Test BPTT=1
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 1);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 1);

    EXPECT_NEAR(layer.get_w_grads()[0],   0.23520037, 1e-6);
    EXPECT_NEAR(layer.get_rw_grads()[0],  0.05066328, 1e-6);
    EXPECT_NEAR(layer.get_b_grads()[0],   0.23520037, 1e-6);
    EXPECT_NEAR(layer.get_f_w_grads()[0], 0.04226026, 1e-6);
    EXPECT_NEAR(layer.get_f_rw_grads()[0],0.00910306, 1e-6);
    EXPECT_NEAR(layer.get_i_w_grads()[0], 0.06588160, 1e-6);
    EXPECT_NEAR(layer.get_i_rw_grads()[0],0.01419121, 1e-6);
    EXPECT_NEAR(layer.get_o_w_grads()[0], 0.10571330, 1e-6);
    EXPECT_NEAR(layer.get_o_rw_grads()[0],0.02277115, 1e-6);

    // Test BPTT=2 (Full sequence)
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 2);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 2);

    EXPECT_NEAR(layer.get_w_grads()[0],   0.40978412, 1e-6);
    EXPECT_NEAR(layer.get_rw_grads()[0],  0.05066328, 1e-6);
    EXPECT_NEAR(layer.get_b_grads()[0],   0.40978412, 1e-6);
    EXPECT_NEAR(layer.get_f_w_grads()[0], 0.04226026, 1e-6);
    EXPECT_NEAR(layer.get_f_rw_grads()[0],0.00910306, 1e-6);
    EXPECT_NEAR(layer.get_i_w_grads()[0], 0.11562776, 1e-6);
    EXPECT_NEAR(layer.get_i_rw_grads()[0],0.01419121, 1e-6);
    EXPECT_NEAR(layer.get_o_w_grads()[0], 0.11594396, 1e-6);
    EXPECT_NEAR(layer.get_o_rw_grads()[0],0.02277115, 1e-6);
}

TEST_F(LSTMLayerTest, ApplyStoredGradientsCacheUpdate)
{
    LSTMLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0 });   layer.set_rw_values({ 0.5 });
    layer.set_f_w_values({ 0.0 }); layer.set_f_rw_values({ 0.0 });
    layer.set_i_w_values({ 0.0 }); layer.set_i_rw_values({ 0.0 });
    layer.set_o_w_values({ 0.0 }); layer.set_o_rw_values({ 0.0 });

    layer.set_f_b_values({ 10.0 });
    layer.set_i_b_values({ 10.0 });
    layer.set_o_b_values({ 10.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 1 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2, LSTMLayer::Multiplier); 

    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    auto outputs = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs[0], 0.999909, 1e-4);
    EXPECT_NEAR(outputs[1], 2.49968, 1e-4);

    layer.set_f_rw_grads({ 0.1 });
    layer.set_i_rw_grads({ 0.1 });
    layer.set_o_rw_grads({ 0.1 });
    layer.set_rw_grads({ 0.1 });
    layer.apply_stored_gradients(1.0, 1.0);

    EXPECT_NEAR(layer.get_f_rw_values()[0], -0.1, 1e-9);
    EXPECT_NEAR(layer.get_i_rw_values()[0], -0.1, 1e-9);
    EXPECT_NEAR(layer.get_o_rw_values()[0], -0.1, 1e-9);
    EXPECT_NEAR(layer.get_rw_values()[0], 0.4, 1e-9);

    auto batch_hs2 = create_batch_hidden_states(topology, 1, 2, LSTMLayer::Multiplier); 
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs2, 1, false);

    auto outputs2 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs2[0], 0.999909, 1e-4);
    EXPECT_NEAR(outputs2[1], 2.39968, 1e-4);
}

TEST_F(LSTMLayerTest, BiasCachingCorrectness)
{
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0 });   layer.set_rw_values({ 0.0 });
    layer.set_f_w_values({ 0.0 }); layer.set_f_rw_values({ 0.0 });
    layer.set_i_w_values({ 0.0 }); layer.set_i_rw_values({ 0.0 });
    layer.set_o_w_values({ 0.0 }); layer.set_o_rw_values({ 0.0 });

    layer.set_f_b_values({ 10.0 });
    layer.set_i_b_values({ 10.0 });
    layer.set_o_b_values({ 10.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);

    batch_go[0].set_rnn_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);
    auto outputs1 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs1[0], 1.0, 1e-3);

    layer.set_b_values({ 10.0, 10.0, 10.0, 2.0 });

    auto batch_hs2 = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs2, 1, false);
    auto outputs2 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs2[0], 3.0, 1e-3);
}

TEST_F(LSTMLayerTest, TransposedWeightsAndFastBpttPassCorrectness) {
    // 2 inputs, 2 neurons, batch size 2, 2 time steps
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    // Populate weights
    layer.set_f_w_values({ 0.1, 0.2, 0.3, 0.4 });
    layer.set_f_rw_values({ 0.15, 0.25, 0.35, 0.45 });
    layer.set_f_b_values({ 0.05, 0.15 });

    layer.set_i_w_values({ 0.2, 0.3, 0.4, 0.5 });
    layer.set_i_rw_values({ 0.25, 0.35, 0.45, 0.55 });
    layer.set_i_b_values({ 0.06, 0.16 });

    layer.set_o_w_values({ 0.3, 0.4, 0.5, 0.6 });
    layer.set_o_rw_values({ 0.35, 0.45, 0.55, 0.65 });
    layer.set_o_b_values({ 0.07, 0.17 });

    layer.set_w_values({ 0.4, 0.5, 0.6, 0.7 });
    layer.set_rw_values({ 0.45, 0.55, 0.65, 0.75 });
    layer.set_b_values({ 0.08, 0.18 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2); // batch size 2
    auto batch_hs = create_batch_hidden_states(topology, 2, 2, LSTMLayer::Multiplier); // 2 steps

    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0, 0.5, 0.5 });
    batch_go[1].set_rnn_outputs(0, { 0.8, 0.8, 0.4, 0.4 });

    // Forward pass
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, false);

    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0, 0.5, 0.2, 0.8 });
    std::vector<std::vector<double>> batch_next_grads = {
        { 0.1, 0.2, 0.3, 0.4 },
        { 0.5, 0.6, 0.7, 0.8 }
    };

    // Initialize cell state values so that gradients propagate through tanh's derivatives
    batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0 });
    batch_hs[0].at(1, 1).set_cell_state_values({ 1.0, 1.0 });
    batch_hs[1].at(1, 0).set_cell_state_values({ 1.0, 1.0 });
    batch_hs[1].at(1, 1).set_cell_state_values({ 1.0, 1.0 });

    // Backward pass (BPTT = 2)
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 2, 2);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 2, 2);

    // Verify gradients are non-zero and accumulated successfully
    EXPECT_GT(std::abs(layer.get_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_rw_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_f_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_f_rw_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_i_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_i_rw_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_o_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_o_rw_grads()[0]), 0.0);
}

TEST_F(LSTMLayerTest, BPTTWorkspaceResizeCorrectness) {
    // 2 inputs, 2 outputs, batch size 2, 2 time steps
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_f_w_values({ 0.1, 0.2, 0.3, 0.4 });
    layer.set_f_rw_values({ 0.15, 0.25, 0.35, 0.45 });
    layer.set_f_b_values({ 0.05, 0.15 });
    layer.set_i_w_values({ 0.2, 0.3, 0.4, 0.5 });
    layer.set_i_rw_values({ 0.25, 0.35, 0.45, 0.55 });
    layer.set_i_b_values({ 0.06, 0.16 });
    layer.set_o_w_values({ 0.3, 0.4, 0.5, 0.6 });
    layer.set_o_rw_values({ 0.35, 0.45, 0.55, 0.65 });
    layer.set_o_b_values({ 0.07, 0.17 });
    layer.set_w_values({ 0.4, 0.5, 0.6, 0.7 });
    layer.set_rw_values({ 0.45, 0.55, 0.65, 0.75 });
    layer.set_b_values({ 0.08, 0.18 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

    // Run backprop first time (creates workspace initially)
    auto batch_go1 = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs1 = create_batch_hidden_states(topology, 2, 2, LSTMLayer::Multiplier);
    batch_go1[0].set_rnn_outputs(0, { 1.0, 1.0, 0.5, 0.5 });
    batch_go1[1].set_rnn_outputs(0, { 0.8, 0.8, 0.4, 0.4 });
    layer.calculate_forward_feed(batch_go1, prev_layer, {}, batch_hs1, 2, false);

    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0, 0.5, 0.2, 0.8 });
    std::vector<std::vector<double>> batch_next_grads = {
        { 0.1, 0.2, 0.3, 0.4 },
        { 0.5, 0.6, 0.7, 0.8 }
    };

    layer.calculate_hidden_gradients(batch_go1, next_layer, batch_next_grads, batch_hs1, 2, 2);
    layer.calculate_and_store_gradients(batch_go1, batch_hs1, prev_layer, 2, 2);
    double initial_norm = layer.get_gradient_norm_sq();
    EXPECT_GT(initial_norm, 0.0);

    // Run backprop second time with the SAME sizes (tests std::fill workspace reuse path)
    layer.zero_gradients();
    layer.calculate_hidden_gradients(batch_go1, next_layer, batch_next_grads, batch_hs1, 2, 2);
    layer.calculate_and_store_gradients(batch_go1, batch_hs1, prev_layer, 2, 2);
    EXPECT_NEAR(layer.get_gradient_norm_sq(), initial_norm, 1e-9);

    // Run backprop third time with DIFFERENT sizes (tests resize/assign reallocation path)
    layer.zero_gradients();
    auto batch_go2 = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs2 = create_batch_hidden_states(topology, 1, 2, LSTMLayer::Multiplier);
    batch_go2[0].set_rnn_outputs(0, { 1.0, 1.0, 0.5, 0.5 });
    layer.calculate_forward_feed(batch_go2, prev_layer, {}, batch_hs2, 1, false);

    std::vector<std::vector<double>> batch_next_grads2 = { { 0.1, 0.2, 0.3, 0.4 } };
    layer.calculate_hidden_gradients(batch_go2, next_layer, batch_next_grads2, batch_hs2, 1, 2);
    layer.calculate_and_store_gradients(batch_go2, batch_hs2, prev_layer, 1, 2);
    EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
}

TEST_F(LSTMLayerTest, SingleVSMultiThreadedEquivalence)
{
  unsigned num_inputs = 100;
  unsigned num_outputs = 100;
  size_t batch_size = 100;
  size_t num_time_steps = 20;

  // Layer 1: single threaded
  LSTMLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

  // Layer 2: multi threaded
  LSTMLayer layer_mt(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 4, true, 0.0);

  // Helper to fill vectors with identical values
  auto initialize_weights = [&](LSTMLayer& l)
  {
    l.set_w_values(std::vector<double>(num_inputs * num_outputs, 0.05));
    l.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.08));
    l.set_b_values(std::vector<double>(num_outputs, 0.01));

    l.set_f_w_values(std::vector<double>(num_inputs * num_outputs, 0.06));
    l.set_f_rw_values(std::vector<double>(num_outputs * num_outputs, 0.09));
    l.set_f_b_values(std::vector<double>(num_outputs, 0.02));

    l.set_i_w_values(std::vector<double>(num_inputs * num_outputs, 0.07));
    l.set_i_rw_values(std::vector<double>(num_outputs * num_outputs, 0.10));
    l.set_i_b_values(std::vector<double>(num_outputs, 0.03));

    l.set_o_w_values(std::vector<double>(num_inputs * num_outputs, 0.04));
    l.set_o_rw_values(std::vector<double>(num_outputs * num_outputs, 0.07));
    l.set_o_b_values(std::vector<double>(num_outputs, 0.04));
  };

  initialize_weights(layer_st);
  initialize_weights(layer_mt);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  // Setup batch inputs and next gradients
  auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_time_steps, LSTMLayer::Multiplier);
  auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_time_steps, LSTMLayer::Multiplier);

  std::vector<double> inputs(num_time_steps * num_inputs, 0.5);
  std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_time_steps * num_outputs, 0.25));

  for (size_t b = 0; b < batch_size; ++b)
  {
    batch_go_st[b].set_rnn_outputs(0, inputs);
    batch_go_mt[b].set_rnn_outputs(0, inputs);
  }

  // Forward feed
  layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, false);
  layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, false);

  // Backward feed
  MockLayer next_layer(2, num_outputs);
  std::vector<double> next_weights(num_outputs * num_outputs, 0.1);
  next_layer.set_w_values(next_weights);

  layer_st.calculate_hidden_gradients(batch_go_st, next_layer, batch_next_grads, batch_hs_st, batch_size, static_cast<int>(num_time_steps));
  layer_mt.calculate_hidden_gradients(batch_go_mt, next_layer, batch_next_grads, batch_hs_mt, batch_size, static_cast<int>(num_time_steps));

  // Store gradients
  layer_st.calculate_and_store_gradients(batch_go_st, batch_hs_st, prev_layer, batch_size, static_cast<int>(num_time_steps));
  layer_mt.calculate_and_store_gradients(batch_go_mt, batch_hs_mt, prev_layer, batch_size, static_cast<int>(num_time_steps));

  // Helper to assert two vectors are equal within tolerance
  auto assert_vectors_equal = [](const std::vector<double>& v1, const std::vector<double>& v2)
  {
    ASSERT_EQ(v1.size(), v2.size());
    for (size_t i = 0; i < v1.size(); ++i)
    {
      EXPECT_NEAR(v1[i], v2[i], 1e-9);
    }
  };

  // Assert all gradients are identical
  assert_vectors_equal(layer_st.get_w_grads(), layer_mt.get_w_grads());
  assert_vectors_equal(layer_st.get_rw_grads(), layer_mt.get_rw_grads());
  assert_vectors_equal(layer_st.get_b_grads(), layer_mt.get_b_grads());

  assert_vectors_equal(layer_st.get_f_w_grads(), layer_mt.get_f_w_grads());
  assert_vectors_equal(layer_st.get_f_rw_grads(), layer_mt.get_f_rw_grads());
  assert_vectors_equal(layer_st.get_f_b_grads(), layer_mt.get_f_b_grads());

  assert_vectors_equal(layer_st.get_i_w_grads(), layer_mt.get_i_w_grads());
  assert_vectors_equal(layer_st.get_i_rw_grads(), layer_mt.get_i_rw_grads());
  assert_vectors_equal(layer_st.get_i_b_grads(), layer_mt.get_i_b_grads());

  assert_vectors_equal(layer_st.get_o_w_grads(), layer_mt.get_o_w_grads());
  assert_vectors_equal(layer_st.get_o_rw_grads(), layer_mt.get_o_rw_grads());
  assert_vectors_equal(layer_st.get_o_b_grads(), layer_mt.get_o_b_grads());
}


