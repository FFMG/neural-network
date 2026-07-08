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

