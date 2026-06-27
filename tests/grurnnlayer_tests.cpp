#include <gtest/gtest.h>
#include "layers/grurnnlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
using namespace test_helper;

class GRURNNLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(GRURNNLayerTest, Construction) {
    GRURNNLayer layer(1, 2, 3, 0.01, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::Adam, -1, 0.0, nullptr, 1, true, 0.9);
    EXPECT_EQ(layer.get_layer_index(), 1);
    EXPECT_EQ(layer.get_number_input_neurons(), 2);
    EXPECT_EQ(layer.get_number_neurons(), 3);
    EXPECT_EQ(layer.get_pre_activation_multiplier(), 5);
}

TEST_F(GRURNNLayerTest, ForwardFeedMathematicalVerification) {
    // 1 input, 1 neuron GRU
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    
    layer.set_z_w_values({ 0.5 }); layer.set_z_rw_values({ 0.1 }); layer.set_z_b_values({ 0.2 });
    layer.set_r_w_values({ 0.6 }); layer.set_r_rw_values({ 0.2 }); layer.set_r_b_values({ 0.3 });
    layer.set_w_values({ 0.7 });   layer.set_rw_values({ 0.3 });   layer.set_b_values({ 0.4 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 1 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5);
    
    batch_go[0].set_outputs(0, { 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], 0.534884, 1e-6);
}

TEST_F(GRURNNLayerTest, BPTTMathematicalVerification) {
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    
    layer.set_z_w_values({ 0.5 }); layer.set_z_rw_values({ 0.1 }); layer.set_z_b_values({ 0.2 });
    layer.set_r_w_values({ 0.6 }); layer.set_r_rw_values({ 0.2 }); layer.set_r_b_values({ 0.3 });
    layer.set_w_values({ 0.7 });   layer.set_rw_values({ 0.3 });   layer.set_b_values({ 0.4 });

    MockLayer prev_layer(0, 1);
    MockLayer next_layer(2, 1);
    next_layer.set_w_values({ 1.0 });
    
    std::vector<unsigned> topology = { 1, 1, 1 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5);
    
    batch_go[0].set_outputs(0, { 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    std::vector<std::vector<double>> batch_next_grads = { { 1.0 } };
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

    const auto& gate_grads = batch_go[0].get_rnn_gate_gradients(1);
    EXPECT_NEAR(gate_grads[0], 0.24001, 1e-4); // dh_hat
    EXPECT_NEAR(gate_grads[1], 0.17748, 1e-4); // dz
    EXPECT_NEAR(gate_grads[2], 0.0, 1e-4);     // dr
    
    const auto& in_grads = batch_go[0].get_rnn_gradients(1);
    EXPECT_NEAR(in_grads[0], 0.256747, 1e-4);
}

TEST_F(GRURNNLayerTest, DropoutConsistency) {
    // Test that dropout mask is preserved and applied correctly in BPTT
    // Use high dropout rate (0.5) to ensure it triggers
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.5, nullptr, 1, true, 0.0);
    
    layer.set_z_w_values({ 0.5 }); layer.set_z_rw_values({ 0.1 }); layer.set_z_b_values({ 0.2 });
    layer.set_r_w_values({ 0.6 }); layer.set_r_rw_values({ 0.2 }); layer.set_r_b_values({ 0.3 });
    layer.set_w_values({ 0.7 });   layer.set_rw_values({ 0.3 });   layer.set_b_values({ 0.4 });

    MockLayer prev_layer(0, 1);
    MockLayer next_layer(2, 1);
    next_layer.set_w_values({ 1.0 });
    
    std::vector<unsigned> topology = { 1, 1, 1 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5);
    
    batch_go[0].set_outputs(0, { 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
    
    const auto& packed = batch_hs[0].at(1)[0].get_pre_activation_sums();
    double mask = packed[4]; // Our stored mask
    EXPECT_TRUE(mask == 0.0 || approx_equal(mask, 2.0)); // 1/(1-0.5) = 2.0
    
    const auto& outputs = batch_go[0].get_outputs(1);
    
    if (mask == 0.0) {
        EXPECT_NEAR(outputs[0], 0.0, 1e-6);
    } else {
        EXPECT_NEAR(outputs[0], 0.534884 * 2.0, 1e-6);
    }
    
    // Backprop
    std::vector<std::vector<double>> batch_next_grads = { { 1.0 } };
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);
    
    const auto& gate_grads = batch_go[0].get_rnn_gate_gradients(1);
    EXPECT_NEAR(gate_grads[0], 0.24001 * mask, 1e-4);
    
    double expected_dz = 1.0 * (packed[3] * mask - 0.0) * 0.668188 * 0.331812;
    EXPECT_NEAR(gate_grads[1], expected_dz, 1e-4);
}

TEST_F(GRURNNLayerTest, SequenceUnrolling3Steps) {
    // 1 input, 1 neuron GRU
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    
    // Set weights to simple values
    layer.set_z_w_values({ 0.1 }); layer.set_z_rw_values({ 0.1 }); layer.set_z_b_values({ 0.0 });
    layer.set_r_w_values({ 0.1 }); layer.set_r_rw_values({ 0.1 }); layer.set_r_b_values({ 0.0 });
    layer.set_w_values({ 0.1 });   layer.set_rw_values({ 0.1 });   layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 1 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 3, 5); // 3 steps
    
    // Feed sequence [1.0, 0.5, -1.0]
    batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, -1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_rnn_outputs(1);
    
    EXPECT_NEAR(outputs[0], 0.052323, 1e-5);
    EXPECT_NEAR(outputs[1], 0.052486, 1e-5);
    EXPECT_NEAR(outputs[2], -0.018810, 1e-5);
}

TEST_F(GRURNNLayerTest, DropoutStatisticalVerification) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0);

    // Identity weights for hidden candidate, zero for gates to keep it simple
    layer.set_w_values(std::vector<double>(num_outputs, 1.0));
    layer.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));
    
    layer.set_z_w_values(std::vector<double>(num_outputs, 0.0));
    layer.set_z_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));
    layer.set_z_b_values(std::vector<double>(num_outputs, 10.0)); // large bias for z means z ~ 1 (always update)

    layer.set_r_w_values(std::vector<double>(num_outputs, 0.0));
    layer.set_r_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));
    layer.set_r_b_values(std::vector<double>(num_outputs, 10.0)); // large bias for r means r ~ 1 (no reset)

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5);

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
            Logger::error("GRU Neuron ", i, " output unexpected value: ", out, " (expected 0.0 or ~", expected_kept, ")");
        }
    }

    EXPECT_EQ(dropped_count + kept_count, (int)num_outputs);
    EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.05);
}

TEST_F(GRURNNLayerTest, DropoutNotInference) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0);

    layer.set_w_values(std::vector<double>(num_outputs, 1.0));
    layer.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));
    
    layer.set_z_b_values(std::vector<double>(num_outputs, 10.0));
    layer.set_r_b_values(std::vector<double>(num_outputs, 10.0));

    // zero out the weights so we only test the bias.
    layer.set_z_w_values(std::vector<double>(num_outputs * num_inputs, 0.0));
    layer.set_z_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));
    layer.set_r_w_values(std::vector<double>(num_outputs * num_inputs, 0.0));
    layer.set_r_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& outputs = batch_go[0].get_outputs(1);
    for (double out : outputs) {
        EXPECT_NEAR(out, 1.0, 1e-2); // Relaxed tolerance due to sigmoid(10) compounding
    }
}

TEST_F(GRURNNLayerTest, LearningRateRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    std::vector<double> learning_rates = { 0.0, 0.0001, 0.01, 0.5, 1.0, 2.0 };
    
    for (double lr : learning_rates) {
        layer.set_w_values({ 1.0 });
        layer.set_rw_values({ 1.0 });
        layer.set_b_values({ 0.5 });
        layer.set_z_w_values({ 1.0 });
        layer.set_z_rw_values({ 1.0 });
        layer.set_z_b_values({ 0.5 });
        layer.set_r_w_values({ 1.0 });
        layer.set_r_rw_values({ 1.0 });
        layer.set_r_b_values({ 0.5 });
        
        layer.set_w_grads({ 0.1 });
        layer.set_rw_grads({ 0.1 });
        layer.set_b_grads({ 0.05 });
        layer.set_z_w_grads({ 0.1 });
        layer.set_z_rw_grads({ 0.1 });
        layer.set_z_b_grads({ 0.05 });
        layer.set_r_w_grads({ 0.1 });
        layer.set_r_rw_grads({ 0.1 });
        layer.set_r_b_grads({ 0.05 });

        layer.apply_stored_gradients(lr, 1.0);

        EXPECT_NEAR(layer.get_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_b_values()[0], 0.5 - lr * 0.05, 1e-9);
        EXPECT_NEAR(layer.get_z_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_z_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_z_b_values()[0], 0.5 - lr * 0.05, 1e-9);
        EXPECT_NEAR(layer.get_r_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_r_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_r_b_values()[0], 0.5 - lr * 0.05, 1e-9);
    }
}

TEST_F(GRURNNLayerTest, BPTTRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 0.5 });   layer.set_rw_values({ 0.1 });   layer.set_b_values({ 0.2 });
    layer.set_z_w_values({ 0.5 }); layer.set_z_rw_values({ 0.1 }); layer.set_z_b_values({ 0.2 });
    layer.set_r_w_values({ 0.6 }); layer.set_r_rw_values({ 0.2 }); layer.set_r_b_values({ 0.3 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs }; // prev, this, next
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2, 5); // 2 steps, multiplier 5

    // Forward pass sequence x_0=1, x_1=1
    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0 });
    std::vector<std::vector<double>> batch_next_grads = { { 0.0, 1.0 } }; // t=0: 0.0, t=1: 1.0

    // Test BPTT=1
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 1);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 1);

    EXPECT_NEAR(layer.get_w_grads()[0],   0.41455598, 1e-6);
    EXPECT_NEAR(layer.get_rw_grads()[0],  0.12175109, 1e-6);
    EXPECT_NEAR(layer.get_b_grads()[0],   0.41455598, 1e-6);
    EXPECT_NEAR(layer.get_z_w_grads()[0], 0.04784955, 1e-6);
    EXPECT_NEAR(layer.get_z_rw_grads()[0],0.01932314, 1e-6);
    EXPECT_NEAR(layer.get_z_b_grads()[0], 0.04784955, 1e-6);
    EXPECT_NEAR(layer.get_r_w_grads()[0], 0.00332064, 1e-6);
    EXPECT_NEAR(layer.get_r_rw_grads()[0],0.00134098, 1e-6);
    EXPECT_NEAR(layer.get_r_b_grads()[0], 0.00332064, 1e-6);

    // Test BPTT=2 (Full sequence)
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 2);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 2);

    EXPECT_NEAR(layer.get_w_grads()[0],   0.56661270, 1e-6);
    EXPECT_NEAR(layer.get_rw_grads()[0],  0.12175109, 1e-6);
    EXPECT_NEAR(layer.get_b_grads()[0],   0.56661270, 1e-6);
    EXPECT_NEAR(layer.get_z_w_grads()[0], 0.09588963, 1e-6);
    EXPECT_NEAR(layer.get_z_rw_grads()[0],0.01932314, 1e-6);
    EXPECT_NEAR(layer.get_z_b_grads()[0], 0.09588963, 1e-6);
    EXPECT_NEAR(layer.get_r_w_grads()[0], 0.00332064, 1e-6);
    EXPECT_NEAR(layer.get_r_rw_grads()[0],0.00134098, 1e-6);
    EXPECT_NEAR(layer.get_r_b_grads()[0], 0.00332064, 1e-6);
}

TEST_F(GRURNNLayerTest, ApplyStoredGradientsCacheUpdate)
{
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    
    layer.set_w_values({ 1.0 });   layer.set_rw_values({ 0.5 });
    layer.set_z_w_values({ 0.0 }); layer.set_z_rw_values({ 0.0 });
    layer.set_r_w_values({ 0.0 }); layer.set_r_rw_values({ 0.0 });

    layer.set_z_b_values({ 0.0 });
    layer.set_r_b_values({ 10.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 1 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2, 5); 

    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    auto outputs = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs[0], 0.5, 1e-4);
    EXPECT_NEAR(outputs[1], 0.875, 1e-4);

    layer.set_z_rw_grads({ 0.1 });
    layer.set_r_rw_grads({ 0.1 });
    layer.set_rw_grads({ 0.1 });
    layer.apply_stored_gradients(1.0, 1.0);

    EXPECT_NEAR(layer.get_z_rw_values()[0], -0.1, 1e-9);
    EXPECT_NEAR(layer.get_r_rw_values()[0], -0.1, 1e-9);
    EXPECT_NEAR(layer.get_rw_values()[0], 0.4, 1e-9);

    auto batch_hs2 = create_batch_hidden_states(topology, 1, 2, 5); 
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs2, 1, false);

    auto outputs2 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs2[0], 0.5, 1e-4);
    EXPECT_NEAR(outputs2[1], 0.8412518, 1e-4);
}

TEST_F(GRURNNLayerTest, InputGatesPrecalculationConsistency)
{
    // Test that our pre-calculate input gates optimization matches sequential reference mathematical expectations.
    GRURNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    
    // Set weights and biases to deterministic values
    layer.set_z_w_values({ 0.1, 0.2, 0.3, 0.4 });
    layer.set_z_rw_values({ 0.15, 0.25, 0.35, 0.45 });
    layer.set_z_b_values({ 0.05, 0.15 });

    layer.set_r_w_values({ 0.2, 0.3, 0.4, 0.5 });
    layer.set_r_rw_values({ 0.25, 0.35, 0.45, 0.55 });
    layer.set_r_b_values({ 0.15, 0.25 });

    layer.set_w_values({ 0.3, 0.4, 0.5, 0.6 });
    layer.set_rw_values({ 0.35, 0.45, 0.55, 0.65 });
    layer.set_b_values({ 0.25, 0.35 });

    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 2 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2, 5); 

    // Input sequence: [[1.0, 0.5], [-0.5, 1.0]]
    batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, -0.5, 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& outputs = batch_go[0].get_rnn_outputs(1);
    ASSERT_EQ(outputs.size(), 4);

    // Verify mathematical output values at t = 0
    // x_0 = [1.0, 0.5], prev_h = [0.0, 0.0]
    // z_pre[0] = 1.0 * 0.1 + 0.5 * 0.3 + 0.05 = 0.3
    // z_pre[1] = 1.0 * 0.2 + 0.5 * 0.4 + 0.15 = 0.55
    // z[0] = 1 / (1 + exp(-0.3)) = 0.5744425
    // z[1] = 1 / (1 + exp(-0.55)) = 0.63413559
    // r_pre[0] = 1.0 * 0.2 + 0.5 * 0.4 + 0.15 = 0.55
    // r_pre[1] = 1.0 * 0.3 + 0.5 * 0.5 + 0.25 = 0.8
    // r[0] = 1 / (1 + exp(-0.55)) = 0.63413559
    // r[1] = 1 / (1 + exp(-0.8)) = 0.68997448
    // h_hat_pre[0] = 1.0 * 0.3 + 0.5 * 0.5 + 0.25 = 0.8
    // h_hat_pre[1] = 1.0 * 0.4 + 0.5 * 0.6 + 0.35 = 1.05
    // gated_h = [0.0, 0.0] -> U_h * gated_h = [0.0, 0.0] -> h_hat_pre stays [0.8, 1.05]
    // h_hat_activated = tanh(h_hat_pre) = [tanh(0.8), tanh(1.05)] = [0.6640367, 0.7818055]
    // final h_0[0] = (1 - z[0]) * 0 + z[0] * h_hat_activated[0] = 0.5744425 * 0.6640367 = 0.381451
    // final h_0[1] = (1 - z[1]) * 0 + z[1] * h_hat_activated[1] = 0.6341356 * 0.7818055 = 0.495772

    EXPECT_NEAR(outputs[0], 0.381451, 1e-5);
    EXPECT_NEAR(outputs[1], 0.495772, 1e-5);
}

TEST_F(GRURNNLayerTest, BiasCachingCorrectness)
{
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    
    layer.set_w_values({ 1.0 });   layer.set_rw_values({ 0.0 });
    layer.set_z_w_values({ 0.0 }); layer.set_z_rw_values({ 0.0 });
    layer.set_r_w_values({ 0.0 }); layer.set_r_rw_values({ 0.0 });

    layer.set_z_b_values({ 0.0 });
    layer.set_r_b_values({ 10.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 1 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5);

    batch_go[0].set_rnn_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);
    auto outputs1 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs1[0], 0.5, 1e-4);

    layer.set_b_values({ 10.0, 10.0, 2.0 });

    auto batch_hs2 = create_batch_hidden_states(topology, 1, 1, 5);
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs2, 1, false);
    auto outputs2 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs2[0], 3.0, 1e-3);
}

