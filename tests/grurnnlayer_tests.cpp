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

TEST_F(GRURNNLayerTest, LayerNormForwardNormalizesHiddenState) {
    // 1 input, 2 neurons, single timestep, LayerNorm enabled on h_t.
    GRURNNLayer layer(1, 1, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, true);

    layer.set_z_w_values({ 0.1, 0.2 }); layer.set_z_rw_values({ 0.0, 0.0, 0.0, 0.0 }); layer.set_z_b_values({ 0.0, 0.0 });
    layer.set_r_w_values({ 0.0, 0.0 }); layer.set_r_rw_values({ 0.0, 0.0, 0.0, 0.0 }); layer.set_r_b_values({ 0.0, 0.0 });
    layer.set_w_values({ 0.3, 0.4 });   layer.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });   layer.set_b_values({ 0.0, 0.0 });
    layer.set_ln_h_gain_values({ 2.0, 2.0 });
    layer.set_ln_h_bias_values({ 0.5, -0.5 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 2 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, GRURNNLayer::LayerNormMultiplier);

    batch_go[0].set_outputs(0, { 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    // z = sigmoid([0.1, 0.2]) = [0.524979, 0.549834]
    // h_hat = tanh([0.3, 0.4]) = [0.291313, 0.379949]
    // h_t (raw) = z * h_hat (h_prev=0) = [0.152933, 0.208909]
    // mean = 0.180921, var = 0.000783, inv_std = 1/sqrt(var+1e-5) = 35.501
    // a_hat = [-0.993662, 0.993662]
    // y = gain*a_hat + bias = [2*(-0.993662)+0.5, 2*(0.993662)-0.5] = [-1.487324, 1.487324]
    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], -1.487324, 1e-4);
    EXPECT_NEAR(outputs[1], 1.487324, 1e-4);

    // Every element of a normalized 2-element vector is equidistant from the
    // mean (mean(y) == mean(bias) here since a_hat sums to zero).
    EXPECT_NEAR(outputs[0] + outputs[1], 0.0, 1e-9);
}

TEST_F(GRURNNLayerTest, LayerNormDisabledMatchesUnnormalizedForwardFeed) {
    // Same weights as LayerNormForwardNormalizesHiddenState but with
    // use_layer_norm left at its default (false): output must be the raw
    // (unnormalized) h_t, confirming the flag is a true no-op when unset.
    GRURNNLayer layer(1, 1, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_z_w_values({ 0.1, 0.2 }); layer.set_z_rw_values({ 0.0, 0.0, 0.0, 0.0 }); layer.set_z_b_values({ 0.0, 0.0 });
    layer.set_r_w_values({ 0.0, 0.0 }); layer.set_r_rw_values({ 0.0, 0.0, 0.0, 0.0 }); layer.set_r_b_values({ 0.0, 0.0 });
    layer.set_w_values({ 0.3, 0.4 });   layer.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });   layer.set_b_values({ 0.0, 0.0 });

    EXPECT_FALSE(layer.get_use_layer_norm());

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 2 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, GRURNNLayer::Multiplier);

    batch_go[0].set_outputs(0, { 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], 0.152933, 1e-4);
    EXPECT_NEAR(outputs[1], 0.208909, 1e-4);
}

TEST_F(GRURNNLayerTest, LayerNormGainBiasGradientsMatchNumericalGradient) {
    // Numerical-gradient check of the LayerNorm backward wiring added to
    // calculate_bptt_batch_chunk: seeds an arbitrary upstream gradient dy
    // via batch_next_grads (same direct-injection mechanism as
    // BPTTMathematicalVerification above) and treats
    // loss(gain, bias) = dot(dy, h_t(gain, bias)) as a plain scalar function,
    // independently verified via central finite differences.
    const unsigned num_inputs = 1;
    const unsigned num_outputs = 2;
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

    auto make_layer = [&](const std::vector<double>& gain, const std::vector<double>& bias)
    {
        GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, true);
        layer.set_z_w_values({ 0.15, -0.22 }); layer.set_z_rw_values({ 0.05, -0.03, 0.02, 0.04 }); layer.set_z_b_values({ 0.01, -0.02 });
        layer.set_r_w_values({ -0.18, 0.27 }); layer.set_r_rw_values({ -0.04, 0.06, 0.03, -0.05 }); layer.set_r_b_values({ 0.0, 0.03 });
        layer.set_w_values({ 0.31, 0.44 });    layer.set_rw_values({ 0.07, -0.02, -0.06, 0.08 });   layer.set_b_values({ -0.01, 0.02 });
        layer.set_ln_h_gain_values(gain);
        layer.set_ln_h_bias_values(bias);
        return layer;
    };

    auto run_loss = [&](const std::vector<double>& gain, const std::vector<double>& bias, const std::vector<double>& dy) -> double
    {
        auto layer = make_layer(gain, bias);
        MockLayer prev_layer(0, num_inputs);
        auto batch_go = create_batch_gradients_and_outputs(topology, 1);
        auto batch_hs = create_batch_hidden_states(topology, 1, 1, GRURNNLayer::LayerNormMultiplier);
        batch_go[0].set_outputs(0, { 1.0 });
        layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
        const auto& outputs = batch_go[0].get_outputs(1);
        double loss = 0.0;
        for (unsigned j = 0; j < num_outputs; ++j)
        {
            loss += dy[j] * outputs[j];
        }
        return loss;
    };

    const std::vector<double> gain = { 1.3, 0.8 };
    const std::vector<double> bias = { 0.1, -0.2 };
    const std::vector<double> dy = { 0.6, -0.9 };

    auto layer = make_layer(gain, bias);
    MockLayer prev_layer(0, num_inputs);
    MockLayer next_layer(2, num_outputs);
    // Identity weight matrix so batch_next_grads is projected through
    // unchanged (dh_t = I^T * dy = dy), matching BPTTMathematicalVerification's
    // use of a 1x1 identity above; without this, next_layer's weight vector
    // is empty and the backward GEMV reads out of bounds.
    {
      std::vector<double> identity(num_outputs * num_outputs, 0.0);
      for (unsigned j = 0; j < num_outputs; ++j)
      {
        identity[j * num_outputs + j] = 1.0;
      }
      next_layer.set_w_values(identity);
    }
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, GRURNNLayer::LayerNormMultiplier);
    batch_go[0].set_outputs(0, { 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    std::vector<std::vector<double>> batch_next_grads = { dy };
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

    const auto& gain_grads = layer.get_ln_h_gain_grads();
    const auto& bias_grads = layer.get_ln_h_bias_grads();
    ASSERT_EQ(gain_grads.size(), num_outputs);
    ASSERT_EQ(bias_grads.size(), num_outputs);

    const double h = 1e-6;
    for (unsigned j = 0; j < num_outputs; ++j)
    {
        std::vector<double> gain_plus = gain; gain_plus[j] += h;
        std::vector<double> gain_minus = gain; gain_minus[j] -= h;
        const double numerical_gain_grad = (run_loss(gain_plus, bias, dy) - run_loss(gain_minus, bias, dy)) / (2.0 * h);
        EXPECT_NEAR(gain_grads[j], numerical_gain_grad, 1e-4) << "gain[" << j << "]";

        std::vector<double> bias_plus = bias; bias_plus[j] += h;
        std::vector<double> bias_minus = bias; bias_minus[j] -= h;
        const double numerical_bias_grad = (run_loss(gain, bias_plus, dy) - run_loss(gain, bias_minus, dy)) / (2.0 * h);
        EXPECT_NEAR(bias_grads[j], numerical_bias_grad, 1e-4) << "bias[" << j << "]";
    }
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
    unsigned num_outputs = 5000;
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
    EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.08);
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

TEST_F(GRURNNLayerTest, StateAndMemoryAllocationOptimizationVerification)
{
    // A 2-input, 2-neuron GRU with 2 batches and 3 time steps
    GRURNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

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
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 3, 5); 

    // Batch 0: [[1.0, 0.5], [-0.5, 1.0], [0.0, 0.0]]
    // Batch 1: [[0.5, -0.5], [1.0, 1.0], [-1.0, 0.5]]
    batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, -0.5, 1.0, 0.0, 0.0 });
    batch_go[1].set_rnn_outputs(0, { 0.5, -0.5, 1.0, 1.0, -1.0, 0.5 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, false);

    // Verify outputs size and correct state retention across batches/time-steps
    const auto& outputs_0 = batch_go[0].get_rnn_outputs(1);
    const auto& outputs_1 = batch_go[1].get_rnn_outputs(1);

    ASSERT_EQ(outputs_0.size(), 6);
    ASSERT_EQ(outputs_1.size(), 6);

    // For Batch 0, t=0, the outputs should match the single batch results exactly
    EXPECT_NEAR(outputs_0[0], 0.381451, 1e-5);
    EXPECT_NEAR(outputs_0[1], 0.495772, 1e-5);

    // Verify we have non-zero results that propagate correctly
    EXPECT_NE(outputs_0[2], 0.0);
    EXPECT_NE(outputs_0[3], 0.0);
    EXPECT_NE(outputs_1[4], 0.0);
    EXPECT_NE(outputs_1[5], 0.0);
}

TEST_F(GRURNNLayerTest, TransposedWeightsAndFastBpttPassCorrectness) {
    // 2 inputs, 2 neurons, batch size 2, 2 time steps
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    // Populate weights
    layer.set_z_w_values({ 0.1, 0.2, 0.3, 0.4 });
    layer.set_z_rw_values({ 0.15, 0.25, 0.35, 0.45 });
    layer.set_z_b_values({ 0.05, 0.15 });

    layer.set_r_w_values({ 0.2, 0.3, 0.4, 0.5 });
    layer.set_r_rw_values({ 0.25, 0.35, 0.45, 0.55 });
    layer.set_r_b_values({ 0.15, 0.25 });

    layer.set_w_values({ 0.3, 0.4, 0.5, 0.6 });
    layer.set_rw_values({ 0.35, 0.45, 0.55, 0.65 });
    layer.set_b_values({ 0.25, 0.35 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2); // batch size 2
    auto batch_hs = create_batch_hidden_states(topology, 2, 2, GRURNNLayer::Multiplier); // 2 steps

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

    // Backward pass (BPTT = 2)
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 2, 2);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 2, 2);

    // Verify gradients are non-zero and accumulated successfully
    EXPECT_GT(std::abs(layer.get_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_rw_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_z_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_z_rw_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_r_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_r_rw_grads()[0]), 0.0);
}

TEST_F(GRURNNLayerTest, BPTTCorrectnessAfterFillOptimization)
{
  // Verify that the std::fill optimization for temp_Uh_T_dh_hat maintains BPTT correctness.
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

  layer.set_z_w_values({ 0.15, 0.25, 0.35, 0.45 });
  layer.set_z_rw_values({ 0.1, 0.2, 0.3, 0.4 });
  layer.set_z_b_values({ 0.05, 0.15 });

  layer.set_r_w_values({ 0.25, 0.35, 0.45, 0.55 });
  layer.set_r_rw_values({ 0.2, 0.3, 0.4, 0.5 });
  layer.set_r_b_values({ 0.15, 0.25 });

  layer.set_w_values({ 0.35, 0.45, 0.55, 0.65 });
  layer.set_rw_values({ 0.3, 0.4, 0.5, 0.6 });
  layer.set_b_values({ 0.25, 0.35 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs = create_batch_hidden_states(topology, 2, 2, GRURNNLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, 0.8, -0.2 });
  batch_go[1].set_rnn_outputs(0, { 0.5, 0.8, -0.2, 0.4 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, true);

  MockLayer next_layer(2, num_outputs);
  next_layer.set_w_values({ 0.5, 0.8, 0.1, 0.9 });
  std::vector<std::vector<double>> batch_next_grads = {
    { 0.1, 0.2, 0.15, 0.25 },
    { 0.2, 0.3, 0.25, 0.35 }
  };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 2, 2);
  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 2, 2);

  // Verify gradients are non-zero
  EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
}

TEST_F(GRURNNLayerTest, BPTTWorkspaceResizeCorrectness)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

  layer.set_z_w_values({ 0.15, 0.25, 0.35, 0.45 });
  layer.set_z_rw_values({ 0.1, 0.2, 0.3, 0.4 });
  layer.set_z_b_values({ 0.05, 0.15 });

  layer.set_r_w_values({ 0.25, 0.35, 0.45, 0.55 });
  layer.set_r_rw_values({ 0.2, 0.3, 0.4, 0.5 });
  layer.set_r_b_values({ 0.15, 0.25 });

  layer.set_w_values({ 0.35, 0.45, 0.55, 0.65 });
  layer.set_rw_values({ 0.3, 0.4, 0.5, 0.6 });
  layer.set_b_values({ 0.25, 0.35 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  // Run backprop first time (creates workspace initially)
  auto batch_go1 = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs1 = create_batch_hidden_states(topology, 2, 2, GRURNNLayer::Multiplier);
  batch_go1[0].set_rnn_outputs(0, { 1.0, 0.5, 0.8, -0.2 });
  batch_go1[1].set_rnn_outputs(0, { 0.5, 0.8, -0.2, 0.4 });
  layer.calculate_forward_feed(batch_go1, prev_layer, {}, batch_hs1, 2, true);

  MockLayer next_layer(2, num_outputs);
  next_layer.set_w_values({ 0.5, 0.8, 0.1, 0.9 });
  std::vector<std::vector<double>> batch_next_grads = {
    { 0.1, 0.2, 0.15, 0.25 },
    { 0.2, 0.3, 0.25, 0.35 }
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
  auto batch_hs2 = create_batch_hidden_states(topology, 1, 2, GRURNNLayer::Multiplier);
  batch_go2[0].set_rnn_outputs(0, { 1.0, 0.5, 0.8, -0.2 });
  layer.calculate_forward_feed(batch_go2, prev_layer, {}, batch_hs2, 1, true);

  std::vector<std::vector<double>> batch_next_grads2 = { { 0.1, 0.2, 0.15, 0.25 } };
  layer.calculate_hidden_gradients(batch_go2, next_layer, batch_next_grads2, batch_hs2, 1, 2);
  layer.calculate_and_store_gradients(batch_go2, batch_hs2, prev_layer, 1, 2);
  EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
}

TEST_F(GRURNNLayerTest, SingleVSMultiThreadedEquivalence)
{
  unsigned num_inputs = 100;
  unsigned num_outputs = 100;
  size_t batch_size = 100;
  size_t num_time_steps = 20;

  // Layer 1: single threaded
  GRURNNLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

  // Layer 2: multi threaded
  GRURNNLayer layer_mt(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 4, true, 0.0);

  // Helper to fill vectors with identical values
  auto initialize_weights = [&](GRURNNLayer& l)
  {
    l.set_w_values(std::vector<double>(num_inputs * num_outputs, 0.05));
    l.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.08));
    l.set_b_values(std::vector<double>(num_outputs, 0.01));

    l.set_z_w_values(std::vector<double>(num_inputs * num_outputs, 0.06));
    l.set_z_rw_values(std::vector<double>(num_outputs * num_outputs, 0.09));
    l.set_z_b_values(std::vector<double>(num_outputs, 0.02));

    l.set_r_w_values(std::vector<double>(num_inputs * num_outputs, 0.07));
    l.set_r_rw_values(std::vector<double>(num_outputs * num_outputs, 0.10));
    l.set_r_b_values(std::vector<double>(num_outputs, 0.03));
  };

  initialize_weights(layer_st);
  initialize_weights(layer_mt);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  // Setup batch inputs and next gradients
  auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);
  auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

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

  assert_vectors_equal(layer_st.get_z_w_grads(), layer_mt.get_z_w_grads());
  assert_vectors_equal(layer_st.get_z_rw_grads(), layer_mt.get_z_rw_grads());
  assert_vectors_equal(layer_st.get_z_b_grads(), layer_mt.get_z_b_grads());

  assert_vectors_equal(layer_st.get_r_w_grads(), layer_mt.get_r_w_grads());
  assert_vectors_equal(layer_st.get_r_rw_grads(), layer_mt.get_r_rw_grads());
  assert_vectors_equal(layer_st.get_r_b_grads(), layer_mt.get_r_b_grads());
}

TEST_F(GRURNNLayerTest, BPTTMultiStepBatchVerification)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  size_t batch_size = 5;
  size_t num_time_steps = 3;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

  layer.set_z_w_values({ 0.15, 0.25, 0.35, 0.45 });
  layer.set_z_rw_values({ 0.1, 0.2, 0.3, 0.4 });
  layer.set_z_b_values({ 0.05, 0.15 });

  layer.set_r_w_values({ 0.25, 0.35, 0.45, 0.55 });
  layer.set_r_rw_values({ 0.2, 0.3, 0.4, 0.5 });
  layer.set_r_b_values({ 0.15, 0.25 });

  layer.set_w_values({ 0.35, 0.45, 0.55, 0.65 });
  layer.set_rw_values({ 0.3, 0.4, 0.5, 0.6 });
  layer.set_b_values({ 0.25, 0.35 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  std::vector<double> inputs = { 0.5, -0.5, 0.2, -0.2, 0.1, -0.1 };
  for (size_t b = 0; b < batch_size; ++b)
  {
    batch_go[b].set_rnn_outputs(0, inputs);
  }

  // Forward feed
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

  MockLayer next_layer(2, num_outputs);
  next_layer.set_w_values({ 1.0, 0.5, 0.2, 0.8 });

  std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_time_steps * num_outputs, 0.1));

  // Backward feed
  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, batch_size, static_cast<int>(num_time_steps));

  // Store gradients
  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, static_cast<int>(num_time_steps));

  // Verify gradients computed are reasonable numbers (non-zero and finite)
  EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
  for (const double w : layer.get_w_grads())
  {
    EXPECT_TRUE(std::isfinite(w));
  }
}

TEST_F(GRURNNLayerTest, BPTTSequenceLengthsVerification)
{
  unsigned num_inputs = 3;
  unsigned num_outputs = 4;
  size_t batch_size = 3;

  for (size_t num_time_steps = 1; num_time_steps <= 12; ++num_time_steps)
  {
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_z_w_values(std::vector<double>(num_inputs * num_outputs, 0.1));
    layer.set_z_rw_values(std::vector<double>(num_outputs * num_outputs, 0.15));
    layer.set_z_b_values(std::vector<double>(num_outputs, 0.05));

    layer.set_r_w_values(std::vector<double>(num_inputs * num_outputs, 0.2));
    layer.set_r_rw_values(std::vector<double>(num_outputs * num_outputs, 0.25));
    layer.set_r_b_values(std::vector<double>(num_outputs, 0.15));

    layer.set_w_values(std::vector<double>(num_inputs * num_outputs, 0.3));
    layer.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.35));
    layer.set_b_values(std::vector<double>(num_outputs, 0.25));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

    auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

    std::vector<double> inputs(num_time_steps * num_inputs, 0.5);
    for (size_t b = 0; b < batch_size; ++b)
    {
      batch_go[b].set_rnn_outputs(0, inputs);
    }

    // Forward feed
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

    MockLayer next_layer(2, num_outputs);
    std::vector<double> next_w_vals(num_outputs * num_outputs, 0.3);
    next_layer.set_w_values(next_w_vals);

    std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_time_steps * num_outputs, 0.15));

    // Backward feed
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, batch_size, static_cast<int>(num_time_steps));

    // Store gradients
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, static_cast<int>(num_time_steps));

    // Verify gradients computed are reasonable numbers (non-zero and finite)
    EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
    for (const double w : layer.get_w_grads())
    {
      EXPECT_TRUE(std::isfinite(w));
    }
    for (const double rw : layer.get_rw_grads())
    {
      EXPECT_TRUE(std::isfinite(rw));
    }
  }
}

TEST_F(GRURNNLayerTest, TempBufferReuseAndMultiIterationConsistency) {
  GRURNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::Adam, -1, 0.0, nullptr, 1, true, 0.0);

  MockLayer prev_layer(0, 2);
  std::vector<unsigned> topology = { 2, 2 };

  std::vector<double> first_pass_outputs;
  std::vector<double> second_pass_outputs;

  for (int iter = 0; iter < 2; ++iter)
  {
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 3);

    batch_go[0].set_rnn_outputs(0, { 0.5, 0.2, -0.1, 0.8, 0.3, 0.4 });
    batch_go[1].set_rnn_outputs(0, { 0.1, -0.4, 0.6, 0.2, -0.5, 0.1 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, true);

    const auto rnn_out_0 = batch_go[0].get_rnn_outputs(1);
    const auto rnn_out_1 = batch_go[1].get_rnn_outputs(1);

    if (iter == 0)
    {
      first_pass_outputs = rnn_out_0;
      first_pass_outputs.insert(first_pass_outputs.end(), rnn_out_1.begin(), rnn_out_1.end());
    }
    else
    {
      second_pass_outputs = rnn_out_0;
      second_pass_outputs.insert(second_pass_outputs.end(), rnn_out_1.begin(), rnn_out_1.end());
    }
  }

  ASSERT_EQ(first_pass_outputs.size(), second_pass_outputs.size());
  for (size_t i = 0; i < first_pass_outputs.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(first_pass_outputs[i], second_pass_outputs[i]);
  }
}

TEST_F(GRURNNLayerTest, GRURNNLayerCalculateAndStoreGradientsMathematicalSoundness) {
  const unsigned num_inputs = 3;
  const unsigned num_outputs = 3;
  const size_t batch_size = 4;
  const size_t num_time_steps = 3;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

  std::vector<unsigned> topology = { num_inputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, 5);

  std::vector<std::vector<double>> inputs_data(batch_size * num_time_steps, std::vector<double>(num_inputs, 0.0));
  std::vector<std::vector<double>> gh_data(batch_size * num_time_steps, std::vector<double>(num_outputs, 0.0));
  std::vector<std::vector<double>> gz_data(batch_size * num_time_steps, std::vector<double>(num_outputs, 0.0));
  std::vector<std::vector<double>> gr_data(batch_size * num_time_steps, std::vector<double>(num_outputs, 0.0));
  std::vector<std::vector<double>> prev_h_data(batch_size * num_time_steps, std::vector<double>(num_outputs, 0.0));
  std::vector<std::vector<double>> r_vals_data(batch_size * num_time_steps, std::vector<double>(num_outputs, 0.0));

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> rnn_inputs(num_time_steps * num_inputs);
    std::vector<double> gate_grads(num_time_steps * 3 * num_outputs);

    auto& layer_states = batch_hs[b].at(1);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const size_t idx = b * num_time_steps + t;
      for (size_t k = 0; k < num_inputs; ++k)
      {
        const double x_val = static_cast<double>(idx * 5 + k + 1) * 0.1;
        inputs_data[idx][k] = x_val;
        rnn_inputs[t * num_inputs + k] = x_val;
      }

      const size_t base_idx = t * 3 * num_outputs;
      for (size_t j = 0; j < num_outputs; ++j)
      {
        const double gh_val = static_cast<double>(idx * 3 + j + 1) * 0.04;
        const double gz_val = static_cast<double>(idx * 4 + j + 2) * 0.03;
        const double gr_val = static_cast<double>(idx * 2 + j + 3) * 0.05;

        gh_data[idx][j] = gh_val;
        gz_data[idx][j] = gz_val;
        gr_data[idx][j] = gr_val;

        gate_grads[base_idx + j] = gh_val;
        gate_grads[base_idx + num_outputs + j] = gz_val;
        gate_grads[base_idx + 2 * num_outputs + j] = gr_val;
      }

      std::vector<double> h_state(num_outputs);
      std::vector<double> packed_sums(5 * num_outputs, 0.0);
      for (size_t rk = 0; rk < num_outputs; ++rk)
      {
        h_state[rk] = static_cast<double>(idx * 2 + rk + 1) * 0.15;
        const double r_val = static_cast<double>(idx + rk + 1) * 0.2;
        r_vals_data[idx][rk] = r_val;
        packed_sums[num_outputs + rk] = r_val;
      }
      layer_states[t].set_hidden_state_values(h_state.data(), num_outputs);
      layer_states[t].set_pre_activation_sums(packed_sums.data(), packed_sums.size());

      if (t > 0)
      {
        const auto& prev_h = layer_states[t - 1].get_hidden_state_values();
        prev_h_data[idx].assign(prev_h.begin(), prev_h.end());
      }
    }
    batch_go[b].set_rnn_outputs(0, rnn_inputs);
    batch_go[b].set_rnn_gate_gradients(1, gate_grads);
  }

  MockLayer prev_layer(0, num_inputs);
  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, 0);

  std::vector<double> expected_w_grads(num_inputs * num_outputs, 0.0);
  std::vector<double> expected_z_w_grads(num_inputs * num_outputs, 0.0);
  std::vector<double> expected_r_w_grads(num_inputs * num_outputs, 0.0);

  std::vector<double> expected_rw_grads(num_outputs * num_outputs, 0.0);
  std::vector<double> expected_z_rw_grads(num_outputs * num_outputs, 0.0);
  std::vector<double> expected_r_rw_grads(num_outputs * num_outputs, 0.0);

  std::vector<double> expected_b_grads(num_outputs, 0.0);
  std::vector<double> expected_z_b_grads(num_outputs, 0.0);
  std::vector<double> expected_r_b_grads(num_outputs, 0.0);

  for (size_t b = 0; b < batch_size; ++b)
  {
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const size_t idx = b * num_time_steps + t;
      for (size_t i = 0; i < num_inputs; ++i)
      {
        for (size_t j = 0; j < num_outputs; ++j)
        {
          const double x_val = inputs_data[idx][i];
          expected_w_grads[i * num_outputs + j] += x_val * gh_data[idx][j];
          expected_z_w_grads[i * num_outputs + j] += x_val * gz_data[idx][j];
          expected_r_w_grads[i * num_outputs + j] += x_val * gr_data[idx][j];
        }
      }
      if (t > 0)
      {
        for (size_t k = 0; k < num_outputs; ++k)
        {
          const double hp = prev_h_data[idx][k];
          const double rv = r_vals_data[idx][k];
          for (size_t j = 0; j < num_outputs; ++j)
          {
            expected_rw_grads[k * num_outputs + j] += (rv * hp) * gh_data[idx][j];
            expected_z_rw_grads[k * num_outputs + j] += hp * gz_data[idx][j];
            expected_r_rw_grads[k * num_outputs + j] += hp * gr_data[idx][j];
          }
        }
      }
      for (size_t j = 0; j < num_outputs; ++j)
      {
        expected_b_grads[j] += gh_data[idx][j];
        expected_z_b_grads[j] += gz_data[idx][j];
        expected_r_b_grads[j] += gr_data[idx][j];
      }
    }
  }

  const double inv_batch = 1.0 / static_cast<double>(batch_size);
  for (size_t m = 0; m < expected_w_grads.size(); ++m)
  {
    expected_w_grads[m] *= inv_batch;
    expected_z_w_grads[m] *= inv_batch;
    expected_r_w_grads[m] *= inv_batch;

    expected_rw_grads[m] *= inv_batch;
    expected_z_rw_grads[m] *= inv_batch;
    expected_r_rw_grads[m] *= inv_batch;
  }
  for (size_t j = 0; j < num_outputs; ++j)
  {
    expected_b_grads[j] *= inv_batch;
    expected_z_b_grads[j] *= inv_batch;
    expected_r_b_grads[j] *= inv_batch;
  }

  const auto& actual_w_grads = layer.get_w_grads();
  const auto& actual_z_w_grads = layer.get_z_w_grads();
  const auto& actual_r_w_grads = layer.get_r_w_grads();

  const auto& actual_rw_grads = layer.get_rw_grads();
  const auto& actual_z_rw_grads = layer.get_z_rw_grads();
  const auto& actual_r_rw_grads = layer.get_r_rw_grads();

  const auto& actual_b_grads = layer.get_b_grads();
  const auto& actual_z_b_grads = layer.get_z_b_grads();
  const auto& actual_r_b_grads = layer.get_r_b_grads();

  for (size_t m = 0; m < expected_w_grads.size(); ++m)
  {
    EXPECT_NEAR(actual_w_grads[m], expected_w_grads[m], 1e-14);
    EXPECT_NEAR(actual_z_w_grads[m], expected_z_w_grads[m], 1e-14);
    EXPECT_NEAR(actual_r_w_grads[m], expected_r_w_grads[m], 1e-14);

    EXPECT_NEAR(actual_rw_grads[m], expected_rw_grads[m], 1e-14);
    EXPECT_NEAR(actual_z_rw_grads[m], expected_z_rw_grads[m], 1e-14);
    EXPECT_NEAR(actual_r_rw_grads[m], expected_r_rw_grads[m], 1e-14);
  }

  for (size_t j = 0; j < num_outputs; ++j)
  {
    EXPECT_NEAR(actual_b_grads[j], expected_b_grads[j], 1e-14);
    EXPECT_NEAR(actual_z_b_grads[j], expected_z_b_grads[j], 1e-14);
    EXPECT_NEAR(actual_r_b_grads[j], expected_r_b_grads[j], 1e-14);
  }
}

// ---------------------------------------------------------------------------
// Batched recurrent forward-pass regression tests.
//
// run_forward_pass batches the recurrent (hidden-to-hidden) GEMV across up to
// 4 batch items per timestep instead of processing one batch item at a time.
// These tests verify that batching a batch item together with other,
// different batch items never changes that item's own result (no cross-talk
// between batch items sharing a group), and that every 4-wide/2-wide/1-wide
// cleanup path is exercised at least once.
// ---------------------------------------------------------------------------
namespace {

  std::vector<double> make_deterministic_weights(size_t rows, size_t cols, double scale, double offset)
  {
    std::vector<double> w(rows * cols);
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        w[i * cols + j] = offset + scale * std::sin(static_cast<double>(i * 7 + j * 3 + 1));
      }
    }
    return w;
  }

  GRURNNLayer make_cross_talk_test_layer(unsigned num_inputs, unsigned num_outputs, bool use_layer_norm = false)
  {
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, use_layer_norm);
    layer.set_z_w_values(make_deterministic_weights(num_inputs, num_outputs, 0.15, 0.02));
    layer.set_z_rw_values(make_deterministic_weights(num_outputs, num_outputs, 0.12, -0.01));
    layer.set_z_b_values(make_deterministic_weights(1, num_outputs, 0.05, 0.0));
    layer.set_r_w_values(make_deterministic_weights(num_inputs, num_outputs, -0.13, 0.03));
    layer.set_r_rw_values(make_deterministic_weights(num_outputs, num_outputs, 0.10, 0.02));
    layer.set_r_b_values(make_deterministic_weights(1, num_outputs, -0.04, 0.01));
    layer.set_w_values(make_deterministic_weights(num_inputs, num_outputs, 0.18, -0.02));
    layer.set_rw_values(make_deterministic_weights(num_outputs, num_outputs, -0.09, 0.04));
    layer.set_b_values(make_deterministic_weights(1, num_outputs, 0.06, -0.01));
    if (use_layer_norm)
    {
      layer.set_ln_h_gain_values(make_deterministic_weights(1, num_outputs, 1.2, 0.05));
      layer.set_ln_h_bias_values(make_deterministic_weights(1, num_outputs, -0.08, 0.02));
    }
    return layer;
  }

  std::vector<double> make_cross_talk_sequence(double base, size_t num_time_steps, size_t num_inputs)
  {
    std::vector<double> seq(num_time_steps * num_inputs);
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      for (size_t i = 0; i < num_inputs; ++i)
      {
        seq[t * num_inputs + i] = base + 0.13 * static_cast<double>(t) - 0.07 * static_cast<double>(i);
      }
    }
    return seq;
  }

  // Runs one distinctive input sequence ("X") both alone (batch_size=1) and
  // batched at position x_index among batch_size-1 other, different sequences,
  // then asserts X's stored pre-activation sums, hidden state values (every
  // timestep) and final rnn_outputs are unaffected by which other batch items
  // happened to share its 4-wide/2-wide/1-wide group.
  void assert_no_batch_cross_talk(unsigned num_inputs, unsigned num_outputs, size_t batch_size, size_t num_time_steps, size_t x_index, bool is_training, bool use_layer_norm = false)
  {
    ASSERT_LT(x_index, batch_size);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    const auto x_seq = make_cross_talk_sequence(0.37, num_time_steps, num_inputs);
    const unsigned multiplier = use_layer_norm ? GRURNNLayer::LayerNormMultiplier : GRURNNLayer::Multiplier;

    GRURNNLayer layer_alone = make_cross_talk_test_layer(num_inputs, num_outputs, use_layer_norm);
    MockLayer prev_layer_alone(0, num_inputs);
    auto batch_go_alone = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs_alone = create_batch_hidden_states(topology, 1, num_time_steps, multiplier);
    batch_go_alone[0].set_rnn_outputs(0, x_seq);
    layer_alone.calculate_forward_feed(batch_go_alone, prev_layer_alone, {}, batch_hs_alone, 1, is_training);

    GRURNNLayer layer_batched = make_cross_talk_test_layer(num_inputs, num_outputs, use_layer_norm);
    MockLayer prev_layer_batched(0, num_inputs);
    auto batch_go_batched = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_batched = create_batch_hidden_states(topology, batch_size, num_time_steps, multiplier);
    for (size_t b = 0; b < batch_size; ++b)
    {
      if (b == x_index)
      {
        batch_go_batched[b].set_rnn_outputs(0, x_seq);
      }
      else
      {
        const double base = -0.6 + 0.23 * static_cast<double>(b);
        batch_go_batched[b].set_rnn_outputs(0, make_cross_talk_sequence(base, num_time_steps, num_inputs));
      }
    }
    layer_batched.calculate_forward_feed(batch_go_batched, prev_layer_batched, {}, batch_hs_batched, batch_size, is_training);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const auto pre_alone = batch_hs_alone[0].at(1, t).get_pre_activation_sums();
      const auto pre_batched = batch_hs_batched[x_index].at(1, t).get_pre_activation_sums();
      ASSERT_EQ(pre_alone.size(), pre_batched.size());
      for (size_t i = 0; i < pre_alone.size(); ++i)
      {
        EXPECT_NEAR(pre_alone[i], pre_batched[i], 1e-9) << "t=" << t << " i=" << i;
      }

      const auto hidden_alone = batch_hs_alone[0].at(1, t).get_hidden_state_values();
      const auto hidden_batched = batch_hs_batched[x_index].at(1, t).get_hidden_state_values();
      ASSERT_EQ(hidden_alone.size(), hidden_batched.size());
      for (size_t i = 0; i < hidden_alone.size(); ++i)
      {
        EXPECT_NEAR(hidden_alone[i], hidden_batched[i], 1e-9) << "t=" << t << " i=" << i;
      }
    }

    const auto rnn_out_alone = batch_go_alone[0].get_rnn_outputs(1);
    const auto rnn_out_batched = batch_go_batched[x_index].get_rnn_outputs(1);
    ASSERT_EQ(rnn_out_alone.size(), rnn_out_batched.size());
    for (size_t i = 0; i < rnn_out_alone.size(); ++i)
    {
      EXPECT_NEAR(rnn_out_alone[i], rnn_out_batched[i], 1e-9) << "i=" << i;
    }
  }

} // namespace

TEST_F(GRURNNLayerTest, NoBatchCrossTalkFourWideGroupInference)
{
  // batch_size=7 -> groups of 4, 2, 1. X at index 3 is the last slot of the 4-wide group.
  assert_no_batch_cross_talk(2, 3, 7, 3, 3, false);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkOneWideCleanupInference)
{
  // batch_size=7 -> groups of 4, 2, 1. X at index 6 is the 1-wide cleanup item.
  assert_no_batch_cross_talk(2, 3, 7, 3, 6, false);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkFourWideGroupTraining)
{
  // Same as above but is_training=true (dropout=0.0, so still deterministic):
  // exercises the activation()'s training-mode code path without RNG noise.
  assert_no_batch_cross_talk(2, 3, 7, 3, 3, true);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkOneWideCleanupTraining)
{
  assert_no_batch_cross_talk(2, 3, 7, 3, 6, true);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkExactFourMultiple)
{
  // batch_size=4: exactly one 4-wide group, no cleanup at all.
  assert_no_batch_cross_talk(2, 3, 4, 2, 0, false);
  assert_no_batch_cross_talk(2, 3, 4, 2, 3, false);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkOneWideCleanupRemainder)
{
  // batch_size=5 -> 4 + 1: X in the 1-wide cleanup.
  assert_no_batch_cross_talk(2, 3, 5, 2, 4, false);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkTwoWideCleanupRemainder)
{
  // batch_size=6 -> 4 + 2: X in the 2-wide cleanup.
  assert_no_batch_cross_talk(2, 3, 6, 2, 5, false);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkTwoFullFourWideGroups)
{
  // batch_size=8 -> 4 + 4: X at the start of the second 4-wide group.
  assert_no_batch_cross_talk(2, 3, 8, 2, 4, false);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkLargerHiddenSize)
{
  // N_this=10 crosses gemm_four_batches' internal 8-wide/4-wide/scalar-tail
  // AVX2 boundaries; batch_size=9 -> 4 + 4 + 1.
  assert_no_batch_cross_talk(4, 10, 9, 3, 4, false);
  assert_no_batch_cross_talk(4, 10, 9, 3, 8, false);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkLayerNormFourWideGroup)
{
  // Same 4-wide-group layout as NoBatchCrossTalkFourWideGroupInference, but
  // with use_layer_norm enabled: verifies each batch item's LayerNorm
  // statistics (mean/inv_std, cached per-item) are computed independently
  // and are not contaminated by its 4-wide SIMD group neighbors.
  assert_no_batch_cross_talk(2, 3, 7, 3, 3, false, true);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkLayerNormOneWideCleanup)
{
  assert_no_batch_cross_talk(2, 3, 5, 2, 4, false, true);
}

TEST_F(GRURNNLayerTest, NoBatchCrossTalkLayerNormTraining)
{
  assert_no_batch_cross_talk(2, 3, 7, 3, 3, true, true);
}





