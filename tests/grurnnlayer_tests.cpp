#include <gtest/gtest.h>
#include "layers/grurnnlayer.h"
#include "layers/fflayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>


using namespace myoddweb::nn;
using namespace test_helper;

class GRURNNLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(GRURNNLayerTest, Construction) {
    GRURNNLayer layer(1, 2, 3, 0.01, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::Adam, -1, 0.0, nullptr, 1, true, 0.9, false, std::nullopt);
    EXPECT_EQ(layer.get_layer_index(), 1);
    EXPECT_EQ(layer.get_number_input_neurons(), 2);
    EXPECT_EQ(layer.get_number_neurons(), 3);
    EXPECT_EQ(layer.get_pre_activation_multiplier(), 5);
}

TEST_F(GRURNNLayerTest, ForwardFeedMathematicalVerification) {
    // 1 input, 1 neuron GRU
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    
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
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    
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
    GRURNNLayer layer(1, 1, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, true, std::nullopt);

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
    // use_layer_normalisation left at its default (false): output must be the raw
    // (unnormalized) h_t, confirming the flag is a true no-op when unset.
    GRURNNLayer layer(1, 1, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    layer.set_z_w_values({ 0.1, 0.2 }); layer.set_z_rw_values({ 0.0, 0.0, 0.0, 0.0 }); layer.set_z_b_values({ 0.0, 0.0 });
    layer.set_r_w_values({ 0.0, 0.0 }); layer.set_r_rw_values({ 0.0, 0.0, 0.0, 0.0 }); layer.set_r_b_values({ 0.0, 0.0 });
    layer.set_w_values({ 0.3, 0.4 });   layer.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });   layer.set_b_values({ 0.0, 0.0 });

    EXPECT_FALSE(layer.get_use_layer_normalisation());

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
        GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, true, std::nullopt);
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
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.5, nullptr, 1, true, 0.0, false, std::nullopt);
    
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
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    
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
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, false, std::nullopt);

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
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, false, std::nullopt);

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
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    
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
    GRURNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    
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
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    
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
    GRURNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
  GRURNNLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  // Layer 2: multi threaded
  GRURNNLayer layer_mt(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 4, true, 0.0, false, std::nullopt);

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

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
  GRURNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::Adam, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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

  GRURNNLayer make_cross_talk_test_layer(unsigned num_inputs, unsigned num_outputs, bool use_layer_normalisation = false)
  {
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, use_layer_normalisation, std::nullopt);
    layer.set_z_w_values(make_deterministic_weights(num_inputs, num_outputs, 0.15, 0.02));
    layer.set_z_rw_values(make_deterministic_weights(num_outputs, num_outputs, 0.12, -0.01));
    layer.set_z_b_values(make_deterministic_weights(1, num_outputs, 0.05, 0.0));
    layer.set_r_w_values(make_deterministic_weights(num_inputs, num_outputs, -0.13, 0.03));
    layer.set_r_rw_values(make_deterministic_weights(num_outputs, num_outputs, 0.10, 0.02));
    layer.set_r_b_values(make_deterministic_weights(1, num_outputs, -0.04, 0.01));
    layer.set_w_values(make_deterministic_weights(num_inputs, num_outputs, 0.18, -0.02));
    layer.set_rw_values(make_deterministic_weights(num_outputs, num_outputs, -0.09, 0.04));
    layer.set_b_values(make_deterministic_weights(1, num_outputs, 0.06, -0.01));
    if (use_layer_normalisation)
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
  void assert_no_batch_cross_talk(unsigned num_inputs, unsigned num_outputs, size_t batch_size, size_t num_time_steps, size_t x_index, bool is_training, bool use_layer_normalisation = false)
  {
    ASSERT_LT(x_index, batch_size);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    const auto x_seq = make_cross_talk_sequence(0.37, num_time_steps, num_inputs);
    const unsigned multiplier = use_layer_normalisation ? GRURNNLayer::LayerNormMultiplier : GRURNNLayer::Multiplier;

    GRURNNLayer layer_alone = make_cross_talk_test_layer(num_inputs, num_outputs, use_layer_normalisation);
    MockLayer prev_layer_alone(0, num_inputs);
    auto batch_go_alone = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs_alone = create_batch_hidden_states(topology, 1, num_time_steps, multiplier);
    batch_go_alone[0].set_rnn_outputs(0, x_seq);
    layer_alone.calculate_forward_feed(batch_go_alone, prev_layer_alone, {}, batch_hs_alone, 1, is_training);

    GRURNNLayer layer_batched = make_cross_talk_test_layer(num_inputs, num_outputs, use_layer_normalisation);
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
  // with use_layer_normalisation enabled: verifies each batch item's LayerNorm
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

TEST_F(GRURNNLayerTest, CalculateAndStoreGradientsVariousTopologiesMathematicalProof)
{
  // 7 inputs, 5 neurons (crosses 4-wide, 2-wide, 1-wide SIMD lanes)
  const unsigned num_inputs = 7;
  const unsigned num_outputs = 5;
  const size_t batch_size = 3;
  const size_t num_time_steps = 4;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

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
        const double x_val = static_cast<double>(idx * 7 + k + 1) * 0.05;
        inputs_data[idx][k] = x_val;
        rnn_inputs[t * num_inputs + k] = x_val;
      }

      const size_t base_idx = t * 3 * num_outputs;
      for (size_t j = 0; j < num_outputs; ++j)
      {
        const double gh_val = static_cast<double>(idx * 5 + j + 1) * 0.03;
        const double gz_val = static_cast<double>(idx * 3 + j + 2) * 0.02;
        const double gr_val = static_cast<double>(idx * 4 + j + 3) * 0.04;

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
        h_state[rk] = static_cast<double>(idx * 3 + rk + 1) * 0.12;
        const double r_val = static_cast<double>(idx * 2 + rk + 1) * 0.15;
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
  }
  for (size_t m = 0; m < expected_rw_grads.size(); ++m)
  {
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
    EXPECT_NEAR(actual_w_grads[m], expected_w_grads[m], 1e-12);
    EXPECT_NEAR(actual_z_w_grads[m], expected_z_w_grads[m], 1e-12);
    EXPECT_NEAR(actual_r_w_grads[m], expected_r_w_grads[m], 1e-12);
  }

  for (size_t m = 0; m < expected_rw_grads.size(); ++m)
  {
    EXPECT_NEAR(actual_rw_grads[m], expected_rw_grads[m], 1e-12);
    EXPECT_NEAR(actual_z_rw_grads[m], expected_z_rw_grads[m], 1e-12);
    EXPECT_NEAR(actual_r_rw_grads[m], expected_r_rw_grads[m], 1e-12);
  }

  for (size_t j = 0; j < num_outputs; ++j)
  {
    EXPECT_NEAR(actual_b_grads[j], expected_b_grads[j], 1e-12);
    EXPECT_NEAR(actual_z_b_grads[j], expected_z_b_grads[j], 1e-12);
    EXPECT_NEAR(actual_r_b_grads[j], expected_r_b_grads[j], 1e-12);
  }
}

TEST_F(GRURNNLayerTest, CalculateAndStoreGradientsSingleStepEquivalence)
{
  const unsigned num_inputs = 4;
  const unsigned num_outputs = 4;
  const size_t batch_size = 2;
  const size_t num_time_steps = 1;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  std::vector<unsigned> topology = { num_inputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, 5);

  for (size_t b = 0; b < batch_size; ++b)
  {
    batch_go[b].set_rnn_outputs(0, { 1.0, 0.5, -0.5, 0.25 });
    batch_go[b].set_rnn_gate_gradients(1, {
      0.1, 0.2, 0.3, 0.4, // gh
      0.2, 0.3, 0.4, 0.5, // gz
      0.3, 0.4, 0.5, 0.6  // gr
    });
  }

  MockLayer prev_layer(0, num_inputs);
  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, 0);

  // Recurrent weight gradients must be exactly 0 since t=0 has no previous hidden state
  for (double rw_g : layer.get_rw_grads())
  {
    EXPECT_DOUBLE_EQ(rw_g, 0.0);
  }
  for (double z_rw_g : layer.get_z_rw_grads())
  {
    EXPECT_DOUBLE_EQ(z_rw_g, 0.0);
  }
  for (double r_rw_g : layer.get_r_rw_grads())
  {
    EXPECT_DOUBLE_EQ(r_rw_g, 0.0);
  }

  // Input weights and biases must be non-zero
  EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
}

TEST_F(GRURNNLayerTest, ResidualCandidateGateInjectionAcrossTimesteps)
{
  const size_t num_inputs = 1;
  const size_t num_outputs = 1;
  const size_t num_time_steps = 2;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  layer.set_w_values({ 0.5 });
  layer.set_rw_values({ 0.0 });
  layer.set_z_w_values({ 0.2 });
  layer.set_z_rw_values({ 0.0 });
  layer.set_r_w_values({ 0.4 });
  layer.set_r_rw_values({ 0.0 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps, GRURNNLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });

  std::vector<std::vector<double>> residual = { { 0.3 } };
  layer.calculate_forward_feed(batch_go, prev_layer, residual, batch_hs, 1, false);

  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  ASSERT_EQ(rnn_out.size(), 2u);
  EXPECT_TRUE(std::isfinite(rnn_out[0]));
  EXPECT_TRUE(std::isfinite(rnn_out[1]));
}

TEST_F(GRURNNLayerTest, ResidualWithDropoutAndInferenceEquivalence)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 2;
  const size_t num_time_steps = 2;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.5, nullptr, 1, true, 0.0, false, 77);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { 2, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps, GRURNNLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 0.8, -0.4, 0.2, -0.6 });

  std::vector<std::vector<double>> residual = { { 0.2, -0.2 } };

  // Inference mode: dropout bypassed deterministically
  layer.calculate_forward_feed(batch_go, prev_layer, residual, batch_hs, 1, false);
  const auto out_inference = batch_go[0].get_rnn_outputs(1);
  ASSERT_EQ(out_inference.size(), 4u);
  for (double val : out_inference)
  {
    EXPECT_TRUE(std::isfinite(val));
  }

  // Training mode: forward and backward pass stability
  auto batch_go_train = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs_train = create_batch_hidden_states(topology, 1, num_time_steps, GRURNNLayer::Multiplier);
  batch_go_train[0].set_rnn_outputs(0, { 0.8, -0.4, 0.2, -0.6 });

  layer.calculate_forward_feed(batch_go_train, prev_layer, residual, batch_hs_train, 1, true);

  MockLayer next_layer(2, num_outputs, num_outputs);
  std::vector<double> identity(num_outputs * num_outputs, 0.0);
  for (size_t j = 0; j < num_outputs; ++j)
  {
    identity[j * num_outputs + j] = 1.0;
  }
  next_layer.set_w_values(identity);

  std::vector<std::vector<double>> upstream_grads = { { 0.1, 0.2, 0.3, 0.4 } };
  layer.calculate_hidden_gradients(batch_go_train, next_layer, upstream_grads, batch_hs_train, 1, num_time_steps);

  const auto gate_grads = batch_go_train[0].get_rnn_gate_gradients(1);
  for (double g : gate_grads)
  {
    EXPECT_TRUE(std::isfinite(g));
  }
}

TEST_F(GRURNNLayerTest, BackpropFromFFLayerWithTransposedWeightsEquivalence)
{
  const size_t num_inputs = 3;
  const size_t num_outputs = 4;
  const size_t next_outputs = 2;
  const size_t num_time_steps = 3;
  const size_t batch_size = 2;

  GRURNNLayer gru_layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 42);

  // FFLayer with transposed weights
  FFLayer ff_next(2, num_outputs, next_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, false, 0.0, std::nullopt);
  std::vector<double> ff_weights = {
    0.1, 0.2,
    0.3, 0.4,
    0.5, 0.6,
    0.7, 0.8
  };
  ff_next.set_w_values(ff_weights);

  // MockLayer with identical weights (does not provide get_w_values_T)
  MockLayer mock_next(2, next_outputs, num_outputs);
  mock_next.set_w_values(ff_weights);

  std::vector<unsigned> topology = { static_cast<unsigned>(num_inputs), static_cast<unsigned>(num_outputs), static_cast<unsigned>(next_outputs) };
  auto batch_go_ff = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_go_mock = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_ff = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);
  auto batch_hs_mock = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  MockLayer prev_layer(0, num_inputs);
  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> in_seq(num_time_steps * num_inputs);
    for (size_t k = 0; k < in_seq.size(); ++k)
    {
      in_seq[k] = std::sin(static_cast<double>(b * num_time_steps + k + 1)) * 0.2;
    }
    batch_go_ff[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
    batch_go_mock[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
  }

  gru_layer.calculate_forward_feed(batch_go_ff, prev_layer, {}, batch_hs_ff, batch_size, true);
  gru_layer.calculate_forward_feed(batch_go_mock, prev_layer, {}, batch_hs_mock, batch_size, true);

  std::vector<std::vector<double>> upstream_grads(batch_size, std::vector<double>(num_time_steps * next_outputs));
  for (size_t b = 0; b < batch_size; ++b)
  {
    for (size_t k = 0; k < upstream_grads[b].size(); ++k)
    {
      upstream_grads[b][k] = std::cos(static_cast<double>(b * num_time_steps + k + 1)) * 0.3;
    }
  }

  gru_layer.calculate_hidden_gradients(batch_go_ff, ff_next, upstream_grads, batch_hs_ff, batch_size, num_time_steps);
  gru_layer.calculate_hidden_gradients(batch_go_mock, mock_next, upstream_grads, batch_hs_mock, batch_size, num_time_steps);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto grads_ff = batch_go_ff[b].get_rnn_gate_gradients(1);
    const auto grads_mock = batch_go_mock[b].get_rnn_gate_gradients(1);
    ASSERT_EQ(grads_ff.size(), grads_mock.size());
    for (size_t k = 0; k < grads_ff.size(); ++k)
    {
      EXPECT_NEAR(grads_ff[k], grads_mock[k], 1e-12) << "batch " << b << " index " << k;
    }
  }
}

TEST_F(GRURNNLayerTest, IdentityProxyBypassEquivalence)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 3;
  const size_t num_time_steps = 2;
  const size_t batch_size = 2;

  GRURNNLayer gru_layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 99);

  // MockLayer with identity matrix of size num_outputs x num_outputs
  MockLayer mock_identity(2, num_outputs, num_outputs);
  std::vector<double> identity_weights(num_outputs * num_outputs, 0.0);
  for (size_t i = 0; i < num_outputs; ++i)
  {
    identity_weights[i * num_outputs + i] = 1.0;
  }
  mock_identity.set_w_values(identity_weights);

  std::vector<unsigned> topology = { static_cast<unsigned>(num_inputs), static_cast<unsigned>(num_outputs), static_cast<unsigned>(num_outputs) };
  auto batch_go_self = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_go_mock = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_self = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);
  auto batch_hs_mock = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  MockLayer prev_layer(0, num_inputs);
  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> in_seq(num_time_steps * num_inputs);
    for (size_t k = 0; k < in_seq.size(); ++k)
    {
      in_seq[k] = 0.1 * static_cast<double>(b * num_time_steps + k + 1);
    }
    batch_go_self[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
    batch_go_mock[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
  }

  gru_layer.calculate_forward_feed(batch_go_self, prev_layer, {}, batch_hs_self, batch_size, true);
  gru_layer.calculate_forward_feed(batch_go_mock, prev_layer, {}, batch_hs_mock, batch_size, true);

  std::vector<std::vector<double>> upstream_grads(batch_size, std::vector<double>(num_time_steps * num_outputs));
  for (size_t b = 0; b < batch_size; ++b)
  {
    for (size_t k = 0; k < upstream_grads[b].size(); ++k)
    {
      upstream_grads[b][k] = 0.05 * static_cast<double>(b + k + 1);
    }
  }

  // Next layer = gru_layer itself (triggers is_identity fast path)
  gru_layer.calculate_hidden_gradients(batch_go_self, gru_layer, upstream_grads, batch_hs_self, batch_size, num_time_steps);
  // Next layer = mock_identity (triggers general matrix multiply)
  gru_layer.calculate_hidden_gradients(batch_go_mock, mock_identity, upstream_grads, batch_hs_mock, batch_size, num_time_steps);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto grads_self = batch_go_self[b].get_rnn_gate_gradients(1);
    const auto grads_mock = batch_go_mock[b].get_rnn_gate_gradients(1);
    ASSERT_EQ(grads_self.size(), grads_mock.size());
    for (size_t k = 0; k < grads_self.size(); ++k)
    {
      EXPECT_NEAR(grads_self[k], grads_mock[k], 1e-12) << "batch " << b << " index " << k;
    }
  }
}

TEST_F(GRURNNLayerTest, TruncatedBpttZeroesUnprocessedTimesteps)
{
  const unsigned num_inputs = 2;
  const unsigned num_outputs = 2;
  const size_t num_time_steps = 4;
  const size_t batch_size = 2;
  const int bptt_ticks = 2;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 123);

  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs, num_outputs);
  std::vector<double> identity(num_outputs * num_outputs, 0.0);
  for (size_t j = 0; j < num_outputs; ++j)
  {
    identity[j * num_outputs + j] = 1.0;
  }
  next_layer.set_w_values(identity);

  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> in_seq(num_time_steps * num_inputs);
    for (size_t k = 0; k < in_seq.size(); ++k)
    {
      in_seq[k] = 0.2 * static_cast<double>(b * num_time_steps + k + 1);
    }
    batch_go[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
  }

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

  std::vector<std::vector<double>> upstream_grads(batch_size, std::vector<double>(num_time_steps * num_outputs, 0.5));
  layer.calculate_hidden_gradients(batch_go, next_layer, upstream_grads, batch_hs, batch_size, bptt_ticks);

  const size_t t_end = num_time_steps - static_cast<size_t>(bptt_ticks);
  const size_t gate_count_elements_per_t = GRURNNLayer::GateCount * num_outputs;

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& gate_grads = batch_go[b].get_rnn_gate_gradients(1);
    ASSERT_EQ(gate_grads.size(), num_time_steps * gate_count_elements_per_t);
    for (size_t t = 0; t < t_end; ++t)
    {
      for (size_t g = 0; g < gate_count_elements_per_t; ++g)
      {
        EXPECT_EQ(gate_grads[t * gate_count_elements_per_t + g], 0.0) << "t=" << t << " g=" << g;
      }
    }

    bool has_nonzero = false;
    for (size_t t = t_end; t < num_time_steps; ++t)
    {
      for (size_t g = 0; g < gate_count_elements_per_t; ++g)
      {
        if (std::abs(gate_grads[t * gate_count_elements_per_t + g]) > 1e-9)
        {
          has_nonzero = true;
        }
      }
    }
    EXPECT_TRUE(has_nonzero);

    const auto& input_grads = batch_go[b].get_rnn_gradients(0);
    if (!input_grads.empty())
    {
      for (size_t t = 0; t < t_end; ++t)
      {
        for (size_t i = 0; i < num_inputs; ++i)
        {
          EXPECT_EQ(input_grads[t * num_inputs + i], 0.0) << "t=" << t << " i=" << i;
        }
      }
    }
  }
}

TEST_F(GRURNNLayerTest, BatchItemWithoutPrevLayerInputStillAccumulatesBiasAndRecurrentGradients)
{
  const unsigned num_inputs = 1;
  const unsigned num_outputs = 1;
  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  layer.set_w_values({ 0.5 });   layer.set_rw_values({ 0.1 });   layer.set_b_values({ 0.2 });
  layer.set_z_w_values({ 0.5 }); layer.set_z_rw_values({ 0.1 }); layer.set_z_b_values({ 0.2 });
  layer.set_r_w_values({ 0.6 }); layer.set_r_rw_values({ 0.2 }); layer.set_r_b_values({ 0.3 });

  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs);
  next_layer.set_w_values({ 1.0 });
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  auto batch_go_ref = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs_ref = create_batch_hidden_states(topology, 2, 2, GRURNNLayer::Multiplier);
  batch_go_ref[0].set_rnn_outputs(0, { 1.0, 1.0 });
  batch_go_ref[1].set_rnn_outputs(0, { 1.0, 1.0 });
  layer.calculate_forward_feed(batch_go_ref, prev_layer, {}, batch_hs_ref, 2, true);
  std::vector<std::vector<double>> batch_next_grads = { { 0.0, 1.0 }, { 0.0, 1.0 } };
  layer.calculate_hidden_gradients(batch_go_ref, next_layer, batch_next_grads, batch_hs_ref, 2, 2);
  layer.calculate_and_store_gradients(batch_go_ref, batch_hs_ref, prev_layer, 2, 2);

  const double ref_w = layer.get_w_grads()[0];
  const double ref_b = layer.get_b_grads()[0];
  const double ref_rw = layer.get_rw_grads()[0];
  const double ref_z_w = layer.get_z_w_grads()[0];
  const double ref_z_b = layer.get_z_b_grads()[0];
  const double ref_z_rw = layer.get_z_rw_grads()[0];
  const double ref_r_w = layer.get_r_w_grads()[0];
  const double ref_r_b = layer.get_r_b_grads()[0];
  const double ref_r_rw = layer.get_r_rw_grads()[0];

  auto batch_go = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs = create_batch_hidden_states(topology, 2, 2, GRURNNLayer::Multiplier);
  batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });
  batch_go[1].set_rnn_outputs(0, { 1.0, 1.0 });
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, true);
  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 2, 2);

  batch_go[1].set_rnn_outputs(0, std::vector<double>{});

  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 2, 2);

  EXPECT_NEAR(layer.get_b_grads()[0], ref_b, 1e-9);
  EXPECT_NEAR(layer.get_z_b_grads()[0], ref_z_b, 1e-9);
  EXPECT_NEAR(layer.get_r_b_grads()[0], ref_r_b, 1e-9);
  EXPECT_NEAR(layer.get_rw_grads()[0], ref_rw, 1e-9);
  EXPECT_NEAR(layer.get_z_rw_grads()[0], ref_z_rw, 1e-9);
  EXPECT_NEAR(layer.get_r_rw_grads()[0], ref_r_rw, 1e-9);

  EXPECT_NEAR(layer.get_w_grads()[0], ref_w / 2.0, 1e-9);
  EXPECT_NEAR(layer.get_z_w_grads()[0], ref_z_w / 2.0, 1e-9);
  EXPECT_NEAR(layer.get_r_w_grads()[0], ref_r_w / 2.0, 1e-9);
}

TEST_F(GRURNNLayerTest, CalculateAndStoreGradientsChunkLargeBatchHeapGrowth)
{
  const unsigned num_inputs = 2;
  const unsigned num_outputs = 2;
  const size_t batch_size = 70;
  const size_t num_time_steps = 2;

  GRURNNLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 42);

  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs, num_outputs);
  std::vector<double> identity(num_outputs * num_outputs, 0.0);
  for (size_t j = 0; j < num_outputs; ++j)
  {
    identity[j * num_outputs + j] = 1.0;
  }
  next_layer.set_w_values(identity);

  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> in_seq(num_time_steps * num_inputs);
    for (size_t k = 0; k < in_seq.size(); ++k)
    {
      in_seq[k] = std::sin(static_cast<double>(b * 10 + k)) * 0.3;
    }
    batch_go[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
  }

  layer_st.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

  std::vector<std::vector<double>> upstream_grads(batch_size, std::vector<double>(num_time_steps * num_outputs));
  for (size_t b = 0; b < batch_size; ++b)
  {
    for (size_t k = 0; k < upstream_grads[b].size(); ++k)
    {
      upstream_grads[b][k] = std::cos(static_cast<double>(b * 7 + k)) * 0.2;
    }
  }

  layer_st.calculate_hidden_gradients(batch_go, next_layer, upstream_grads, batch_hs, batch_size, num_time_steps);
  layer_st.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, static_cast<int>(num_time_steps));

  for (double v : layer_st.get_w_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
  for (double v : layer_st.get_rw_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
  for (double v : layer_st.get_b_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
  for (double v : layer_st.get_z_w_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
  for (double v : layer_st.get_z_rw_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
  for (double v : layer_st.get_z_b_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
  for (double v : layer_st.get_r_w_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
  for (double v : layer_st.get_r_rw_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
  for (double v : layer_st.get_r_b_grads())
  {
    EXPECT_TRUE(std::isfinite(v));
  }
}

namespace
{
  static void assert_gru_gradients_near(const std::vector<double>& g_st, const std::vector<double>& g_mt, const char* name)
  {
    ASSERT_EQ(g_st.size(), g_mt.size()) << name;
    for (size_t k = 0; k < g_st.size(); ++k)
    {
      EXPECT_NEAR(g_st[k], g_mt[k], 1e-12) << name << " index " << k;
    }
  }
}

TEST_F(GRURNNLayerTest, ThreadedGradientAccumulationEquivalence)
{
  const size_t num_inputs = 4;
  const size_t num_outputs = 4;
  const size_t num_time_steps = 3;
  const size_t batch_size = 8;

  GRURNNLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 42);
  GRURNNLayer layer_mt(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 4, true, 0.0, false, 42);

  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs, num_outputs);
  std::vector<double> identity(num_outputs * num_outputs, 0.0);
  for (size_t j = 0; j < num_outputs; ++j)
  {
    identity[j * num_outputs + j] = 1.0;
  }
  next_layer.set_w_values(identity);

  std::vector<unsigned> topology = { static_cast<unsigned>(num_inputs), static_cast<unsigned>(num_outputs), static_cast<unsigned>(num_outputs) };
  auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);
  auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> in_seq(num_time_steps * num_inputs);
    for (size_t k = 0; k < in_seq.size(); ++k)
    {
      in_seq[k] = std::sin(static_cast<double>(b * 10 + k)) * 0.5;
    }
    batch_go_st[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
    batch_go_mt[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
  }

  layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, true);
  layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, true);

  std::vector<std::vector<double>> upstream_grads(batch_size, std::vector<double>(num_time_steps * num_outputs));
  for (size_t b = 0; b < batch_size; ++b)
  {
    for (size_t k = 0; k < upstream_grads[b].size(); ++k)
    {
      upstream_grads[b][k] = std::cos(static_cast<double>(b * 5 + k)) * 0.2;
    }
  }

  layer_st.calculate_hidden_gradients(batch_go_st, next_layer, upstream_grads, batch_hs_st, batch_size, num_time_steps);
  layer_mt.calculate_hidden_gradients(batch_go_mt, next_layer, upstream_grads, batch_hs_mt, batch_size, num_time_steps);

  layer_st.calculate_and_store_gradients(batch_go_st, batch_hs_st, prev_layer, batch_size, num_time_steps);
  layer_mt.calculate_and_store_gradients(batch_go_mt, batch_hs_mt, prev_layer, batch_size, num_time_steps);

  assert_gru_gradients_near(layer_st.get_w_grads(), layer_mt.get_w_grads(), "w_grads");
  assert_gru_gradients_near(layer_st.get_rw_grads(), layer_mt.get_rw_grads(), "rw_grads");
  assert_gru_gradients_near(layer_st.get_b_grads(), layer_mt.get_b_grads(), "b_grads");
  assert_gru_gradients_near(layer_st.get_z_w_grads(), layer_mt.get_z_w_grads(), "z_w_grads");
  assert_gru_gradients_near(layer_st.get_z_rw_grads(), layer_mt.get_z_rw_grads(), "z_rw_grads");
  assert_gru_gradients_near(layer_st.get_z_b_grads(), layer_mt.get_z_b_grads(), "z_b_grads");
  assert_gru_gradients_near(layer_st.get_r_w_grads(), layer_mt.get_r_w_grads(), "r_w_grads");
  assert_gru_gradients_near(layer_st.get_r_rw_grads(), layer_mt.get_r_rw_grads(), "r_rw_grads");
  assert_gru_gradients_near(layer_st.get_r_b_grads(), layer_mt.get_r_b_grads(), "r_b_grads");
}

TEST_F(GRURNNLayerTest, AnyHasRnnInputIsComputedOverWholeBatchNotPerChunk)
{
  const unsigned num_inputs = 1;
  const unsigned num_outputs = 1;
  const size_t num_time_steps = 2;
  const size_t batch_size = 3;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 7);

  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs, num_outputs);
  next_layer.set_w_values({ 1.0 });

  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> in_seq(num_time_steps * num_inputs);
    for (size_t k = 0; k < in_seq.size(); ++k)
    {
      in_seq[k] = 0.1 * static_cast<double>(b * num_time_steps + k + 1);
    }
    batch_go[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
  }

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

  std::vector<std::vector<double>> upstream_grads(batch_size, std::vector<double>(num_time_steps * num_outputs, 0.3));
  layer.calculate_hidden_gradients(batch_go, next_layer, upstream_grads, batch_hs, batch_size, num_time_steps);

  const double item2_last_step_value = 0.1 * static_cast<double>(2 * num_time_steps + num_time_steps);
  batch_go[2].set_rnn_outputs(0, std::vector<double>{});
  batch_go[2].set_outputs(0, { item2_last_step_value });

  const size_t w_size = static_cast<size_t>(num_inputs) * num_outputs;
  const size_t rw_size = static_cast<size_t>(num_outputs) * num_outputs;
  const size_t b_size = num_outputs;

  const unsigned prev_layer_index = prev_layer.get_layer_index();
  const int t_start = static_cast<int>(num_time_steps) - 1;
  const int t_end = 0;

  struct ChunkRunner
  {
    const GRURNNLayer& layer;
    const std::vector<GradientsAndOutputs>& batch_go;
    const std::vector<HiddenStates>& batch_hs;
    unsigned prev_layer_index;
    unsigned num_inputs;
    unsigned num_outputs;
    size_t num_time_steps;
    int t_start;
    int t_end;
    size_t batch_size;
    size_t w_size;
    size_t rw_size;
    size_t b_size;

    std::array<std::vector<double>, 9> operator()(bool any_has_rnn_input) const
    {
      std::vector<double> w(w_size, 0.0), rw(rw_size, 0.0);
      std::vector<double> zw(w_size, 0.0), zrw(rw_size, 0.0);
      std::vector<double> rw2(w_size, 0.0), rrw(rw_size, 0.0);
      std::vector<double> b(b_size, 0.0), zb(b_size, 0.0), rb(b_size, 0.0);
      layer.calculate_and_store_gradients_chunk(
        0, batch_size,
        batch_go, batch_hs,
        prev_layer_index, num_inputs, num_outputs, num_time_steps,
        t_start, t_end, any_has_rnn_input,
        w, rw, zw, zrw, rw2, rrw, b, zb, rb
      );
      return std::array<std::vector<double>, 9>{ w, rw, zw, zrw, rw2, rrw, b, zb, rb };
    }
  };

  ChunkRunner run_chunk{
    layer, batch_go, batch_hs,
    prev_layer_index, num_inputs, num_outputs, num_time_steps,
    t_start, t_end, batch_size,
    w_size, rw_size, b_size
  };

  const auto rejected = run_chunk(/*any_has_rnn_input=*/true);
  const auto accepted = run_chunk(/*any_has_rnn_input=*/false);

  // Bias and recurrent-weight gradients never depend on prev-layer input, so
  // they must be identical regardless of any_has_rnn_input.
  for (size_t idx : { 1, 3, 5 })
  {
    ASSERT_EQ(rejected[idx].size(), accepted[idx].size());
    for (size_t k = 0; k < rejected[idx].size(); ++k)
    {
      EXPECT_NEAR(rejected[idx][k], accepted[idx][k], 1e-12) << "recurrent-weight index " << idx << "," << k;
    }
  }
  for (size_t idx : { 6, 7, 8 })
  {
    ASSERT_EQ(rejected[idx].size(), accepted[idx].size());
    for (size_t k = 0; k < rejected[idx].size(); ++k)
    {
      EXPECT_NEAR(rejected[idx][k], accepted[idx][k], 1e-12) << "bias index " << idx << "," << k;
    }
  }

  // Input-weight gradients must differ: any_has_rnn_input=false wrongly lets
  // item 2's last-step value broadcast across every timestep.
  bool any_input_weight_differs = false;
  for (size_t idx : { 0, 2, 4 })
  {
    for (size_t k = 0; k < rejected[idx].size(); ++k)
    {
      if (std::abs(rejected[idx][k] - accepted[idx][k]) > 1e-9)
      {
        any_input_weight_differs = true;
      }
    }
  }
  EXPECT_TRUE(any_input_weight_differs);
}

TEST_F(GRURNNLayerTest, SwaAndLookaheadWeightCaching)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 2;

  GRURNNLayer layer_base(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 11);
  GRURNNLayer layer_snapshot(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 22);

  std::vector<double> rw_snap = { 0.1, 0.2, 0.3, 0.4 };
  std::vector<double> z_rw_snap = { 0.5, 0.6, 0.7, 0.8 };
  std::vector<double> r_rw_snap = { 0.9, 1.0, 1.1, 1.2 };
  layer_snapshot.set_rw_values(rw_snap);
  layer_snapshot.set_z_rw_values(z_rw_snap);
  layer_snapshot.set_r_rw_values(r_rw_snap);

  layer_base.accumulate_swa_average(layer_snapshot, 1);

  const auto& rw_base = layer_base.get_rw_values();
  const auto& rw_base_T = layer_base.get_rw_values_T();
  for (size_t r = 0; r < num_outputs; ++r)
  {
    for (size_t c = 0; c < num_outputs; ++c)
    {
      EXPECT_DOUBLE_EQ(rw_base_T[c * num_outputs + r], rw_base[r * num_outputs + c]);
    }
  }

  const auto& z_rw_base = layer_base.get_z_rw_values();
  const auto& z_rw_base_T = layer_base.get_z_rw_values_T();
  for (size_t r = 0; r < num_outputs; ++r)
  {
    for (size_t c = 0; c < num_outputs; ++c)
    {
      EXPECT_DOUBLE_EQ(z_rw_base_T[c * num_outputs + r], z_rw_base[r * num_outputs + c]);
    }
  }

  GRURNNLayer slow_layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 33);
  GRURNNLayer fast_layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, 44);

  std::vector<double> fast_rw = { 0.25, -0.15, 0.35, -0.45 };
  fast_layer.set_rw_values(fast_rw);

  slow_layer.update_lookahead_slow_weights(fast_layer, 0.5);

  const auto& slow_rw = slow_layer.get_rw_values();
  const auto& slow_rw_T = slow_layer.get_rw_values_T();
  for (size_t r = 0; r < num_outputs; ++r)
  {
    for (size_t c = 0; c < num_outputs; ++c)
    {
      EXPECT_DOUBLE_EQ(slow_rw_T[c * num_outputs + r], slow_rw[r * num_outputs + c]);
    }
  }

  const auto& fast_rw_current = fast_layer.get_rw_values();
  const auto& fast_rw_T = fast_layer.get_rw_values_T();
  for (size_t r = 0; r < num_outputs; ++r)
  {
    for (size_t c = 0; c < num_outputs; ++c)
    {
      EXPECT_DOUBLE_EQ(fast_rw_T[c * num_outputs + r], fast_rw_current[r * num_outputs + c]);
    }
  }
}

namespace
{
  static double gru_multi_batch_average_loss(
    GRURNNLayer& layer,
    const MockLayer& prev_layer,
    const std::vector<unsigned>& topology,
    const std::vector<std::vector<double>>& inputs_batch,
    const std::vector<std::vector<double>>& targets_batch,
    size_t num_time_steps,
    unsigned num_outputs)
  {
    const size_t batch_size = inputs_batch.size();
    auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);
    for (size_t b = 0; b < batch_size; ++b)
    {
      batch_go[b].set_rnn_outputs(0, inputs_batch[b]);
    }

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

    double total_loss = 0.0;
    for (size_t b = 0; b < batch_size; ++b)
    {
      const auto& outputs = batch_go[b].get_rnn_outputs(1);
      for (size_t i = 0; i < num_time_steps * num_outputs; ++i)
      {
        const double diff = outputs[i] - targets_batch[b][i];
        total_loss += 0.5 * diff * diff;
      }
    }
    return total_loss / static_cast<double>(batch_size);
  }
}

TEST_F(GRURNNLayerTest, RecurrentAndInputWeightsFiniteDifferenceMultiBatchNumericalEquivalence)
{
  const unsigned num_inputs = 2;
  const unsigned num_outputs = 2;
  const size_t num_time_steps = 3;
  const size_t batch_size = 2;

  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  std::vector<double> zw = { 0.2, -0.3, 0.1, 0.4 };
  std::vector<double> rw = { -0.1, 0.2, 0.3, -0.2 };
  std::vector<double> gw = { 0.3, -0.1, -0.2, 0.2 };

  std::vector<double> zrw = { 0.15, -0.1, 0.05, 0.2 };
  std::vector<double> rrw = { -0.1, 0.15, -0.2, 0.1 };
  std::vector<double> grw = { 0.2, -0.05, 0.1, -0.15 };

  std::vector<double> zb = { 0.05, -0.02 };
  std::vector<double> rb = { -0.03, 0.04 };
  std::vector<double> gb = { 0.02, -0.01 };

  layer.set_z_w_values(zw);
  layer.set_r_w_values(rw);
  layer.set_w_values(gw);

  layer.set_z_rw_values(zrw);
  layer.set_r_rw_values(rrw);
  layer.set_rw_values(grw);

  layer.set_z_b_values(zb);
  layer.set_r_b_values(rb);
  layer.set_b_values(gb);

  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs);
  {
    std::vector<double> identity(num_outputs * num_outputs, 0.0);
    for (unsigned j = 0; j < num_outputs; ++j)
    {
      identity[j * num_outputs + j] = 1.0;
    }
    next_layer.set_w_values(identity);
  }

  std::vector<std::vector<double>> inputs_batch(batch_size, std::vector<double>(num_time_steps * num_inputs));
  std::vector<std::vector<double>> targets_batch(batch_size, std::vector<double>(num_time_steps * num_outputs));
  for (size_t b = 0; b < batch_size; ++b)
  {
    for (size_t i = 0; i < inputs_batch[b].size(); ++i)
    {
      inputs_batch[b][i] = 0.2 * std::sin(static_cast<double>((b + 1) * 10 + i + 1));
    }
    for (size_t i = 0; i < targets_batch[b].size(); ++i)
    {
      targets_batch[b][i] = 0.15 * std::cos(static_cast<double>((b + 1) * 7 + i + 1));
    }
  }

  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);
  for (size_t b = 0; b < batch_size; ++b)
  {
    batch_go[b].set_rnn_outputs(0, inputs_batch[b]);
  }

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

  std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_time_steps * num_outputs));
  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& outputs = batch_go[b].get_rnn_outputs(1);
    for (size_t i = 0; i < num_time_steps * num_outputs; ++i)
    {
      batch_next_grads[b][i] = outputs[i] - targets_batch[b][i];
    }
  }

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, batch_size, 0);
  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, 0);

  const auto analytical_rw_grads = layer.get_rw_grads();
  const auto analytical_gw_grads = layer.get_w_grads();
  const auto analytical_z_rw_grads = layer.get_z_rw_grads();
  const auto analytical_z_w_grads = layer.get_z_w_grads();
  const auto analytical_r_rw_grads = layer.get_r_rw_grads();
  const auto analytical_r_w_grads = layer.get_r_w_grads();

  const double eps = 1e-6;

  for (size_t k = 0; k < num_outputs * num_outputs; ++k)
  {
    auto grw_pert = grw;
    grw_pert[k] = grw[k] + eps;
    layer.set_rw_values(grw_pert);
    const double loss_plus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    grw_pert[k] = grw[k] - eps;
    layer.set_rw_values(grw_pert);
    const double loss_minus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    layer.set_rw_values(grw);
    const double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(analytical_rw_grads[k], num_grad, 1e-5) << "rw index " << k;
  }

  for (size_t k = 0; k < num_inputs * num_outputs; ++k)
  {
    auto gw_pert = gw;
    gw_pert[k] = gw[k] + eps;
    layer.set_w_values(gw_pert);
    const double loss_plus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    gw_pert[k] = gw[k] - eps;
    layer.set_w_values(gw_pert);
    const double loss_minus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    layer.set_w_values(gw);
    const double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(analytical_gw_grads[k], num_grad, 1e-5) << "w index " << k;
  }

  for (size_t k = 0; k < num_outputs * num_outputs; ++k)
  {
    auto zrw_pert = zrw;
    zrw_pert[k] = zrw[k] + eps;
    layer.set_z_rw_values(zrw_pert);
    const double loss_plus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    zrw_pert[k] = zrw[k] - eps;
    layer.set_z_rw_values(zrw_pert);
    const double loss_minus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    layer.set_z_rw_values(zrw);
    const double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(analytical_z_rw_grads[k], num_grad, 1e-5) << "z_rw index " << k;
  }

  for (size_t k = 0; k < num_inputs * num_outputs; ++k)
  {
    auto zw_pert = zw;
    zw_pert[k] = zw[k] + eps;
    layer.set_z_w_values(zw_pert);
    const double loss_plus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    zw_pert[k] = zw[k] - eps;
    layer.set_z_w_values(zw_pert);
    const double loss_minus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    layer.set_z_w_values(zw);
    const double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(analytical_z_w_grads[k], num_grad, 1e-5) << "z_w index " << k;
  }

  for (size_t k = 0; k < num_outputs * num_outputs; ++k)
  {
    auto rrw_pert = rrw;
    rrw_pert[k] = rrw[k] + eps;
    layer.set_r_rw_values(rrw_pert);
    const double loss_plus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    rrw_pert[k] = rrw[k] - eps;
    layer.set_r_rw_values(rrw_pert);
    const double loss_minus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    layer.set_r_rw_values(rrw);
    const double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(analytical_r_rw_grads[k], num_grad, 1e-5) << "r_rw index " << k;
  }

  for (size_t k = 0; k < num_inputs * num_outputs; ++k)
  {
    auto rw_pert = rw;
    rw_pert[k] = rw[k] + eps;
    layer.set_r_w_values(rw_pert);
    const double loss_plus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    rw_pert[k] = rw[k] - eps;
    layer.set_r_w_values(rw_pert);
    const double loss_minus = gru_multi_batch_average_loss(layer, prev_layer, topology, inputs_batch, targets_batch, num_time_steps, num_outputs);

    layer.set_r_w_values(rw);
    const double num_grad = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(analytical_r_w_grads[k], num_grad, 1e-5) << "r_w index " << k;
  }
}

TEST_F(GRURNNLayerTest, DropoutMathematicalSoundnessInBPTT)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 2;
  const size_t num_time_steps = 2;
  const double dropout_rate = 0.5;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, false, 888);

  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs, num_outputs);
  std::vector<double> identity(num_outputs * num_outputs, 0.0);
  for (size_t j = 0; j < num_outputs; ++j)
  {
    identity[j * num_outputs + j] = 1.0;
  }
  next_layer.set_w_values(identity);

  std::vector<unsigned> topology = { 2, 2, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps, GRURNNLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 0.5, -0.5, 0.3, -0.3 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  const auto& item_states = batch_hs[0].at(1);
  for (size_t t = 0; t < num_time_steps; ++t)
  {
    const auto& packed = item_states[t].get_pre_activation_sums();
    ASSERT_GE(packed.size(), 5 * num_outputs);

    for (size_t j = 0; j < num_outputs; ++j)
    {
      const double raw_candidate_act = packed[3 * num_outputs + j];
      EXPECT_GE(raw_candidate_act, -1.0);
      EXPECT_LE(raw_candidate_act, 1.0);

      const double mask = packed[4 * num_outputs + j];
      EXPECT_TRUE(mask == 0.0 || std::abs(mask - 2.0) < 1e-9);
    }
  }

  std::vector<std::vector<double>> upstream_grads = { { 0.2, -0.1, 0.3, -0.2 } };
  layer.calculate_hidden_gradients(batch_go, next_layer, upstream_grads, batch_hs, 1, num_time_steps);
  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, num_time_steps);

  for (double g : layer.get_w_grads())
  {
    EXPECT_TRUE(std::isfinite(g));
  }
  for (double g : layer.get_rw_grads())
  {
    EXPECT_TRUE(std::isfinite(g));
  }
  for (double g : layer.get_b_grads())
  {
    EXPECT_TRUE(std::isfinite(g));
  }
}

TEST_F(GRURNNLayerTest, DirectGradientsFallbackToLayerIndexRnnGradients)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 2;
  const size_t num_time_steps = 2;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
  layer.set_z_w_values({ 0.1, 0.2, 0.3, 0.4 });
  layer.set_z_rw_values({ 0.1, 0.0, 0.0, 0.1 });
  layer.set_z_b_values({ 0.0, 0.0 });
  layer.set_r_w_values({ 0.2, 0.1, 0.4, 0.3 });
  layer.set_r_rw_values({ 0.0, 0.1, 0.1, 0.0 });
  layer.set_r_b_values({ 0.0, 0.0 });
  layer.set_w_values({ 0.5, 0.6, 0.7, 0.8 });
  layer.set_rw_values({ 0.2, 0.3, 0.4, 0.5 });
  layer.set_b_values({ 0.0, 0.0 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { 2, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps, GRURNNLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, -0.5, 1.0 });
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  // Set sequence gradients directly at layer 1 (this layer index)
  std::vector<double> incoming_seq_grads = { 0.25, -0.5, 0.75, -0.25 };
  batch_go[0].set_rnn_gradients(1, incoming_seq_grads);

  // Call calculate_hidden_gradients_from_output_gradients with empty batch_output_gradients
  // so direct gradients are used via _identity_proxy
  layer.calculate_hidden_gradients_from_output_gradients(batch_go, {}, batch_hs, 1, num_time_steps);

  const auto& gate_grads = batch_go[0].get_rnn_gate_gradients(1);
  ASSERT_EQ(gate_grads.size(), num_time_steps * GRURNNLayer::GateCount * num_outputs);

  bool any_non_zero = false;
  for (double g : gate_grads)
  {
    if (std::abs(g) > 1e-6)
    {
      any_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(any_non_zero);
}

TEST_F(GRURNNLayerTest, ForwardFeedFullSequenceResiduals)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 2;
  const size_t num_time_steps = 3;

  GRURNNLayer layer_no_res(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
  GRURNNLayer layer_res(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  std::vector<double> zw = { 0.1, 0.2, 0.3, 0.4 };
  std::vector<double> zrw = { 0.1, 0.0, 0.0, 0.1 };
  std::vector<double> zb = { 0.0, 0.0 };
  std::vector<double> rw = { 0.2, 0.1, 0.4, 0.3 };
  std::vector<double> rrw = { 0.0, 0.1, 0.1, 0.0 };
  std::vector<double> rb = { 0.0, 0.0 };
  std::vector<double> w = { 0.5, 0.6, 0.7, 0.8 };
  std::vector<double> r_w = { 0.2, 0.3, 0.4, 0.5 };
  std::vector<double> b = { 0.0, 0.0 };

  layer_no_res.set_z_w_values(zw); layer_no_res.set_z_rw_values(zrw); layer_no_res.set_z_b_values(zb);
  layer_no_res.set_r_w_values(rw); layer_no_res.set_r_rw_values(rrw); layer_no_res.set_r_b_values(rb);
  layer_no_res.set_w_values(w);   layer_no_res.set_rw_values(r_w);   layer_no_res.set_b_values(b);

  layer_res.set_z_w_values(zw); layer_res.set_z_rw_values(zrw); layer_res.set_z_b_values(zb);
  layer_res.set_r_w_values(rw); layer_res.set_r_rw_values(rrw); layer_res.set_r_b_values(rb);
  layer_res.set_w_values(w);   layer_res.set_rw_values(r_w);   layer_res.set_b_values(b);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { 2, 2 };
  auto batch_go_no_res = create_batch_gradients_and_outputs(topology, 1);
  auto batch_go_res = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs_no_res = create_batch_hidden_states(topology, 1, num_time_steps, GRURNNLayer::Multiplier);
  auto batch_hs_res = create_batch_hidden_states(topology, 1, num_time_steps, GRURNNLayer::Multiplier);

  std::vector<double> inputs = { 0.5, -0.5, 0.2, -0.2, 0.8, -0.8 };
  batch_go_no_res[0].set_rnn_outputs(0, inputs);
  batch_go_res[0].set_rnn_outputs(0, inputs);

  std::vector<std::vector<double>> full_seq_residuals = {
    { 0.1, -0.1, 0.2, -0.2, 0.3, -0.3 }
  };

  layer_no_res.calculate_forward_feed(batch_go_no_res, prev_layer, {}, batch_hs_no_res, 1, false);
  layer_res.calculate_forward_feed(batch_go_res, prev_layer, full_seq_residuals, batch_hs_res, 1, false);

  const auto& out_no_res = batch_go_no_res[0].get_rnn_outputs(1);
  const auto& out_res = batch_go_res[0].get_rnn_outputs(1);

  ASSERT_EQ(out_no_res.size(), num_time_steps * num_outputs);
  ASSERT_EQ(out_res.size(), num_time_steps * num_outputs);

  for (size_t i = 0; i < out_res.size(); ++i)
  {
    EXPECT_NE(out_no_res[i], out_res[i]);
    EXPECT_TRUE(std::isfinite(out_res[i]));
  }
}

TEST_F(GRURNNLayerTest, ForwardFeedPartialSequenceBroadcastingSafety)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 2;
  const size_t num_time_steps = 3;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { 2, 2 };

  auto batch_go = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs = create_batch_hidden_states(topology, 2, num_time_steps, GRURNNLayer::Multiplier);

  // Item 0: Single-step static output (size 2)
  batch_go[0].set_outputs(0, { 0.5, -0.5 });
  // Item 1: Full-sequence output (size 6 = 3 timesteps x 2 inputs)
  batch_go[1].set_rnn_outputs(0, { 0.1, -0.1, 0.2, -0.2, 0.3, -0.3 });

  // Must not crash or buffer-overflow
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, false);

  const auto& out0 = batch_go[0].get_rnn_outputs(1);
  const auto& out1 = batch_go[1].get_rnn_outputs(1);

  EXPECT_EQ(out0.size(), num_time_steps * num_outputs);
  EXPECT_EQ(out1.size(), num_time_steps * num_outputs);

  for (double val : out0)
  {
    EXPECT_TRUE(std::isfinite(val));
  }
  for (double val : out1)
  {
    EXPECT_TRUE(std::isfinite(val));
  }
}

TEST_F(GRURNNLayerTest, SingleStepFastPathGradientEquivalence)
{
  const size_t num_inputs = 3;
  const size_t num_outputs = 2;
  const size_t num_time_steps = 1;
  const size_t batch_size = 2;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
  layer.set_z_w_values({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
  layer.set_z_rw_values({ 0.1, 0.0, 0.0, 0.1 });
  layer.set_z_b_values({ 0.1, -0.1 });

  layer.set_r_w_values({ 0.2, 0.1, 0.4, 0.3, 0.6, 0.5 });
  layer.set_r_rw_values({ 0.0, 0.1, 0.1, 0.0 });
  layer.set_r_b_values({ -0.1, 0.1 });

  layer.set_w_values({ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 });
  layer.set_rw_values({ 0.2, 0.3, 0.4, 0.5 });
  layer.set_b_values({ 0.05, -0.05 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { 3, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  batch_go[0].set_outputs(0, { 1.0, 0.5, -0.5 });
  batch_go[1].set_outputs(0, { -0.5, 1.0, 0.5 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

  // Set upstream gradients directly
  std::vector<double> gate_grads_b0 = { 0.1, -0.2, 0.3, -0.4, 0.5, -0.6 }; // 1 tick x 3 gates x 2 neurons
  std::vector<double> gate_grads_b1 = { -0.2, 0.1, -0.4, 0.3, -0.6, 0.5 };
  batch_go[0].set_rnn_gate_gradients(1, gate_grads_b0);
  batch_go[1].set_rnn_gate_gradients(1, gate_grads_b1);

  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, num_time_steps);

  // For single-step (T=1), recurrent weight gradients must be 0 (no transitions t >= 1)
  for (double g : layer.get_rw_grads())
  {
    EXPECT_NEAR(g, 0.0, 1e-12);
  }
  for (double g : layer.get_z_rw_grads())
  {
    EXPECT_NEAR(g, 0.0, 1e-12);
  }
  for (double g : layer.get_r_rw_grads())
  {
    EXPECT_NEAR(g, 0.0, 1e-12);
  }

  // Bias gradients are mean of gate gradients across batch
  // Candidate bias (gh): (0.1 + (-0.2))/2 = -0.05, (-0.2 + 0.1)/2 = -0.05
  EXPECT_NEAR(layer.get_b_grads()[0], -0.05, 1e-10);
  EXPECT_NEAR(layer.get_b_grads()[1], -0.05, 1e-10);

  // Update gate bias (gz): (0.3 + (-0.4))/2 = -0.05, (-0.4 + 0.3)/2 = -0.05
  EXPECT_NEAR(layer.get_z_b_grads()[0], -0.05, 1e-10);
  EXPECT_NEAR(layer.get_z_b_grads()[1], -0.05, 1e-10);

  // Reset gate bias (gr): (0.5 + (-0.6))/2 = -0.05, (-0.6 + 0.5)/2 = -0.05
  EXPECT_NEAR(layer.get_r_b_grads()[0], -0.05, 1e-10);
  EXPECT_NEAR(layer.get_r_b_grads()[1], -0.05, 1e-10);
}

TEST_F(GRURNNLayerTest, DropoutBPTTConsistencyMultiBatch)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 2;
  const size_t num_time_steps = 2;
  const size_t batch_size = 2;
  const double dropout_rate = 0.5;

  GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, false, 999);

  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs, num_outputs);
  std::vector<double> identity(num_outputs * num_outputs, 0.0);
  for (size_t j = 0; j < num_outputs; ++j)
  {
    identity[j * num_outputs + j] = 1.0;
  }
  next_layer.set_w_values(identity);

  std::vector<unsigned> topology = { 2, 2, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, GRURNNLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 0.4, -0.4, 0.2, -0.2 });
  batch_go[1].set_rnn_outputs(0, { -0.3, 0.3, -0.1, 0.1 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

  std::vector<std::vector<double>> upstream_grads = {
    { 0.2, -0.1, 0.3, -0.2 },
    { -0.1, 0.2, -0.2, 0.3 }
  };
  layer.calculate_hidden_gradients(batch_go, next_layer, upstream_grads, batch_hs, batch_size, num_time_steps);
  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, num_time_steps);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& item_states = batch_hs[b].at(1);
    const auto& gate_grads = batch_go[b].get_rnn_gate_gradients(1);
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const auto& packed = item_states[t].get_pre_activation_sums();
      for (size_t j = 0; j < num_outputs; ++j)
      {
        const double mask = packed[4 * num_outputs + j];
        const double dh_hat = gate_grads[(t * GRURNNLayer::GateCount + 0) * num_outputs + j];
        if (mask == 0.0)
        {
          EXPECT_NEAR(dh_hat, 0.0, 1e-12);
        }
        else
        {
          EXPECT_TRUE(std::isfinite(dh_hat));
        }
      }
    }
  }
}
