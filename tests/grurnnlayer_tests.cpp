#include <gtest/gtest.h>
#include "../src/neuralnetwork/grurnnlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>

using namespace test_helper;

class GRURNNLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(GRURNNLayerTest, ConstructorAndClone) {
    unsigned num_inputs = 3;
    unsigned num_outputs = 2;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.01, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::Adam, -1, 0.1, nullptr, 1, true, 0.9);

    EXPECT_EQ(layer.get_layer_index(), 1);
    EXPECT_EQ(layer.get_number_input_neurons(), num_inputs);
    EXPECT_EQ(layer.get_number_output_neurons(), num_outputs);
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Gru);

    std::unique_ptr<Layer> cloned(layer.clone());
    EXPECT_EQ(cloned->get_layer_index(), 1);
    EXPECT_EQ(cloned->get_number_input_neurons(), num_inputs);
    EXPECT_EQ(cloned->get_number_output_neurons(), num_outputs);
    EXPECT_EQ(cloned->get_layer_architecture(), Layer::Architecture::Gru);
}

TEST_F(GRURNNLayerTest, ForwardFeedBasic) {
    // Single neuron GRU for simple math verification
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    // All weights zero, except W_h
    layer.set_z_w_values({ 0.0 });
    layer.set_z_rw_values({ 0.0 });
    layer.set_z_b_values({ 0.0 });
    
    layer.set_r_w_values({ 0.0 });
    layer.set_r_rw_values({ 0.0 });
    layer.set_r_b_values({ 0.0 });
    
    layer.set_w_values({ 1.0 });  // W_h
    layer.set_rw_values({ 1.0 }); // U_h
    layer.set_b_values({ 0.0 });  // b_h

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 3);

    // Step 1: x = 1.0, h_prev = 0.0
    batch_go[0].set_outputs(0, { 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    // z = sig(0*1 + 0*0 + 0) = 0.5
    // r = sig(0*1 + 0*0 + 0) = 0.5
    // h_hat = tanh(1*1 + (0.5*0)*1 + 0) = tanh(1)
    // h = (1-0.5)*0 + 0.5*tanh(1) = 0.5 * tanh(1)
    double expected_h1 = 0.5 * std::tanh(1.0);
    EXPECT_NEAR(batch_go[0].get_output(1, 0), expected_h1, 1e-9);
}

TEST_F(GRURNNLayerTest, ForwardFeedRecurrent) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    // Set weights so h_t depends on h_{t-1}
    layer.set_z_w_values({ -10.0 }); // Update gate near 0 (h_t approx h_{t-1})
    layer.set_z_rw_values({ 0.0 });
    layer.set_z_b_values({ 0.0 });
    
    layer.set_r_w_values({ 10.0 }); // Reset gate near 1
    layer.set_r_rw_values({ 0.0 });
    layer.set_r_b_values({ 0.0 });
    
    layer.set_w_values({ 0.0 });
    layer.set_rw_values({ 0.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2); // 2 time steps

    // Provide sequence [1.0, 1.0]
    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });
    
    // Manually set initial hidden state if possible? 
    // The forward feed in GRURNNLayer resets hidden state to zero for each sample.
    // So h_0 = 0.
    
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);
    
    // Step 1:
    // z1 = sig(-10*1) approx 0
    // h1 = (1-z1)*0 + z1*h_hat1 approx 0
    
    // Actually let's use easier weights
    layer.set_z_w_values({ 10.0 }); // z approx 1 (h_t approx h_hat_t)
    layer.set_w_values({ 0.5 });    // h_hat = tanh(0.5*x)
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);
    
    double h1 = batch_go[0].get_rnn_outputs(1)[0];
    EXPECT_NEAR(h1, std::tanh(0.5), 0.01);
    
    // Step 2:
    // z2 approx 1
    // r2 approx 1
    // h_hat2 = tanh(0.5*x2 + 0*h1) = tanh(0.5)
    // h2 approx tanh(0.5)
    double h2 = batch_go[0].get_rnn_outputs(1)[1];
    EXPECT_NEAR(h2, std::tanh(0.5), 0.01);
}

TEST_F(GRURNNLayerTest, AllActivationTypes) {
    std::vector<activation::method> methods = {
        activation::method::linear,
        activation::method::sigmoid,
        activation::method::relu,
        activation::method::tanh,
        activation::method::leakyRelu,
        activation::method::PRelu,
        activation::method::selu,
        activation::method::swish,
        activation::method::mish,
        activation::method::gelu,
        activation::method::elu
    };

    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };

    for (auto m : methods) {
        GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(m, 0.1), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
        auto batch_go = create_batch_gradients_and_outputs(topology, 1);
        auto batch_hs = create_batch_hidden_states(topology, 1, 1, 3);
        batch_go[0].set_outputs(0, { 0.5 });

        EXPECT_NO_THROW(layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false));
        double out = batch_go[0].get_output(1, 0);
        EXPECT_TRUE(std::isfinite(out));
    }
}

TEST_F(GRURNNLayerTest, CalculateHiddenGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    // Weights: W_z=0, W_h=1
    layer.set_z_w_values({ 0.0 });
    layer.set_z_rw_values({ 0.0 });
    layer.set_z_b_values({ 0.0 });
    layer.set_r_w_values({ 0.0 });
    layer.set_r_rw_values({ 0.0 });
    layer.set_r_b_values({ 0.0 });
    layer.set_w_values({ 1.0 });
    layer.set_rw_values({ 0.0 });
    layer.set_b_values({ 0.0 });

    MockLayer next_layer(2, num_outputs);
    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 3);

    // Forward pass to populate hidden states
    batch_go[0].set_outputs(0, { 1.0 }); // x = 1.0
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Next layer grad: [1.0]
    std::vector<std::vector<double>> batch_next_grads = { { 1.0 } };

    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

    // From math above: 
    // dL/dh_hat_pre = dL/dh * z * tanh'(h_hat_pre)
    // dL/dh = 1.0 (from next_layer)
    // z = sig(0) = 0.5
    // h_hat_pre = 1.0 (from W_h=1, x=1)
    // grad_x = dL/dh_hat_pre * W_h = 1.0 * 0.5 * (1.0 - tanh(1.0)^2) * 1.0
    double expected_grad_x = 0.5 * (1.0 - std::pow(std::tanh(1.0), 2));
    
    const auto& grads = batch_go[0].get_rnn_gradients(1);
    EXPECT_NEAR(grads[0], expected_grad_x, 1e-7);
}

TEST_F(GRURNNLayerTest, CalculateAndStoreGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 3);

    // Setup state
    batch_go[0].set_outputs(0, { 1.0 }); // x = 1.0
    // Manually set gate gradients (from BPTT normally)
    // rnn_gate_gradients: [d_h_hat, d_z, d_r]
    batch_go[0].set_rnn_gate_gradients(1, { 0.5, 0.2, 0.1 });

    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 0);

    // dL/dW_h = dL/dh_hat * x = 0.5 * 1.0 = 0.5
    // dL/dW_z = dL/dz * x = 0.2 * 1.0 = 0.2
    // dL/dW_r = dL/dr * x = 0.1 * 1.0 = 0.1
    EXPECT_NEAR(layer.get_w_grads()[0], 0.5, 1e-9);
    EXPECT_NEAR(layer.get_z_w_grads()[0], 0.2, 1e-9);
    EXPECT_NEAR(layer.get_r_w_grads()[0], 0.1, 1e-9);
}

TEST_F(GRURNNLayerTest, ZeroGradients) {
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
    layer.set_w_grads({ 1.0 });
    layer.set_rw_grads({ 1.0 });
    layer.set_z_w_grads({ 1.0 });
    layer.set_r_w_grads({ 1.0 });
    
    layer.zero_gradients();
    
    EXPECT_DOUBLE_EQ(layer.get_w_grads()[0], 0.0);
    EXPECT_DOUBLE_EQ(layer.get_rw_grads()[0], 0.0);
    EXPECT_DOUBLE_EQ(layer.get_z_w_grads()[0], 0.0);
    EXPECT_DOUBLE_EQ(layer.get_r_w_grads()[0], 0.0);
}

TEST_F(GRURNNLayerTest, ApplyStoredGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    GRURNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0 });
    layer.set_z_w_values({ 1.0 });
    layer.set_r_w_values({ 1.0 });

    layer.set_w_grads({ 0.1 });
    layer.set_z_w_grads({ 0.1 });
    layer.set_r_w_grads({ 0.1 });

    layer.apply_stored_gradients(0.1, 1.0); // LR=0.1, clipping=1.0

    // New weight = Old - LR * Grad = 1.0 - 0.1 * 0.1 = 0.99
    EXPECT_NEAR(layer.get_w_values()[0], 0.99, 1e-9);
    EXPECT_NEAR(layer.get_z_w_values()[0], 0.99, 1e-9);
    EXPECT_NEAR(layer.get_r_w_values()[0], 0.99, 1e-9);
}

TEST_F(GRURNNLayerTest, HelperMethods) {
    GRURNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
    layer.set_rw_values({ 0.1, 0.2, 0.3, 0.4 });
    
    // get_recurrent_weight_value(from, to) -> index = from * N + to
    EXPECT_DOUBLE_EQ(layer.get_recurrent_weight_value(0, 1), 0.2);
    EXPECT_DOUBLE_EQ(layer.get_recurrent_weight_value(1, 0), 0.3);
}

TEST_F(GRURNNLayerTest, GradientNorm) {
    GRURNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
    layer.set_w_grads({ 3.0 });
    layer.set_z_w_grads({ 4.0 });
    // NormSq = 3^2 + 4^2 = 25
    EXPECT_DOUBLE_EQ(layer.get_gradient_norm_sq(), 25.0);
}
