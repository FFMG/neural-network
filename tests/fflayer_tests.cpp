#include <gtest/gtest.h>
#include "../src/neuralnetwork/fflayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>

using namespace test_helper;

class FFLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(FFLayerTest, ConstructorAndClone) {
    unsigned num_inputs = 3;
    unsigned num_outputs = 2;
    FFLayer layer(1, num_inputs, num_outputs, 0.01, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::Adam, -1, 0.1, nullptr, 1, true, 0.9);

    EXPECT_EQ(layer.get_layer_index(), 1);
    EXPECT_EQ(layer.get_number_input_neurons(), num_inputs);
    EXPECT_EQ(layer.get_number_output_neurons(), num_outputs);
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::FF);
    EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);

    std::unique_ptr<Layer> cloned(layer.clone());
    EXPECT_EQ(cloned->get_layer_index(), 1);
    EXPECT_EQ(cloned->get_number_input_neurons(), num_inputs);
    EXPECT_EQ(cloned->get_number_output_neurons(), num_outputs);
    EXPECT_EQ(cloned->get_layer_architecture(), Layer::Architecture::FF);
    EXPECT_EQ(cloned->get_pre_activation_multiplier(), 1);
}

TEST_F(FFLayerTest, ForwardFeedLinear) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 0.5, -0.2 });
    layer.set_b_values({ 0.1 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 1.0, 2.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    double expected = 1.0 * 0.5 + 2.0 * (-0.2) + 0.1;
    EXPECT_NEAR(batch_go[0].get_output(1, 0), expected, 1e-9);
}

TEST_F(FFLayerTest, ForwardFeedSigmoid) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::sigmoid, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 2.0 });
    layer.set_b_values({ -1.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 0.5 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    double z = 0.5 * 2.0 - 1.0; // 0.0
    double expected = 1.0 / (1.0 + std::exp(-z)); // sigmoid(0) = 0.5
    EXPECT_NEAR(batch_go[0].get_output(1, 0), expected, 1e-9);
}

TEST_F(FFLayerTest, ForwardFeedReLU) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0, 1.0, -1.0, 1.0 }); // W[in][out]: W[0][0]=1, W[0][1]=1, W[1][0]=-1, W[1][1]=1
    layer.set_b_values({ 0.0, 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 1.0, 2.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    // z0 = 1*1 + 2*(-1) = -1 -> relu(-1) = 0
    // z1 = 1*1 + 2*1 = 3  -> relu(3) = 3
    EXPECT_NEAR(batch_go[0].get_output(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(batch_go[0].get_output(1, 1), 3.0, 1e-9);
}

TEST_F(FFLayerTest, ForwardFeedTanh) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 0.5 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    double expected = std::tanh(0.5);
    EXPECT_NEAR(batch_go[0].get_output(1, 0), expected, 1e-9);
}

TEST_F(FFLayerTest, ForwardFeedSoftmax) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::softmax, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 }); // Identity
    layer.set_b_values({ 0.0, 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 1.0, 2.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    // z = [1.0, 2.0]
    // softmax([1.0, 2.0]) = [exp(1)/(exp(1)+exp(2)), exp(2)/(exp(1)+exp(2))]
    double sum = std::exp(1.0) + std::exp(2.0);
    EXPECT_NEAR(batch_go[0].get_output(1, 0), std::exp(1.0) / sum, 1e-9);
    EXPECT_NEAR(batch_go[0].get_output(1, 1), std::exp(2.0) / sum, 1e-9);
}

TEST_F(FFLayerTest, ForwardFeedSequential) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 0.5, -0.2 });
    layer.set_b_values({ 0.1 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2);

    batch_go[0].set_rnn_outputs(0, { 1.0, 2.0, 0.5, 0.5 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& rnn_out = batch_go[0].get_rnn_outputs(1);
    ASSERT_EQ(rnn_out.size(), 2);
    EXPECT_NEAR(rnn_out[0], 0.2, 1e-9);
    EXPECT_NEAR(rnn_out[1], 0.25, 1e-9);
}

TEST_F(FFLayerTest, AllActivationTypes) {
    std::vector<activation::method> methods = {
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
        FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(m, 0.1), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
        layer.set_w_values({ 1.0 });
        layer.set_b_values({ 0.0 });

        auto batch_go = create_batch_gradients_and_outputs(topology, 1);
        auto batch_hs = create_batch_hidden_states(topology, 1, 1);
        batch_go[0].set_outputs(0, { 0.5 });

        // Just ensure it doesn't crash and produces a finite number
        EXPECT_NO_THROW(layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false));
        double out = batch_go[0].get_output(1, 0);
        EXPECT_TRUE(std::isfinite(out));
    }
}

TEST_F(FFLayerTest, CalculateHiddenGradients) {
    // Layer 1: 2 inputs, 2 outputs
    // Layer 2: 2 inputs, 1 output (next layer)
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    unsigned next_outputs = 1;

    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
    FFLayer next_layer(2, num_outputs, next_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 }); // Identity for simplicity
    layer.set_b_values({ 0.0, 0.0 });

    next_layer.set_w_values({ 0.5, 0.8 }); // W_next = [0.5, 0.8]
    next_layer.set_b_values({ 0.0 });

    std::vector<unsigned> topology = { num_inputs, num_outputs, next_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    // Set hidden states for layer 1 (needed for derivative)
    // For linear, derivative is 1.0 regardless of state.
    batch_hs[0].at(1, 0).set_pre_activation_sum(0, 0.5);
    batch_hs[0].at(1, 0).set_pre_activation_sum(1, 0.5);

    // Next gradients: [1.0]
    std::vector<std::vector<double>> batch_next_grads = { { 1.0 } };

    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

    // Expected gradient for neuron j in layer 1: sum_k(grad_k * W_jk) * act_deriv(z_j)
    // grad_0 = 1.0, W_00 = 0.5, W_10 = 0.8
    // neuron 0: 1.0 * 0.5 * 1.0 = 0.5
    // neuron 1: 1.0 * 0.8 * 1.0 = 0.8
    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], 0.5, 1e-9);
    EXPECT_NEAR(grads[1], 0.8, 1e-9);
}

TEST_F(FFLayerTest, CalculateAndStoreGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    // Input x = 2.0
    batch_go[0].set_outputs(0, { 2.0 });
    // Gradient of loss w.r.t output: dL/da = 0.5
    // For linear, dL/dz = dL/da * 1.0 = 0.5
    batch_go[0].set_gradients(1, { 0.5 });

    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 0);

    // dL/dW = dL/dz * x = 0.5 * 2.0 = 1.0
    // dL/db = dL/dz * 1.0 = 0.5
    EXPECT_NEAR(layer.get_w_grads()[0], 1.0, 1e-9);
    EXPECT_NEAR(layer.get_b_grads()[0], 0.5, 1e-9);
}

TEST_F(FFLayerTest, ApplyStoredGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    // Use SGD-like optimizer (OptimiserType::None or similar)
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0 });
    layer.set_b_values({ 0.5 });
    
    // Manually set gradients
    std::vector<double> w_grads = { 0.1 };
    std::vector<double> b_grads = { 0.05 };
    layer.set_w_grads(w_grads);
    layer.set_b_grads(b_grads);

    layer.apply_stored_gradients(0.1, 1.0); // learning_rate = 0.1, clipping_scale = 1.0 (no clipping)

    // New W = 1.0 - 0.1 * 0.1 = 0.99
    // New b = 0.5 - 0.1 * 0.05 = 0.495
    EXPECT_NEAR(layer.get_w_values()[0], 0.99, 1e-9);
    EXPECT_NEAR(layer.get_b_values()[0], 0.495, 1e-9);
}
