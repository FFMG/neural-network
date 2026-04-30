#include <gtest/gtest.h>
#include "../src/neuralnetwork/ffoutputlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>

using namespace test_helper;

class FFOutputLayerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FFOutputLayerTest, ConstructorAndClone) {
    unsigned num_inputs = 4;
    unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::Adam, 0.9)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

    EXPECT_EQ(layer.get_layer_index(), 1);
    EXPECT_EQ(layer.get_number_input_neurons(), num_inputs);
    EXPECT_EQ(layer.get_number_output_neurons(), num_outputs);
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::FF);
    EXPECT_EQ(layer.get_layer_role(), Layer::Role::Output);
    EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);

    std::unique_ptr<Layer> cloned(layer.clone());
    EXPECT_EQ(cloned->get_layer_index(), 1);
    EXPECT_EQ(cloned->get_number_input_neurons(), num_inputs);
    EXPECT_EQ(cloned->get_number_output_neurons(), num_outputs);
    EXPECT_EQ(cloned->get_pre_activation_multiplier(), 1);
}

TEST_F(FFOutputLayerTest, CalculateOutputGradientsMSE) {
    // MSE with Linear activation: dL/dz = (y - t) / N
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    // Given outputs (y) = [0.8, 0.4]
    // Target outputs (t) = [1.0, 0.0]
    batch_hs[0].at(1, 0).set_hidden_state_values({ 0.8, 0.4 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.8, 0.4 }); // linear

    std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    // Expected deltas:
    // delta_0 = (0.8 - 1.0) / 2 = -0.1
    // delta_1 = (0.4 - 0.0) / 2 = 0.2
    // Since linear activation derivative is 1.0, grad = delta
    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], -0.1, 1e-9);
    EXPECT_NEAR(grads[1], 0.2, 1e-9);
}

TEST_F(FFOutputLayerTest, CalculateOutputGradientsBCE) {
    // BCE with Sigmoid activation: dL/dz = (y - t) / N (Simplified)
    unsigned num_inputs = 2;
    unsigned num_outputs = 1;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::bce_loss, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    // y = sigmoid(z). Let z = 0, so y = 0.5
    batch_hs[0].at(1, 0).set_hidden_state_values({ 0.5 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.0 });

    std::vector<std::vector<double>> targets = { { 1.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    // Expected delta: (0.5 - 1.0) / 1 = -0.5
    // BCE + Sigmoid usually cancels the derivative in the code if 'is_not_using_activation_derivative' is true.
    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], -0.5, 1e-9);
}

TEST_F(FFOutputLayerTest, CalculateOutputGradientsCE) {
    // Cross-Entropy with Softmax: dL/dz = (y - t) (Note: usually no 1/N in CE implementation in this codebase)
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::softmax, 0.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    // y = softmax(z). Let z = [0, 0], so y = [0.5, 0.5]
    batch_hs[0].at(1, 0).set_hidden_state_values({ 0.5, 0.5 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.0, 0.0 });

    std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    // Expected grads: 
    // grad_0 = 0.5 - 1.0 = -0.5
    // grad_1 = 0.5 - 0.0 = 0.5
    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], -0.5, 1e-9);
    EXPECT_NEAR(grads[1], 0.5, 1e-9);
}

TEST_F(FFOutputLayerTest, MultiHeadOutput) {
    // Test multiple output heads with different activations and optimizers
    unsigned num_inputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0),
        OutputLayerDetails(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::bce_loss, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    unsigned num_outputs = 2;
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    // Head 0: y=0.8, t=1.0 -> delta = (0.8-1.0)/1 = -0.2
    // Head 1: y=0.5, t=1.0 -> delta = (0.5-1.0)/1 = -0.5
    batch_hs[0].at(1, 0).set_hidden_state_values({ 0.8, 0.5 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.8, 0.0 });

    std::vector<std::vector<double>> targets = { { 1.0, 1.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], -0.2, 1e-9);
    EXPECT_NEAR(grads[1], -0.5, 1e-9);
}

TEST_F(FFOutputLayerTest, CalculateOutputMetrics) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 1;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

    std::vector<std::vector<double>> predictions = { { 0.8 }, { 0.4 } };
    std::vector<std::vector<double>> targets = { { 1.0 }, { 0.0 } };

    auto metrics = layer.calculate_output_metrics({ ErrorCalculation::type::mse }, targets, predictions);

    // MSE = ( (0.8-1.0)^2 + (0.4-0.0)^2 ) / 2 = (0.04 + 0.16) / 2 = 0.1
    EXPECT_EQ(metrics.size(), 1); // 1 head
    EXPECT_EQ(metrics[0].size(), 1); // 1 error type
    EXPECT_NEAR(metrics[0][0].error(), 0.1, 1e-9);
}

TEST_F(FFOutputLayerTest, AllActivationTypes) {
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
        activation::method::elu,
        activation::method::softmax
    };

    unsigned num_inputs = 1;
    unsigned num_outputs = 1;

    for (auto m : methods) {
        ErrorCalculation::type err = (m == activation::method::softmax) ? ErrorCalculation::type::cross_entropy : ErrorCalculation::type::mse;
        std::vector<OutputLayerDetails> details = {
            OutputLayerDetails(num_outputs, activation(m, 0.1), err, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
        };
        FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
        
        std::vector<unsigned> topology = { num_inputs, num_outputs };
        auto batch_go = create_batch_gradients_and_outputs(topology, 1);
        auto batch_hs = create_batch_hidden_states(topology, 1, 1);
        
        batch_hs[0].at(1, 0).set_hidden_state_values({ 0.5 });
        batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.5 });
        
        std::vector<std::vector<double>> targets = { { 1.0 } };
        EXPECT_NO_THROW(layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1));
        
        double grad = batch_go[0].get_gradients(1)[0];
        EXPECT_TRUE(std::isfinite(grad));
    }
}

TEST_F(FFOutputLayerTest, GetMomentum) {
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.1),
        OutputLayerDetails(3, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.5)
    };
    FFOutputLayer layer(1, details, 1, 5, 1, true);

    EXPECT_DOUBLE_EQ(layer.get_momentum(0), 0.1);
    EXPECT_DOUBLE_EQ(layer.get_momentum(1), 0.1);
    EXPECT_DOUBLE_EQ(layer.get_momentum(2), 0.5);
    EXPECT_DOUBLE_EQ(layer.get_momentum(4), 0.5);
}

TEST_F(FFOutputLayerTest, ApplyStoredGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

    layer.set_w_values({ 1.0 });
    layer.set_w_grads({ 0.1 });
    layer.apply_stored_gradients(0.1, 1.0); // LR=0.1

    // 1.0 - 0.1 * 0.1 = 0.99
    EXPECT_NEAR(layer.get_w_values()[0], 0.99, 1e-9);
    // Grads should be cleared
    EXPECT_DOUBLE_EQ(layer.get_w_grads()[0], 0.0);
}

TEST_F(FFOutputLayerTest, ForwardFeed) {
    unsigned num_inputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0),
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    unsigned num_outputs = 2;
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

    // Identity weights, zero bias
    layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 });
    layer.set_b_values({ 0.0, 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 0.5, -0.2 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    // Expected outputs: [0.5, -0.2]
    EXPECT_NEAR(batch_go[0].get_output(1, 0), 0.5, 1e-9);
    EXPECT_NEAR(batch_go[0].get_output(1, 1), -0.2, 1e-9);
}
