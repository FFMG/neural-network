#include <gtest/gtest.h>
#include "../src/neuralnetwork/multioutputlayer.h"
#include "../src/neuralnetwork/multioutputlayerdetails.h"
#include "../src/neuralnetwork/layerdetails.h"
#include "../src/neuralnetwork/outputlayerdetails.h"
#include "test_helper.h"
#include <vector>
#include <cmath>

using namespace test_helper;

class MultiOutputLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(MultiOutputLayerTest, ConstructionAndTopology) {
    // Branch 1: 2 hidden neurons, 2 output neurons
    std::vector<LayerDetails> h1 = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails o1(2, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod1(h1, o1);

    // Branch 2: 3 output neurons (no hidden)
    std::vector<LayerDetails> h2 = {};
    OutputLayerDetails o2(3, activation(activation::method::sigmoid, 1.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod2(h2, o2);

    std::vector<MultiOutputLayerDetails> details = { mod1, mod2 };
    
    MultiOutputLayer layer(5, 4, 5, details, 1, true);

    EXPECT_EQ(layer.get_layer_index(), 5);
    EXPECT_EQ(layer.get_number_input_neurons(), 4);
    EXPECT_EQ(layer.get_number_output_neurons(), 5);
    EXPECT_EQ(layer.get_branches().size(), 2);
    
    EXPECT_EQ(layer.get_branches()[0].layers.size(), 2);
    EXPECT_EQ(layer.get_branches()[1].layers.size(), 1);
}

TEST_F(MultiOutputLayerTest, ForwardFeedMathematicalVerification) {
    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oA(2, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = {};
    OutputLayerDetails oB(1, activation(activation::method::sigmoid, 1.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    std::vector<MultiOutputLayerDetails> details = { modA, modB };
    MultiOutputLayer layer(1, 2, 3, details, 1, true);

    auto& branches = layer.get_mutable_branches();
    
    branches[0].layers[0]->set_w_values({ 0.2, 0.4, 0.6, 0.8 });
    branches[0].layers[0]->set_b_values({ 0.1, -0.1 });
    branches[0].layers[1]->set_w_values({ 0.5, 0.6, 0.7, 0.8 });
    branches[0].layers[1]->set_b_values({ 0.0, 0.0 });
    branches[1].layers[0]->set_w_values({ 0.9, -0.1 });
    branches[1].layers[0]->set_b_values({ 0.0 });

    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    
    batch_go[0].set_outputs(0, { 0.5, -0.2 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], 0.498000, 1e-6);
    EXPECT_NEAR(outputs[1], 0.502000, 1e-6);
    EXPECT_NEAR(outputs[2], 0.615383, 1e-6);
}

TEST_F(MultiOutputLayerTest, OutputGradientsMathematicalVerification) {
    EvaluationConfig clean_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12);

    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oA(2, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = {};
    OutputLayerDetails oB(1, activation(activation::method::sigmoid, 1.0, 1.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    MultiOutputLayer layer(1, 2, 3, { modA, modB }, 1, true);
    auto& branches = layer.get_mutable_branches();
    branches[0].layers[0]->set_w_values({ 0.2, 0.4, 0.6, 0.8 });
    branches[0].layers[0]->set_b_values({ 0.1, -0.1 });
    branches[0].layers[1]->set_w_values({ 0.5, 0.6, 0.7, 0.8 });
    branches[0].layers[1]->set_b_values({ 0.0, 0.0 });
    branches[1].layers[0]->set_w_values({ 0.9, -0.1 });
    branches[1].layers[0]->set_b_values({ 0.0, 0.0 });

    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 0.5, -0.2 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    std::vector<std::vector<double>> targets = { { 1.0, 0.0, 0.8 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    const auto& bA_grads = branches[0].gradients_and_outputs[0].get_gradients(branches[0].layers[1]->get_layer_index());
    EXPECT_NEAR(bA_grads[0], -0.502000, 1e-4);
    EXPECT_NEAR(bA_grads[1], 0.502000, 1e-4);

    const auto& bB_grads = branches[1].gradients_and_outputs[0].get_gradients(branches[1].layers[0]->get_layer_index());
    EXPECT_NEAR(bB_grads[0], -0.043696, 1e-4);
}

TEST_F(MultiOutputLayerTest, BackpropAndTrunkGradients) {
    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oA(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = {};
    OutputLayerDetails oB(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    MultiOutputLayer layer(1, 2, 3, { modA, modB }, 1, true);
    auto& branches = layer.get_mutable_branches();

    branches[0].layers[0]->set_w_values({ 0.1, 0.2, 0.3, 0.4 });
    branches[0].layers[0]->set_b_values({ 0, 0 });
    branches[0].layers[1]->set_w_values({ 0.5, 0.6, 0.7, 0.8 });
    branches[0].layers[1]->set_b_values({ 0, 0 });

    branches[1].layers[0]->set_w_values({ 0.9, -0.1 });
    branches[1].layers[0]->set_b_values({ 0 });

    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 1.0, 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    std::vector<std::vector<double>> targets = { { 1.0, 1.0, 1.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    layer.backprop_branches(1, 0);

    auto trunk_grads = layer.get_trunk_gradients(1);
    EXPECT_NEAR(trunk_grads[0][0], -0.2469, 1e-6);
    EXPECT_NEAR(trunk_grads[0][1], -0.1317, 1e-6);
}

TEST_F(MultiOutputLayerTest, TemperatureMethods) {
    std::vector<LayerDetails> h = {};
    OutputLayerDetails o1(2, activation(activation::method::softmax, 0.0, 1.5), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::softmax, 0.0, 2.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod1(h, o1);
    MultiOutputLayerDetails mod2(h, o2);
    
    MultiOutputLayer layer(0, 1, 3, { mod1, mod2 }, 1, true);

    EXPECT_NEAR(layer.get_temperature(0), 1.5, 1e-9);
    EXPECT_NEAR(layer.get_temperature(1), 2.0, 1e-9);
    
    layer.set_inference_temperature(0, 0.5);
    EXPECT_NEAR(layer.get_inference_temperature(0), 0.5, 1e-9);
}

TEST_F(MultiOutputLayerTest, CalculateOutputMetrics) {
    OutputLayerDetails o1(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod1({}, o1);
    MultiOutputLayerDetails mod2({}, o2);
    
    MultiOutputLayer layer(1, 1, 2, { mod1, mod2 }, 1, true);
    
    std::vector<std::vector<double>> predictions = { { 0.8, 0.4 } };
    std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };
    
    auto metrics = layer.calculate_output_metrics({ ErrorCalculation::type::mse }, targets, predictions);
    
    EXPECT_EQ(metrics.size(), 2);
    EXPECT_NEAR((double)metrics[0][0].error(), 0.04, 1e-6);
    EXPECT_NEAR((double)metrics[1][0].error(), 0.16, 1e-6);
}

TEST_F(MultiOutputLayerTest, ActivationBranches) {
    OutputLayerDetails o1(1, activation(activation::method::tanh, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::elu, 1.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o3(1, activation(activation::method::swish, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    
    MultiOutputLayerDetails mod1({}, o1);
    MultiOutputLayerDetails mod2({}, o2);
    MultiOutputLayerDetails mod3({}, o3);
    
    MultiOutputLayer layer(1, 1, 3, { mod1, mod2, mod3 }, 1, true);
    auto& branches = layer.get_mutable_branches();
    branches[0].layers[0]->set_w_values({ 1.0 }); branches[0].layers[0]->set_b_values({ 0.0 });
    branches[1].layers[0]->set_w_values({ 1.0 }); branches[1].layers[0]->set_b_values({ 0.0 });
    branches[2].layers[0]->set_w_values({ 1.0 }); branches[2].layers[0]->set_b_values({ 0.0 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { -1.0 });
    
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
    
    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], std::tanh(-1.0), 1e-6);
    EXPECT_NEAR(outputs[1], 1.0 * (std::exp(-1.0) - 1.0), 1e-6);
    EXPECT_NEAR(outputs[2], -1.0 * (1.0 / (1.0 + std::exp(1.0))), 1e-6);
}

TEST_F(MultiOutputLayerTest, MultiTimeStepForwardFeed) {
    OutputLayerDetails o1(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayer layer(1, 1, 2, { MultiOutputLayerDetails({}, o1), MultiOutputLayerDetails({}, o2) }, 1, true);
    auto& branches = layer.get_mutable_branches();
    branches[0].layers[0]->set_w_values({ 2.0 }); branches[0].layers[0]->set_b_values({ 0.0 });
    branches[1].layers[0]->set_w_values({ 3.0 }); branches[1].layers[0]->set_b_values({ 0.0 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 2 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2);
    
    batch_go[0].set_rnn_outputs(0, { 0.5, 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto rnn_out = batch_go[0].get_rnn_outputs(1);
    EXPECT_EQ(rnn_out.size(), 4);
    EXPECT_NEAR(rnn_out[0], 1.0, 1e-6);
    EXPECT_NEAR(rnn_out[1], 1.5, 1e-6);
    EXPECT_NEAR(rnn_out[2], 2.0, 1e-6);
    EXPECT_NEAR(rnn_out[3], 3.0, 1e-6);
    
    const auto std_out = batch_go[0].get_outputs(1);
    EXPECT_NEAR(std_out[0], 2.0, 1e-6);
    EXPECT_NEAR(std_out[1], 3.0, 1e-6);
}

TEST_F(MultiOutputLayerTest, MultiInputProxyLayerTest) {
    MultiInputProxyLayer proxy(2);
    EXPECT_EQ(proxy.get_number_neurons(), 2);
    EXPECT_EQ(proxy.get_layer_architecture(), Layer::Architecture::MultiOutput);

    MockLayer next_layer(1, 3);
    next_layer.set_w_values({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
    
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    std::vector<std::vector<double>> batch_next_grads = { { 1.0, 0.5, 0.0, 0.0, 0.5, 1.0 } };
    
    proxy.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, {}, 1, 0);

    const auto proxy_rnn_grads = batch_go[0].get_rnn_gradients(0);
    EXPECT_EQ(proxy_rnn_grads.size(), 4);
    EXPECT_NEAR(proxy_rnn_grads[0], 0.2, 1e-6);
    EXPECT_NEAR(proxy_rnn_grads[1], 0.65, 1e-6);
    EXPECT_NEAR(proxy_rnn_grads[2], 0.4, 1e-6);
    EXPECT_NEAR(proxy_rnn_grads[3], 0.85, 1e-6);
    
    const auto proxy_std_grads = batch_go[0].get_gradients(0);
    EXPECT_NEAR(proxy_std_grads[0], 0.6, 1e-6);
    EXPECT_NEAR(proxy_std_grads[1], 1.5, 1e-6);
}

TEST_F(MultiOutputLayerTest, ComplexArchitectureVerification) {
    EvaluationConfig clean_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12);

    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oA(1, activation(activation::method::sigmoid, 1.0, 1.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = { LayerDetails(Layer::Architecture::FF, 1, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oB(2, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    MultiOutputLayer layer(1, 2, 3, { modA, modB }, 1, true);
    auto& branches = layer.get_mutable_branches();

    branches[0].layers[0]->set_w_values({ 0.2, 0.4, 0.6, 0.8 });
    branches[0].layers[0]->set_b_values({ 0.1, -0.1 });
    branches[0].layers[1]->set_w_values({ 0.5, 0.6 });
    branches[0].layers[1]->set_b_values({ 0.2 });

    branches[1].layers[0]->set_w_values({ 0.7, 0.8 });
    branches[1].layers[0]->set_b_values({ 0.3 });
    branches[1].layers[1]->set_w_values({ 0.9, 1.0 });
    branches[1].layers[1]->set_b_values({ -0.2, 0.2 });

    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 0.5, -0.5 });
    
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], 0.549834, 1e-5);
    EXPECT_NEAR(outputs[1], 0.395442, 1e-5);
    EXPECT_NEAR(outputs[2], 0.604558, 1e-5);

    std::vector<std::vector<double>> targets = { { 1.0, 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);
    
    layer.backprop_branches(1, 0);
    
    auto trunk_grads = layer.get_trunk_gradients(1);
    EXPECT_NEAR(trunk_grads[0][0], 0.03978, 1e-4);
    EXPECT_NEAR(trunk_grads[0][1], 0.04546, 1e-4);
}

TEST_F(MultiOutputLayerTest, DropoutStatisticalVerification) {
    const unsigned num_inputs = 100;
    const unsigned num_outputs = 1000;
    const double dropout_rate = 0.5;

    std::vector<LayerDetails> h = { LayerDetails(Layer::Architecture::FF, num_outputs, activation(activation::method::linear, 0.0), dropout_rate, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails o(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod(h, o);

    MultiOutputLayer layer(1, num_inputs, num_outputs, { mod }, 1, true);
    
    // Set weights to 1.0 so output = input * 1.0 (before dropout)
    auto& branch_h = *layer.get_mutable_branches()[0].layers[0];
    branch_h.set_w_values(std::vector<double>(num_inputs * num_outputs, 1.0));
    branch_h.set_b_values(std::vector<double>(num_outputs, 0.0));
    
    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs }; // Trunk topology
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    
    // Input 1.0
    batch_go[0].set_outputs(0, std::vector<double>(num_inputs, 1.0));

    // Training mode
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Get the hidden layer output from the branch
    const auto& branch_hidden_out = layer.get_branches()[0].gradients_and_outputs[0].get_outputs(1);
    
    unsigned dropped = 0;
    double sum = 0.0;
    double expected_scaled_value = num_inputs * (1.0 / (1.0 - dropout_rate));

    for (double val : branch_hidden_out) {
        if (val == 0.0) dropped++;
        else sum += val;
    }

    double actual_rate = static_cast<double>(dropped) / num_outputs;
    EXPECT_NEAR(actual_rate, dropout_rate, 0.05); // 5% tolerance for 10k samples
    
    if (num_outputs - dropped > 0) {
        double avg_active = sum / (num_outputs - dropped);
        EXPECT_NEAR(avg_active, expected_scaled_value, 1e-7);
    }
}

TEST_F(MultiOutputLayerTest, DropoutNotInference) {
    const unsigned num_inputs = 10;
    const unsigned num_outputs = 100;
    const double dropout_rate = 0.5;

    std::vector<LayerDetails> h = { LayerDetails(Layer::Architecture::FF, num_outputs, activation(activation::method::linear, 0.0), dropout_rate, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails o(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod(h, o);

    MultiOutputLayer layer(1, num_inputs, num_outputs, { mod }, 1, true);
    layer.get_mutable_branches()[0].layers[0]->set_w_values(std::vector<double>(num_inputs * num_outputs, 1.0));
    layer.get_mutable_branches()[0].layers[0]->set_b_values(std::vector<double>(num_outputs, 0.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, std::vector<double>(num_inputs, 1.0));

    // Inference mode (is_training = false)
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& branch_hidden_out = layer.get_branches()[0].gradients_and_outputs[0].get_outputs(1);
    for (double val : branch_hidden_out) {
        EXPECT_NEAR(val, (double)num_inputs, 1e-7); // No dropout, no scaling
    }
}
