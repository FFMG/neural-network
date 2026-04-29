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
    OutputLayerDetails o1(2, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod1(h1, o1);

    // Branch 2: 3 output neurons (no hidden)
    std::vector<LayerDetails> h2 = {};
    OutputLayerDetails o2(3, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod2(h2, o2);

    std::vector<MultiOutputLayerDetails> details = { mod1, mod2 };
    
    // MultiOutputLayer(index, num_inputs, num_outputs, details, threads, has_bias)
    MultiOutputLayer layer(5, 4, 5, details, 1, true);

    EXPECT_EQ(layer.get_layer_index(), 5);
    EXPECT_EQ(layer.get_number_input_neurons(), 4);
    EXPECT_EQ(layer.get_number_output_neurons(), 5);
    EXPECT_EQ(layer.get_branches().size(), 2);
    
    // Branch 1 should have 2 layers (hidden + output)
    EXPECT_EQ(layer.get_branches()[0].layers.size(), 2);
    // Branch 2 should have 1 layer (output)
    EXPECT_EQ(layer.get_branches()[1].layers.size(), 1);
}

TEST_F(MultiOutputLayerTest, ForwardFeedMathematicalVerification) {
    // Setup a simple MultiOutputLayer
    // Input: 2 neurons
    // Branch A: 1 hidden layer (2 neurons, ReLU), 1 output layer (2 neurons, Softmax)
    // Branch B: 1 output layer (1 neuron, Sigmoid)
    
    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oA(2, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = {};
    OutputLayerDetails oB(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    std::vector<MultiOutputLayerDetails> details = { modA, modB };
    MultiOutputLayer layer(1, 2, 3, details, 1, true);

    // Access branches to set specific weights
    auto& branches = layer.get_mutable_branches();
    
    // Branch A Hidden Layer (ReLU)
    // W_ah = [[0.1, 0.2], [0.3, 0.4]]
    // B_ah = [0.0, 0.0]
    branches[0].layers[0]->set_w_values({ 0.1, 0.2, 0.3, 0.4 });
    branches[0].layers[0]->set_b_values({ 0.0, 0.0 });

    // Branch A Output Layer (Softmax)
    // W_ao = [[0.5, 0.6], [0.7, 0.8]]
    // B_ao = [0.0, 0.0]
    branches[0].layers[1]->set_w_values({ 0.5, 0.6, 0.7, 0.8 });
    branches[0].layers[1]->set_b_values({ 0.0, 0.0 });

    // Branch B Output Layer (Sigmoid)
    // W_bo = [[0.9], [-0.1]]
    // B_bo = [0.0]
    branches[1].layers[0]->set_w_values({ 0.9, -0.1 });
    branches[1].layers[0]->set_b_values({ 0.0 });

    // Previous layer for proxy
    MockLayer prev_layer(0, 2);
    
    // Batch size 1
    std::vector<unsigned> topology = { 2, 3 }; // Trunk size 2, MultiOutput size 3
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    
    // Set input values to previous layer output
    batch_go[0].set_outputs(0, { 0.5, -0.2 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Mathematical verification:
    // Input: [0.5, -0.2]
    // Branch A Hidden:
    // Z_ah = [0.5*0.1 - 0.2*0.3, 0.5*0.2 - 0.2*0.4] = [-0.01, 0.02]
    // A_ah = [0.0, 0.02]
    // Branch A Output:
    // Z_ao = [0.0*0.5 + 0.02*0.7, 0.0*0.6 + 0.02*0.8] = [0.014, 0.016]
    // Softmax([0.014, 0.016]) -> exp(0.014)=1.014098, exp(0.016)=1.016128, sum=2.030226
    // A_ao = [0.4994999, 0.5005001]
    
    // Branch B Output:
    // Z_bo = [0.5*0.9 - 0.2*-0.1] = [0.47]
    // A_bo = [sigmoid(0.47)] = [0.6153829]

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_EQ(outputs.size(), 3);
    EXPECT_NEAR(outputs[0], 0.4994999, 1e-6);
    EXPECT_NEAR(outputs[1], 0.5005001, 1e-6);
    EXPECT_NEAR(outputs[2], 0.6153829, 1e-6);
}

TEST_F(MultiOutputLayerTest, OutputGradientsMathematicalVerification) {
    // Re-use same setup as Forward Feed
    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oA(2, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = {};
    OutputLayerDetails oB(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    std::vector<MultiOutputLayerDetails> details = { modA, modB };
    MultiOutputLayer layer(1, 2, 3, details, 1, true);

    auto& branches = layer.get_mutable_branches();
    branches[0].layers[1]->set_w_values({ 0.5, 0.6, 0.7, 0.8 }); // Needed for topology init if any
    
    // We need to simulate a forward feed to initialize buffers
    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 0.5, -0.2 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Targets: Branch A: [1.0, 0.0], Branch B: [0.8]
    std::vector<std::vector<double>> targets = { { 1.0, 0.0, 0.8 } };
    
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    // Branch A Output (Softmax + CE): grad = y - t
    // From forward feed: A_ao = [0.4994999, 0.5005001]
    // Grad_ao = [0.4994999 - 1.0, 0.5005001 - 0.0] = [-0.5005001, 0.5005001]
    
    // Branch B Output (Sigmoid + MSE): grad = (y - t) / N
    // N=1 for this head.
    // From forward feed: A_bo = [0.6153829]
    // Grad_bo = [0.6153829 - 0.8] = [-0.1846171]

    const auto& bA_grads = branches[0].gradients_and_outputs[0].get_gradients(branches[0].layers[1]->get_layer_index());
    EXPECT_NEAR(bA_grads[0], -0.5005001, 1e-6);
    EXPECT_NEAR(bA_grads[1], 0.5005001, 1e-6);

    const auto& bB_grads = branches[1].gradients_and_outputs[0].get_gradients(branches[1].layers[0]->get_layer_index());
    EXPECT_NEAR(bB_grads[0], -0.1846171, 1e-6);
}

TEST_F(MultiOutputLayerTest, BackpropAndTrunkGradients) {
    // Re-use same setup
    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) }; // Linear for simpler math
    OutputLayerDetails oA(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = {};
    OutputLayerDetails oB(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    std::vector<MultiOutputLayerDetails> details = { modA, modB };
    MultiOutputLayer layer(1, 2, 3, details, 1, true);
    auto& branches = layer.get_mutable_branches();

    // Branch A Weights
    // Hidden: W_ah = [[0.1, 0.2], [0.3, 0.4]], B_ah = 0
    // Output: W_ao = [[0.5, 0.6], [0.7, 0.8]], B_ao = 0
    branches[0].layers[0]->set_w_values({ 0.1, 0.2, 0.3, 0.4 });
    branches[0].layers[0]->set_b_values({ 0, 0 });
    branches[0].layers[1]->set_w_values({ 0.5, 0.6, 0.7, 0.8 });
    branches[0].layers[1]->set_b_values({ 0, 0 });

    // Branch B Weights
    // Output: W_bo = [[0.9], [-0.1]], B_bo = 0
    branches[1].layers[0]->set_w_values({ 0.9, -0.1 });
    branches[1].layers[0]->set_b_values({ 0 });

    // Forward
    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 1.0, 1.0 }); // Input [1, 1]
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Branch A outputs:
    // Hidden: [1*0.1 + 1*0.3, 1*0.2 + 1*0.4] = [0.4, 0.6]
    // Output: [0.4*0.5 + 0.6*0.7, 0.4*0.6 + 0.6*0.8] = [0.2 + 0.42, 0.24 + 0.48] = [0.62, 0.72]
    // Branch B outputs:
    // Output: [1*0.9 + 1*-0.1] = [0.8]

    // Set Targets
    std::vector<std::vector<double>> targets = { { 1.0, 1.0, 1.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    // Branch A Output Gradients (MSE, N=2):
    // G_ao = [(0.62 - 1.0)/2, (0.72 - 1.0)/2] = [-0.19, -0.14]
    // Branch B Output Gradients (MSE, N=1):
    // G_bo = [(0.8 - 1.0)/1] = [-0.2]

    // Backprop branches
    layer.backprop_branches(1, 0);

    // Branch A Hidden Gradients:
    // G_ah = [ G_ao[0]*W_ao[0,0] + G_ao[1]*W_ao[0,1], G_ao[0]*W_ao[1,0] + G_ao[1]*W_ao[1,1] ]
    //      = [ -0.19*0.5 + -0.14*0.6, -0.19*0.7 + -0.14*0.8 ]
    //      = [ -0.095 - 0.084, -0.133 - 0.112 ]
    //      = [ -0.179, -0.245 ]
    
    // Trunk Gradients from Branch A:
    // G_tr_A = [ G_ah[0]*W_ah[0,0] + G_ah[1]*W_ah[0,1], G_ah[0]*W_ah[1,0] + G_ah[1]*W_ah[1,1] ]
    //        = [ -0.179*0.1 + -0.245*0.2, -0.179*0.3 + -0.245*0.4 ]
    //        = [ -0.0179 - 0.049, -0.0537 - 0.098 ]
    //        = [ -0.0669, -0.1517 ]

    // Trunk Gradients from Branch B:
    // G_tr_B = [ G_bo[0]*W_bo[0], G_bo[0]*W_bo[1] ]
    //        = [ -0.2*0.9, -0.2*-0.1 ]
    //        = [ -0.18, 0.02 ]

    // Total Trunk Gradients:
    // G_tr = G_tr_A + G_tr_B = [ -0.0669 - 0.18, -0.1517 + 0.02 ] = [ -0.2469, -0.1317 ]

    auto trunk_grads = layer.get_trunk_gradients(1);
    EXPECT_NEAR(trunk_grads[0][0], -0.2469, 1e-6);
    EXPECT_NEAR(trunk_grads[0][1], -0.1317, 1e-6);
}

TEST_F(MultiOutputLayerTest, TemperatureMethods) {
    std::vector<LayerDetails> h = {};
    OutputLayerDetails o1(2, activation(activation::method::softmax, 1.5), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::softmax, 2.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod1(h, o1);
    MultiOutputLayerDetails mod2(h, o2);
    
    MultiOutputLayer layer(0, 1, 3, { mod1, mod2 }, 1, true);

    EXPECT_NEAR(layer.get_temperature(0), 1.5, 1e-9);
    EXPECT_NEAR(layer.get_temperature(1), 2.0, 1e-9);
    
    layer.set_inference_temperature(0, 0.5);
    EXPECT_NEAR(layer.get_inference_temperature(0), 0.5, 1e-9);
}

TEST_F(MultiOutputLayerTest, CalculateOutputMetrics) {
    // 2 branches, 1 output each
    OutputLayerDetails o1(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails mod1({}, o1);
    MultiOutputLayerDetails mod2({}, o2);
    
    MultiOutputLayer layer(1, 1, 2, { mod1, mod2 }, 1, true);
    
    std::vector<std::vector<double>> predictions = { { 0.8, 0.4 } };
    std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };
    
    auto metrics = layer.calculate_output_metrics({ ErrorCalculation::type::mse }, targets, predictions);
    
    // metrics[output_layer_index][error_type_index]
    EXPECT_EQ(metrics.size(), 2);
    // Head 1: MSE = (0.8 - 1.0)^2 = 0.04
    EXPECT_NEAR(metrics[0][0].error(), 0.04, 1e-6);
    // Head 2: MSE = (0.4 - 0.0)^2 = 0.16
    EXPECT_NEAR(metrics[1][0].error(), 0.16, 1e-6);
}

TEST_F(MultiOutputLayerTest, ActivationBranches) {
    // Verify that different activation types in different branches work
    OutputLayerDetails o1(1, activation(activation::method::tanh, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::elu, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o3(1, activation(activation::method::swish, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    
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
    // tanh(-1.0) approx -0.76159
    EXPECT_NEAR(outputs[0], std::tanh(-1.0), 1e-6);
    // elu(-1.0) = alpha * (exp(-1.0) - 1). Default alpha=1.0? code says 1.0
    EXPECT_NEAR(outputs[1], 1.0 * (std::exp(-1.0) - 1.0), 1e-6);
    // swish(-1.0) = -1.0 * sigmoid(-1.0) = -1.0 * (1 / (1 + exp(1.0))) approx -0.2689
    EXPECT_NEAR(outputs[2], -1.0 * (1.0 / (1.0 + std::exp(1.0))), 1e-6);
}

TEST_F(MultiOutputLayerTest, MultiTimeStepForwardFeed) {
    // 1 input neuron, 2 output neurons (2 branches)
    // 2 time steps
    OutputLayerDetails o1(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayer layer(1, 1, 2, { MultiOutputLayerDetails({}, o1), MultiOutputLayerDetails({}, o2) }, 1, true);
    auto& branches = layer.get_mutable_branches();
    branches[0].layers[0]->set_w_values({ 2.0 }); branches[0].layers[0]->set_b_values({ 0.0 }); // Head 1: y = 2x
    branches[1].layers[0]->set_w_values({ 3.0 }); branches[1].layers[0]->set_b_values({ 0.0 }); // Head 2: y = 3x

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 2 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2); // 2 time steps
    
    // Set RNN outputs for previous layer: [x_t0, x_t1] = [0.5, 1.0]
    batch_go[0].set_rnn_outputs(0, { 0.5, 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Expected Output RNN Sequence: [y1_t0, y2_t0, y1_t1, y2_t1]
    // Head 1: 2 * 0.5 = 1.0 (t0), 2 * 1.0 = 2.0 (t1)
    // Head 2: 3 * 0.5 = 1.5 (t0), 3 * 1.0 = 3.0 (t1)
    // Concatenated: [1.0, 1.5, 2.0, 3.0]
    const auto rnn_out = batch_go[0].get_rnn_outputs(1);
    EXPECT_EQ(rnn_out.size(), 4);
    EXPECT_NEAR(rnn_out[0], 1.0, 1e-6);
    EXPECT_NEAR(rnn_out[1], 1.5, 1e-6);
    EXPECT_NEAR(rnn_out[2], 2.0, 1e-6);
    EXPECT_NEAR(rnn_out[3], 3.0, 1e-6);
    
    // Last step standard output: [2.0, 3.0]
    const auto std_out = batch_go[0].get_outputs(1);
    EXPECT_NEAR(std_out[0], 2.0, 1e-6);
    EXPECT_NEAR(std_out[1], 3.0, 1e-6);
}

TEST_F(MultiOutputLayerTest, MultiInputProxyLayerTest) {
    // MultiInputProxyLayer is used internally but we can test it directly
    MultiInputProxyLayer proxy(2);
    EXPECT_EQ(proxy.get_number_neurons(), 2);
    EXPECT_EQ(proxy.get_layer_architecture(), Layer::Architecture::MultiOutput);

    // Test hidden gradient calculation through proxy
    // proxy(2) -> next_layer(3)
    MockLayer next_layer(1, 3);
    next_layer.set_w_values({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }); // 2x3 weights
    
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    
    // Next layer gradients: 2 time steps, size 3 each = 6 values
    // [g1_t0, g2_t0, g3_t0, g1_t1, g2_t1, g3_t1]
    std::vector<std::vector<double>> batch_next_grads = { { 1.0, 0.5, 0.0, 0.0, 0.5, 1.0 } };
    
    proxy.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, {}, 1, 0);

    // Expected Proxy RNN Gradients (size 2*2 = 4):
    // t=0:
    // grad[0] = 1.0*0.1 + 0.5*0.2 + 0.0*0.3 = 0.1 + 0.1 = 0.2
    // grad[1] = 1.0*0.4 + 0.5*0.5 + 0.0*0.6 = 0.4 + 0.25 = 0.65
    // t=1:
    // grad[0] = 0.0*0.1 + 0.5*0.2 + 1.0*0.3 = 0.1 + 0.3 = 0.4
    // grad[1] = 0.0*0.4 + 0.5*0.5 + 1.0*0.6 = 0.25 + 0.6 = 0.85
    // Sequence: [0.2, 0.65, 0.4, 0.85]
    
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
    // Trunk: 2 inputs -> 2 neurons (Layer 0)
    // Branch A: 2 neurons -> 2 hidden (ReLU, Layer 1) -> 1 output (Sigmoid, Layer 2)
    // Branch B: 2 neurons -> 1 hidden (Tanh, Layer 1) -> 2 outputs (Softmax, Layer 2)
    
    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oA(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = { LayerDetails(Layer::Architecture::FF, 1, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oB(2, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    MultiOutputLayer layer(1, 2, 3, { modA, modB }, 1, true);
    auto& branches = layer.get_mutable_branches();

    // Branch A Weights
    // H1 (ReLU): W=[[0.1, 0.2], [0.3, 0.4]], B=[0.1, -0.1]
    branches[0].layers[0]->set_w_values({ 0.1, 0.2, 0.3, 0.4 });
    branches[0].layers[0]->set_b_values({ 0.1, -0.1 });
    // O (Sigmoid): W=[[0.5], [0.6]], B=[0.2]
    branches[0].layers[1]->set_w_values({ 0.5, 0.6 });
    branches[0].layers[1]->set_b_values({ 0.2 });

    // Branch B Weights
    // H1 (Tanh): W=[[0.7], [0.8]], B=[0.3]
    branches[1].layers[0]->set_w_values({ 0.7, 0.8 });
    branches[1].layers[0]->set_b_values({ 0.3 });
    // O (Softmax): W=[[0.9, 1.0]], B=[-0.2, 0.2]
    branches[1].layers[1]->set_w_values({ 0.9, 1.0 });
    branches[1].layers[1]->set_b_values({ -0.2, 0.2 });

    // Forward pass
    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 0.5, -0.5 }); // Input
    
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Math check:
    // Input: [0.5, -0.5]
    
    // Branch A:
    // H1_z = [0.5*0.1 - 0.5*0.3 + 0.1, 0.5*0.2 - 0.5*0.4 - 0.1] 
    //      = [0.05 - 0.15 + 0.1, 0.1 - 0.2 - 0.1] = [0.0, -0.2]
    // H1_a = [ReLU(0.0), ReLU(-0.2)] = [0.0, 0.0]
    // O_z  = [0.0*0.5 + 0.0*0.6 + 0.2] = [0.2]
    // O_a  = [Sigmoid(0.2)] = [1/(1+exp(-0.2))] = [1/(1+0.8187)] = [0.549833997]
    
    // Branch B:
    // H1_z = [0.5*0.7 - 0.5*0.8 + 0.3] = [0.35 - 0.4 + 0.3] = [0.25]
    // H1_a = [Tanh(0.25)] = [0.24491866]
    // O_z  = [0.24491866*0.9 - 0.2, 0.24491866*1.0 + 0.2] 
    //      = [0.22042679 - 0.2, 0.24491866 + 0.2] = [0.02042679, 0.44491866]
    // Softmax([0.0204, 0.4449]):
    // exp(0.0204)=1.020636, exp(0.4449)=1.560363, sum=2.580999
    // O_a = [0.395442, 0.604558]

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], 0.549834, 1e-5);
    EXPECT_NEAR(outputs[1], 0.395442, 1e-5);
    EXPECT_NEAR(outputs[2], 0.604558, 1e-5);

    // Backprop check:
    std::vector<std::vector<double>> targets = { { 1.0, 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);
    
    // Branch A Output Grad (MSE, N=1): (0.549834 - 1.0) = -0.450166
    // Branch B Output Grad (CE): [0.395442 - 1.0, 0.604558 - 0.0] = [-0.604558, 0.604558]
    
    layer.backprop_branches(1, 0);
    
    // Branch A H1 Grad (ReLU at 0.0, derivative is 0.0):
    // G_ah1 = [ -0.450166 * 0.5 * 0.0, -0.450166 * 0.6 * 0.0 ] = [0.0, 0.0]
    // Trunk Grad A: [0.0, 0.0]
    
    // Branch B H1 Grad (Tanh deriv: 1 - a^2 = 1 - 0.24491866^2 = 1 - 0.059985 = 0.940015):
    // G_bh1 = [ (-0.604558 * 0.9 + 0.604558 * 1.0) * 0.940015 ]
    //       = [ (0.604558 * 0.1) * 0.940015 ] = [ 0.0604558 * 0.940015 ] = [ 0.056829 ]
    // Trunk Grad B:
    // G_tr_b = [ 0.056829 * 0.7, 0.056829 * 0.8 ] = [ 0.03978, 0.04546 ]
    
    // Total Trunk Grad: [0.03978, 0.04546]
    
    auto trunk_grads = layer.get_trunk_gradients(1);
    EXPECT_NEAR(trunk_grads[0][0], 0.03978, 1e-4);
    EXPECT_NEAR(trunk_grads[0][1], 0.04546, 1e-4);
}
