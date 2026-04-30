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
    OutputLayerDetails oA(2, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = {};
    OutputLayerDetails oB(1, activation(activation::method::sigmoid, 1.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    std::vector<MultiOutputLayerDetails> details = { modA, modB };
    MultiOutputLayer layer(1, 2, 3, details, 1, true);

    // Access branches to set specific weights
    auto& branches = layer.get_mutable_branches();
    
    // Branch A Hidden Layer (ReLU)
    // Use weights that move z away from 0.0 for robustness
    branches[0].layers[0]->set_w_values({ 0.2, 0.4, 0.6, 0.8 });
    branches[0].layers[0]->set_b_values({ 0.1, -0.1 });

    // Branch A Output Layer (Softmax)
    branches[0].layers[1]->set_w_values({ 0.5, 0.6, 0.7, 0.8 });
    branches[0].layers[1]->set_b_values({ 0.0, 0.0 });

    // Branch B Output Layer (Sigmoid)
    branches[1].layers[0]->set_w_values({ 0.9, -0.1 });
    branches[1].layers[0]->set_b_values({ 0.0 });

    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    
    batch_go[0].set_outputs(0, { 0.5, -0.2 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Math:
    // Input: [0.5, -0.2]
    // Branch A Hidden:
    // Z_ah = [0.5*0.2 - 0.2*0.6 + 0.1, 0.5*0.4 - 0.2*0.8 - 0.1]
    //      = [0.1 - 0.12 + 0.1, 0.2 - 0.16 - 0.1] = [0.08, -0.06]
    // A_ah = [0.08, 0.0]
    // Branch A Output:
    // Z_ao = [0.08*0.5 + 0.0*0.7, 0.08*0.6 + 0.0*0.8] = [0.04, 0.048]
    // Softmax([0.04, 0.048]):
    // exp(0.04)=1.0408108, exp(0.048)=1.0491708, sum=2.0899816
    // A_ao = [0.4979999, 0.5020001]
    
    // Branch B Output:
    // Z_bo = [0.5*0.9 - 0.2*-0.1] = [0.47]
    // A_bo = [sigmoid(0.47)] = [0.6153829]

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], 0.498000, 1e-6);
    EXPECT_NEAR(outputs[1], 0.502000, 1e-6);
    EXPECT_NEAR(outputs[2], 0.615383, 1e-6);
}

TEST_F(MultiOutputLayerTest, OutputGradientsMathematicalVerification) {
    // Explicitly clean EvaluationConfig to disable any direction boost/scaling
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

    // Targets: Branch A: [1.0, 0.0], Branch B: [0.8]
    std::vector<std::vector<double>> targets = { { 1.0, 0.0, 0.8 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    // Branch A Output (Softmax + CE): grad = y - t
    // A_ao = [0.4979999, 0.5020001]
    // Grad_ao = [-0.502000, 0.502000]
    
    // Branch B Output (Sigmoid + MSE): grad = (y - t) * sig_deriv
    // delta = (0.615383 - 0.8) = -0.184617
    // deriv = 0.615383 * (1 - 0.615383) = 0.236686
    // grad = -0.184617 * 0.236686 = -0.043696

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
    EXPECT_NEAR(metrics[0][0].error(), 0.04, 1e-6);
    EXPECT_NEAR(metrics[1][0].error(), 0.16, 1e-6);
}

TEST_F(MultiOutputLayerTest, ActivationBranches) {
    OutputLayerDetails o1(1, activation(activation::method::tanh, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    OutputLayerDetails o2(1, activation(activation::method::elu, 1.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
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
    // Trunk: 2 inputs -> 2 neurons
    // Branch A: 2 inputs -> 2 hidden (ReLU) -> 1 output (Sigmoid, MSE)
    // Branch B: 2 inputs -> 1 hidden (Tanh) -> 2 outputs (Softmax, CE)
    
    EvaluationConfig clean_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12);

    std::vector<LayerDetails> hA = { LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oA(1, activation(activation::method::sigmoid, 1.0, 1.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA(hA, oA);

    std::vector<LayerDetails> hB = { LayerDetails(Layer::Architecture::FF, 1, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0) };
    OutputLayerDetails oB(2, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB(hB, oB);

    MultiOutputLayer layer(1, 2, 3, { modA, modB }, 1, true);
    auto& branches = layer.get_mutable_branches();

    // Set weights to move z away from 0.0 for robustness
    // Branch A H1 (ReLU): W=[[0.2, 0.4], [0.6, 0.8]], B=[0.1, -0.1]
    branches[0].layers[0]->set_w_values({ 0.2, 0.4, 0.6, 0.8 });
    branches[0].layers[0]->set_b_values({ 0.1, -0.1 });
    // Branch A O (Sigmoid): W=[[0.5], [0.6]], B=[0.2]
    branches[0].layers[1]->set_w_values({ 0.5, 0.6 });
    branches[0].layers[1]->set_b_values({ 0.2 });

    // Branch B H1 (Tanh): W=[[0.7], [0.8]], B=[0.3]
    branches[1].layers[0]->set_w_values({ 0.7, 0.8 });
    branches[1].layers[0]->set_b_values({ 0.3 });
    // Branch B O (Softmax): W=[[0.9, 1.0]], B=[-0.2, 0.2]
    branches[1].layers[1]->set_w_values({ 0.9, 1.0 });
    branches[1].layers[1]->set_b_values({ -0.2, 0.2 });

    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 3 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 0.5, -0.5 });
    
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    // Forward Math:
    // Input: [0.5, -0.5]
    // Branch A H1: z = [0.5*0.2 - 0.5*0.6 + 0.1, 0.5*0.4 - 0.5*0.8 - 0.1] = [-0.1, -0.1]
    // Branch A H1 a: [0.0, 0.0]
    // Branch A O: z = [0.0*0.5 + 0.0*0.6 + 0.2] = [0.2], a = [sigmoid(0.2)] = [0.549834]
    
    // Branch B H1: z = [0.5*0.7 - 0.5*0.8 + 0.3] = [0.35 - 0.4 + 0.3] = [0.25]
    // Branch B H1 a: [tanh(0.25)] = [0.24491866]
    // Branch B O: z = [0.24491866*0.9 - 0.2, 0.24491866*1.0 + 0.2] = [0.02042679, 0.44491866]
    // Branch B O a: exp(0.0204)=1.0206, exp(0.4449)=1.5603, sum=2.5809, a = [0.395442, 0.604558]

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], 0.549834, 1e-5);
    EXPECT_NEAR(outputs[1], 0.395442, 1e-5);
    EXPECT_NEAR(outputs[2], 0.604558, 1e-5);

    // Backprop Math:
    std::vector<std::vector<double>> targets = { { 1.0, 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);
    
    // Branch A O grad: (0.549834 - 1.0) * sig'(0.2) = -0.450166 * 0.247517 = -0.111424
    // Branch B O grad (CE): [0.395442 - 1.0, 0.604558 - 0.0] = [-0.604558, 0.604558]
    
    layer.backprop_branches(1, 0);
    
    // Branch A H1: z = [-0.1, -0.1], ReLU' = [0.0, 0.0] -> Branch A Trunk grad = [0, 0]
    
    // Branch B H1: z = 0.25, tanh' = 0.940015
    // Branch B H1 grad: (-0.604558*0.9 + 0.604558*1.0) * 0.940015 = 0.0604558 * 0.940015 = 0.056829
    // Branch B Trunk grad: [0.056829*0.7, 0.056829*0.8] = [0.03978, 0.04546]
    
    auto trunk_grads = layer.get_trunk_gradients(1);
    EXPECT_NEAR(trunk_grads[0][0], 0.03978, 1e-4);
    EXPECT_NEAR(trunk_grads[0][1], 0.04546, 1e-4);
}
