#include <gtest/gtest.h>
#include "../src/neuralnetwork/fflayer.h"
#include "../src/neuralnetwork/grurnnlayer.h"
#include "../src/neuralnetwork/ffoutputlayer.h"
#include "test_helper.h"
#include <vector>


using namespace myoddweb::nn;
using namespace test_helper;

TEST(NetworkIntegrationTest, CrossLayerGradientPropagation) {
    // Topology: 1 (Input) -> 1 (FF) -> 1 (GRU) -> 1 (FFOutput)
    unsigned num_inputs = 1;
    unsigned num_neurons = 1;
    std::vector<unsigned> topology = { num_inputs, num_neurons, num_neurons, num_neurons };

    FFLayer layer1(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    GRURNNLayer layer2(2, num_neurons, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    
    OutputLayerDetails out_details(num_neurons, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 0.0, 0.0, false, 1.0 }, 0.0, OptimiserType::SGD, 0.0);
    FFOutputLayer layer3(3, { out_details }, num_neurons, num_neurons, 1, true);

    // Set weights to identity for all layers
    layer1.set_w_values({ 1.0 }); layer1.set_b_values({ 0.0 });
    layer2.set_w_values({ 1.0 }); layer2.set_rw_values({ 0.0 }); layer2.set_b_values({ 0.0 });
    layer2.set_z_w_values({ 0.0 }); layer2.set_z_rw_values({ 0.0 }); layer2.set_z_b_values({ 100.0 }); // z=1 -> h = h_hat
    layer2.set_r_w_values({ 0.0 }); layer2.set_r_rw_values({ 0.0 }); layer2.set_r_b_values({ 100.0 }); // r=1
    layer3.set_w_values({ 1.0 }); layer3.set_b_values({ 0.0 });

    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 5); // 1 step

    // Input x = 1.0
    batch_go[0].set_outputs(0, { 1.0 });

    // Forward pass
    layer1.calculate_forward_feed(batch_go, MockLayer(0, 1), {}, batch_hs, 1, false);
    layer2.calculate_forward_feed(batch_go, layer1, {}, batch_hs, 1, false);
    layer3.calculate_forward_feed(batch_go, layer2, {}, batch_hs, 1, false);

    // Expected output: 1.0 (all linear identity)
    EXPECT_NEAR(batch_go[0].get_output(3, 0), 1.0, 1e-9);

    // Target y = 0.0. Loss = (1-0)^2 = 1. dLoss/dy = 2*(1-0) = 2.0
    // BUT the library uses (a-y)/N for MSE gradient.
    // Given outputs = 1.0, Target = 0.0, N = 1.
    // dL/dz3 = (1.0 - 0.0) / 1.0 = 1.0
    std::vector<std::vector<double>> targets = { { 0.0 } };
    
    // Backward pass
    layer3.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1); 
    // dL/dz3 = 1.0
    double grad3 = batch_go[0].get_gradients(3)[0];
    EXPECT_NEAR(grad3, 1.0, 1e-9);

    // Propagate to layer 2
    auto span3 = batch_go[0].get_gradients(3);
    std::vector<std::vector<double>> next_grads = { std::vector<double>(span3.begin(), span3.end()) };
    layer2.calculate_hidden_gradients(batch_go, layer3, next_grads, batch_hs, 1, 0);
    // dL/dz2 = dL/dz3 * W3 * act_deriv2 = 1.0 * 1.0 * 1.0 = 1.0
    double grad2 = batch_go[0].get_rnn_gradients(2)[0];
    EXPECT_NEAR(grad2, 1.0, 1e-9);

    // Propagate to layer 1
    auto span2 = batch_go[0].get_rnn_gradients(2);
    next_grads = { std::vector<double>(span2.begin(), span2.end()) };
    layer1.calculate_hidden_gradients(batch_go, layer2, next_grads, batch_hs, 1, 0);
    // dL/dz1 = dL/dz2 * W2 * act_deriv1 = 1.0 * 1.0 * 1.0 = 1.0
    double grad1 = batch_go[0].get_gradients(1)[0];
    EXPECT_NEAR(grad1, 1.0, 1e-9);

    // Store gradients for layer 1
    layer1.calculate_and_store_gradients(batch_go, batch_hs, MockLayer(0, 1, 1), 1, 0);
    // dL/dW1 = dL/dz1 * x = 1.0 * 1.0 = 1.0
    EXPECT_NEAR(layer1.get_w_grads()[0], 1.0, 1e-9);
}
