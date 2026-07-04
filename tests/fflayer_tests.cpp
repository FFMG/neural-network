#include <gtest/gtest.h>
#include "layers/fflayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
using namespace test_helper;

class FFLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(FFLayerTest, ConstructionAndTopology) {
    FFLayer layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

    EXPECT_EQ(layer.get_layer_index(), 1);
    EXPECT_EQ(layer.get_number_input_neurons(), 2);
    EXPECT_EQ(layer.get_number_output_neurons(), 3);
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::FF);
    EXPECT_FALSE(layer.use_bptt());
    EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);
}

TEST_F(FFLayerTest, DropoutStatisticalVerification) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0);

    layer.set_w_values(std::vector<double>(num_outputs, 1.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_outputs(1);
    int dropped_count = 0;
    int kept_count = 0;
    for (double out : outputs) {
        if (out == 0.0) dropped_count++;
        else if (approx_equal(out, 1.0 / (1.0 - dropout_rate))) kept_count++;
    }

    EXPECT_EQ(dropped_count + kept_count, (int)num_outputs);
    EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.05); // within 5% tolerance
}

TEST_F(FFLayerTest, DropoutNotInference) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0);

    layer.set_w_values(std::vector<double>(num_outputs, 1.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false); // is_training = false

    const auto& outputs = batch_go[0].get_outputs(1);
    for (double out : outputs) {
        EXPECT_NEAR(out, 1.0, 1e-9); // No scaling, no dropping
    }
}

TEST_F(FFLayerTest, DropoutConsistencyVerification) {
    // 1 neuron with 100% dropout
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 1.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    // Forward pass: should drop (output 0.0)
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
    EXPECT_NEAR(batch_go[0].get_outputs(1)[0], 0.0, 1e-9);

    // Backward pass: gradient should also be 0.0
    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0 });
    std::vector<std::vector<double>> batch_next_grads = { { 10.0 } };

    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

    // The gradient should be 0.0 because the neuron was dropped.
    EXPECT_NEAR(batch_go[0].get_gradients(1)[0], 0.0, 1e-9);
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

        EXPECT_NO_THROW(layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false));
        double out = batch_go[0].get_output(1, 0);
        EXPECT_TRUE(std::isfinite(out));
    }
}

TEST_F(FFLayerTest, CalculateHiddenGradients) {
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

    batch_hs[0].at(1, 0).set_pre_activation_sum(0, 0.5);
    batch_hs[0].at(1, 0).set_pre_activation_sum(1, 0.5);
    batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0 });

    std::vector<std::vector<double>> batch_next_grads = { { 1.0 } };

    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

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

    batch_go[0].set_outputs(0, { 2.0 });
    batch_go[0].set_gradients(1, { 0.5 });

    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 0);

    EXPECT_NEAR(layer.get_w_grads()[0], 1.0, 1e-9);
    EXPECT_NEAR(layer.get_b_grads()[0], 0.5, 1e-9);
}

TEST_F(FFLayerTest, ApplyStoredGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    layer.set_w_values({ 1.0 });
    layer.set_b_values({ 0.5 });
    
    std::vector<double> w_grads = { 0.1 };
    std::vector<double> b_grads = { 0.05 };
    layer.set_w_grads(w_grads);
    layer.set_b_grads(b_grads);

    layer.apply_stored_gradients(0.1, 1.0); 

    EXPECT_NEAR(layer.get_w_values()[0], 0.99, 1e-9);
    EXPECT_NEAR(layer.get_b_values()[0], 0.495, 1e-9);
}

TEST_F(FFLayerTest, LearningRateRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    std::vector<double> learning_rates = { 0.0, 0.0001, 0.01, 0.5, 1.0, 2.0 };
    
    for (double lr : learning_rates) {
        double initial_w = 1.0;
        double initial_b = 0.5;
        layer.set_w_values({ initial_w });
        layer.set_b_values({ initial_b });
        
        double w_grad = 0.1;
        double b_grad = 0.05;
        layer.set_w_grads({ w_grad });
        layer.set_b_grads({ b_grad });

        layer.apply_stored_gradients(lr, 1.0);

        double expected_w = initial_w - lr * w_grad;
        double expected_b = initial_b - lr * b_grad;
        
        EXPECT_NEAR(layer.get_w_values()[0], expected_w, 1e-9);
        EXPECT_NEAR(layer.get_b_values()[0], expected_b, 1e-9);
    }
}

TEST_F(FFLayerTest, SequentialGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2); 

    batch_go[0].set_rnn_outputs(0, { 1.0, 2.0 });
    batch_go[0].set_rnn_gradients(1, { 0.5, 0.3 });

    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 0);

    EXPECT_NEAR(layer.get_w_grads()[0], 1.1, 1e-9);
    EXPECT_NEAR(layer.get_b_grads()[0], 0.8, 1e-9);
}

TEST_F(FFLayerTest, SequentialGradientsBatch2) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 2);

    batch_go[0].set_rnn_outputs(0, { 1.0, 2.0 });
    batch_go[0].set_rnn_gradients(1, { 0.5, 0.3 });
    batch_go[1].set_rnn_outputs(0, { 0.5, 1.5 });
    batch_go[1].set_rnn_gradients(1, { 0.2, 0.4 });

    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 2, 0);

    EXPECT_NEAR(layer.get_w_grads()[0], 0.9, 1e-9);
    EXPECT_NEAR(layer.get_b_grads()[0], 0.7, 1e-9);
}

TEST_F(FFLayerTest, ForwardFeedAndGradientsBiasBehaviour)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  
  // Create layer with bias
  FFLayer layer_with_bias(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
  layer_with_bias.set_w_values({ 1.0, 0.5, 0.2, 1.5 });
  layer_with_bias.set_b_values({ 0.3, -0.4 });
  
  // Create layer without bias
  FFLayer layer_no_bias(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, false, 0.0);
  layer_no_bias.set_w_values({ 1.0, 0.5, 0.2, 1.5 });
  
  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs };
  
  // Verify forward pass with bias
  auto batch_go_wb = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs_wb = create_batch_hidden_states(topology, 1, 1);
  batch_go_wb[0].set_outputs(0, { 1.5, 2.0 });
  
  layer_with_bias.calculate_forward_feed(batch_go_wb, prev_layer, {}, batch_hs_wb, 1, false);
  // Expected outputs: 
  // out0 = 1.5 * 1.0 + 2.0 * 0.2 + 0.3 = 1.5 + 0.4 + 0.3 = 2.2
  // out1 = 1.5 * 0.5 + 2.0 * 1.5 - 0.4 = 0.75 + 3.0 - 0.4 = 3.35
  EXPECT_NEAR(batch_go_wb[0].get_output(1, 0), 2.2, 1e-9);
  EXPECT_NEAR(batch_go_wb[0].get_output(1, 1), 3.35, 1e-9);
  
  // Verify forward pass without bias
  auto batch_go_nb = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs_nb = create_batch_hidden_states(topology, 1, 1);
  batch_go_nb[0].set_outputs(0, { 1.5, 2.0 });
  
  layer_no_bias.calculate_forward_feed(batch_go_nb, prev_layer, {}, batch_hs_nb, 1, false);
  // Expected outputs: 
  // out0 = 1.5 * 1.0 + 2.0 * 0.2 = 1.9
  // out1 = 1.5 * 0.5 + 2.0 * 1.5 = 3.75
  EXPECT_NEAR(batch_go_nb[0].get_output(1, 0), 1.9, 1e-9);
  EXPECT_NEAR(batch_go_nb[0].get_output(1, 1), 3.75, 1e-9);

  // Verify backward pass gradient accumulation with bias
  batch_go_wb[0].set_gradients(1, { 0.5, -0.2 });
  layer_with_bias.calculate_and_store_gradients(batch_go_wb, batch_hs_wb, prev_layer, 1, 0);
  EXPECT_NEAR(layer_with_bias.get_b_grads()[0], 0.5, 1e-9);
  EXPECT_NEAR(layer_with_bias.get_b_grads()[1], -0.2, 1e-9);
}

TEST_F(FFLayerTest, OversizedBiasVectorSafety)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 1;
  
  // Create layer with bias
  FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
  layer.set_w_values({ 1.0, 0.5 });
  
  // Set oversized bias vector (size 3, whereas output size is 1)
  layer.set_b_values({ 0.3, 0.9, -1.2 });
  
  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs };
  
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1);
  batch_go[0].set_outputs(0, { 1.5, 2.0 });
  
  // Verify that it doesn't crash and only uses the first bias value (0.3)
  EXPECT_NO_THROW(layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false));
  // Expected output: 1.5 * 1.0 + 2.0 * 0.5 + 0.3 = 1.5 + 1.0 + 0.3 = 2.8
  EXPECT_NEAR(batch_go[0].get_output(1, 0), 2.8, 1e-9);
}

TEST_F(FFLayerTest, DirectNextGradientsRetrieval)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  unsigned next_outputs = 1;
  FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);
  MockLayer next_layer(2, next_outputs);

  layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 });
  layer.set_b_values({ 0.0, 0.0 });

  next_layer.set_w_values({ 0.5, 0.8 });
  next_layer.set_b_values({ 0.0 });

  std::vector<unsigned> topology = { num_inputs, num_outputs, next_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1);

  batch_hs[0].at(1, 0).set_pre_activation_sum(0, 0.5);
  batch_hs[0].at(1, 0).set_pre_activation_sum(1, 0.5);
  batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0 });

  // Store the next gradients directly in the GradientsAndOutputs object
  batch_go[0].set_gradients(2, { 1.0 });

  // Pass an empty vector to trigger the direct next gradient retrieval path
  std::vector<std::vector<double>> empty_next_grads = {};
  layer.calculate_hidden_gradients(batch_go, next_layer, empty_next_grads, batch_hs, 1, 0);

  const auto grads = batch_go[0].get_gradients(1);
  EXPECT_NEAR(grads[0], 0.5, 1e-9);
  EXPECT_NEAR(grads[1], 0.8, 1e-9);
}

TEST_F(FFLayerTest, DirectOutputGradientsRetrieval)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  unsigned next_outputs = 2;
  FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

  layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 });
  layer.set_b_values({ 0.0, 0.0 });

  std::vector<unsigned> topology = { num_inputs, num_outputs, next_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1);

  batch_hs[0].at(1, 0).set_pre_activation_sum(0, 0.5);
  batch_hs[0].at(1, 0).set_pre_activation_sum(1, 0.5);
  batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0 });

  // Store the output layer's input gradients directly (at layer index 2)
  batch_go[0].set_gradients(2, { 0.5, 0.8 });

  // Pass an empty vector to trigger the direct retrieval path
  std::vector<std::vector<double>> empty_output_grads = {};
  layer.calculate_hidden_gradients_from_output_gradients(batch_go, empty_output_grads, batch_hs, 1, 0);

  const auto grads = batch_go[0].get_gradients(1);
  EXPECT_NEAR(grads[0], 0.5, 1e-9);
  EXPECT_NEAR(grads[1], 0.8, 1e-9);
}

TEST_F(FFLayerTest, StateAndMemoryAllocationOptimizationVerification)
{
  // A 2-input, 2-neuron FFLayer with 2 batches and 3 time steps
  FFLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

  layer.set_w_values({ 0.1, 0.2, 0.3, 0.4 });
  layer.set_b_values({ 0.05, 0.15 });

  MockLayer prev_layer(0, 2);
  std::vector<unsigned> topology = { 2, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs = create_batch_hidden_states(topology, 2, 3); // 3 steps

  // Batch 0: [[1.0, 0.5], [-0.5, 1.0], [0.0, 0.0]]
  // Batch 1: [[0.5, -0.5], [1.0, 1.0], [-1.0, 0.5]]
  batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, -0.5, 1.0, 0.0, 0.0 });
  batch_go[1].set_rnn_outputs(0, { 0.5, -0.5, 1.0, 1.0, -1.0, 0.5 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, false);

  // Verify outputs
  const auto& outputs_0 = batch_go[0].get_rnn_outputs(1);
  const auto& outputs_1 = batch_go[1].get_rnn_outputs(1);

  ASSERT_EQ(outputs_0.size(), 6);
  ASSERT_EQ(outputs_1.size(), 6);

  // t=0, Batch 0:
  // pre_act[0] = 1.0 * 0.1 + 0.5 * 0.3 + 0.05 = 0.3 -> relu -> 0.3
  // pre_act[1] = 1.0 * 0.2 + 0.5 * 0.4 + 0.15 = 0.55 -> relu -> 0.55
  EXPECT_NEAR(outputs_0[0], 0.3, 1e-9);
  EXPECT_NEAR(outputs_0[1], 0.55, 1e-9);

  // Verify non-zero/retention propagate correctly
  EXPECT_NEAR(outputs_0[4], 0.05, 1e-9); // relu(0.0 * 0.1 + 0.0 * 0.3 + 0.05) = 0.05
  EXPECT_NEAR(outputs_0[5], 0.15, 1e-9); // relu(0.0 * 0.2 + 0.0 * 0.4 + 0.15) = 0.15
}
