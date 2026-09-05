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
    FFLayer layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

    EXPECT_EQ(layer.get_layer_index(), 1);
    EXPECT_EQ(layer.get_number_input_neurons(), 2);
    EXPECT_EQ(layer.get_number_output_neurons(), 3);
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::FF);
    EXPECT_FALSE(layer.use_bptt());
    EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);
}

TEST_F(FFLayerTest, DropoutStatisticalVerification) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 5000;
    double dropout_rate = 0.5;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, std::nullopt);

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
    EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.08); // within 8% tolerance
}

TEST_F(FFLayerTest, DropoutNotInference) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 1.0, nullptr, 1, true, 0.0, std::nullopt);

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

TEST_F(FFLayerTest, DropoutWithTanhActivationDerivative) {
    const unsigned num_inputs = 1;
    const unsigned num_outputs = 200;
    const double dropout_rate = 0.5;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, std::nullopt);

    layer.set_w_values(std::vector<double>(num_outputs, 1.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    std::vector<std::vector<double>> deltas(1, std::vector<double>(num_outputs, 1.0));
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, deltas, batch_hs, 1, 0);

    const auto& grads = batch_go[0].get_gradients(1);
    const double tanh_val = std::tanh(1.0);
    const double expected_kept_grad = 1.0 * (1.0 - tanh_val * tanh_val) * (1.0 / (1.0 - dropout_rate));

    int kept_count = 0;
    int dropped_count = 0;
    for (size_t j = 0; j < num_outputs; ++j)
    {
        if (grads[j] == 0.0)
        {
            dropped_count++;
        }
        else
        {
            kept_count++;
            // Must be positive and strictly equal to f'(z) * scale, NOT corrupted by scaled y_vals
            EXPECT_GT(grads[j], 0.0);
            EXPECT_NEAR(grads[j], expected_kept_grad, 1e-9);
        }
    }

    EXPECT_GT(kept_count, 0);
    EXPECT_GT(dropped_count, 0);
    EXPECT_EQ(kept_count + dropped_count, static_cast<int>(num_outputs));
}

TEST_F(FFLayerTest, DropoutWithSigmoidActivationDerivative) {
    const unsigned num_inputs = 1;
    const unsigned num_outputs = 200;
    const double dropout_rate = 0.5;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::sigmoid, 1.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, std::nullopt);

    layer.set_w_values(std::vector<double>(num_outputs, 1.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    std::vector<std::vector<double>> deltas(1, std::vector<double>(num_outputs, 1.0));
    layer.calculate_hidden_gradients_from_output_gradients(batch_go, deltas, batch_hs, 1, 0);

    const auto& grads = batch_go[0].get_gradients(1);
    const double sig_val = 1.0 / (1.0 + std::exp(-1.0));
    const double expected_kept_grad = 1.0 * (sig_val * (1.0 - sig_val)) * (1.0 / (1.0 - dropout_rate));

    int kept_count = 0;
    int dropped_count = 0;
    for (size_t j = 0; j < num_outputs; ++j)
    {
        if (grads[j] == 0.0)
        {
            dropped_count++;
        }
        else
        {
            kept_count++;
            // Must be positive and strictly equal to f'(z) * scale, NOT negative
            EXPECT_GT(grads[j], 0.0);
            EXPECT_NEAR(grads[j], expected_kept_grad, 1e-6);
        }
    }

    EXPECT_GT(kept_count, 0);
    EXPECT_GT(dropped_count, 0);
    EXPECT_EQ(kept_count + dropped_count, static_cast<int>(num_outputs));
}

TEST_F(FFLayerTest, DropoutMultiTimestepSequenceForwardFeed) {
    const unsigned num_inputs = 2;
    const unsigned num_outputs = 100;
    const size_t num_time_steps = 4;
    const double dropout_rate = 0.5;

    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, std::nullopt);

    layer.set_w_values(std::vector<double>(num_inputs * num_outputs, 0.5));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, num_time_steps);

    std::vector<double> rnn_in(num_time_steps * num_inputs, 1.0);
    batch_go[0].set_rnn_outputs(0, rnn_in.data(), rnn_in.size());

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& rnn_out = batch_go[0].get_rnn_outputs(1);
    ASSERT_EQ(rnn_out.size(), num_time_steps * num_outputs);

    // Each timestep should have both dropped and kept outputs
    for (size_t t = 0; t < num_time_steps; ++t)
    {
        int dropped = 0;
        int kept = 0;
        for (size_t j = 0; j < num_outputs; ++j)
        {
            double val = rnn_out[t * num_outputs + j];
            if (val == 0.0)
            {
                dropped++;
            }
            else
            {
                kept++;
                EXPECT_GT(val, 0.0);
            }
        }
        EXPECT_GT(dropped, 0);
        EXPECT_GT(kept, 0);
    }
}

TEST_F(FFLayerTest, ForwardFeedReLU) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::softmax, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
        FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(m, 0.1), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
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

    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
    FFLayer next_layer(2, num_outputs, next_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
  FFLayer layer_with_bias(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
  layer_with_bias.set_w_values({ 1.0, 0.5, 0.2, 1.5 });
  layer_with_bias.set_b_values({ 0.3, -0.4 });
  
  // Create layer without bias
  FFLayer layer_no_bias(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, false, 0.0, std::nullopt);
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
  FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
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
  FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
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
  FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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
  FFLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

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

TEST_F(FFLayerTest, TransposedWeightsCacheAndFastBackwardPass)
{
  // Create a layer: 3 inputs, 4 outputs
  FFLayer layer(1, 3, 4, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  // Set weights: shape [3, 4]
  std::vector<double> weights = {
    0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 1.1, 1.2
  };
  layer.set_w_values(weights);

  // Trigger transposition/caching
  layer.cache_recurrent_weights();

  // Verify transposed cache is correct: shape [4, 3]
  const auto& weights_T = layer.get_w_values_T();
  ASSERT_EQ(weights_T.size(), 12);
  EXPECT_DOUBLE_EQ(weights_T[0], 0.1);
  EXPECT_DOUBLE_EQ(weights_T[1], 0.5);
  EXPECT_DOUBLE_EQ(weights_T[2], 0.9);
  EXPECT_DOUBLE_EQ(weights_T[3], 0.2);
  EXPECT_DOUBLE_EQ(weights_T[4], 0.6);
  EXPECT_DOUBLE_EQ(weights_T[5], 1.0);
  EXPECT_DOUBLE_EQ(weights_T[6], 0.3);
  EXPECT_DOUBLE_EQ(weights_T[7], 0.7);
  EXPECT_DOUBLE_EQ(weights_T[8], 1.1);
  EXPECT_DOUBLE_EQ(weights_T[9], 0.4);
  EXPECT_DOUBLE_EQ(weights_T[10], 0.8);
  EXPECT_DOUBLE_EQ(weights_T[11], 1.2);

  // Test backpropagation results:
  // Next layer has 4 inputs, 2 outputs.
  FFLayer next_layer(2, 4, 2, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
  std::vector<double> next_weights = {
    0.15, 0.25,
    0.35, 0.45,
    0.55, 0.65,
    0.75, 0.85
  };
  next_layer.set_w_values(next_weights);
  next_layer.cache_recurrent_weights();

  // Create batch gradients and outputs and hidden states
  std::vector<unsigned> topology = { 3, 4, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 2); // batch size 2
  auto batch_hs = create_batch_hidden_states(topology, 2, 1);

  // Initialize cell state values so that gradients are not zeroed out during post-gemm masking
  batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0, 1.0, 1.0 });
  batch_hs[1].at(1, 0).set_cell_state_values({ 1.0, 1.0, 1.0, 1.0 });

  // Set next layer incoming gradients (at index 2)
  batch_go[0].set_gradients(2, { 0.1, 0.2 });
  batch_go[1].set_gradients(2, { 0.3, 0.4 });

  // Calculate hidden gradients for layer index 1
  layer.calculate_hidden_gradients(batch_go, next_layer, {}, batch_hs, 2, 0);

  const auto grads_0 = batch_go[0].get_gradients(1);
  const auto grads_1 = batch_go[1].get_gradients(1);

  ASSERT_EQ(grads_0.size(), 4);
  ASSERT_EQ(grads_1.size(), 4);

  // Hand calculate the expected backpropagated gradients:
  // grad_prev_j = sum_k next_grad_k * W_next_jk
  // For batch 0 (incoming grad = [0.1, 0.2]):
  // grad_0[0] = 0.1 * 0.15 + 0.2 * 0.25 = 0.015 + 0.050 = 0.065
  // grad_0[1] = 0.1 * 0.35 + 0.2 * 0.45 = 0.035 + 0.090 = 0.125
  // grad_0[2] = 0.1 * 0.55 + 0.2 * 0.65 = 0.055 + 0.130 = 0.185
  // grad_0[3] = 0.1 * 0.75 + 0.2 * 0.85 = 0.075 + 0.170 = 0.245
  EXPECT_NEAR(grads_0[0], 0.065, 1e-9);
  EXPECT_NEAR(grads_0[1], 0.125, 1e-9);
  EXPECT_NEAR(grads_0[2], 0.185, 1e-9);
  EXPECT_NEAR(grads_0[3], 0.245, 1e-9);

  // For batch 1 (incoming grad = [0.3, 0.4]):
  // grad_1[0] = 0.3 * 0.15 + 0.4 * 0.25 = 0.045 + 0.100 = 0.145
  // grad_1[1] = 0.3 * 0.35 + 0.4 * 0.45 = 0.105 + 0.180 = 0.285
  // grad_1[2] = 0.3 * 0.55 + 0.4 * 0.65 = 0.165 + 0.260 = 0.425
  // grad_1[3] = 0.3 * 0.75 + 0.4 * 0.85 = 0.225 + 0.340 = 0.565
  EXPECT_NEAR(grads_1[0], 0.145, 1e-9);
  EXPECT_NEAR(grads_1[1], 0.285, 1e-9);
  EXPECT_NEAR(grads_1[2], 0.425, 1e-9);
  EXPECT_NEAR(grads_1[3], 0.565, 1e-9);
}

TEST_F(FFLayerTest, BatchForwardFeedInputCopyingSequenceAndBiasVerification)
{
  const unsigned num_inputs = 4;
  const unsigned num_neurons = 2;
  FFLayer layer(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  // Set weights: W is size 4 x 2: row 0 = [1, 0], row 1 = [0, 1], row 2 = [0, 0], row 3 = [0, 0]
  std::vector<double> weights = { 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
  std::vector<double> biases = { 0.5, 1.5 };
  layer.set_w_values(weights);
  layer.set_b_values(biases);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_neurons };

  // Test 1: Standard input copying across batch size 3
  auto batch_go = create_batch_gradients_and_outputs(topology, 3);
  auto batch_hs = create_batch_hidden_states(topology, 3, 1);

  batch_go[0].set_outputs(0, { 1.0, 2.0, 3.0, 4.0 });
  batch_go[1].set_outputs(0, { 2.0, 3.0, 4.0, 5.0 });
  batch_go[2].set_outputs(0, { 3.0, 4.0, 5.0, 6.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 3, false);

  // Expected output = x * W + b
  // Batch 0: [1*1 + 2*0 + 0.5, 1*0 + 2*1 + 1.5] = [1.5, 3.5]
  // Batch 1: [2*1 + 3*0 + 0.5, 2*0 + 3*1 + 1.5] = [2.5, 4.5]
  // Batch 2: [3*1 + 4*0 + 0.5, 3*0 + 4*1 + 1.5] = [3.5, 5.5]
  const auto& out0 = batch_go[0].get_outputs(1);
  const auto& out1 = batch_go[1].get_outputs(1);
  const auto& out2 = batch_go[2].get_outputs(1);

  ASSERT_EQ(out0.size(), 2u);
  EXPECT_DOUBLE_EQ(out0[0], 1.5);
  EXPECT_DOUBLE_EQ(out0[1], 3.5);

  ASSERT_EQ(out1.size(), 2u);
  EXPECT_DOUBLE_EQ(out1[0], 2.5);
  EXPECT_DOUBLE_EQ(out1[1], 4.5);

  ASSERT_EQ(out2.size(), 2u);
  EXPECT_DOUBLE_EQ(out2[0], 3.5);
  EXPECT_DOUBLE_EQ(out2[1], 5.5);

  // Test 2: Sequence rnn_outputs copying across 2 time steps
  auto batch_go_seq = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs_seq = create_batch_hidden_states(topology, 1, 2);

  // 2 time steps for batch item 0 (4 inputs * 2 ticks = 8 doubles)
  std::vector<double> rnn_inputs = { 1.0, 2.0, 0.0, 0.0,  10.0, 20.0, 0.0, 0.0 };
  batch_go_seq[0].set_rnn_outputs(0, rnn_inputs.data(), rnn_inputs.size());

  layer.calculate_forward_feed(batch_go_seq, prev_layer, {}, batch_hs_seq, 1, false);

  const auto& rnn_out = batch_go_seq[0].get_rnn_outputs(1);
  ASSERT_EQ(rnn_out.size(), 4u); // 2 ticks * 2 neurons
  EXPECT_DOUBLE_EQ(rnn_out[0], 1.5);
  EXPECT_DOUBLE_EQ(rnn_out[1], 3.5);
  EXPECT_DOUBLE_EQ(rnn_out[2], 10.5);
  EXPECT_DOUBLE_EQ(rnn_out[3], 21.5);
}

TEST_F(FFLayerTest, GradientAccumulationFourWideEquivalence)
{
  const std::vector<std::pair<unsigned, unsigned>> dimensions = {
    { 1, 1 }, { 2, 3 }, { 3, 5 }, { 4, 4 }, { 5, 7 }, { 7, 8 }, { 9, 6 }, { 11, 13 }
  };

  for (const auto& [num_inputs, num_outputs] : dimensions)
  {
    const unsigned batch_size = 4;
    const unsigned num_time_steps = 3;

    FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };

    auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

    std::vector<std::vector<double>> all_inputs(batch_size);
    std::vector<std::vector<double>> all_grads(batch_size);

    for (size_t b = 0; b < batch_size; ++b)
    {
      all_inputs[b].resize(num_time_steps * num_inputs);
      for (size_t k = 0; k < all_inputs[b].size(); ++k)
      {
        all_inputs[b][k] = std::sin(static_cast<double>(b * 100 + k + 1));
      }
      batch_go[b].set_rnn_outputs(0, all_inputs[b].data(), all_inputs[b].size());

      all_grads[b].resize(num_time_steps * num_outputs);
      for (size_t k = 0; k < all_grads[b].size(); ++k)
      {
        all_grads[b][k] = std::cos(static_cast<double>(b * 50 + k + 1));
      }
      batch_go[b].set_rnn_gradients(1, all_grads[b].data(), all_grads[b].size());
    }

    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, -1);

    const auto& actual_w_grads = layer.get_w_grads();
    const auto& actual_b_grads = layer.get_b_grads();

    std::vector<double> expected_w_grads(num_inputs * num_outputs, 0.0);
    std::vector<double> expected_b_grads(num_outputs, 0.0);

    for (size_t b = 0; b < batch_size; ++b)
    {
      for (size_t t = 0; t < num_time_steps; ++t)
      {
        const double* x_t = &all_inputs[b][t * num_inputs];
        const double* g_t = &all_grads[b][t * num_outputs];

        for (size_t j = 0; j < num_outputs; ++j)
        {
          expected_b_grads[j] += g_t[j];
        }

        for (size_t i = 0; i < num_inputs; ++i)
        {
          for (size_t j = 0; j < num_outputs; ++j)
          {
            expected_w_grads[i * num_outputs + j] += x_t[i] * g_t[j];
          }
        }
      }
    }

    const double inv_batch = 1.0 / static_cast<double>(batch_size);
    for (size_t k = 0; k < expected_w_grads.size(); ++k)
    {
      expected_w_grads[k] *= inv_batch;
      EXPECT_NEAR(actual_w_grads[k], expected_w_grads[k], 1e-12)
        << "W grad mismatch at inputs=" << num_inputs << ", outputs=" << num_outputs << ", index=" << k;
    }

    for (size_t k = 0; k < expected_b_grads.size(); ++k)
    {
      expected_b_grads[k] *= inv_batch;
      EXPECT_NEAR(actual_b_grads[k], expected_b_grads[k], 1e-12)
        << "B grad mismatch at inputs=" << num_inputs << ", outputs=" << num_outputs << ", index=" << k;
    }
  }
}

TEST_F(FFLayerTest, InferenceMultiThreadingConsistency)
{
  const unsigned num_inputs = 32;
  const unsigned num_outputs = 64;
  const unsigned batch_size = 64;
  const unsigned num_time_steps = 8;
  const unsigned num_threads = 4;

  FFLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
  FFLayer layer_mt(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0, std::nullopt);

  std::vector<double> w_vals(num_inputs * num_outputs);
  for (size_t i = 0; i < w_vals.size(); ++i)
  {
    w_vals[i] = std::sin(static_cast<double>(i + 1));
  }
  std::vector<double> b_vals(num_outputs);
  for (size_t i = 0; i < b_vals.size(); ++i)
  {
    b_vals[i] = std::cos(static_cast<double>(i + 1));
  }

  layer_st.set_w_values(w_vals);
  layer_st.set_b_values(b_vals);
  layer_mt.set_w_values(w_vals);
  layer_mt.set_b_values(b_vals);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs };

  auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_time_steps);

  auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_time_steps);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> in_seq(num_time_steps * num_inputs);
    for (size_t k = 0; k < in_seq.size(); ++k)
    {
      in_seq[k] = std::sin(static_cast<double>(b * num_time_steps + k));
    }
    batch_go_st[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
    batch_go_mt[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
  }

  // Execute forward feed in inference mode (is_training = false)
  layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, false);
  layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, false);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto& out_st = batch_go_st[b].get_rnn_outputs(1);
    const auto& out_mt = batch_go_mt[b].get_rnn_outputs(1);
    ASSERT_EQ(out_st.size(), out_mt.size());
    for (size_t i = 0; i < out_st.size(); ++i)
    {
      EXPECT_NEAR(out_st[i], out_mt[i], 1e-12) << "Inference mismatch at batch " << b << ", index " << i;
    }
  }
}

TEST_F(FFLayerTest, SingleSampleContiguousBypassVerification)
{
  const unsigned num_inputs = 8;
  const unsigned num_outputs = 16;
  const unsigned num_time_steps = 5;

  FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  std::vector<double> w_vals(num_inputs * num_outputs);
  for (size_t i = 0; i < w_vals.size(); ++i)
  {
    w_vals[i] = 0.05 * static_cast<double>(i % 7);
  }
  std::vector<double> b_vals(num_outputs, 0.1);
  layer.set_w_values(w_vals);
  layer.set_b_values(b_vals);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs };

  for (int iter = 0; iter < 3; ++iter)
  {
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps);

    std::vector<double> inputs(num_time_steps * num_inputs);
    for (size_t k = 0; k < inputs.size(); ++k)
    {
      inputs[k] = 0.1 * static_cast<double>((iter + 1) * (k + 1));
    }
    batch_go[0].set_rnn_outputs(0, inputs.data(), inputs.size());

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& rnn_outputs = batch_go[0].get_rnn_outputs(1);
    ASSERT_EQ(rnn_outputs.size(), num_time_steps * num_outputs);

    double expected_sum = 0.1;
    for (size_t i = 0; i < num_inputs; ++i)
    {
      expected_sum += inputs[i] * w_vals[i * num_outputs];
    }
    double expected_act = std::max(0.0, expected_sum);
    EXPECT_NEAR(rnn_outputs[0], expected_act, 1e-12);
  }
}

