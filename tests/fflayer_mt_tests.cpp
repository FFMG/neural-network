#include <gtest/gtest.h>
#include "../src/neuralnetwork/fflayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
using namespace test_helper;

class FFLayerMTTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
    }

    // Helper to initialize layer weights deterministically
    void init_layer_weights(FFLayer& layer) 
    {
        unsigned num_neurons = layer.get_number_output_neurons();
        unsigned num_inputs = layer.get_number_input_neurons();

        std::vector<double> w_vals(num_neurons * num_inputs);
        std::vector<double> b_vals(num_neurons);

        for (size_t i = 0; i < w_vals.size(); ++i) 
        {
          w_vals[i] = std::sin(static_cast<double>(i));
        }
        for (size_t i = 0; i < b_vals.size(); ++i) 
        {
          b_vals[i] = std::cos(static_cast<double>(i) * 0.5);
        }

        layer.set_w_values(w_vals);
        layer.set_b_values(b_vals);
    }
};

TEST_F(FFLayerMTTest, ForwardFeedMTConsistency) 
{
    const unsigned num_inputs = 16;
    const unsigned num_neurons = 32;
    const unsigned batch_size = 256;
    const unsigned num_threads = get_test_threads();

    FFLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    FFLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_neurons };

    // Setup inputs
    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
    
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);

    for (size_t b = 0; b < batch_size; ++b) 
    {
        std::vector<double> inputs(num_inputs);
        for (size_t i = 0; i < inputs.size(); ++i) 
        {
          inputs[i] = std::cos(static_cast<double>(b * i));
        }
        batch_go_st[b].set_outputs(0, inputs);
        batch_go_mt[b].set_outputs(0, inputs);
    }

    layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, false);
    layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, false);

    // Verify consistency
    for (size_t b = 0; b < batch_size; ++b) 
    {
        const auto& out_st = batch_go_st[b].get_outputs(1);
        const auto& out_mt = batch_go_mt[b].get_outputs(1);
        ASSERT_EQ(out_st.size(), out_mt.size());
        for (size_t i = 0; i < out_st.size(); ++i) 
        {
            EXPECT_NEAR(out_st[i], out_mt[i], 1e-12) << "Mismatch at batch " << b << " index " << i;
        }
    }
}

TEST_F(FFLayerMTTest, BackwardFeedMTConsistency) 
{
    const unsigned num_inputs = 16;
    const unsigned num_neurons = 32;
    const unsigned next_neurons = 16;
    const unsigned batch_size = 256;
    const unsigned num_threads = get_test_threads();

    FFLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    FFLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);

    FFLayer next_layer(2, num_neurons, next_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    std::vector<double> next_w(num_neurons * next_neurons);
    for (size_t i = 0; i < next_w.size(); ++i) 
    {
      next_w[i] = std::sin(static_cast<double>(i) * 0.3);
    }
    next_layer.set_w_values(next_w);

    std::vector<unsigned> topology = { num_inputs, num_neurons, next_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);

    for (size_t b = 0; b < batch_size; ++b) 
    {
        std::vector<double> inputs(num_inputs);
        for (size_t i = 0; i < inputs.size(); ++i) 
        {
          inputs[i] = std::cos(static_cast<double>(b + i));
        }
        batch_go_st[b].set_outputs(0, inputs);
        batch_go_mt[b].set_outputs(0, inputs);
    }

    // Run forward first to populate hidden states
    MockLayer prev_layer(0, num_inputs);
    layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, true);
    layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, true);

    // Setup next gradients
    std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(next_neurons));
    for (size_t b = 0; b < batch_size; ++b) 
    {
        for (size_t i = 0; i < next_neurons; ++i) 
        {
          batch_next_grads[b][i] = std::sin(static_cast<double>(b * i));
        }
    }

    layer_st.calculate_hidden_gradients(batch_go_st, next_layer, batch_next_grads, batch_hs_st, batch_size, 0);
    layer_mt.calculate_hidden_gradients(batch_go_mt, next_layer, batch_next_grads, batch_hs_mt, batch_size, 0);

    // Verify consistency of input gradients
    for (size_t b = 0; b < batch_size; ++b) 
    {
        const auto& in_g_st = batch_go_st[b].get_gradients(1);
        const auto& in_g_mt = batch_go_mt[b].get_gradients(1);
        ASSERT_EQ(in_g_st.size(), in_g_mt.size());
        for (size_t i = 0; i < in_g_st.size(); ++i) 
        {
            EXPECT_NEAR(in_g_st[i], in_g_mt[i], 1e-12) << "Input grad mismatch at batch " << b << " index " << i;
        }
    }
}

TEST_F(FFLayerMTTest, GradientStorageMTConsistency) 
{
    const unsigned num_inputs = 16;
    const unsigned num_neurons = 32;
    const unsigned next_neurons = 16;
    const unsigned batch_size = 256;
    const unsigned num_threads = get_test_threads();

    FFLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    FFLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);

    FFLayer next_layer(2, num_neurons, next_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    std::vector<double> next_w(num_neurons * next_neurons, 0.1);
    next_layer.set_w_values(next_w);

    std::vector<unsigned> topology = { num_inputs, num_neurons, next_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, 1, 1);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, 1, 1);

    for (size_t b = 0; b < batch_size; ++b) 
    {
        std::vector<double> inputs(num_inputs);
        for (size_t i = 0; i < inputs.size(); ++i) 
        {
          inputs[i] = std::cos(static_cast<double>(b + i));
        }
        batch_go_st[b].set_outputs(0, inputs);
        batch_go_mt[b].set_outputs(0, inputs);
    }

    MockLayer prev_layer(0, num_inputs);
    layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, true);
    layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, true);

    std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(next_neurons));
    for (size_t b = 0; b < batch_size; ++b) 
    {
        for (size_t i = 0; i < next_neurons; ++i) 
        {
          batch_next_grads[b][i] = std::sin(static_cast<double>(b * i));
        }
    }

    layer_st.calculate_hidden_gradients(batch_go_st, next_layer, batch_next_grads, batch_hs_st, batch_size, 0);
    layer_mt.calculate_hidden_gradients(batch_go_mt, next_layer, batch_next_grads, batch_hs_mt, batch_size, 0);

    layer_st.calculate_and_store_gradients(batch_go_st, batch_hs_st, prev_layer, batch_size, 0);
    layer_mt.calculate_and_store_gradients(batch_go_mt, batch_hs_mt, prev_layer, batch_size, 0);

    // Verify weight gradients
    const auto& w_grads_st = layer_st.get_w_grads();
    const auto& w_grads_mt = layer_mt.get_w_grads();
    ASSERT_EQ(w_grads_st.size(), w_grads_mt.size());
    for (size_t i = 0; i < w_grads_st.size(); ++i) 
    {
        EXPECT_NEAR(w_grads_st[i], w_grads_mt[i], 1e-12) << "Weight grad mismatch at index " << i;
    }

    const auto& b_grads_st = layer_st.get_b_grads();
    const auto& b_grads_mt = layer_mt.get_b_grads();
    ASSERT_EQ(b_grads_st.size(), b_grads_mt.size());
    for (size_t i = 0; i < b_grads_st.size(); ++i) 
    {
        EXPECT_NEAR(b_grads_st[i], b_grads_mt[i], 1e-12) << "Bias grad mismatch at index " << i;
    }
}
