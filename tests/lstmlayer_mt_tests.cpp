#include <gtest/gtest.h>
#include "../src/neuralnetwork/lstmlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
using namespace test_helper;

class LSTMLayerMTTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
    }

    // Helper to initialize layer weights deterministically
    void init_layer_weights(LSTMLayer& layer) 
    {
        unsigned num_neurons = layer.get_number_neurons();
        unsigned num_inputs = layer.get_number_input_neurons();

        // LSTM has 4 gates: candidate, forget, input, output
        std::vector<double> w_vals(num_neurons * num_inputs * 4);
        std::vector<double> rw_vals(num_neurons * num_neurons * 4);
        std::vector<double> b_vals(num_neurons * 4);

        for (size_t i = 0; i < w_vals.size(); ++i) 
        {
          w_vals[i] = std::sin(static_cast<double>(i));
        }
        for (size_t i = 0; i < rw_vals.size(); ++i) 
        {
          rw_vals[i] = std::cos(static_cast<double>(i));
        }
        for (size_t i = 0; i < b_vals.size(); ++i) 
        {
          b_vals[i] = std::sin(static_cast<double>(i) * 0.5);
        }

        layer.set_w_values(w_vals);
        layer.set_rw_values(rw_vals);
        layer.set_b_values(b_vals);
        
        // Specifically set gate weights as well to be sure
        layer.set_f_w_values(std::vector<double>(w_vals.begin(), w_vals.begin() + num_neurons * num_inputs));
        layer.set_i_w_values(std::vector<double>(w_vals.begin() + num_neurons * num_inputs, w_vals.begin() + 2 * num_neurons * num_inputs));
        layer.set_o_w_values(std::vector<double>(w_vals.begin() + 2 * num_neurons * num_inputs, w_vals.begin() + 3 * num_neurons * num_inputs));
        
        layer.set_f_rw_values(std::vector<double>(rw_vals.begin(), rw_vals.begin() + num_neurons * num_neurons));
        layer.set_i_rw_values(std::vector<double>(rw_vals.begin() + num_neurons * num_neurons, rw_vals.begin() + 2 * num_neurons * num_neurons));
        layer.set_o_rw_values(std::vector<double>(rw_vals.begin() + 2 * num_neurons * num_neurons, rw_vals.begin() + 3 * num_neurons * num_neurons));
    }
};

TEST_F(LSTMLayerMTTest, ForwardFeedMTConsistency) 
{
    const unsigned num_inputs = 8;
    const unsigned num_neurons = 16;
    const unsigned batch_size = 128;
    const unsigned num_threads = get_test_threads();
    const unsigned num_timesteps = 10;

    LSTMLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    LSTMLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_timesteps, 5);
    
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_timesteps, 5);

    for (size_t b = 0; b < batch_size; ++b) 
    {
        std::vector<double> inputs(num_inputs * num_timesteps);
        for (size_t i = 0; i < inputs.size(); ++i) 
        {
          inputs[i] = std::cos(static_cast<double>(b * i));
        }
        batch_go_st[b].set_rnn_outputs(0, inputs);
        batch_go_mt[b].set_rnn_outputs(0, inputs);
    }

    layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, false);
    layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, false);

    for (size_t b = 0; b < batch_size; ++b) 
    {
        const auto& out_st = batch_go_st[b].get_rnn_outputs(1);
        const auto& out_mt = batch_go_mt[b].get_rnn_outputs(1);
        ASSERT_EQ(out_st.size(), out_mt.size());
        for (size_t i = 0; i < out_st.size(); ++i) 
        {
            EXPECT_NEAR(out_st[i], out_mt[i], 1e-12);
        }
    }
}

TEST_F(LSTMLayerMTTest, BackwardFeedMTConsistency) 
{
    const unsigned num_inputs = 8;
    const unsigned num_neurons = 16;
    const unsigned batch_size = 128;
    const unsigned num_threads = get_test_threads();
    const unsigned num_timesteps = 10;

    LSTMLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    LSTMLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);

    MockLayer prev_layer(0, num_inputs);
    MockLayer next_layer(2, num_neurons);
    next_layer.set_w_values(std::vector<double>(num_neurons * num_neurons, 0.1));

    std::vector<unsigned> topology = { num_inputs, num_neurons, num_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_timesteps, 5);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_timesteps, 5);

    for (size_t b = 0; b < batch_size; ++b) 
    {
        std::vector<double> inputs(num_inputs * num_timesteps);
        for (size_t i = 0; i < inputs.size(); ++i) 
        {
          inputs[i] = std::cos(static_cast<double>(b + i));
        }
        batch_go_st[b].set_rnn_outputs(0, inputs);
        batch_go_mt[b].set_rnn_outputs(0, inputs);
    }

    layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, true);
    layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, true);

    std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_neurons * num_timesteps));
    for (size_t b = 0; b < batch_size; ++b) 
    {
        for (size_t i = 0; i < batch_next_grads[b].size(); ++i) 
        {
          batch_next_grads[b][i] = std::sin(static_cast<double>(b * i));
        }
    }

    layer_st.calculate_hidden_gradients(batch_go_st, next_layer, batch_next_grads, batch_hs_st, batch_size, 0);
    layer_mt.calculate_hidden_gradients(batch_go_mt, next_layer, batch_next_grads, batch_hs_mt, batch_size, 0);

    for (size_t b = 0; b < batch_size; ++b) 
    {
        const auto& g_st = batch_go_st[b].get_rnn_gate_gradients(1);
        const auto& g_mt = batch_go_mt[b].get_rnn_gate_gradients(1);
        ASSERT_EQ(g_st.size(), g_mt.size());
        for (size_t i = 0; i < g_st.size(); ++i) 
        {
            EXPECT_NEAR(g_st[i], g_mt[i], 1e-12);
        }

        const auto& in_g_st = batch_go_st[b].get_rnn_gradients(1);
        const auto& in_g_mt = batch_go_mt[b].get_rnn_gradients(1);
        ASSERT_EQ(in_g_st.size(), in_g_mt.size());
        for (size_t i = 0; i < in_g_st.size(); ++i) 
        {
            EXPECT_NEAR(in_g_st[i], in_g_mt[i], 1e-12);
        }
    }
}
