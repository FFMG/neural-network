#include <gtest/gtest.h>
#include "layers/lstmlayer.h"
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

    LSTMLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    LSTMLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0, false, std::nullopt);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);
    
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);

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

    LSTMLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    LSTMLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0, false, std::nullopt);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);

    MockLayer prev_layer(0, num_inputs);
    MockLayer next_layer(2, num_neurons);
    next_layer.set_w_values(std::vector<double>(num_neurons * num_neurons, 0.1));

    std::vector<unsigned> topology = { num_inputs, num_neurons, num_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);

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

TEST_F(LSTMLayerMTTest, LayerNormForwardAndBackwardMTConsistency)
{
    // Same shape as BackwardFeedMTConsistency (batch_size=128 is large
    // enough to push calculate_hidden_gradients past its multithreading
    // threshold, dispatching multiple BPTTWorkspace-backed chunks), but
    // with use_layer_normalisation enabled and extended through
    // calculate_and_store_gradients: specifically exercises the
    // per-workspace LayerNorm gain/bias gradient accumulation and its
    // merge back into _ln_c_gain_grads/_ln_c_bias_grads in
    // calculate_hidden_gradients.
    const unsigned num_inputs = 8;
    const unsigned num_neurons = 16;
    const unsigned batch_size = 128;
    const unsigned num_threads = get_test_threads();
    const unsigned num_timesteps = 10;

    LSTMLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, true, std::nullopt);
    LSTMLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0, true, std::nullopt);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);
    std::vector<double> gain(num_neurons), bias(num_neurons);
    for (size_t i = 0; i < num_neurons; ++i)
    {
        gain[i] = 1.0 + 0.1 * std::sin(static_cast<double>(i));
        bias[i] = 0.05 * std::cos(static_cast<double>(i));
    }
    layer_st.set_ln_c_gain_values(gain); layer_st.set_ln_c_bias_values(bias);
    layer_mt.set_ln_c_gain_values(gain); layer_mt.set_ln_c_bias_values(bias);
    layer_st.cache_recurrent_weights();
    layer_mt.cache_recurrent_weights();

    MockLayer prev_layer(0, num_inputs);
    MockLayer next_layer(2, num_neurons);
    next_layer.set_w_values(std::vector<double>(num_neurons * num_neurons, 0.1));

    std::vector<unsigned> topology = { num_inputs, num_neurons, num_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::LayerNormMultiplier);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::LayerNormMultiplier);

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

    for (size_t b = 0; b < batch_size; ++b)
    {
        const auto& out_st = batch_go_st[b].get_rnn_outputs(1);
        const auto& out_mt = batch_go_mt[b].get_rnn_outputs(1);
        ASSERT_EQ(out_st.size(), out_mt.size());
        for (size_t i = 0; i < out_st.size(); ++i)
        {
            EXPECT_NEAR(out_st[i], out_mt[i], 1e-12) << "Forward mismatch at batch " << b << " index " << i;
        }
    }

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

    layer_st.calculate_and_store_gradients(batch_go_st, batch_hs_st, prev_layer, batch_size, 0);
    layer_mt.calculate_and_store_gradients(batch_go_mt, batch_hs_mt, prev_layer, batch_size, 0);

    const auto& ln_gain_grads_st = layer_st.get_ln_c_gain_grads();
    const auto& ln_gain_grads_mt = layer_mt.get_ln_c_gain_grads();
    ASSERT_EQ(ln_gain_grads_st.size(), ln_gain_grads_mt.size());
    for (size_t i = 0; i < ln_gain_grads_st.size(); ++i)
    {
        EXPECT_NEAR(ln_gain_grads_st[i], ln_gain_grads_mt[i], 1e-9) << "LN gain grad mismatch at index " << i;
    }

    const auto& ln_bias_grads_st = layer_st.get_ln_c_bias_grads();
    const auto& ln_bias_grads_mt = layer_mt.get_ln_c_bias_grads();
    ASSERT_EQ(ln_bias_grads_st.size(), ln_bias_grads_mt.size());
    for (size_t i = 0; i < ln_bias_grads_st.size(); ++i)
    {
        EXPECT_NEAR(ln_bias_grads_st[i], ln_bias_grads_mt[i], 1e-9) << "LN bias grad mismatch at index " << i;
    }
}

TEST_F(LSTMLayerMTTest, SmallBatchSizeThresholdFallback)
{
    const unsigned num_inputs = 8;
    const unsigned num_neurons = 16;
    const unsigned num_threads = 8;
    const unsigned batch_size = 4; 
    const unsigned num_timesteps = 10;

    LSTMLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    LSTMLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0, false, std::nullopt);

    std::vector<double> w_vals(num_neurons * num_inputs * 4, 0.15);
    std::vector<double> rw_vals(num_neurons * num_neurons * 4, 0.25);
    std::vector<double> b_vals(num_neurons * 4, 0.05);

    layer_st.set_w_values(w_vals);
    layer_st.set_rw_values(rw_vals);
    layer_st.set_b_values(b_vals);

    layer_mt.set_w_values(w_vals);
    layer_mt.set_rw_values(rw_vals);
    layer_mt.set_b_values(b_vals);

    layer_st.cache_recurrent_weights();
    layer_mt.cache_recurrent_weights();

    MockLayer prev_layer(0, num_inputs);
    MockLayer next_layer(2, num_neurons);
    next_layer.set_w_values(std::vector<double>(num_neurons * num_neurons, 0.1));

    std::vector<unsigned> topology = { num_inputs, num_neurons, num_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);

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
    }
}

TEST_F(LSTMLayerMTTest, OddBatchSizeAllGradsThreadCountInvariance)
{
    const unsigned num_inputs = 8;
    const unsigned num_neurons = 16;
    const unsigned batch_size = 33;
    const unsigned num_timesteps = 7;

    std::vector<unsigned> thread_counts = { 1, 2, 4, 8 };
    std::vector<LSTMLayer> layers;
    for (unsigned tc : thread_counts)
    {
        layers.emplace_back(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, tc, true, 0.0, false, std::nullopt);
        init_layer_weights(layers.back());
        layers.back().cache_recurrent_weights();
    }

    MockLayer prev_layer(0, num_inputs);
    MockLayer next_layer(2, num_neurons);
    next_layer.set_w_values(std::vector<double>(num_neurons * num_neurons, 0.1));

    std::vector<unsigned> topology = { num_inputs, num_neurons, num_neurons };

    std::vector<std::vector<GradientsAndOutputs>> all_batch_go;
    std::vector<std::vector<HiddenStates>> all_batch_hs;

    for (size_t i = 0; i < thread_counts.size(); ++i)
    {
        auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
        auto batch_hs = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);

        for (size_t b = 0; b < batch_size; ++b)
        {
            std::vector<double> inputs(num_inputs * num_timesteps);
            for (size_t k = 0; k < inputs.size(); ++k)
            {
                inputs[k] = std::sin(static_cast<double>((b + 1) * (k + 1)));
            }
            batch_go[b].set_rnn_outputs(0, inputs);
        }

        all_batch_go.push_back(std::move(batch_go));
        all_batch_hs.push_back(std::move(batch_hs));
    }

    for (size_t i = 0; i < thread_counts.size(); ++i)
    {
        layers[i].calculate_forward_feed(all_batch_go[i], prev_layer, {}, all_batch_hs[i], batch_size, true);
    }

    std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_neurons * num_timesteps));
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t k = 0; k < batch_next_grads[b].size(); ++k)
        {
            batch_next_grads[b][k] = std::cos(static_cast<double>(b * k + 1));
        }
    }

    for (size_t i = 0; i < thread_counts.size(); ++i)
    {
        layers[i].calculate_hidden_gradients(all_batch_go[i], next_layer, batch_next_grads, all_batch_hs[i], batch_size, 0);
        layers[i].calculate_and_store_gradients(all_batch_go[i], all_batch_hs[i], prev_layer, batch_size, 0);
    }

    const auto& ref_w = layers[0].get_w_grads();
    const auto& ref_rw = layers[0].get_rw_grads();
    const auto& ref_b = layers[0].get_b_grads();
    const auto& ref_f_w = layers[0].get_f_w_grads();
    const auto& ref_f_rw = layers[0].get_f_rw_grads();
    const auto& ref_f_b = layers[0].get_f_b_grads();
    const auto& ref_i_w = layers[0].get_i_w_grads();
    const auto& ref_i_rw = layers[0].get_i_rw_grads();
    const auto& ref_i_b = layers[0].get_i_b_grads();
    const auto& ref_o_w = layers[0].get_o_w_grads();
    const auto& ref_o_rw = layers[0].get_o_rw_grads();
    const auto& ref_o_b = layers[0].get_o_b_grads();

    for (size_t i = 1; i < thread_counts.size(); ++i)
    {
        const auto& cur_w = layers[i].get_w_grads();
        const auto& cur_rw = layers[i].get_rw_grads();
        const auto& cur_b = layers[i].get_b_grads();
        const auto& cur_f_w = layers[i].get_f_w_grads();
        const auto& cur_f_rw = layers[i].get_f_rw_grads();
        const auto& cur_f_b = layers[i].get_f_b_grads();
        const auto& cur_i_w = layers[i].get_i_w_grads();
        const auto& cur_i_rw = layers[i].get_i_rw_grads();
        const auto& cur_i_b = layers[i].get_i_b_grads();
        const auto& cur_o_w = layers[i].get_o_w_grads();
        const auto& cur_o_rw = layers[i].get_o_rw_grads();
        const auto& cur_o_b = layers[i].get_o_b_grads();

        ASSERT_EQ(ref_w.size(), cur_w.size());
        for (size_t k = 0; k < ref_w.size(); ++k)
        {
            EXPECT_NEAR(ref_w[k], cur_w[k], 1e-12) << "w_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_f_w[k], cur_f_w[k], 1e-12) << "f_w_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_i_w[k], cur_i_w[k], 1e-12) << "i_w_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_o_w[k], cur_o_w[k], 1e-12) << "o_w_grad mismatch at index " << k << " with thread count " << thread_counts[i];
        }

        ASSERT_EQ(ref_rw.size(), cur_rw.size());
        for (size_t k = 0; k < ref_rw.size(); ++k)
        {
            EXPECT_NEAR(ref_rw[k], cur_rw[k], 1e-12) << "rw_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_f_rw[k], cur_f_rw[k], 1e-12) << "f_rw_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_i_rw[k], cur_i_rw[k], 1e-12) << "i_rw_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_o_rw[k], cur_o_rw[k], 1e-12) << "o_rw_grad mismatch at index " << k << " with thread count " << thread_counts[i];
        }

        ASSERT_EQ(ref_b.size(), cur_b.size());
        for (size_t k = 0; k < ref_b.size(); ++k)
        {
            EXPECT_NEAR(ref_b[k], cur_b[k], 1e-12) << "b_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_f_b[k], cur_f_b[k], 1e-12) << "f_b_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_i_b[k], cur_i_b[k], 1e-12) << "i_b_grad mismatch at index " << k << " with thread count " << thread_counts[i];
            EXPECT_NEAR(ref_o_b[k], cur_o_b[k], 1e-12) << "o_b_grad mismatch at index " << k << " with thread count " << thread_counts[i];
        }
    }
}

TEST_F(LSTMLayerMTTest, InferenceForwardFeedMTConsistency)
{
    const unsigned num_inputs = 8;
    const unsigned num_neurons = 16;
    const unsigned batch_size = 128;
    const unsigned num_threads = get_test_threads();
    const unsigned num_timesteps = 10;

    LSTMLayer layer_st(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);
    LSTMLayer layer_mt(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, num_threads, true, 0.0, false, std::nullopt);

    init_layer_weights(layer_st);
    init_layer_weights(layer_mt);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_neurons };

    auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);
    
    auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_timesteps, LSTMLayer::Multiplier);

    for (size_t b = 0; b < batch_size; ++b)
    {
        std::vector<double> inputs(num_inputs * num_timesteps);
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            inputs[i] = std::sin(static_cast<double>(b * (i + 1)));
        }
        batch_go_st[b].set_rnn_outputs(0, inputs);
        batch_go_mt[b].set_rnn_outputs(0, inputs);
    }

    // is_training = false
    layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, false);
    layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, false);

    for (size_t b = 0; b < batch_size; ++b)
    {
        const auto& out_st = batch_go_st[b].get_rnn_outputs(1);
        const auto& out_mt = batch_go_mt[b].get_rnn_outputs(1);
        ASSERT_EQ(out_st.size(), out_mt.size());
        for (size_t i = 0; i < out_st.size(); ++i)
        {
            EXPECT_NEAR(out_st[i], out_mt[i], 1e-12) << "Inference mismatch at batch " << b << " index " << i;
        }
    }
}
