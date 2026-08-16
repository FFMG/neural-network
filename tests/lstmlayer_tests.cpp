#include <gtest/gtest.h>
#include "layers/lstmlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
using namespace test_helper;

class LSTMLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
  }
};

TEST_F(LSTMLayerTest, ConstructionAndTopology) {
  LSTMLayer layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  EXPECT_EQ(layer.get_layer_index(), 1);
  EXPECT_EQ(layer.get_number_input_neurons(), 2);
  EXPECT_EQ(layer.get_number_output_neurons(), 3);
  EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Lstm);
  EXPECT_TRUE(layer.use_bptt());
  EXPECT_EQ(layer.get_pre_activation_multiplier(), LSTMLayer::Multiplier);
}

TEST_F(LSTMLayerTest, ForwardFeedMathematicalVerification) {
  // 1 input, 1 hidden neuron
  LSTMLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0, false, std::nullopt);

  // Set weights to simple values
  layer.set_w_values({ 0.5 });
  layer.set_rw_values({ 0.1 });
  layer.set_f_w_values({ 0.2 });
  layer.set_f_rw_values({ 0.3 });
  layer.set_i_w_values({ 0.4 });
  layer.set_i_rw_values({ 0.5 });
  layer.set_o_w_values({ 0.6 });
  layer.set_o_rw_values({ 0.7 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier); // 1 time step

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 0.174207488, 1e-6);
}

TEST_F(LSTMLayerTest, LayerNormForwardNormalizesCellState) {
  // 1 input, 2 neurons, single timestep, LayerNorm enabled on c_t.
  LSTMLayer layer(1, 1, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0, true, std::nullopt);

  layer.set_f_w_values({ 0.1, -0.3 }); layer.set_f_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  layer.set_i_w_values({ 0.5, 0.2 });  layer.set_i_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  layer.set_o_w_values({ 0.2, 0.4 });  layer.set_o_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  layer.set_w_values({ 0.6, -0.4 });   layer.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  layer.set_ln_c_gain_values({ 1.5, 1.5 });
  layer.set_ln_c_bias_values({ 0.2, -0.2 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::LayerNormMultiplier);

  batch_go[0].set_rnn_outputs(0, { 1.0 });
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

  // f = sigmoid([0.1,-0.3]), i = sigmoid([0.5,0.2]), o = sigmoid([0.2,0.4]), g = tanh([0.6,-0.4])
  // c_t (raw) = i * g (c_prev=0) = [0.334291, -0.208909]
  // mean = 0.062691, dev = +/-0.271600, inv_std = 1/sqrt(dev^2+1e-5) ~ 3.68152
  // a_hat ~ [0.999932, -0.999932]; y = gain*a_hat+bias = [1.699898, -1.699898]
  // h_t = o * tanh(y) ~ [0.549834*tanh(1.699898), 0.598688*tanh(-1.699898)]
  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 0.514322, 1e-3);
  EXPECT_NEAR(rnn_out[1], -0.560018, 1e-3);
}

TEST_F(LSTMLayerTest, LayerNormDisabledMatchesUnnormalizedForwardFeed) {
  LSTMLayer layer(1, 1, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0, false, std::nullopt);

  layer.set_f_w_values({ 0.1, -0.3 }); layer.set_f_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  layer.set_i_w_values({ 0.5, 0.2 });  layer.set_i_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  layer.set_o_w_values({ 0.2, 0.4 });  layer.set_o_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  layer.set_w_values({ 0.6, -0.4 });   layer.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });

  EXPECT_FALSE(layer.get_use_layer_normalisation());

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 1.0 });
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

  // h_t = o * tanh(c_t_raw), c_t_raw = i*g = [0.334291, -0.208909]
  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 0.549834 * std::tanh(0.334291), 1e-4);
  EXPECT_NEAR(rnn_out[1], 0.598688 * std::tanh(-0.208909), 1e-4);
}

TEST_F(LSTMLayerTest, LayerNormGainBiasGradientsMatchNumericalGradient) {
  // Numerical-gradient check of the LayerNorm backward wiring added to
  // calculate_bptt_batch_chunk: seeds an arbitrary upstream gradient dy via
  // batch_next_grads (direct-injection, same mechanism used elsewhere in
  // this file) and treats loss(gain, bias) = dot(dy, h_t(gain, bias)) as a
  // plain scalar function, independently verified via central finite
  // differences. This also exercises the do_out (output-gate gradient)
  // dh_curr/dc_next substitution carefully, since (unlike GRU) it depends
  // on dh_curr directly rather than solely through dc.
  const unsigned num_inputs = 1;
  const unsigned num_outputs = 2;
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  auto make_layer = [&](const std::vector<double>& gain, const std::vector<double>& bias)
  {
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, true, std::nullopt);
    layer.set_f_w_values({ 0.12, -0.24 }); layer.set_f_rw_values({ 0.05, -0.03, 0.02, 0.04 }); layer.set_f_b_values({ 0.01, -0.02 });
    layer.set_i_w_values({ 0.31, 0.18 });  layer.set_i_rw_values({ -0.04, 0.06, 0.03, -0.05 }); layer.set_i_b_values({ 0.0, 0.03 });
    layer.set_o_w_values({ -0.22, 0.27 }); layer.set_o_rw_values({ 0.07, -0.02, -0.06, 0.08 }); layer.set_o_b_values({ -0.01, 0.02 });
    layer.set_w_values({ 0.35, -0.19 });   layer.set_rw_values({ -0.03, 0.05, 0.04, -0.06 });   layer.set_b_values({ 0.02, -0.01 });
    layer.set_ln_c_gain_values(gain);
    layer.set_ln_c_bias_values(bias);
    return layer;
  };

  auto run_loss = [&](const std::vector<double>& gain, const std::vector<double>& bias, const std::vector<double>& dy) -> double
  {
    auto layer = make_layer(gain, bias);
    MockLayer prev_layer(0, num_inputs);
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::LayerNormMultiplier);
    batch_go[0].set_rnn_outputs(0, { 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
    const auto out = batch_go[0].get_rnn_outputs(1);
    double loss = 0.0;
    for (unsigned j = 0; j < num_outputs; ++j)
    {
      loss += dy[j] * out[j];
    }
    return loss;
  };

  const std::vector<double> gain = { 1.2, 0.9 };
  const std::vector<double> bias = { 0.05, -0.15 };
  const std::vector<double> dy = { 0.5, -0.7 };

  auto layer = make_layer(gain, bias);
  MockLayer prev_layer(0, num_inputs);
  MockLayer next_layer(2, num_outputs);
  // Identity weight matrix so batch_next_grads is projected through
  // unchanged (dh_t = I^T * dy = dy); without this, next_layer's weight
  // vector is empty and the backward GEMV reads out of bounds.
  {
    std::vector<double> identity(num_outputs * num_outputs, 0.0);
    for (unsigned j = 0; j < num_outputs; ++j)
    {
      identity[j * num_outputs + j] = 1.0;
    }
    next_layer.set_w_values(identity);
  }
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::LayerNormMultiplier);
  batch_go[0].set_rnn_outputs(0, { 1.0 });
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  std::vector<std::vector<double>> batch_next_grads = { dy };
  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

  const auto& gain_grads = layer.get_ln_c_gain_grads();
  const auto& bias_grads = layer.get_ln_c_bias_grads();
  ASSERT_EQ(gain_grads.size(), num_outputs);
  ASSERT_EQ(bias_grads.size(), num_outputs);

  const double h = 1e-6;
  for (unsigned j = 0; j < num_outputs; ++j)
  {
    std::vector<double> gain_plus = gain; gain_plus[j] += h;
    std::vector<double> gain_minus = gain; gain_minus[j] -= h;
    const double numerical_gain_grad = (run_loss(gain_plus, bias, dy) - run_loss(gain_minus, bias, dy)) / (2.0 * h);
    EXPECT_NEAR(gain_grads[j], numerical_gain_grad, 1e-4) << "gain[" << j << "]";

    std::vector<double> bias_plus = bias; bias_plus[j] += h;
    std::vector<double> bias_minus = bias; bias_minus[j] -= h;
    const double numerical_bias_grad = (run_loss(gain, bias_plus, dy) - run_loss(gain, bias_minus, dy)) / (2.0 * h);
    EXPECT_NEAR(bias_grads[j], numerical_bias_grad, 1e-4) << "bias[" << j << "]";
  }
}

TEST_F(LSTMLayerTest, DropoutConsistencyVerification) {
  // 1 neuron with 100% dropout
  LSTMLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 1.0, nullptr, 1, false, 0.0, false, std::nullopt);
  layer.set_w_values({ 1.0 });
  layer.set_rw_values({ 1.0 });
  layer.set_f_w_values({ 1.0 });
  layer.set_f_rw_values({ 1.0 });
  layer.set_i_w_values({ 1.0 });
  layer.set_i_rw_values({ 1.0 });
  layer.set_o_w_values({ 1.0 });
  layer.set_o_rw_values({ 1.0 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
  EXPECT_NEAR(batch_go[0].get_rnn_outputs(1)[0], 0.0, 1e-9);

  MockLayer next_layer(2, 1);
  next_layer.set_w_values({ 1.0 });
  std::vector<std::vector<double>> batch_next_grads = { { 10.0 } };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

  EXPECT_NEAR(batch_go[0].get_rnn_gate_gradients(1)[0], 0.0, 1e-9);
}

TEST_F(LSTMLayerTest, DropoutStatisticalVerification) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 5000;
    double dropout_rate = 0.5;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, false, std::nullopt);

    layer.set_w_values(std::vector<double>(num_outputs * 4, 1.0));
    layer.set_rw_values(std::vector<double>(num_outputs * num_outputs * 4, 0.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));
    
    layer.set_i_b_values(std::vector<double>(num_outputs, 10.0)); // input gate open
    layer.set_f_b_values(std::vector<double>(num_outputs, -10.0)); // forget gate closed
    layer.set_o_b_values(std::vector<double>(num_outputs, 10.0)); // output gate open

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_outputs(1);
    int dropped_count = 0;
    int kept_count = 0;
    const double expected_kept = 1.0 / (1.0 - dropout_rate);
    for (size_t i = 0; i < outputs.size(); ++i) {
        double out = outputs[i];
        if (out == 0.0) {
            dropped_count++;
        } else if (std::abs(out - expected_kept) < 0.05) {
            kept_count++;
        } else {
            Logger::error("Neuron ", i, " output unexpected value: ", out, " (expected 0.0 or ~", expected_kept, ")");
        }
    }

    EXPECT_EQ(dropped_count + kept_count, (int)num_outputs);
    EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.08);
}

TEST_F(LSTMLayerTest, DropoutNotInference) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, false, std::nullopt);

    layer.set_w_values(std::vector<double>(num_outputs * 4, 1.0));
    layer.set_rw_values(std::vector<double>(num_outputs * num_outputs * 4, 0.0));
    layer.set_b_values(std::vector<double>(num_outputs, 0.0));
    
    layer.set_i_b_values(std::vector<double>(num_outputs, 10.0));
    layer.set_f_b_values(std::vector<double>(num_outputs, -10.0));
    layer.set_o_b_values(std::vector<double>(num_outputs, 10.0));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& outputs = batch_go[0].get_outputs(1);
    for (double out : outputs) {
        EXPECT_NEAR(out, 1.0, 1e-2);
    }
}

TEST_F(LSTMLayerTest, LearningRateRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    std::vector<double> learning_rates = { 0.0, 0.0001, 0.01, 0.5, 1.0, 2.0 };
    
    for (double lr : learning_rates) {
        layer.set_w_values({ 1.0 });
        layer.set_rw_values({ 1.0 });
        layer.set_b_values({ 0.5 });
        layer.set_f_w_values({ 1.0 });
        layer.set_f_rw_values({ 1.0 });
        layer.set_f_b_values({ 0.5 });
        layer.set_i_w_values({ 1.0 });
        layer.set_i_rw_values({ 1.0 });
        layer.set_i_b_values({ 0.5 });
        layer.set_o_w_values({ 1.0 });
        layer.set_o_rw_values({ 1.0 });
        layer.set_o_b_values({ 0.5 });
        
        layer.set_w_grads({ 0.1 });
        layer.set_rw_grads({ 0.1 });
        layer.set_b_grads({ 0.05 });
        layer.set_f_w_grads({ 0.1 });
        layer.set_f_rw_grads({ 0.1 });
        layer.set_f_b_grads({ 0.05 });
        layer.set_i_w_grads({ 0.1 });
        layer.set_i_rw_grads({ 0.1 });
        layer.set_i_b_grads({ 0.05 });
        layer.set_o_w_grads({ 0.1 });
        layer.set_o_rw_grads({ 0.1 });
        layer.set_o_b_grads({ 0.05 });

        layer.apply_stored_gradients(lr, 1.0);

        EXPECT_NEAR(layer.get_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_b_values()[0], 0.5 - lr * 0.05, 1e-9);
        EXPECT_NEAR(layer.get_f_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_f_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_f_b_values()[0], 0.5 - lr * 0.05, 1e-9);
        EXPECT_NEAR(layer.get_i_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_i_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_i_b_values()[0], 0.5 - lr * 0.05, 1e-9);
        EXPECT_NEAR(layer.get_o_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_o_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_o_b_values()[0], 0.5 - lr * 0.05, 1e-9);
    }
}

TEST_F(LSTMLayerTest, BPTTRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    layer.set_f_w_values({ 0.2 }); layer.set_f_rw_values({ 0.3 }); layer.set_f_b_values({ 0.1 });
    layer.set_i_w_values({ 0.4 }); layer.set_i_rw_values({ 0.5 }); layer.set_i_b_values({ 0.1 });
    layer.set_o_w_values({ 0.6 }); layer.set_o_rw_values({ 0.7 }); layer.set_o_b_values({ 0.1 });
    layer.set_w_values({ 0.5 });   layer.set_rw_values({ 0.1 });   layer.set_b_values({ 0.1 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2, LSTMLayer::Multiplier); // 2 steps, correct multiplier

    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0 });
    std::vector<std::vector<double>> batch_next_grads = { { 0.0, 1.0 } }; // dL/dh_1 = 1.0

    // Test BPTT=1
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 1);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 1);

    EXPECT_NEAR(layer.get_w_grads()[0],   0.23520037, 1e-6);
    EXPECT_NEAR(layer.get_rw_grads()[0],  0.05066328, 1e-6);
    EXPECT_NEAR(layer.get_b_grads()[0],   0.23520037, 1e-6);
    EXPECT_NEAR(layer.get_f_w_grads()[0], 0.04226026, 1e-6);
    EXPECT_NEAR(layer.get_f_rw_grads()[0],0.00910306, 1e-6);
    EXPECT_NEAR(layer.get_i_w_grads()[0], 0.06588160, 1e-6);
    EXPECT_NEAR(layer.get_i_rw_grads()[0],0.01419121, 1e-6);
    EXPECT_NEAR(layer.get_o_w_grads()[0], 0.10571330, 1e-6);
    EXPECT_NEAR(layer.get_o_rw_grads()[0],0.02277115, 1e-6);

    // Test BPTT=2 (Full sequence)
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 2);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 2);

    EXPECT_NEAR(layer.get_w_grads()[0],   0.40978412, 1e-6);
    EXPECT_NEAR(layer.get_rw_grads()[0],  0.05066328, 1e-6);
    EXPECT_NEAR(layer.get_b_grads()[0],   0.40978412, 1e-6);
    EXPECT_NEAR(layer.get_f_w_grads()[0], 0.04226026, 1e-6);
    EXPECT_NEAR(layer.get_f_rw_grads()[0],0.00910306, 1e-6);
    EXPECT_NEAR(layer.get_i_w_grads()[0], 0.11562776, 1e-6);
    EXPECT_NEAR(layer.get_i_rw_grads()[0],0.01419121, 1e-6);
    EXPECT_NEAR(layer.get_o_w_grads()[0], 0.11594396, 1e-6);
    EXPECT_NEAR(layer.get_o_rw_grads()[0],0.02277115, 1e-6);
}

TEST_F(LSTMLayerTest, ForwardFeedCachesActivatedCandidateAndCellStateForBptt) {
  // BPTT re-derives dtanh(g)/dtanh(c) every timestep purely from cached state,
  // so the forward pass must store the *activated* candidate (g) and cell (c)
  // values it already computed, rather than making BPTT recompute tanh() from
  // scratch. This test pins that contract directly against HiddenState storage,
  // independently of any end-to-end gradient values.
  const unsigned num_inputs = 2;
  const unsigned num_neurons = 2;
  LSTMLayer layer(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  layer.set_w_values({ 0.6, -0.4, 0.3, -0.2 });
  layer.set_rw_values({ 0.15, -0.25, 0.35, -0.1 });
  layer.set_b_values({ 0.05, -0.05 });

  layer.set_f_w_values({ 0.2, 0.1, -0.1, 0.3 });
  layer.set_f_rw_values({ 0.1, 0.2, -0.2, 0.1 });
  layer.set_f_b_values({ 0.0, 0.0 });

  layer.set_i_w_values({ 0.3, -0.2, 0.1, 0.2 });
  layer.set_i_rw_values({ -0.1, 0.2, 0.1, -0.2 });
  layer.set_i_b_values({ 0.0, 0.0 });

  layer.set_o_w_values({ 0.25, 0.15, -0.15, 0.2 });
  layer.set_o_rw_values({ 0.2, -0.1, 0.1, 0.3 });
  layer.set_o_b_values({ 0.0, 0.0 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_neurons };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  const size_t num_time_steps = 2;
  auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps, LSTMLayer::Multiplier);

  batch_go[0].set_rnn_outputs(0, { 0.5, -0.3, 0.2, 0.7 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  const auto& states = batch_hs[0].at(layer.get_layer_index());
  ASSERT_EQ(states.size(), num_time_steps);

  for (size_t t = 0; t < num_time_steps; ++t)
  {
    const auto packed = states[t].get_pre_activation_sums();
    ASSERT_EQ(packed.size(), LSTMLayer::Multiplier * num_neurons);

    const auto cell = states[t].get_cell_state_values();
    ASSERT_EQ(cell.size(), num_neurons);

    for (unsigned n = 0; n < num_neurons; ++n)
    {
      const double raw_g = packed[3 * num_neurons + n];
      const double cached_activated_g = packed[5 * num_neurons + n];
      EXPECT_NEAR(cached_activated_g, std::tanh(raw_g), 1e-12);
      // Confirm the test setup isn't degenerate (raw g == 0 would make
      // tanh(g) == g and hide a caching bug that stored the raw value).
      EXPECT_NE(raw_g, 0.0);

      const double raw_c = cell[n];
      const double cached_activated_c = packed[6 * num_neurons + n];
      EXPECT_NEAR(cached_activated_c, std::tanh(raw_c), 1e-12);
      EXPECT_NE(raw_c, 0.0);
    }
  }
}

TEST_F(LSTMLayerTest, ApplyStoredGradientsCacheUpdate)
{
    LSTMLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    layer.set_w_values({ 1.0 });   layer.set_rw_values({ 0.5 });
    layer.set_f_w_values({ 0.0 }); layer.set_f_rw_values({ 0.0 });
    layer.set_i_w_values({ 0.0 }); layer.set_i_rw_values({ 0.0 });
    layer.set_o_w_values({ 0.0 }); layer.set_o_rw_values({ 0.0 });

    layer.set_f_b_values({ 10.0 });
    layer.set_i_b_values({ 10.0 });
    layer.set_o_b_values({ 10.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 1 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2, LSTMLayer::Multiplier); 

    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    auto outputs = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs[0], 0.999909, 1e-4);
    EXPECT_NEAR(outputs[1], 2.49968, 1e-4);

    layer.set_f_rw_grads({ 0.1 });
    layer.set_i_rw_grads({ 0.1 });
    layer.set_o_rw_grads({ 0.1 });
    layer.set_rw_grads({ 0.1 });
    layer.apply_stored_gradients(1.0, 1.0);

    EXPECT_NEAR(layer.get_f_rw_values()[0], -0.1, 1e-9);
    EXPECT_NEAR(layer.get_i_rw_values()[0], -0.1, 1e-9);
    EXPECT_NEAR(layer.get_o_rw_values()[0], -0.1, 1e-9);
    EXPECT_NEAR(layer.get_rw_values()[0], 0.4, 1e-9);

    auto batch_hs2 = create_batch_hidden_states(topology, 1, 2, LSTMLayer::Multiplier); 
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs2, 1, false);

    auto outputs2 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs2[0], 0.999909, 1e-4);
    EXPECT_NEAR(outputs2[1], 2.39968, 1e-4);
}

TEST_F(LSTMLayerTest, BiasCachingCorrectness)
{
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    layer.set_w_values({ 1.0 });   layer.set_rw_values({ 0.0 });
    layer.set_f_w_values({ 0.0 }); layer.set_f_rw_values({ 0.0 });
    layer.set_i_w_values({ 0.0 }); layer.set_i_rw_values({ 0.0 });
    layer.set_o_w_values({ 0.0 }); layer.set_o_rw_values({ 0.0 });

    layer.set_f_b_values({ 10.0 });
    layer.set_i_b_values({ 10.0 });
    layer.set_o_b_values({ 10.0 });
    layer.set_b_values({ 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);

    batch_go[0].set_rnn_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);
    auto outputs1 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs1[0], 1.0, 1e-3);

    layer.set_b_values({ 10.0, 10.0, 10.0, 2.0 });

    auto batch_hs2 = create_batch_hidden_states(topology, 1, 1, LSTMLayer::Multiplier);
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs2, 1, false);
    auto outputs2 = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs2[0], 3.0, 1e-3);
}

TEST_F(LSTMLayerTest, TransposedWeightsAndFastBpttPassCorrectness) {
    // 2 inputs, 2 neurons, batch size 2, 2 time steps
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    // Populate weights
    layer.set_f_w_values({ 0.1, 0.2, 0.3, 0.4 });
    layer.set_f_rw_values({ 0.15, 0.25, 0.35, 0.45 });
    layer.set_f_b_values({ 0.05, 0.15 });

    layer.set_i_w_values({ 0.2, 0.3, 0.4, 0.5 });
    layer.set_i_rw_values({ 0.25, 0.35, 0.45, 0.55 });
    layer.set_i_b_values({ 0.06, 0.16 });

    layer.set_o_w_values({ 0.3, 0.4, 0.5, 0.6 });
    layer.set_o_rw_values({ 0.35, 0.45, 0.55, 0.65 });
    layer.set_o_b_values({ 0.07, 0.17 });

    layer.set_w_values({ 0.4, 0.5, 0.6, 0.7 });
    layer.set_rw_values({ 0.45, 0.55, 0.65, 0.75 });
    layer.set_b_values({ 0.08, 0.18 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2); // batch size 2
    auto batch_hs = create_batch_hidden_states(topology, 2, 2, LSTMLayer::Multiplier); // 2 steps

    batch_go[0].set_rnn_outputs(0, { 1.0, 1.0, 0.5, 0.5 });
    batch_go[1].set_rnn_outputs(0, { 0.8, 0.8, 0.4, 0.4 });

    // Forward pass
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, false);

    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0, 0.5, 0.2, 0.8 });
    std::vector<std::vector<double>> batch_next_grads = {
        { 0.1, 0.2, 0.3, 0.4 },
        { 0.5, 0.6, 0.7, 0.8 }
    };

    // Initialize cell state values so that gradients propagate through tanh's derivatives
    batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0 });
    batch_hs[0].at(1, 1).set_cell_state_values({ 1.0, 1.0 });
    batch_hs[1].at(1, 0).set_cell_state_values({ 1.0, 1.0 });
    batch_hs[1].at(1, 1).set_cell_state_values({ 1.0, 1.0 });

    // Backward pass (BPTT = 2)
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 2, 2);
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 2, 2);

    // Verify gradients are non-zero and accumulated successfully
    EXPECT_GT(std::abs(layer.get_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_rw_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_f_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_f_rw_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_i_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_i_rw_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_o_w_grads()[0]), 0.0);
    EXPECT_GT(std::abs(layer.get_o_rw_grads()[0]), 0.0);
}

TEST_F(LSTMLayerTest, BPTTWorkspaceResizeCorrectness) {
    // 2 inputs, 2 outputs, batch size 2, 2 time steps
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    layer.set_f_w_values({ 0.1, 0.2, 0.3, 0.4 });
    layer.set_f_rw_values({ 0.15, 0.25, 0.35, 0.45 });
    layer.set_f_b_values({ 0.05, 0.15 });
    layer.set_i_w_values({ 0.2, 0.3, 0.4, 0.5 });
    layer.set_i_rw_values({ 0.25, 0.35, 0.45, 0.55 });
    layer.set_i_b_values({ 0.06, 0.16 });
    layer.set_o_w_values({ 0.3, 0.4, 0.5, 0.6 });
    layer.set_o_rw_values({ 0.35, 0.45, 0.55, 0.65 });
    layer.set_o_b_values({ 0.07, 0.17 });
    layer.set_w_values({ 0.4, 0.5, 0.6, 0.7 });
    layer.set_rw_values({ 0.45, 0.55, 0.65, 0.75 });
    layer.set_b_values({ 0.08, 0.18 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

    // Run backprop first time (creates workspace initially)
    auto batch_go1 = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs1 = create_batch_hidden_states(topology, 2, 2, LSTMLayer::Multiplier);
    batch_go1[0].set_rnn_outputs(0, { 1.0, 1.0, 0.5, 0.5 });
    batch_go1[1].set_rnn_outputs(0, { 0.8, 0.8, 0.4, 0.4 });
    layer.calculate_forward_feed(batch_go1, prev_layer, {}, batch_hs1, 2, false);

    MockLayer next_layer(2, num_outputs);
    next_layer.set_w_values({ 1.0, 0.5, 0.2, 0.8 });
    std::vector<std::vector<double>> batch_next_grads = {
        { 0.1, 0.2, 0.3, 0.4 },
        { 0.5, 0.6, 0.7, 0.8 }
    };

    layer.calculate_hidden_gradients(batch_go1, next_layer, batch_next_grads, batch_hs1, 2, 2);
    layer.calculate_and_store_gradients(batch_go1, batch_hs1, prev_layer, 2, 2);
    double initial_norm = layer.get_gradient_norm_sq();
    EXPECT_GT(initial_norm, 0.0);

    // Run backprop second time with the SAME sizes (tests std::fill workspace reuse path)
    layer.zero_gradients();
    layer.calculate_hidden_gradients(batch_go1, next_layer, batch_next_grads, batch_hs1, 2, 2);
    layer.calculate_and_store_gradients(batch_go1, batch_hs1, prev_layer, 2, 2);
    EXPECT_NEAR(layer.get_gradient_norm_sq(), initial_norm, 1e-9);

    // Run backprop third time with DIFFERENT sizes (tests resize/assign reallocation path)
    layer.zero_gradients();
    auto batch_go2 = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs2 = create_batch_hidden_states(topology, 1, 2, LSTMLayer::Multiplier);
    batch_go2[0].set_rnn_outputs(0, { 1.0, 1.0, 0.5, 0.5 });
    layer.calculate_forward_feed(batch_go2, prev_layer, {}, batch_hs2, 1, false);

    std::vector<std::vector<double>> batch_next_grads2 = { { 0.1, 0.2, 0.3, 0.4 } };
    layer.calculate_hidden_gradients(batch_go2, next_layer, batch_next_grads2, batch_hs2, 1, 2);
    layer.calculate_and_store_gradients(batch_go2, batch_hs2, prev_layer, 1, 2);
    EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
}

TEST_F(LSTMLayerTest, SingleVSMultiThreadedEquivalence)
{
  unsigned num_inputs = 100;
  unsigned num_outputs = 100;
  size_t batch_size = 100;
  size_t num_time_steps = 20;

  // Layer 1: single threaded
  LSTMLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  // Layer 2: multi threaded
  LSTMLayer layer_mt(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 4, true, 0.0, false, std::nullopt);

  // Helper to fill vectors with identical values
  auto initialize_weights = [&](LSTMLayer& l)
  {
    l.set_w_values(std::vector<double>(num_inputs * num_outputs, 0.05));
    l.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.08));
    l.set_b_values(std::vector<double>(num_outputs, 0.01));

    l.set_f_w_values(std::vector<double>(num_inputs * num_outputs, 0.06));
    l.set_f_rw_values(std::vector<double>(num_outputs * num_outputs, 0.09));
    l.set_f_b_values(std::vector<double>(num_outputs, 0.02));

    l.set_i_w_values(std::vector<double>(num_inputs * num_outputs, 0.07));
    l.set_i_rw_values(std::vector<double>(num_outputs * num_outputs, 0.10));
    l.set_i_b_values(std::vector<double>(num_outputs, 0.03));

    l.set_o_w_values(std::vector<double>(num_inputs * num_outputs, 0.04));
    l.set_o_rw_values(std::vector<double>(num_outputs * num_outputs, 0.07));
    l.set_o_b_values(std::vector<double>(num_outputs, 0.04));
  };

  initialize_weights(layer_st);
  initialize_weights(layer_mt);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  // Setup batch inputs and next gradients
  auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_time_steps, LSTMLayer::Multiplier);
  auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_time_steps, LSTMLayer::Multiplier);

  std::vector<double> inputs(num_time_steps * num_inputs, 0.5);
  std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_time_steps * num_outputs, 0.25));

  for (size_t b = 0; b < batch_size; ++b)
  {
    batch_go_st[b].set_rnn_outputs(0, inputs);
    batch_go_mt[b].set_rnn_outputs(0, inputs);
  }

  // Forward feed
  layer_st.calculate_forward_feed(batch_go_st, prev_layer, {}, batch_hs_st, batch_size, false);
  layer_mt.calculate_forward_feed(batch_go_mt, prev_layer, {}, batch_hs_mt, batch_size, false);

  // Backward feed
  MockLayer next_layer(2, num_outputs);
  std::vector<double> next_weights(num_outputs * num_outputs, 0.1);
  next_layer.set_w_values(next_weights);

  layer_st.calculate_hidden_gradients(batch_go_st, next_layer, batch_next_grads, batch_hs_st, batch_size, static_cast<int>(num_time_steps));
  layer_mt.calculate_hidden_gradients(batch_go_mt, next_layer, batch_next_grads, batch_hs_mt, batch_size, static_cast<int>(num_time_steps));

  // Store gradients
  layer_st.calculate_and_store_gradients(batch_go_st, batch_hs_st, prev_layer, batch_size, static_cast<int>(num_time_steps));
  layer_mt.calculate_and_store_gradients(batch_go_mt, batch_hs_mt, prev_layer, batch_size, static_cast<int>(num_time_steps));

  // Helper to assert two vectors are equal within tolerance
  auto assert_vectors_equal = [](const std::vector<double>& v1, const std::vector<double>& v2)
  {
    ASSERT_EQ(v1.size(), v2.size());
    for (size_t i = 0; i < v1.size(); ++i)
    {
      EXPECT_NEAR(v1[i], v2[i], 1e-9);
    }
  };

  // Assert all gradients are identical
  assert_vectors_equal(layer_st.get_w_grads(), layer_mt.get_w_grads());
  assert_vectors_equal(layer_st.get_rw_grads(), layer_mt.get_rw_grads());
  assert_vectors_equal(layer_st.get_b_grads(), layer_mt.get_b_grads());

  assert_vectors_equal(layer_st.get_f_w_grads(), layer_mt.get_f_w_grads());
  assert_vectors_equal(layer_st.get_f_rw_grads(), layer_mt.get_f_rw_grads());
  assert_vectors_equal(layer_st.get_f_b_grads(), layer_mt.get_f_b_grads());

  assert_vectors_equal(layer_st.get_i_w_grads(), layer_mt.get_i_w_grads());
  assert_vectors_equal(layer_st.get_i_rw_grads(), layer_mt.get_i_rw_grads());
  assert_vectors_equal(layer_st.get_i_b_grads(), layer_mt.get_i_b_grads());

  assert_vectors_equal(layer_st.get_o_w_grads(), layer_mt.get_o_w_grads());
  assert_vectors_equal(layer_st.get_o_rw_grads(), layer_mt.get_o_rw_grads());
  assert_vectors_equal(layer_st.get_o_b_grads(), layer_mt.get_o_b_grads());
}

TEST_F(LSTMLayerTest, BPTTSequenceLengthsVerification)
{
  unsigned num_inputs = 3;
  unsigned num_outputs = 4;
  size_t batch_size = 3;

  for (size_t num_time_steps = 1; num_time_steps <= 12; ++num_time_steps)
  {
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    layer.set_w_values(std::vector<double>(num_inputs * num_outputs, 0.1));
    layer.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.15));
    layer.set_b_values(std::vector<double>(num_outputs, 0.05));

    layer.set_f_w_values(std::vector<double>(num_inputs * num_outputs, 0.12));
    layer.set_f_rw_values(std::vector<double>(num_outputs * num_outputs, 0.16));
    layer.set_f_b_values(std::vector<double>(num_outputs, 0.06));

    layer.set_i_w_values(std::vector<double>(num_inputs * num_outputs, 0.14));
    layer.set_i_rw_values(std::vector<double>(num_outputs * num_outputs, 0.18));
    layer.set_i_b_values(std::vector<double>(num_outputs, 0.07));

    layer.set_o_w_values(std::vector<double>(num_inputs * num_outputs, 0.16));
    layer.set_o_rw_values(std::vector<double>(num_outputs * num_outputs, 0.20));
    layer.set_o_b_values(std::vector<double>(num_outputs, 0.08));

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

    auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps, LSTMLayer::Multiplier);

    std::vector<double> inputs(num_time_steps * num_inputs, 0.5);
    for (size_t b = 0; b < batch_size; ++b)
    {
      batch_go[b].set_rnn_outputs(0, inputs);
    }

    // Forward feed
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

    MockLayer next_layer(2, num_outputs);
    std::vector<double> next_w_vals(num_outputs * num_outputs, 0.3);
    next_layer.set_w_values(next_w_vals);

    std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_time_steps * num_outputs, 0.15));

    // Backward feed
    layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, batch_size, static_cast<int>(num_time_steps));

    // Store gradients
    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, static_cast<int>(num_time_steps));

    // Verify gradients computed are reasonable numbers (non-zero and finite)
    EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
    for (const double w : layer.get_w_grads())
    {
      EXPECT_TRUE(std::isfinite(w));
    }
    for (const double rw : layer.get_rw_grads())
    {
      EXPECT_TRUE(std::isfinite(rw));
    }
  }
}

TEST_F(LSTMLayerTest, TempBufferReuseAndMultiIterationConsistency) {
  LSTMLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::Adam, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

  MockLayer prev_layer(0, 2);
  std::vector<unsigned> topology = { 2, 2 };

  std::vector<double> first_pass_outputs;
  std::vector<double> second_pass_outputs;

  for (int iter = 0; iter < 2; ++iter)
  {
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 3, LSTMLayer::Multiplier);

    batch_go[0].set_rnn_outputs(0, { 0.5, 0.2, -0.1, 0.8, 0.3, 0.4 });
    batch_go[1].set_rnn_outputs(0, { 0.1, -0.4, 0.6, 0.2, -0.5, 0.1 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, true);

    const auto rnn_out_0 = batch_go[0].get_rnn_outputs(1);
    const auto rnn_out_1 = batch_go[1].get_rnn_outputs(1);

    if (iter == 0)
    {
      first_pass_outputs = rnn_out_0;
      first_pass_outputs.insert(first_pass_outputs.end(), rnn_out_1.begin(), rnn_out_1.end());
    }
    else
    {
      second_pass_outputs = rnn_out_0;
      second_pass_outputs.insert(second_pass_outputs.end(), rnn_out_1.begin(), rnn_out_1.end());
    }
  }

  ASSERT_EQ(first_pass_outputs.size(), second_pass_outputs.size());
  for (size_t i = 0; i < first_pass_outputs.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(first_pass_outputs[i], second_pass_outputs[i]);
  }
}

// ---------------------------------------------------------------------------
// Batched recurrent forward-pass regression tests.
//
// calculate_forward_feed's recurrent pass batches the recurrent
// (hidden-to-hidden) GEMV across up to 4 batch items per timestep instead of
// processing one batch item at a time. These tests verify that batching a
// batch item together with other, different batch items never changes that
// item's own result (no cross-talk between batch items sharing a group), and
// that every 4-wide/2-wide/1-wide cleanup path is exercised at least once.
// ---------------------------------------------------------------------------
namespace {

  std::vector<double> lstm_make_deterministic_weights(size_t rows, size_t cols, double scale, double offset)
  {
    std::vector<double> w(rows * cols);
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        w[i * cols + j] = offset + scale * std::sin(static_cast<double>(i * 7 + j * 3 + 1));
      }
    }
    return w;
  }

  LSTMLayer make_cross_talk_test_layer(unsigned num_inputs, unsigned num_outputs, bool use_layer_normalisation = false)
  {
    LSTMLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, use_layer_normalisation, std::nullopt);
    layer.set_f_w_values(lstm_make_deterministic_weights(num_inputs, num_outputs, 0.15, 0.02));
    layer.set_f_rw_values(lstm_make_deterministic_weights(num_outputs, num_outputs, 0.12, -0.01));
    layer.set_f_b_values(lstm_make_deterministic_weights(1, num_outputs, 0.05, 0.0));
    layer.set_i_w_values(lstm_make_deterministic_weights(num_inputs, num_outputs, -0.13, 0.03));
    layer.set_i_rw_values(lstm_make_deterministic_weights(num_outputs, num_outputs, 0.10, 0.02));
    layer.set_i_b_values(lstm_make_deterministic_weights(1, num_outputs, -0.04, 0.01));
    layer.set_o_w_values(lstm_make_deterministic_weights(num_inputs, num_outputs, 0.11, 0.015));
    layer.set_o_rw_values(lstm_make_deterministic_weights(num_outputs, num_outputs, -0.08, 0.02));
    layer.set_o_b_values(lstm_make_deterministic_weights(1, num_outputs, 0.03, -0.02));
    layer.set_w_values(lstm_make_deterministic_weights(num_inputs, num_outputs, 0.18, -0.02));
    layer.set_rw_values(lstm_make_deterministic_weights(num_outputs, num_outputs, -0.09, 0.04));
    layer.set_b_values(lstm_make_deterministic_weights(1, num_outputs, 0.06, -0.01));
    if (use_layer_normalisation)
    {
      layer.set_ln_c_gain_values(lstm_make_deterministic_weights(1, num_outputs, 1.2, 0.05));
      layer.set_ln_c_bias_values(lstm_make_deterministic_weights(1, num_outputs, -0.08, 0.02));
    }
    return layer;
  }

  std::vector<double> lstm_make_cross_talk_sequence(double base, size_t num_time_steps, size_t num_inputs)
  {
    std::vector<double> seq(num_time_steps * num_inputs);
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      for (size_t i = 0; i < num_inputs; ++i)
      {
        seq[t * num_inputs + i] = base + 0.13 * static_cast<double>(t) - 0.07 * static_cast<double>(i);
      }
    }
    return seq;
  }

  // Runs one distinctive input sequence ("X") both alone (batch_size=1) and
  // batched at position x_index among batch_size-1 other, different sequences,
  // then asserts X's stored pre-activation sums, cell state values, hidden
  // state values (every timestep) and final rnn_outputs are unaffected by
  // which other batch items happened to share its 4-wide/2-wide/1-wide group.
  void assert_no_lstm_batch_cross_talk(unsigned num_inputs, unsigned num_outputs, size_t batch_size, size_t num_time_steps, size_t x_index, bool is_training, bool use_layer_normalisation = false)
  {
    ASSERT_LT(x_index, batch_size);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    const auto x_seq = lstm_make_cross_talk_sequence(0.37, num_time_steps, num_inputs);
    const unsigned multiplier = use_layer_normalisation ? LSTMLayer::LayerNormMultiplier : LSTMLayer::Multiplier;

    LSTMLayer layer_alone = make_cross_talk_test_layer(num_inputs, num_outputs, use_layer_normalisation);
    MockLayer prev_layer_alone(0, num_inputs);
    auto batch_go_alone = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs_alone = create_batch_hidden_states(topology, 1, num_time_steps, multiplier);
    batch_go_alone[0].set_rnn_outputs(0, x_seq);
    layer_alone.calculate_forward_feed(batch_go_alone, prev_layer_alone, {}, batch_hs_alone, 1, is_training);

    LSTMLayer layer_batched = make_cross_talk_test_layer(num_inputs, num_outputs, use_layer_normalisation);
    MockLayer prev_layer_batched(0, num_inputs);
    auto batch_go_batched = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_batched = create_batch_hidden_states(topology, batch_size, num_time_steps, multiplier);
    for (size_t b = 0; b < batch_size; ++b)
    {
      if (b == x_index)
      {
        batch_go_batched[b].set_rnn_outputs(0, x_seq);
      }
      else
      {
        const double base = -0.6 + 0.23 * static_cast<double>(b);
        batch_go_batched[b].set_rnn_outputs(0, lstm_make_cross_talk_sequence(base, num_time_steps, num_inputs));
      }
    }
    layer_batched.calculate_forward_feed(batch_go_batched, prev_layer_batched, {}, batch_hs_batched, batch_size, is_training);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const auto pre_alone = batch_hs_alone[0].at(1, t).get_pre_activation_sums();
      const auto pre_batched = batch_hs_batched[x_index].at(1, t).get_pre_activation_sums();
      ASSERT_EQ(pre_alone.size(), pre_batched.size());
      for (size_t i = 0; i < pre_alone.size(); ++i)
      {
        EXPECT_NEAR(pre_alone[i], pre_batched[i], 1e-9) << "t=" << t << " i=" << i;
      }

      const auto cell_alone = batch_hs_alone[0].at(1, t).get_cell_state_values();
      const auto cell_batched = batch_hs_batched[x_index].at(1, t).get_cell_state_values();
      ASSERT_EQ(cell_alone.size(), cell_batched.size());
      for (size_t i = 0; i < cell_alone.size(); ++i)
      {
        EXPECT_NEAR(cell_alone[i], cell_batched[i], 1e-9) << "t=" << t << " i=" << i;
      }

      const auto hidden_alone = batch_hs_alone[0].at(1, t).get_hidden_state_values();
      const auto hidden_batched = batch_hs_batched[x_index].at(1, t).get_hidden_state_values();
      ASSERT_EQ(hidden_alone.size(), hidden_batched.size());
      for (size_t i = 0; i < hidden_alone.size(); ++i)
      {
        EXPECT_NEAR(hidden_alone[i], hidden_batched[i], 1e-9) << "t=" << t << " i=" << i;
      }
    }

    const auto rnn_out_alone = batch_go_alone[0].get_rnn_outputs(1);
    const auto rnn_out_batched = batch_go_batched[x_index].get_rnn_outputs(1);
    ASSERT_EQ(rnn_out_alone.size(), rnn_out_batched.size());
    for (size_t i = 0; i < rnn_out_alone.size(); ++i)
    {
      EXPECT_NEAR(rnn_out_alone[i], rnn_out_batched[i], 1e-9) << "i=" << i;
    }
  }

} // namespace

TEST_F(LSTMLayerTest, NoBatchCrossTalkFourWideGroupInference)
{
  // batch_size=7 -> groups of 4, 2, 1. X at index 3 is the last slot of the 4-wide group.
  assert_no_lstm_batch_cross_talk(2, 3, 7, 3, 3, false);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkOneWideCleanupInference)
{
  // batch_size=7 -> groups of 4, 2, 1. X at index 6 is the 1-wide cleanup item.
  assert_no_lstm_batch_cross_talk(2, 3, 7, 3, 6, false);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkFourWideGroupTraining)
{
  // Same as above but is_training=true (dropout=0.0, so still deterministic):
  // exercises the activation()'s training-mode code path without RNG noise.
  assert_no_lstm_batch_cross_talk(2, 3, 7, 3, 3, true);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkOneWideCleanupTraining)
{
  assert_no_lstm_batch_cross_talk(2, 3, 7, 3, 6, true);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkExactFourMultiple)
{
  // batch_size=4: exactly one 4-wide group, no cleanup at all.
  assert_no_lstm_batch_cross_talk(2, 3, 4, 2, 0, false);
  assert_no_lstm_batch_cross_talk(2, 3, 4, 2, 3, false);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkOneWideCleanupRemainder)
{
  // batch_size=5 -> 4 + 1: X in the 1-wide cleanup.
  assert_no_lstm_batch_cross_talk(2, 3, 5, 2, 4, false);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkTwoWideCleanupRemainder)
{
  // batch_size=6 -> 4 + 2: X in the 2-wide cleanup.
  assert_no_lstm_batch_cross_talk(2, 3, 6, 2, 5, false);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkTwoFullFourWideGroups)
{
  // batch_size=8 -> 4 + 4: X at the start of the second 4-wide group.
  assert_no_lstm_batch_cross_talk(2, 3, 8, 2, 4, false);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkLargerHiddenSize)
{
  // N_this=10 crosses gemm_four_batches' internal 8-wide/4-wide/scalar-tail
  // AVX2 boundaries; batch_size=9 -> 4 + 4 + 1.
  assert_no_lstm_batch_cross_talk(4, 10, 9, 3, 4, false);
  assert_no_lstm_batch_cross_talk(4, 10, 9, 3, 8, false);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkLayerNormFourWideGroup)
{
  // Same 4-wide-group layout as NoBatchCrossTalkFourWideGroupInference, but
  // with use_layer_normalisation enabled: verifies each batch item's LayerNorm
  // statistics (mean/inv_std, cached per-item) for the cell state are
  // computed independently and are not contaminated by its 4-wide SIMD
  // group neighbors.
  assert_no_lstm_batch_cross_talk(2, 3, 7, 3, 3, false, true);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkLayerNormOneWideCleanup)
{
  assert_no_lstm_batch_cross_talk(2, 3, 5, 2, 4, false, true);
}

TEST_F(LSTMLayerTest, NoBatchCrossTalkLayerNormTraining)
{
  assert_no_lstm_batch_cross_talk(2, 3, 7, 3, 3, true, true);
}




