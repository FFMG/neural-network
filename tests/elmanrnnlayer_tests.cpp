#include <gtest/gtest.h>
#include "layers/elmanrnnlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
using namespace test_helper;

class ElmanRNNLayerTest : public ::testing::Test {
protected:
  void SetUp() override {
  }
};

TEST_F(ElmanRNNLayerTest, ConstructionAndTopology) {
  ElmanRNNLayer layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  EXPECT_EQ(layer.get_layer_index(), 1);
  EXPECT_EQ(layer.get_number_input_neurons(), 2);
  EXPECT_EQ(layer.get_number_output_neurons(), 3);
  EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Elman);
  EXPECT_TRUE(layer.use_bptt());
  EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);
}

TEST_F(ElmanRNNLayerTest, ForwardFeedMathematicalVerification) {
  ElmanRNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  layer.set_w_values({ 0.1, 0.2, 0.3, 0.4 });
  layer.set_rw_values({ 0.5, 0.6, 0.7, 0.8 });
  layer.set_b_values({ 0.1, -0.1 });

  MockLayer prev_layer(0, 2);
  std::vector<unsigned> topology = { 2, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 2); 

  batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, 0.0, 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 0.35, 1e-9);
  EXPECT_NEAR(rnn_out[1], 0.30, 1e-9);
  EXPECT_NEAR(rnn_out[2], 0.785, 1e-9);
  EXPECT_NEAR(rnn_out[3], 0.75, 1e-9);
}

TEST_F(ElmanRNNLayerTest, BPTTMathematicalVerification) {
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0, std::nullopt);

  layer.set_w_values({ 0.5 });
  layer.set_rw_values({ 0.8 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 2); 

  batch_go[0].set_rnn_outputs(0, { 1.0, 2.0 }); 
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  MockLayer next_layer(2, 1);
  next_layer.set_w_values({ 2.0 });
  std::vector<std::vector<double>> batch_next_grads = { { 10.0, 10.0 } };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

  const auto rnn_grads = batch_go[0].get_rnn_gradients(1);
  EXPECT_NEAR(rnn_grads[0], 18.0, 1e-9);
  EXPECT_NEAR(rnn_grads[1], 10.0, 1e-9);

  const auto gate_grads = batch_go[0].get_rnn_gate_gradients(1);
  EXPECT_NEAR(gate_grads[0], 36.0, 1e-9);
  EXPECT_NEAR(gate_grads[1], 20.0, 1e-9);
}

TEST_F(ElmanRNNLayerTest, GradientStorageVerification) {
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0, std::nullopt);
  layer.set_w_values({ 0.5 });
  layer.set_rw_values({ 0.8 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 2);

  batch_go[0].set_rnn_outputs(0, { 1.0, 2.0 });
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  batch_go[0].set_rnn_gate_gradients(1, { 52.0, 40.0 });

  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 0);

  EXPECT_NEAR(layer.get_w_grads()[0], 132.0, 1e-9);
  EXPECT_NEAR(layer.get_rw_grads()[0], 20.0, 1e-9);
}

TEST_F(ElmanRNNLayerTest, DropoutConsistencyVerification) {
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 1.0, nullptr, 1, false, 0.0, std::nullopt);
  layer.set_w_values({ 1.0 });
  layer.set_rw_values({ 1.0 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1);

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
  EXPECT_NEAR(batch_go[0].get_rnn_outputs(1)[0], 0.0, 1e-9);

  MockLayer next_layer(2, 1);
  next_layer.set_w_values({ 1.0 });
  std::vector<std::vector<double>> batch_next_grads = { { 10.0 } };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 0);

  EXPECT_NEAR(batch_go[0].get_rnn_gate_gradients(1)[0], 0.0, 1e-9);
}

TEST_F(ElmanRNNLayerTest, DropoutWithTanhActivationDerivative) {
  const unsigned num_inputs = 1;
  const unsigned num_outputs = 200;
  const double dropout_rate = 0.5;
  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, std::nullopt);

  layer.set_w_values(std::vector<double>(num_outputs, 1.0));
  layer.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));
  layer.set_b_values(std::vector<double>(num_outputs, 0.0));

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1);

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  std::vector<std::vector<double>> deltas(1, std::vector<double>(num_outputs, 1.0));
  layer.calculate_hidden_gradients_from_output_gradients(batch_go, deltas, batch_hs, 1, 0);

  const auto& grads = batch_go[0].get_rnn_gate_gradients(1);
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
      EXPECT_GT(grads[j], 0.0);
      EXPECT_NEAR(grads[j], expected_kept_grad, 1e-9);
    }
  }

  EXPECT_GT(kept_count, 0);
  EXPECT_GT(dropped_count, 0);
  EXPECT_EQ(kept_count + dropped_count, static_cast<int>(num_outputs));
}

TEST_F(ElmanRNNLayerTest, DropoutStatisticalVerification) {
  unsigned num_inputs = 1;
  unsigned num_outputs = 5000;
  double dropout_rate = 0.5;
  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, std::nullopt);

  layer.set_w_values(std::vector<double>(num_outputs, 1.0));
  layer.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));
  layer.set_b_values(std::vector<double>(num_outputs, 0.0));

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1);

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  const auto& outputs = batch_go[0].get_rnn_outputs(1);
  int dropped_count = 0;
  int kept_count = 0;
  for (double out : outputs) {
    if (out == 0.0) dropped_count++;
    else if (approx_equal(out, 1.0 / (1.0 - dropout_rate))) kept_count++;
  }

  EXPECT_EQ(dropped_count + kept_count, (int)num_outputs);
  EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.08);
}

TEST_F(ElmanRNNLayerTest, DropoutNotInference) {
  unsigned num_inputs = 1;
  unsigned num_outputs = 1000;
  double dropout_rate = 0.5;
  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0, std::nullopt);

  layer.set_w_values(std::vector<double>(num_outputs, 1.0));
  layer.set_rw_values(std::vector<double>(num_outputs * num_outputs, 0.0));
  layer.set_b_values(std::vector<double>(num_outputs, 0.0));

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1);

  batch_go[0].set_rnn_outputs(0, { 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

  const auto& outputs = batch_go[0].get_rnn_outputs(1);
  for (double out : outputs) {
    EXPECT_NEAR(out, 1.0, 1e-9);
  }
}

TEST_F(ElmanRNNLayerTest, LearningRateRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

    std::vector<double> learning_rates = { 0.0, 0.0001, 0.01, 0.5, 1.0, 2.0 };
    
    for (double lr : learning_rates) {
        layer.set_w_values({ 1.0 });
        layer.set_rw_values({ 1.0 });
        layer.set_b_values({ 0.5 });
        
        layer.set_w_grads({ 0.1 });
        layer.set_rw_grads({ 0.1 });
        layer.set_b_grads({ 0.05 });

        layer.apply_stored_gradients(lr, 1.0);

        EXPECT_NEAR(layer.get_w_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_rw_values()[0], 1.0 - lr * 0.1, 1e-9);
        EXPECT_NEAR(layer.get_b_values()[0], 0.5 - lr * 0.05, 1e-9);
    }
}

TEST_F(ElmanRNNLayerTest, ApplyStoredGradientsCacheUpdate)
{
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0, std::nullopt);

  layer.set_w_values({ 1.0 });
  layer.set_rw_values({ 0.5 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 2); 

  batch_go[0].set_rnn_outputs(0, { 1.0, 1.0 });

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 1.0, 1e-9);
  EXPECT_NEAR(rnn_out[1], 1.5, 1e-9);

  layer.set_rw_grads({ 0.2 });
  layer.apply_stored_gradients(1.0, 1.0);

  EXPECT_NEAR(layer.get_rw_values()[0], 0.3, 1e-9);

  auto batch_hs2 = create_batch_hidden_states(topology, 1, 2); 

  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs2, 1, true);

  auto rnn_out2 = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out2[0], 1.0, 1e-9);
  EXPECT_NEAR(rnn_out2[1], 1.3, 1e-9);
}

TEST_F(ElmanRNNLayerTest, IdentityProxyCachingAndLifecycle)
{
  ElmanRNNLayer layer1(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
  
  std::vector<unsigned> topology = { 2, 3 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs = create_batch_hidden_states(topology, 2, 2);
  
  std::vector<std::vector<double>> batch_output_grads = { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } };

  // First call to trigger lazy proxy initialization
  EXPECT_NO_THROW(layer1.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));

  // Second call to use the cached proxy
  EXPECT_NO_THROW(layer1.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));

  // Test copy constructor
  ElmanRNNLayer layer2(layer1);
  EXPECT_NO_THROW(layer2.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));

  // Test copy assignment
  ElmanRNNLayer layer3(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
  layer3 = layer1;
  EXPECT_NO_THROW(layer3.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));

  // Test move constructor
  ElmanRNNLayer layer4(std::move(layer2));
  EXPECT_NO_THROW(layer4.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));

  // Test move assignment
  ElmanRNNLayer layer5(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
  layer5 = std::move(layer4);
  EXPECT_NO_THROW(layer5.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));
}

TEST_F(ElmanRNNLayerTest, BPTTWorkspaceResizeCorrectness)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  layer.set_w_values({ 0.1, 0.2, 0.3, 0.4 });
  layer.set_rw_values({ 0.15, 0.25, 0.35, 0.45 });
  layer.set_b_values({ 0.05, 0.15 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  // Run backprop first time (creates workspace initially)
  auto batch_go1 = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs1 = create_batch_hidden_states(topology, 2, 2);
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
  auto batch_hs2 = create_batch_hidden_states(topology, 1, 2);
  batch_go2[0].set_rnn_outputs(0, { 1.0, 1.0, 0.5, 0.5 });
  layer.calculate_forward_feed(batch_go2, prev_layer, {}, batch_hs2, 1, false);

  std::vector<std::vector<double>> batch_next_grads2 = { { 0.1, 0.2, 0.3, 0.4 } };
  layer.calculate_hidden_gradients(batch_go2, next_layer, batch_next_grads2, batch_hs2, 1, 2);
  layer.calculate_and_store_gradients(batch_go2, batch_hs2, prev_layer, 1, 2);
  EXPECT_GT(layer.get_gradient_norm_sq(), 0.0);
}

TEST_F(ElmanRNNLayerTest, SingleVSMultiThreadedEquivalence)
{
  unsigned num_inputs = 100;
  unsigned num_outputs = 100;
  size_t batch_size = 100;
  size_t num_time_steps = 20;

  // Layer 1: single threaded
  ElmanRNNLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  // Layer 2: multi threaded
  ElmanRNNLayer layer_mt(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 4, true, 0.0, std::nullopt);

  // Set identical weights
  std::vector<double> weights(num_inputs * num_outputs, 0.05);
  std::vector<double> recurrent_weights(num_outputs * num_outputs, 0.08);
  std::vector<double> biases(num_outputs, 0.01);

  layer_st.set_w_values(weights);
  layer_mt.set_w_values(weights);

  layer_st.set_rw_values(recurrent_weights);
  layer_mt.set_rw_values(recurrent_weights);

  layer_st.set_b_values(biases);
  layer_mt.set_b_values(biases);

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  // Setup batch inputs and next gradients
  auto batch_go_st = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_go_mt = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs_st = create_batch_hidden_states(topology, batch_size, num_time_steps);
  auto batch_hs_mt = create_batch_hidden_states(topology, batch_size, num_time_steps);

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

  // Assert gradients are identical
  const auto& w_grads_st = layer_st.get_w_grads();
  const auto& w_grads_mt = layer_mt.get_w_grads();
  const auto& rw_grads_st = layer_st.get_rw_grads();
  const auto& rw_grads_mt = layer_mt.get_rw_grads();
  const auto& b_grads_st = layer_st.get_b_grads();
  const auto& b_grads_mt = layer_mt.get_b_grads();

  ASSERT_EQ(w_grads_st.size(), w_grads_mt.size());
  for (size_t i = 0; i < w_grads_st.size(); ++i)
  {
    EXPECT_NEAR(w_grads_st[i], w_grads_mt[i], 1e-9);
  }

  ASSERT_EQ(rw_grads_st.size(), rw_grads_mt.size());
  for (size_t i = 0; i < rw_grads_st.size(); ++i)
  {
    EXPECT_NEAR(rw_grads_st[i], rw_grads_mt[i], 1e-9);
  }

  ASSERT_EQ(b_grads_st.size(), b_grads_mt.size());
  for (size_t i = 0; i < b_grads_st.size(); ++i)
  {
    EXPECT_NEAR(b_grads_st[i], b_grads_mt[i], 1e-9);
  }
}

TEST_F(ElmanRNNLayerTest, BPTTMultiStepBatchVerification)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  size_t batch_size = 5;
  size_t num_time_steps = 3;

  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  layer.set_w_values({ 0.1, 0.2, 0.3, 0.4 });
  layer.set_rw_values({ 0.15, 0.25, 0.35, 0.45 });
  layer.set_b_values({ 0.05, 0.15 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

  std::vector<double> inputs = { 0.5, -0.5, 0.2, -0.2, 0.1, -0.1 };
  for (size_t b = 0; b < batch_size; ++b)
  {
    batch_go[b].set_rnn_outputs(0, inputs);
  }

  // Forward feed
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, true);

  MockLayer next_layer(2, num_outputs);
  next_layer.set_w_values({ 1.0, 0.5, 0.2, 0.8 });

  std::vector<std::vector<double>> batch_next_grads(batch_size, std::vector<double>(num_time_steps * num_outputs, 0.1));

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
}

TEST_F(ElmanRNNLayerTest, BPTTSequenceLengthsVerification)
{
  unsigned num_inputs = 3;
  unsigned num_outputs = 4;
  size_t batch_size = 3;

  for (size_t num_time_steps = 1; num_time_steps <= 12; ++num_time_steps)
  {
    ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

    std::vector<double> w_vals(num_inputs * num_outputs, 0.1);
    std::vector<double> rw_vals(num_outputs * num_outputs, 0.2);
    std::vector<double> b_vals(num_outputs, 0.05);
    layer.set_w_values(w_vals);
    layer.set_rw_values(rw_vals);
    layer.set_b_values(b_vals);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs, num_outputs };

    auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

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

TEST_F(ElmanRNNLayerTest, TempBufferReuseAndMultiIterationConsistency) {
  ElmanRNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::Adam, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  layer.set_w_values({ 0.2, -0.3, 0.4, 0.5 });
  layer.set_rw_values({ 0.1, 0.2, -0.1, 0.3 });
  layer.set_b_values({ 0.05, -0.05 });

  MockLayer prev_layer(0, 2);
  std::vector<unsigned> topology = { 2, 2 };

  std::vector<double> first_pass_outputs;
  std::vector<double> second_pass_outputs;

  for (int iter = 0; iter < 2; ++iter)
  {
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 3);

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

TEST_F(ElmanRNNLayerTest, ElmanRNNLayerCalculateAndStoreGradientsMathematicalSoundness) {
  const unsigned num_inputs = 3;
  const unsigned num_outputs = 3;
  const size_t batch_size = 4;
  const size_t num_time_steps = 3;

  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  std::vector<unsigned> topology = { num_inputs, num_outputs };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

  std::vector<std::vector<double>> inputs_data(batch_size * num_time_steps, std::vector<double>(num_inputs, 0.0));
  std::vector<std::vector<double>> grads_data(batch_size * num_time_steps, std::vector<double>(num_outputs, 0.0));
  std::vector<std::vector<double>> prev_h_data(batch_size * num_time_steps, std::vector<double>(num_outputs, 0.0));

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> rnn_inputs(num_time_steps * num_inputs);
    std::vector<double> gate_grads(num_time_steps * num_outputs);

    auto& layer_states = batch_hs[b].at(1);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const size_t idx = b * num_time_steps + t;
      for (size_t k = 0; k < num_inputs; ++k)
      {
        const double x_val = static_cast<double>(idx * 7 + k + 1) * 0.1;
        inputs_data[idx][k] = x_val;
        rnn_inputs[t * num_inputs + k] = x_val;
      }
      for (size_t j = 0; j < num_outputs; ++j)
      {
        const double g_val = static_cast<double>(idx * 3 + j + 2) * 0.05;
        grads_data[idx][j] = g_val;
        gate_grads[t * num_outputs + j] = g_val;
      }
      std::vector<double> h_state(num_outputs);
      for (size_t rk = 0; rk < num_outputs; ++rk)
      {
        h_state[rk] = static_cast<double>(idx * 4 + rk + 1) * 0.2;
      }
      layer_states[t].set_hidden_state_values(h_state.data(), num_outputs);
      if (t > 0)
      {
        const auto& prev_h = layer_states[t - 1].get_hidden_state_values();
        prev_h_data[idx].assign(prev_h.begin(), prev_h.end());
      }
    }
    batch_go[b].set_rnn_outputs(0, rnn_inputs);
    batch_go[b].set_rnn_gate_gradients(1, gate_grads);
  }

  MockLayer prev_layer(0, num_inputs);
  layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, batch_size, 0);

  // Reference formulas:
  // dW_kj = sum_{b,t} x_{b,t,k} * g_{b,t,j}
  // dRW_rk,j = sum_{b, t>0} h_{b,t-1,rk} * g_{b,t,j}
  // dB_j = sum_{b,t} g_{b,t,j}
  std::vector<double> expected_w_grads(num_inputs * num_outputs, 0.0);
  std::vector<double> expected_rw_grads(num_outputs * num_outputs, 0.0);
  std::vector<double> expected_b_grads(num_outputs, 0.0);

  for (size_t b = 0; b < batch_size; ++b)
  {
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const size_t idx = b * num_time_steps + t;
      for (size_t k = 0; k < num_inputs; ++k)
      {
        for (size_t j = 0; j < num_outputs; ++j)
        {
          expected_w_grads[k * num_outputs + j] += inputs_data[idx][k] * grads_data[idx][j];
        }
      }
      if (t > 0)
      {
        for (size_t rk = 0; rk < num_outputs; ++rk)
        {
          for (size_t j = 0; j < num_outputs; ++j)
          {
            expected_rw_grads[rk * num_outputs + j] += prev_h_data[idx][rk] * grads_data[idx][j];
          }
        }
      }
      for (size_t j = 0; j < num_outputs; ++j)
      {
        expected_b_grads[j] += grads_data[idx][j];
      }
    }
  }

  const double inv_batch = 1.0 / static_cast<double>(batch_size);
  for (size_t m = 0; m < expected_w_grads.size(); ++m)
  {
    expected_w_grads[m] *= inv_batch;
  }
  for (size_t m = 0; m < expected_rw_grads.size(); ++m)
  {
    expected_rw_grads[m] *= inv_batch;
  }
  for (size_t j = 0; j < expected_b_grads.size(); ++j)
  {
    expected_b_grads[j] *= inv_batch;
  }

  const auto& actual_w_grads = layer.get_w_grads();
  const auto& actual_rw_grads = layer.get_rw_grads();
  const auto& actual_b_grads = layer.get_b_grads();

  ASSERT_EQ(actual_w_grads.size(), expected_w_grads.size());
  for (size_t m = 0; m < expected_w_grads.size(); ++m)
  {
    EXPECT_NEAR(actual_w_grads[m], expected_w_grads[m], 1e-14);
  }

  ASSERT_EQ(actual_rw_grads.size(), expected_rw_grads.size());
  for (size_t m = 0; m < expected_rw_grads.size(); ++m)
  {
    EXPECT_NEAR(actual_rw_grads[m], expected_rw_grads[m], 1e-14);
  }

  ASSERT_EQ(actual_b_grads.size(), expected_b_grads.size());
  for (size_t j = 0; j < expected_b_grads.size(); ++j)
  {
    EXPECT_NEAR(actual_b_grads[j], expected_b_grads[j], 1e-14);
  }
}

TEST_F(ElmanRNNLayerTest, CalculateOutputGradientsFullSequenceEquivalence)
{
  ElmanRNNLayer layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  std::vector<unsigned> topology = { 2, 3 };
  const size_t batch_size = 2;
  const size_t num_time_steps = 3;
  const size_t num_outputs = 3;

  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

  std::vector<std::vector<double>> targets(batch_size);
  for (size_t b = 0; b < batch_size; ++b)
  {
    targets[b].resize(num_time_steps * num_outputs);
    auto& states = batch_hs[b].at(1);
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      std::vector<double> h_vals(num_outputs);
      for (size_t j = 0; j < num_outputs; ++j)
      {
        h_vals[j] = static_cast<double>(b * 10 + t * 3 + j + 1) * 0.5;
        targets[b][t * num_outputs + j] = static_cast<double>(b * 10 + t * 3 + j + 1) * 0.2;
      }
      states[t].set_hidden_state_values(h_vals.data(), num_outputs);
    }
  }

  layer.calculate_output_gradients(batch_go, targets.cbegin(), batch_hs, batch_size);

  for (size_t b = 0; b < batch_size; ++b)
  {
    const auto rnn_grads = batch_go[b].get_rnn_gradients(1);
    ASSERT_EQ(rnn_grads.size(), num_time_steps * num_outputs);
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      for (size_t j = 0; j < num_outputs; ++j)
      {
        const size_t idx = t * num_outputs + j;
        const double expected_delta = (static_cast<double>(b * 10 + t * 3 + j + 1) * 0.5) - targets[b][idx];
        EXPECT_NEAR(rnn_grads[idx], expected_delta, 1e-12);
      }
    }

    const auto last_tick_grads = batch_go[b].get_gradients(1);
    ASSERT_EQ(last_tick_grads.size(), num_outputs);
    for (size_t j = 0; j < num_outputs; ++j)
    {
      const size_t idx = (num_time_steps - 1) * num_outputs + j;
      EXPECT_NEAR(last_tick_grads[j], rnn_grads[idx], 1e-12);
    }
  }
}

TEST_F(ElmanRNNLayerTest, BatchedForwardFeedUnrollingEquivalence)
{
  const size_t num_inputs = 3;
  const size_t num_outputs = 4;
  const size_t num_time_steps = 3;
  const size_t batch_size = 7;

  ElmanRNNLayer layer_batched(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, 42);
  ElmanRNNLayer layer_single(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, 42);

  std::vector<double> w(num_inputs * num_outputs);
  for (size_t i = 0; i < w.size(); ++i)
  {
    w[i] = static_cast<double>(i + 1) * 0.05 - 0.2;
  }
  std::vector<double> rw(num_outputs * num_outputs);
  for (size_t i = 0; i < rw.size(); ++i)
  {
    rw[i] = static_cast<double>(i + 1) * 0.04 - 0.15;
  }
  std::vector<double> b_vals(num_outputs);
  for (size_t i = 0; i < b_vals.size(); ++i)
  {
    b_vals[i] = static_cast<double>(i + 1) * 0.02;
  }

  layer_batched.set_w_values(w);
  layer_batched.set_rw_values(rw);
  layer_batched.set_b_values(b_vals);

  layer_single.set_w_values(w);
  layer_single.set_rw_values(rw);
  layer_single.set_b_values(b_vals);

  std::vector<unsigned> topology = { static_cast<unsigned>(num_inputs), static_cast<unsigned>(num_outputs) };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

  MockLayer prev_layer(0, num_inputs);

  std::vector<std::vector<double>> inputs_per_batch(batch_size);
  for (size_t b = 0; b < batch_size; ++b)
  {
    inputs_per_batch[b].resize(num_time_steps * num_inputs);
    for (size_t i = 0; i < inputs_per_batch[b].size(); ++i)
    {
      inputs_per_batch[b][i] = std::sin(static_cast<double>(b * 10 + i + 1));
    }
    batch_go[b].set_rnn_outputs(0, inputs_per_batch[b]);
  }

  layer_batched.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, batch_size, false);

  for (size_t b = 0; b < batch_size; ++b)
  {
    auto single_go = create_batch_gradients_and_outputs(topology, 1);
    auto single_hs = create_batch_hidden_states(topology, 1, num_time_steps);
    single_go[0].set_rnn_outputs(0, inputs_per_batch[b]);

    layer_single.calculate_forward_feed(single_go, prev_layer, {}, single_hs, 1, false);

    const auto batched_rnn_out = batch_go[b].get_rnn_outputs(1);
    const auto single_rnn_out = single_go[0].get_rnn_outputs(1);
    ASSERT_EQ(batched_rnn_out.size(), single_rnn_out.size());
    for (size_t i = 0; i < batched_rnn_out.size(); ++i)
    {
      EXPECT_NEAR(batched_rnn_out[i], single_rnn_out[i], 1e-13);
    }
  }
}

TEST_F(ElmanRNNLayerTest, ResidualConnectionsForward)
{
  const size_t num_inputs = 2;
  const size_t num_outputs = 2;
  const size_t num_time_steps = 2;

  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0, std::nullopt);
  layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 });
  layer.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });

  std::vector<unsigned> topology = { 2, 2 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps);
  batch_go[0].set_rnn_outputs(0, { 1.0, 2.0, 3.0, 4.0 });

  MockLayer prev_layer(0, num_inputs);
  std::vector<std::vector<double>> residual = { { 10.0, 20.0 } };

  layer.calculate_forward_feed(batch_go, prev_layer, residual, batch_hs, 1, true);

  const auto rnn_out = batch_go[0].get_rnn_outputs(1);
  EXPECT_NEAR(rnn_out[0], 11.0, 1e-9);
  EXPECT_NEAR(rnn_out[1], 22.0, 1e-9);
  EXPECT_NEAR(rnn_out[2], 13.0, 1e-9);
  EXPECT_NEAR(rnn_out[3], 24.0, 1e-9);
}

TEST_F(ElmanRNNLayerTest, BPTTMaxTicksTruncation)
{
  const size_t num_inputs = 1;
  const size_t num_outputs = 1;
  const size_t num_time_steps = 4;

  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0, std::nullopt);
  layer.set_w_values({ 1.0 });
  layer.set_rw_values({ 0.5 });

  MockLayer prev_layer(0, 1);
  std::vector<unsigned> topology = { 1, 1 };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps);

  batch_go[0].set_rnn_outputs(0, { 1.0, 1.0, 1.0, 1.0 });
  layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

  MockLayer next_layer(2, 1);
  next_layer.set_w_values({ 1.0 });
  std::vector<std::vector<double>> batch_next_grads = { { 1.0, 1.0, 1.0, 1.0 } };

  layer.calculate_hidden_gradients(batch_go, next_layer, batch_next_grads, batch_hs, 1, 2);

  const auto rnn_gate_grads = batch_go[0].get_rnn_gate_gradients(1);
  ASSERT_EQ(rnn_gate_grads.size(), num_time_steps);
  EXPECT_NEAR(rnn_gate_grads[0], 0.0, 1e-9);
  EXPECT_NEAR(rnn_gate_grads[1], 0.0, 1e-9);
  EXPECT_GT(std::abs(rnn_gate_grads[2]), 0.0);
  EXPECT_GT(std::abs(rnn_gate_grads[3]), 0.0);
}

TEST_F(ElmanRNNLayerTest, RecurrentWeightsTransposedCacheAndAccessors)
{
  const size_t num_inputs = 3;
  const size_t num_outputs = 2;
  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  std::vector<double> w = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
  std::vector<double> rw = { 10.0, 20.0, 30.0, 40.0 };

  layer.set_w_values(w);
  layer.set_rw_values(rw);

  EXPECT_EQ(layer.get_recurrent_weight_value(0, 0), 10.0);
  EXPECT_EQ(layer.get_recurrent_weight_value(0, 1), 20.0);
  EXPECT_EQ(layer.get_recurrent_weight_value(1, 0), 30.0);
  EXPECT_EQ(layer.get_recurrent_weight_value(1, 1), 40.0);

  const auto& rw_T = layer.get_rw_values_T();
  ASSERT_EQ(rw_T.size(), 4);
  EXPECT_EQ(rw_T[0], 10.0);
  EXPECT_EQ(rw_T[1], 30.0);
  EXPECT_EQ(rw_T[2], 20.0);
  EXPECT_EQ(rw_T[3], 40.0);

  const auto& w_T = layer.get_w_values_T();
  ASSERT_EQ(w_T.size(), 6);
  EXPECT_EQ(w_T[0], 1.0);
  EXPECT_EQ(w_T[1], 3.0);
  EXPECT_EQ(w_T[2], 5.0);
  EXPECT_EQ(w_T[3], 2.0);
  EXPECT_EQ(w_T[4], 4.0);
  EXPECT_EQ(w_T[5], 6.0);
}

TEST_F(ElmanRNNLayerTest, AdamOptimiserAndSettersCoverage)
{
  ElmanRNNLayer layer(1, 2, 2, 0.01, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::Adam, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  EXPECT_EQ(layer.get_rw_values().size(), 4);
  EXPECT_EQ(layer.get_rw_grads().size(), 4);
  EXPECT_EQ(layer.get_rw_velocities().size(), 4);
  EXPECT_EQ(layer.get_rw_m1().size(), 4);
  EXPECT_EQ(layer.get_rw_m2().size(), 4);
  EXPECT_EQ(layer.get_rw_timesteps().size(), 4);
  EXPECT_EQ(layer.get_rw_decays().size(), 4);

  layer.set_rw_grads({ 0.1, 0.2, 0.3, 0.4 });
  EXPECT_NEAR(layer.get_rw_grads()[0], 0.1, 1e-9);

  layer.set_rw_velocities({ 0.01, 0.02, 0.03, 0.04 });
  EXPECT_NEAR(layer.get_rw_velocities()[1], 0.02, 1e-9);

  layer.set_rw_m1({ 0.001, 0.002, 0.003, 0.004 });
  EXPECT_NEAR(layer.get_rw_m1()[2], 0.003, 1e-9);

  layer.set_rw_m2({ 0.0001, 0.0002, 0.0003, 0.0004 });
  EXPECT_NEAR(layer.get_rw_m2()[3], 0.0004, 1e-9);

  layer.set_rw_timesteps({ 5, 5, 5, 5 });
  EXPECT_EQ(layer.get_rw_timesteps()[0], 5);

  layer.set_rw_decays({ 0.05, 0.05, 0.05, 0.05 });
  EXPECT_NEAR(layer.get_rw_decays()[0], 0.05, 1e-9);

  layer.apply_stored_gradients(0.001, 1.0);
  EXPECT_EQ(layer.get_rw_timesteps()[0], 6);

  std::unique_ptr<Layer> cloned(layer.clone());
  ASSERT_NE(cloned, nullptr);
  auto* cloned_elman = dynamic_cast<ElmanRNNLayer*>(cloned.get());
  ASSERT_NE(cloned_elman, nullptr);
  EXPECT_EQ(cloned_elman->get_rw_values(), layer.get_rw_values());
}

TEST_F(ElmanRNNLayerTest, SpanAndVectorOverloadEquivalence)
{
  const size_t num_inputs = 7;
  const size_t num_outputs = 6;
  const size_t num_time_steps = 3;
  const size_t batch_size = 2;

  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  std::vector<unsigned> topology = { static_cast<unsigned>(num_inputs), static_cast<unsigned>(num_outputs) };
  auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> rnn_inputs(num_time_steps * num_inputs);
    std::vector<double> gate_grads(num_time_steps * num_outputs);
    for (size_t i = 0; i < rnn_inputs.size(); ++i)
    {
      rnn_inputs[i] = static_cast<double>(b * 100 + i + 1) * 0.03;
    }
    for (size_t i = 0; i < gate_grads.size(); ++i)
    {
      gate_grads[i] = static_cast<double>(b * 50 + i + 1) * 0.02;
    }
    batch_go[b].set_rnn_outputs(0, rnn_inputs);
    batch_go[b].set_rnn_gate_gradients(1, gate_grads);

    auto& states = batch_hs[b].at(1);
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      std::vector<double> h_vals(num_outputs);
      for (size_t j = 0; j < num_outputs; ++j)
      {
        h_vals[j] = static_cast<double>(b * 20 + t * 6 + j + 1) * 0.05;
      }
      states[t].set_hidden_state_values(h_vals.data(), num_outputs);
    }
  }

  std::vector<double> vec_w_grads(num_inputs * num_outputs, 0.0);
  std::vector<double> vec_rw_grads(num_outputs * num_outputs, 0.0);
  std::vector<double> vec_b_grads(num_outputs, 0.0);

  std::vector<double> span_w_grads(num_inputs * num_outputs, 0.0);
  std::vector<double> span_rw_grads(num_outputs * num_outputs, 0.0);
  std::vector<double> span_b_grads(num_outputs, 0.0);

  layer.calculate_and_store_gradients_chunk(
    0, batch_size,
    batch_go, batch_hs,
    0, num_inputs, num_outputs, num_time_steps,
    vec_w_grads, vec_rw_grads, vec_b_grads
  );

  layer.calculate_and_store_gradients_chunk(
    0, batch_size,
    batch_go, batch_hs,
    0, num_inputs, num_outputs, num_time_steps,
    std::span<double>(span_w_grads),
    std::span<double>(span_rw_grads),
    std::span<double>(span_b_grads)
  );

  for (size_t i = 0; i < vec_w_grads.size(); ++i)
  {
    EXPECT_NEAR(vec_w_grads[i], span_w_grads[i], 1e-14);
  }
  for (size_t i = 0; i < vec_rw_grads.size(); ++i)
  {
    EXPECT_NEAR(vec_rw_grads[i], span_rw_grads[i], 1e-14);
  }
  for (size_t i = 0; i < vec_b_grads.size(); ++i)
  {
    EXPECT_NEAR(vec_b_grads[i], span_b_grads[i], 1e-14);
  }
}
