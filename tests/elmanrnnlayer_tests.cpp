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
  ElmanRNNLayer layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

  EXPECT_EQ(layer.get_layer_index(), 1);
  EXPECT_EQ(layer.get_number_input_neurons(), 2);
  EXPECT_EQ(layer.get_number_output_neurons(), 3);
  EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Elman);
  EXPECT_TRUE(layer.use_bptt());
  EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);
}

TEST_F(ElmanRNNLayerTest, ForwardFeedMathematicalVerification) {
  ElmanRNNLayer layer(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);

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
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0);

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
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0);
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
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 1.0, nullptr, 1, false, 0.0);
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

TEST_F(ElmanRNNLayerTest, DropoutStatisticalVerification) {
  unsigned num_inputs = 1;
  unsigned num_outputs = 1000;
  double dropout_rate = 0.5;
  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0);

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
  EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.05);
}

TEST_F(ElmanRNNLayerTest, DropoutNotInference) {
  unsigned num_inputs = 1;
  unsigned num_outputs = 1000;
  double dropout_rate = 0.5;
  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, dropout_rate, nullptr, 1, true, 0.0);

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
    ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

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
  ElmanRNNLayer layer(1, 1, 1, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, false, 0.0);

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
  ElmanRNNLayer layer1(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
  
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
  ElmanRNNLayer layer3(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
  layer3 = layer1;
  EXPECT_NO_THROW(layer3.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));

  // Test move constructor
  ElmanRNNLayer layer4(std::move(layer2));
  EXPECT_NO_THROW(layer4.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));

  // Test move assignment
  ElmanRNNLayer layer5(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
  layer5 = std::move(layer4);
  EXPECT_NO_THROW(layer5.calculate_hidden_gradients_from_output_gradients(batch_go, batch_output_grads, batch_hs, 2, 2));
}

TEST_F(ElmanRNNLayerTest, BPTTWorkspaceResizeCorrectness)
{
  unsigned num_inputs = 2;
  unsigned num_outputs = 2;
  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

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
  ElmanRNNLayer layer_st(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

  // Layer 2: multi threaded
  ElmanRNNLayer layer_mt(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 4, true, 0.0);

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

  ElmanRNNLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::tanh, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0);

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

