#include <gtest/gtest.h>
#include "../src/neuralnetwork/elmanrnnlayer.h"
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
