#include <gtest/gtest.h>
#include "layers/attentionpoollayer.h"
#include "layers/tcnlayer.h"
#include "layers/selfattentionlayer.h"
#include "layers/fflayer.h"
#include "layers/grurnnlayer.h"
#include "layers/elmanrnnlayer.h"
#include "layers/lstmlayer.h"
#include "layers/ffoutputlayer.h"
#include "layers/multioutputlayer.h"
#include "layers/multioutputlayerdetails.h"
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include "helpers/neuralnetworkserializer.h"
#include "test_helper.h"
#include <cmath>
#include <string>
#include <vector>


using namespace myoddweb::nn;
using namespace test_helper;

TEST(NetworkIntegrationTest, CrossLayerGradientPropagation) {
    // Topology: 1 (Input) -> 1 (FF) -> 1 (GRU) -> 1 (FFOutput)
    unsigned num_inputs = 1;
    unsigned num_neurons = 1;
    std::vector<unsigned> topology = { num_inputs, num_neurons, num_neurons, num_neurons };

    FFLayer layer1(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
    GRURNNLayer layer2(2, num_neurons, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, false, std::nullopt);

    OutputLayerDetails out_details(num_neurons, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 0.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::SGD, 0.0);
    FFOutputLayer layer3(3, { out_details }, num_neurons, num_neurons, 1, true, std::nullopt);

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

TEST(NetworkIntegrationTest, LinearRegressionNoBiasConvergence)
{
  auto options = NeuralNetworkOptions::create({ 2, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(100)
    .with_shuffle_training_data(true)
    .with_has_bias(false)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.0))
    .build();

  NeuralNetwork nn(options);
  
  auto& layers = const_cast<Layers&>(nn.get_layers());
  layers[1].set_w_values({ 0.0, 0.0 });

  std::vector<std::vector<double>> inputs = {
    {0.1, 0.2},
    {0.3, 0.4},
    {0.5, 0.6},
    {0.7, 0.8}
  };
  std::vector<std::vector<double>> outputs = {
    {0.3},
    {0.7},
    {1.1},
    {1.5}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(inputs);

  ASSERT_EQ(predictions.size(), 4);
  EXPECT_NEAR(predictions[0][0], 0.3, 1e-2);
  EXPECT_NEAR(predictions[1][0], 0.7, 1e-2);
  EXPECT_NEAR(predictions[2][0], 1.1, 1e-2);
  EXPECT_NEAR(predictions[3][0], 1.5, 1e-2);
}

TEST(NetworkIntegrationTest, LinearRegressionWithBiasConvergence)
{
  auto options = NeuralNetworkOptions::create({ 1, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(100)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.0))
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  layers[1].set_w_values({ 2.0 });
  layers[1].set_b_values({ 1.0 });

  std::vector<std::vector<double>> inputs = {
    {0.0},
    {1.0},
    {2.0},
    {3.0}
  };
  std::vector<std::vector<double>> outputs = {
    {1.0},
    {3.0},
    {5.0},
    {7.0}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(inputs);
  ASSERT_EQ(predictions.size(), 4);
  EXPECT_NEAR(predictions[0][0], 1.0, 1e-2);
  EXPECT_NEAR(predictions[1][0], 3.0, 1e-2);
  EXPECT_NEAR(predictions[2][0], 5.0, 1e-2);
  EXPECT_NEAR(predictions[3][0], 7.0, 1e-2);
}

TEST(NetworkIntegrationTest, XorFFConvergence)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::Adam, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.0))
    .with_learning_rate(0.1)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  layers[1].set_w_values({
    10.0, 10.0, 0.0, 0.0,
    10.0, 10.0, 0.0, 0.0
  });
  layers[1].set_b_values({ -5.0, -15.0, 0.0, 0.0 });
  layers[2].set_w_values({ 10.0, -20.0, 0.0, 0.0 });
  layers[2].set_b_values({ -5.0 });

  std::vector<std::vector<double>> inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(inputs);
  ASSERT_EQ(predictions.size(), 4);
  EXPECT_NEAR(predictions[0][0], 0.0, 0.15);
  EXPECT_NEAR(predictions[1][0], 1.0, 0.15);
  EXPECT_NEAR(predictions[2][0], 1.0, 0.15);
  EXPECT_NEAR(predictions[3][0], 0.0, 0.15);
}

TEST(NetworkIntegrationTest, XorFFConvergenceLion)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::Lion, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Lion, 0.9))
    .with_learning_rate(0.1)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  layers[1].set_w_values({
    10.0, 10.0, 0.0, 0.0,
    10.0, 10.0, 0.0, 0.0
  });
  layers[1].set_b_values({ -5.0, -15.0, 0.0, 0.0 });
  layers[2].set_w_values({ 10.0, -20.0, 0.0, 0.0 });
  layers[2].set_b_values({ -5.0 });

  std::vector<std::vector<double>> inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(inputs);
  ASSERT_EQ(predictions.size(), 4);
  EXPECT_NEAR(predictions[0][0], 0.0, 0.15);
  EXPECT_NEAR(predictions[1][0], 1.0, 0.15);
  EXPECT_NEAR(predictions[2][0], 1.0, 0.15);
  EXPECT_NEAR(predictions[3][0], 0.0, 0.15);
}

TEST(NetworkIntegrationTest, ElmanRNNSequenceConvergence)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Elman, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.0))
    .with_learning_rate(0.05)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  layers[1].set_w_values({ 1.0, 1.0 });
  layers[1].set_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  layers[1].set_b_values({ 0.0, 0.0 });
  layers[2].set_w_values({ 0.5, 0.5 });
  layers[2].set_b_values({ 0.0 });

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };

  nn.train(inputs, outputs);

  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };
  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  EXPECT_NEAR(predictions[0][0], 0.3, 1e-2);
  EXPECT_NEAR(predictions[1][0], 0.6, 1e-2);
  EXPECT_NEAR(predictions[2][0], 0.9, 1e-2);
}

TEST(NetworkIntegrationTest, LSTMSequenceConvergence)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Lstm, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::SGD, 0.0))
    .with_learning_rate(0.05)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  LSTMLayer& lstm = static_cast<LSTMLayer&>(layers[1]);
  lstm.set_w_values({ 1.0, 1.0 });
  lstm.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  lstm.set_b_values({ 0.0, 0.0 });

  lstm.set_f_w_values({ 0.0, 0.0 });
  lstm.set_f_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  lstm.set_f_b_values({ 10.0, 10.0 });

  lstm.set_i_w_values({ 0.0, 0.0 });
  lstm.set_i_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  lstm.set_i_b_values({ 10.0, 10.0 });

  lstm.set_o_w_values({ 0.0, 0.0 });
  lstm.set_o_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  lstm.set_o_b_values({ 10.0, 10.0 });

  layers[2].set_w_values({ 0.16666666666666666, 0.16666666666666666 });
  layers[2].set_b_values({ 0.1 });

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {}, {}, {0.3},
    {}, {}, {0.6},
    {}, {}, {0.9}
  };

  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  EXPECT_NEAR(predictions[0][0], 0.3, 1e-2);
  EXPECT_NEAR(predictions[1][0], 0.6, 1e-2);
  EXPECT_NEAR(predictions[2][0], 0.9, 1e-2);
}

// Smoke tests (not exact-convergence checks like LSTMSequenceConvergence
// above, which hand-seeds near-solution weights that assume no
// normalization is applied): confirm training a GRU/LSTM hidden layer with
// use_layer_normalisation enabled, through the full NeuralNetwork::train pipeline
// (forward, BPTT backward, optimiser step), completes without throwing and
// produces finite, bounded predictions, and that the LayerNorm gain
// actually moves away from its 1.0 identity initialization -- i.e. the
// optimiser is really updating it, not silently skipping it.
TEST(NetworkIntegrationTest, GRUSequenceConvergenceLayerNorm)
{
  // Output layer neuron count is deliberately matched to the GRU hidden
  // size (4): GRURNNLayer::calculate_hidden_gradients_from_output_gradients
  // routes the real output-layer gradient through a same-sized identity
  // proxy layer, and a pre-existing (not LayerNorm-related) bug in that
  // routing silently drops the gradient whenever the next layer's neuron
  // count differs from the GRU's own. Matching sizes here sidesteps that
  // bug so this test can focus purely on verifying LayerNorm wiring
  // end-to-end through NeuralNetwork::train (multi-tick GRU BPTT backward
  // correctness is already covered directly in
  // grurnnlayer_tests.cpp/grurnnlayer_mt_tests.cpp without going through an
  // FFOutputLayer).
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 4, 4 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(4, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(1)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {0.1, 0.1, 0.1, 0.1}, {0.2, 0.2, 0.2, 0.2}, {0.3, 0.3, 0.3, 0.3},
    {0.4, 0.4, 0.4, 0.4}, {0.5, 0.5, 0.5, 0.5}, {0.6, 0.6, 0.6, 0.6},
    {0.7, 0.7, 0.7, 0.7}, {0.8, 0.8, 0.8, 0.8}, {0.9, 0.9, 0.9, 0.9}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto& layers = const_cast<Layers&>(nn.get_layers());
  GRURNNLayer& gru = static_cast<GRURNNLayer&>(layers[1]);
  EXPECT_TRUE(gru.get_use_layer_normalisation());
  const auto& gain_after = gru.get_ln_h_gain_values();
  ASSERT_NE(gain_after, std::vector<double>(gain_after.size(), 1.0));

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, LSTMSequenceConvergenceLayerNorm)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Lstm, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {}, {}, {0.3},
    {}, {}, {0.6},
    {}, {}, {0.9}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto& layers = const_cast<Layers&>(nn.get_layers());
  LSTMLayer& lstm = static_cast<LSTMLayer&>(layers[1]);
  EXPECT_TRUE(lstm.get_use_layer_normalisation());
  const auto& gain_after = lstm.get_ln_c_gain_values();
  ASSERT_NE(gain_after, std::vector<double>(gain_after.size(), 1.0));

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, GRUSequenceConvergenceAttentionPool)
{
  // Output layer neuron count is deliberately matched to the AttentionPool
  // layer's own size (4, itself matched to the GRU hidden size): AttentionPool's
  // own backward pass, like GRU/LSTM's, relies on the "direct gradient
  // injection" mechanism for whatever sits below it, and inherits the same
  // pre-existing identity-proxy gradient-routing limitation documented in
  // [1.1.21]'s Known Issues for whatever sits above it. Matching sizes
  // sidesteps that pre-existing, unrelated limitation so this test can focus
  // on verifying AttentionPool wiring end-to-end through NeuralNetwork::train
  // (the attention forward/backward math itself is already verified
  // independently, via numerical-gradient checks, in
  // attentionpoollayer_tests.cpp).
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::AttentionPool, 4, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 4, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 4, 4, 4 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(4, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn(options);

  auto& layers_before = const_cast<Layers&>(nn.get_layers());
  AttentionPoolLayer& pool_before = static_cast<AttentionPoolLayer&>(layers_before[2]);
  const std::vector<double> v_before = pool_before.get_v_values();

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {}, {}, {0.3, 0.3, 0.3, 0.3},
    {}, {}, {0.6, 0.6, 0.6, 0.6},
    {}, {}, {0.9, 0.9, 0.9, 0.9}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto& layers_after = const_cast<Layers&>(nn.get_layers());
  AttentionPoolLayer& pool_after = static_cast<AttentionPoolLayer&>(layers_after[2]);
  // The scoring vector must have moved from its random initialization,
  // proving gradients actually reached the attention scoring weights.
  ASSERT_NE(pool_after.get_v_values(), v_before);

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, LSTMSequenceConvergenceAttentionPool)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Lstm, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::AttentionPool, 4, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 4, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 4, 4, 4 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(4, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {}, {}, {0.3, 0.3, 0.3, 0.3},
    {}, {}, {0.6, 0.6, 0.6, 0.6},
    {}, {}, {0.9, 0.9, 0.9, 0.9}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, AttentionPoolSerializerSaveLoad)
{
  // Weight-value round-trip test, following the LayerNormGainBiasSerializerSaveLoad
  // pattern: captures the AttentionPool layer's wa/ba/v values before saving
  // and asserts they come back identical after loading.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 3, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::AttentionPool, 3, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 5, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 3, 3, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(3, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(3)
    .with_shuffle_training_data(false)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(2)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.2, 0.1} };
  std::vector<std::vector<double>> outputs = { {}, {0.2, 0.2, 0.2}, {}, {0.4, 0.4, 0.4} };
  nn.train(inputs, outputs);

  auto& layers_before = const_cast<Layers&>(nn.get_layers());
  AttentionPoolLayer& pool_before = static_cast<AttentionPoolLayer&>(layers_before[2]);
  const std::vector<double> wa_before = pool_before.get_wa_values();
  const std::vector<double> ba_before = pool_before.get_ba_values();
  const std::vector<double> v_before = pool_before.get_v_values();
  ASSERT_EQ(pool_before.get_attention_hidden_size(), 5u);

  std::string test_path = "test_attention_pool_serializer.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  auto& layers_after = const_cast<Layers&>(loaded_nn->get_layers());
  AttentionPoolLayer& pool_after = static_cast<AttentionPoolLayer&>(layers_after[2]);
  EXPECT_EQ(pool_after.get_attention_hidden_size(), 5u);
  ASSERT_EQ(pool_after.get_wa_values().size(), wa_before.size());
  ASSERT_EQ(pool_after.get_ba_values().size(), ba_before.size());
  ASSERT_EQ(pool_after.get_v_values().size(), v_before.size());
  for (size_t i = 0; i < wa_before.size(); ++i)
  {
    EXPECT_NEAR(pool_after.get_wa_values()[i], wa_before[i], 1e-9);
  }
  for (size_t i = 0; i < ba_before.size(); ++i)
  {
    EXPECT_NEAR(pool_after.get_ba_values()[i], ba_before[i], 1e-9);
  }
  for (size_t i = 0; i < v_before.size(); ++i)
  {
    EXPECT_NEAR(pool_after.get_v_values()[i], v_before[i], 1e-9);
  }

  EXPECT_EQ(loaded_nn->options().hidden_layers().size(), 2u);
  EXPECT_EQ(loaded_nn->options().hidden_layers()[1].get_layer_architecture(), Layer::Architecture::AttentionPool);
  EXPECT_EQ(loaded_nn->options().hidden_layers()[1].get_attention_hidden_size(), 5u);

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, AttentionPoolRequiresBpttOptionValidation)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 2, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::AttentionPool, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 4, 0, 0, 0, 0, 0, 0)
  };

  EXPECT_THROW(
    NeuralNetworkOptions::create({ 1, 2, 2, 1 })
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(false)
      .build(),
    std::runtime_error);

  EXPECT_NO_THROW(
    NeuralNetworkOptions::create({ 1, 2, 2, 1 })
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(true)
      .build());
}

TEST(NetworkIntegrationTest, TCNSequenceConvergence)
{
  // Single Tcn hidden layer directly below the output layer - Tcn needs no
  // preceding recurrent layer (unlike AttentionPool), so this is the
  // simplest possible end-to-end wiring check for the layer.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Tcn, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 1, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 4, 4 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(4, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn(options);

  auto& layers_before = const_cast<Layers&>(nn.get_layers());
  TcnLayer& tcn_before = static_cast<TcnLayer&>(layers_before[1]);
  const std::vector<double> w_before = tcn_before.get_w_values();

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {}, {}, {0.3, 0.3, 0.3, 0.3},
    {}, {}, {0.6, 0.6, 0.6, 0.6},
    {}, {}, {0.9, 0.9, 0.9, 0.9}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto& layers_after = const_cast<Layers&>(nn.get_layers());
  TcnLayer& tcn_after = static_cast<TcnLayer&>(layers_after[1]);
  // The weights must have moved from their random initialization, proving
  // gradients actually reached them.
  ASSERT_NE(tcn_after.get_w_values(), w_before);

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, TCNSerializerSaveLoad)
{
  // Weight-value round-trip test, following the AttentionPoolSerializerSaveLoad
  // pattern: captures the Tcn layer's weight/bias values and kernel_size/
  // dilation before saving and asserts they come back identical after loading.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Tcn, 3, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 2, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 3, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(3, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(3)
    .with_shuffle_training_data(false)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(5)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.2, 0.1} };
  std::vector<std::vector<double>> outputs = { {}, {0.2, 0.2, 0.2}, {}, {0.4, 0.4, 0.4} };
  nn.train(inputs, outputs);

  auto& layers_before = const_cast<Layers&>(nn.get_layers());
  TcnLayer& tcn_before = static_cast<TcnLayer&>(layers_before[1]);
  const std::vector<double> w_before = tcn_before.get_w_values();
  const std::vector<double> b_before = tcn_before.get_b_values();
  ASSERT_EQ(tcn_before.get_kernel_size(), 3u);
  ASSERT_EQ(tcn_before.get_dilation(), 2u);

  std::string test_path = "test_tcn_serializer.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  auto& layers_after = const_cast<Layers&>(loaded_nn->get_layers());
  TcnLayer& tcn_after = static_cast<TcnLayer&>(layers_after[1]);
  EXPECT_EQ(tcn_after.get_kernel_size(), 3u);
  EXPECT_EQ(tcn_after.get_dilation(), 2u);
  ASSERT_EQ(tcn_after.get_w_values().size(), w_before.size());
  ASSERT_EQ(tcn_after.get_b_values().size(), b_before.size());
  for (size_t i = 0; i < w_before.size(); ++i)
  {
    EXPECT_NEAR(tcn_after.get_w_values()[i], w_before[i], 1e-9);
  }
  for (size_t i = 0; i < b_before.size(); ++i)
  {
    EXPECT_NEAR(tcn_after.get_b_values()[i], b_before[i], 1e-9);
  }

  EXPECT_EQ(loaded_nn->options().hidden_layers().size(), 1u);
  EXPECT_EQ(loaded_nn->options().hidden_layers()[0].get_layer_architecture(), Layer::Architecture::Tcn);
  EXPECT_EQ(loaded_nn->options().hidden_layers()[0].get_kernel_size(), 3u);
  EXPECT_EQ(loaded_nn->options().hidden_layers()[0].get_dilation(), 2u);

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, TCNRequiresBpttOptionValidation)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Tcn, 2, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 1, 0, 0, 0, 0)
  };

  EXPECT_THROW(
    NeuralNetworkOptions::create({ 1, 2, 1 })
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(false)
      .build(),
    std::runtime_error);

  EXPECT_NO_THROW(
    NeuralNetworkOptions::create({ 1, 2, 1 })
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(3)
      .build());
}

TEST(NetworkIntegrationTest, TCNReceptiveFieldValidation)
{
  // receptive_field = 1 + (kernel_size - 1) * dilation = 1 + (3-1)*4 = 9,
  // which exceeds bptt_max_ticks(5).
  std::vector<LayerDetails> hidden_layers_too_wide = {
    LayerDetails(Layer::Architecture::Tcn, 2, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 4, 0, 0, 0, 0)
  };
  EXPECT_THROW(
    NeuralNetworkOptions::create({ 1, 2, 1 })
      .with_hidden_layers(hidden_layers_too_wide)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(5)
      .build(),
    std::runtime_error);

  // receptive_field = 1 + (3-1)*1 = 3, within bptt_max_ticks(5).
  std::vector<LayerDetails> hidden_layers_ok = {
    LayerDetails(Layer::Architecture::Tcn, 2, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 1, 0, 0, 0, 0)
  };
  EXPECT_NO_THROW(
    NeuralNetworkOptions::create({ 1, 2, 1 })
      .with_hidden_layers(hidden_layers_ok)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(5)
      .build());
}

TEST(NetworkIntegrationTest, SelfAttentionSequenceConvergence)
{
  // Single SelfAttention hidden layer directly below the output layer -
  // SelfAttention needs no preceding recurrent layer (unlike AttentionPool),
  // so this is the simplest possible end-to-end wiring check for the layer.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::SelfAttention, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 2, 8, 0, 0)
  };
  // SelfAttention (unlike Tcn) requires its own size to match the layer it
  // attends over - since it is the first (and only) hidden layer here, that
  // means its size (4) must match the input topology width (4), not 1.
  auto options = NeuralNetworkOptions::create({ 4, 4, 4 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(4, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn(options);

  auto& layers_before = const_cast<Layers&>(nn.get_layers());
  SelfAttentionLayer& sa_before = static_cast<SelfAttentionLayer&>(layers_before[1]);
  const std::vector<double> wq_before = sa_before.get_wq_values();

  std::vector<std::vector<double>> inputs = {
    {0.1, 0.1, 0.1, 0.1}, {0.2, 0.2, 0.2, 0.2}, {0.3, 0.3, 0.3, 0.3},
    {0.4, 0.4, 0.4, 0.4}, {0.5, 0.5, 0.5, 0.5}, {0.6, 0.6, 0.6, 0.6},
    {0.7, 0.7, 0.7, 0.7}, {0.8, 0.8, 0.8, 0.8}, {0.9, 0.9, 0.9, 0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {}, {}, {0.3, 0.3, 0.3, 0.3},
    {}, {}, {0.6, 0.6, 0.6, 0.6},
    {}, {}, {0.9, 0.9, 0.9, 0.9}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3},
    {0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6},
    {0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto& layers_after = const_cast<Layers&>(nn.get_layers());
  SelfAttentionLayer& sa_after = static_cast<SelfAttentionLayer&>(layers_after[1]);
  // The Q-projection weights must have moved from their random
  // initialization, proving gradients actually reached them.
  ASSERT_NE(sa_after.get_wq_values(), wq_before);

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, SelfAttentionSerializerSaveLoad)
{
  // Weight-value round-trip test, following the TCNSerializerSaveLoad
  // pattern: captures every one of SelfAttentionLayer's 16 weight families
  // before saving and asserts they all come back identical after loading -
  // a missed serializer key here would otherwise silently corrupt a
  // saved/reloaded model.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::SelfAttention, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 2, 6, 0, 0)
  };
  // SelfAttention (unlike Tcn) requires its own size to match the layer it
  // attends over - since it is the first (and only) hidden layer here, that
  // means its size (4) must match the input topology width (4), not 2.
  auto options = NeuralNetworkOptions::create({ 4, 4, 4 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(4, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(3)
    .with_shuffle_training_data(false)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(4)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.1, 0.2, 0.3, 0.4}, {0.3, 0.4, 0.5, 0.6}, {0.5, 0.6, 0.7, 0.8}, {0.2, 0.1, 0.4, 0.3} };
  std::vector<std::vector<double>> outputs = { {}, {0.2, 0.2, 0.2, 0.2}, {}, {0.4, 0.4, 0.4, 0.4} };
  nn.train(inputs, outputs);

  auto& layers_before = const_cast<Layers&>(nn.get_layers());
  SelfAttentionLayer& sa_before = static_cast<SelfAttentionLayer&>(layers_before[1]);
  ASSERT_EQ(sa_before.get_number_of_heads(), 2u);
  ASSERT_EQ(sa_before.get_feed_forward_hidden_size(), 6u);
  ASSERT_TRUE(sa_before.get_use_layer_normalisation());

  const std::vector<double> wq_before = sa_before.get_wq_values();
  const std::vector<double> wk_before = sa_before.get_wk_values();
  const std::vector<double> wv_before = sa_before.get_wv_values();
  const std::vector<double> wo_before = sa_before.get_wo_values();
  const std::vector<double> ff1_w_before = sa_before.get_ff1_w_values();
  const std::vector<double> ff2_w_before = sa_before.get_ff2_w_values();
  const std::vector<double> ln1_gain_before = sa_before.get_ln1_gain_values();
  const std::vector<double> ln2_bias_before = sa_before.get_ln2_bias_values();

  std::string test_path = "test_self_attention_serializer.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  auto& layers_after = const_cast<Layers&>(loaded_nn->get_layers());
  SelfAttentionLayer& sa_after = static_cast<SelfAttentionLayer&>(layers_after[1]);
  EXPECT_EQ(sa_after.get_number_of_heads(), 2u);
  EXPECT_EQ(sa_after.get_feed_forward_hidden_size(), 6u);
  EXPECT_TRUE(sa_after.get_use_layer_normalisation());

  ASSERT_EQ(sa_after.get_wq_values().size(), wq_before.size());
  for (size_t i = 0; i < wq_before.size(); ++i)
  {
    EXPECT_NEAR(sa_after.get_wq_values()[i], wq_before[i], 1e-9);
  }
  ASSERT_EQ(sa_after.get_wk_values().size(), wk_before.size());
  for (size_t i = 0; i < wk_before.size(); ++i)
  {
    EXPECT_NEAR(sa_after.get_wk_values()[i], wk_before[i], 1e-9);
  }
  ASSERT_EQ(sa_after.get_wv_values().size(), wv_before.size());
  for (size_t i = 0; i < wv_before.size(); ++i)
  {
    EXPECT_NEAR(sa_after.get_wv_values()[i], wv_before[i], 1e-9);
  }
  ASSERT_EQ(sa_after.get_wo_values().size(), wo_before.size());
  for (size_t i = 0; i < wo_before.size(); ++i)
  {
    EXPECT_NEAR(sa_after.get_wo_values()[i], wo_before[i], 1e-9);
  }
  ASSERT_EQ(sa_after.get_ff1_w_values().size(), ff1_w_before.size());
  for (size_t i = 0; i < ff1_w_before.size(); ++i)
  {
    EXPECT_NEAR(sa_after.get_ff1_w_values()[i], ff1_w_before[i], 1e-9);
  }
  ASSERT_EQ(sa_after.get_ff2_w_values().size(), ff2_w_before.size());
  for (size_t i = 0; i < ff2_w_before.size(); ++i)
  {
    EXPECT_NEAR(sa_after.get_ff2_w_values()[i], ff2_w_before[i], 1e-9);
  }
  ASSERT_EQ(sa_after.get_ln1_gain_values().size(), ln1_gain_before.size());
  for (size_t i = 0; i < ln1_gain_before.size(); ++i)
  {
    EXPECT_NEAR(sa_after.get_ln1_gain_values()[i], ln1_gain_before[i], 1e-9);
  }
  ASSERT_EQ(sa_after.get_ln2_bias_values().size(), ln2_bias_before.size());
  for (size_t i = 0; i < ln2_bias_before.size(); ++i)
  {
    EXPECT_NEAR(sa_after.get_ln2_bias_values()[i], ln2_bias_before[i], 1e-9);
  }

  EXPECT_EQ(loaded_nn->options().hidden_layers().size(), 1u);
  EXPECT_EQ(loaded_nn->options().hidden_layers()[0].get_layer_architecture(), Layer::Architecture::SelfAttention);
  EXPECT_EQ(loaded_nn->options().hidden_layers()[0].get_number_of_heads(), 2u);
  EXPECT_EQ(loaded_nn->options().hidden_layers()[0].get_feed_forward_hidden_size(), 6u);

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, SelfAttentionRequiresBpttOptionValidation)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::SelfAttention, 2, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 2, 4, 0, 0)
  };

  EXPECT_THROW(
    NeuralNetworkOptions::create({ 1, 2, 1 })
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(false)
      .build(),
    std::runtime_error);

  EXPECT_THROW(
    NeuralNetworkOptions::create({ 1, 2, 1 })
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(1)
      .build(),
    std::runtime_error);

  EXPECT_NO_THROW(
    NeuralNetworkOptions::create({ 1, 2, 1 })
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(4)
      .build());
}

TEST(NetworkIntegrationTest, GRUSequenceConvergence)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::SGD, 0.0))
    .with_learning_rate(0.05)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  GRURNNLayer& gru = static_cast<GRURNNLayer&>(layers[1]);
  gru.set_w_values({ 1.0, 1.0 });
  gru.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_b_values({ 0.0, 0.0 });

  gru.set_z_w_values({ 0.0, 0.0 });
  gru.set_z_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_z_b_values({ 10.0, 10.0 });

  gru.set_r_w_values({ 0.0, 0.0 });
  gru.set_r_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_r_b_values({ 10.0, 10.0 });

  layers[2].set_w_values({ 0.5, 0.5 });
  layers[2].set_b_values({ 0.0 });

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {}, {}, {0.3},
    {}, {}, {0.6},
    {}, {}, {0.9}
  };

  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  EXPECT_NEAR(predictions[0][0], 0.3, 1e-2);
  EXPECT_NEAR(predictions[1][0], 0.6, 1e-2);
  EXPECT_NEAR(predictions[2][0], 0.9, 1e-2);
}

// Same network/task as GRUSequenceConvergence, but instead of hand-built
// {}-placeholder targets, every row gets a real (dense) target and
// bptt-supervise-last-step-only is enabled to make create_bptt_batches
// itself discard everything but the last tick of each 3-row block. Since
// GRUSequenceConvergence already proves the {}-placeholder pattern converges
// to these exact predictions, this test is an end-to-end equivalence check:
// the new code path should reduce dense per-row targets down to the same
// effective supervision (rows 2, 5, 8 only) and converge identically.
TEST(NetworkIntegrationTest, GRUSequenceConvergenceBpttSuperviseLastStepOnly)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::SGD, 0.0))
    .with_learning_rate(0.05)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .with_bptt_supervise_last_step_only(true)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  GRURNNLayer& gru = static_cast<GRURNNLayer&>(layers[1]);
  gru.set_w_values({ 1.0, 1.0 });
  gru.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_b_values({ 0.0, 0.0 });

  gru.set_z_w_values({ 0.0, 0.0 });
  gru.set_z_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_z_b_values({ 10.0, 10.0 });

  gru.set_r_w_values({ 0.0, 0.0 });
  gru.set_r_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_r_b_values({ 10.0, 10.0 });

  layers[2].set_w_values({ 0.5, 0.5 });
  layers[2].set_b_values({ 0.0 });

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  // Dense targets on every row (unlike GRUSequenceConvergence's {} placeholders).
  std::vector<std::vector<double>> outputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };

  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  EXPECT_NEAR(predictions[0][0], 0.3, 1e-2);
  EXPECT_NEAR(predictions[1][0], 0.6, 1e-2);
  EXPECT_NEAR(predictions[2][0], 0.9, 1e-2);
}

// Production's real config uses `multi-output-layer-details` (2 branches),
// not a single output layer — the two tests above don't exercise
// MultiOutputLayer::calculate_output_gradients's branch-offset slicing
// (multioutputlayer.h:487-506) under BPTT with bptt-supervise-last-step-only.
// This test closes that gap: two branches learn two DIFFERENT targets (x and
// 2x) from the same GRU trunk under the new option. If gradients were
// misrouted or mixed between branches, this would not converge correctly to
// both distinct targets simultaneously.
TEST(NetworkIntegrationTest, GRUSequenceConvergenceMultiOutputBpttSuperviseLastStepOnly)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig clean_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails branch_a_output(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails branch_b_output(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
  std::vector<MultiOutputLayerDetails> multi_output_layer_details = {
    MultiOutputLayerDetails({}, branch_a_output),
    MultiOutputLayerDetails({}, branch_b_output)
  };

  auto options = NeuralNetworkOptions::create({ 1, 2, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(multi_output_layer_details)
    .with_learning_rate(0.05)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .with_bptt_supervise_last_step_only(true)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  GRURNNLayer& gru = static_cast<GRURNNLayer&>(layers[1]);
  gru.set_w_values({ 1.0, 1.0 });
  gru.set_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_b_values({ 0.0, 0.0 });

  gru.set_z_w_values({ 0.0, 0.0 });
  gru.set_z_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_z_b_values({ 10.0, 10.0 });

  gru.set_r_w_values({ 0.0, 0.0 });
  gru.set_r_rw_values({ 0.0, 0.0, 0.0, 0.0 });
  gru.set_r_b_values({ 10.0, 10.0 });

  auto& multi_output = static_cast<MultiOutputLayer&>(layers[2]);
  auto& branches = multi_output.get_mutable_branches();
  ASSERT_EQ(branches.size(), 2u);
  // Branch A learns identity: 0.5*h0 + 0.5*h1 == x (h0 == h1 == x, as in GRUSequenceConvergence).
  branches[0].layers[0]->set_w_values({ 0.5, 0.5 });
  branches[0].layers[0]->set_b_values({ 0.0 });
  // Branch B learns double: 1.0*h0 + 1.0*h1 == 2x — deliberately different from branch A.
  branches[1].layers[0]->set_w_values({ 1.0, 1.0 });
  branches[1].layers[0]->set_b_values({ 0.0 });

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  // Dense per-row targets [x, 2x] on every row; bptt-supervise-last-step-only
  // should reduce this to supervising only rows 2, 5, 8 (the last tick of
  // each 3-row block) for both branches simultaneously.
  std::vector<std::vector<double>> outputs = {
    {0.1, 0.2}, {0.2, 0.4}, {0.3, 0.6},
    {0.4, 0.8}, {0.5, 1.0}, {0.6, 1.2},
    {0.7, 1.4}, {0.8, 1.6}, {0.9, 1.8}
  };

  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  ASSERT_EQ(predictions[0].size(), 2u);
  EXPECT_NEAR(predictions[0][0], 0.3, 1e-2);
  EXPECT_NEAR(predictions[0][1], 0.6, 1e-2);
  EXPECT_NEAR(predictions[1][0], 0.6, 1e-2);
  EXPECT_NEAR(predictions[1][1], 1.2, 1e-2);
  EXPECT_NEAR(predictions[2][0], 0.9, 1e-2);
  EXPECT_NEAR(predictions[2][1], 1.8, 1e-2);
}

TEST(NetworkIntegrationTest, LogTrainingInfo)
{
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_learning_rate(0.05)
    .with_number_of_epoch(1)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { {0.5} };
  std::vector<std::vector<double>> outputs = { {1.0} };

  nn.train(inputs, outputs);
}

TEST(NetworkIntegrationTest, LogTrainingInfoOptionAndSerialization)
{
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_learning_rate(0.05)
    .with_number_of_epoch(1)
    .with_log_training_info(false)
    .build();

  EXPECT_FALSE(options.log_training_info());

  NeuralNetwork nn(options);
  
  std::vector<std::vector<double>> inputs = { {0.5} };
  std::vector<std::vector<double>> outputs = { {1.0} };
  
  nn.train(inputs, outputs);

  std::string test_path = "test_nn_log_option.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);
  EXPECT_FALSE(loaded_nn->options().log_training_info());

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, BpttSuperviseLastStepOnlySerializerSaveLoad)
{
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(1)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .with_bptt_supervise_last_step_only(true)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6} };
  std::vector<std::vector<double>> outputs = { {0.1}, {0.2}, {0.3} };
  nn.train(inputs, outputs);

  std::string test_path = "test_bptt_supervise_last_step_only.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);
  EXPECT_TRUE(loaded_nn->options().bptt_supervise_last_step_only());

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, LayerNormGainBiasSerializerSaveLoad)
{
  // Weight-value round-trip test: unlike BpttSuperviseLastStepOnlySerializerSaveLoad
  // (which only checks an option survives save/load), this captures the
  // GRU layer's LayerNorm gain/bias values before saving and asserts they
  // come back identical after loading, exercising the new
  // use-layer-normalisation/ln-h-gain-*/ln-h-bias-* serializer fields end-to-end.
  // Output layer neuron count is deliberately matched to the GRU hidden
  // size (3): GRURNNLayer::calculate_hidden_gradients_from_output_gradients
  // routes the real output-layer gradient through a same-sized identity
  // proxy layer, and a pre-existing (not LayerNorm-related) bug in that
  // routing silently drops the gradient whenever the next layer's neuron
  // count differs from the GRU's own. Matching sizes here sidesteps that
  // bug so this test can focus purely on LayerNorm gain/bias serialization.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 3, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 3, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(3, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(3)
    .with_shuffle_training_data(false)
    .with_seed(42)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(1)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6} };
  std::vector<std::vector<double>> outputs = { {0.1, 0.1, 0.1}, {0.2, 0.2, 0.2}, {0.3, 0.3, 0.3} };
  nn.train(inputs, outputs);

  auto& layers_before = const_cast<Layers&>(nn.get_layers());
  GRURNNLayer& gru_before = static_cast<GRURNNLayer&>(layers_before[1]);
  ASSERT_TRUE(gru_before.get_use_layer_normalisation());
  const std::vector<double> gain_before = gru_before.get_ln_h_gain_values();
  const std::vector<double> bias_before = gru_before.get_ln_h_bias_values();
  // Sanity: training with a non-zero learning rate must have moved gain
  // away from its 1.0 identity initialization at least somewhere, otherwise
  // this test would trivially pass even if LayerNorm state weren't wired up.
  ASSERT_NE(gain_before, std::vector<double>(gain_before.size(), 1.0));

  std::string test_path = "test_layer_norm_gain_bias_serializer.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  auto& layers_after = const_cast<Layers&>(loaded_nn->get_layers());
  GRURNNLayer& gru_after = static_cast<GRURNNLayer&>(layers_after[1]);
  EXPECT_TRUE(gru_after.get_use_layer_normalisation());
  const auto& gain_after = gru_after.get_ln_h_gain_values();
  const auto& bias_after = gru_after.get_ln_h_bias_values();
  ASSERT_EQ(gain_after.size(), gain_before.size());
  ASSERT_EQ(bias_after.size(), bias_before.size());
  // TinyJSON writes doubles as decimal text with finite precision, so a
  // save/load round-trip is not guaranteed to be bit-for-bit identical --
  // compare with a tight numeric tolerance instead of exact equality.
  for (size_t i = 0; i < gain_after.size(); ++i)
  {
    EXPECT_NEAR(gain_after[i], gain_before[i], 1e-9);
  }
  for (size_t i = 0; i < bias_after.size(); ++i)
  {
    EXPECT_NEAR(bias_after[i], bias_before[i], 1e-9);
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, LSTMLayerNormGainBiasSerializerSaveLoad)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Lstm, 3, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 3, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(3, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(3)
    .with_shuffle_training_data(false)
    .with_seed(42)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(1)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6} };
  std::vector<std::vector<double>> outputs = { {0.1, 0.1, 0.1}, {0.2, 0.2, 0.2}, {0.3, 0.3, 0.3} };
  nn.train(inputs, outputs);

  auto& layers_before = const_cast<Layers&>(nn.get_layers());
  LSTMLayer& lstm_before = static_cast<LSTMLayer&>(layers_before[1]);
  ASSERT_TRUE(lstm_before.get_use_layer_normalisation());
  const std::vector<double> gain_before = lstm_before.get_ln_c_gain_values();
  const std::vector<double> bias_before = lstm_before.get_ln_c_bias_values();
  ASSERT_NE(gain_before, std::vector<double>(gain_before.size(), 1.0));

  std::string test_path = "test_lstm_layer_norm_gain_bias_serializer.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  auto& layers_after = const_cast<Layers&>(loaded_nn->get_layers());
  LSTMLayer& lstm_after = static_cast<LSTMLayer&>(layers_after[1]);
  EXPECT_TRUE(lstm_after.get_use_layer_normalisation());
  const auto& gain_after = lstm_after.get_ln_c_gain_values();
  const auto& bias_after = lstm_after.get_ln_c_bias_values();
  ASSERT_EQ(gain_after.size(), gain_before.size());
  ASSERT_EQ(bias_after.size(), bias_before.size());
  for (size_t i = 0; i < gain_after.size(); ++i)
  {
    EXPECT_NEAR(gain_after[i], gain_before[i], 1e-9);
  }
  for (size_t i = 0; i < bias_after.size(); ++i)
  {
    EXPECT_NEAR(bias_after[i], bias_before[i], 1e-9);
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, UseLayerNormalisationOptionSerialization)
{
  // Unlike LayerNormGainBiasSerializerSaveLoad (which checks the trained
  // gain/bias weight values), this checks that use_layer_normalisation survives as
  // part of the NeuralNetworkOptions hidden-layer configuration itself --
  // NeuralNetworkSerializer::add_hidden_layers/get_hidden_layers is a
  // separate code path from the per-layer weight save/load functions
  // (add_grurnnlayer/create_grurnnlayer etc.), following the
  // BpttSuperviseLastStepOnlySerializerSaveLoad option-survives-round-trip
  // pattern.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 2, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_learning_rate(0.02)
    .with_number_of_epoch(1)
    .build();

  ASSERT_TRUE(options.hidden_layers()[0].get_use_layer_normalisation());

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.5} };
  std::vector<std::vector<double>> outputs = { {1.0} };
  nn.train(inputs, outputs);

  std::string test_path = "test_use_layer_normalisation_option.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);
  ASSERT_EQ(loaded_nn->options().hidden_layers().size(), 1);
  EXPECT_TRUE(loaded_nn->options().hidden_layers()[0].get_use_layer_normalisation());

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, SwaOptionSerialization)
{
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_learning_rate(0.05)
    .with_number_of_epoch(1)
    .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.6, 0.1))
    .build();

  ASSERT_TRUE(options.stochastic_weight_averaging().enabled());
  EXPECT_NEAR(options.stochastic_weight_averaging().start_percent(), 0.6, 1e-9);
  EXPECT_NEAR(options.stochastic_weight_averaging().update_percent(), 0.1, 1e-9);

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.5} };
  std::vector<std::vector<double>> outputs = { {1.0} };
  nn.train(inputs, outputs);

  std::string test_path = "test_swa_option.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);
  EXPECT_TRUE(loaded_nn->options().stochastic_weight_averaging().enabled());
  EXPECT_NEAR(loaded_nn->options().stochastic_weight_averaging().start_percent(), 0.6, 1e-9);
  EXPECT_NEAR(loaded_nn->options().stochastic_weight_averaging().update_percent(), 0.1, 1e-9);

  std::remove(test_path.c_str());
}

static NeuralNetworkOptions create_swa_baseline_comparison_options(bool swa_enabled)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::Adam, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto builder = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.0))
    .with_learning_rate(0.1)
    .with_number_of_epoch(40)
    .with_shuffle_training_data(false)
    .with_shuffle_bptt_batches(false)
    .with_has_bias(true);
  if (swa_enabled)
  {
    builder.with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.25, 0.1));
  }
  return builder.build();
}

static void seed_swa_comparison_weights(NeuralNetwork& nn)
{
  auto& layers = const_cast<Layers&>(nn.get_layers());
  layers[1].set_w_values({ 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0 });
  layers[1].set_b_values({ -5.0, -15.0, 0.0, 0.0 });
  layers[2].set_w_values({ 10.0, -20.0, 0.0, 0.0 });
  layers[2].set_b_values({ -5.0 });
}

TEST(NetworkIntegrationTest, SwaProducesAveragedWeightsDifferentFromBaseline)
{
  // Trains two otherwise-identical FF networks (same hand-seeded starting
  // weights, no shuffling anywhere so both runs are fully deterministic)
  // differing only in whether SWA is enabled, then asserts the SWA-enabled
  // run's final weights actually differ from the plain baseline -- proving
  // the averaging swap-in in NeuralNetwork::train() really replaces the
  // trained weights rather than being a silent no-op.
  std::vector<std::vector<double>> inputs = {
    {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.0}, {1.0}, {1.0}, {0.0}
  };

  NeuralNetwork nn_baseline(create_swa_baseline_comparison_options(false));
  seed_swa_comparison_weights(nn_baseline);
  nn_baseline.train(inputs, outputs);

  NeuralNetwork nn_swa(create_swa_baseline_comparison_options(true));
  seed_swa_comparison_weights(nn_swa);
  nn_swa.train(inputs, outputs);

  const auto& baseline_w = nn_baseline.get_layers()[2].get_w_values();
  const auto& swa_w = nn_swa.get_layers()[2].get_w_values();
  ASSERT_EQ(baseline_w.size(), swa_w.size());
  bool any_different = false;
  for (size_t i = 0; i < baseline_w.size(); ++i)
  {
    EXPECT_TRUE(std::isfinite(swa_w[i]));
    if (std::abs(baseline_w[i] - swa_w[i]) > 1e-9)
    {
      any_different = true;
    }
  }
  EXPECT_TRUE(any_different);
}

TEST(NetworkIntegrationTest, GRUSequenceConvergenceSwa)
{
  // Smoke test mirroring GRUSequenceConvergenceLayerNorm: verifies SWA
  // wiring end-to-end through NeuralNetwork::train on a recurrent layer
  // without throwing and without producing NaN/exploded predictions.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 4, 4 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(4, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(1)
    .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.5, 0.1))
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {0.1, 0.1, 0.1, 0.1}, {0.2, 0.2, 0.2, 0.2}, {0.3, 0.3, 0.3, 0.3},
    {0.4, 0.4, 0.4, 0.4}, {0.5, 0.5, 0.5, 0.5}, {0.6, 0.6, 0.6, 0.6},
    {0.7, 0.7, 0.7, 0.7}, {0.8, 0.8, 0.8, 0.8}, {0.9, 0.9, 0.9, 0.9}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, LSTMSequenceConvergenceSwa)
{
  // Smoke test mirroring LSTMSequenceConvergenceLayerNorm, for SWA instead.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Lstm, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.5, 0.1))
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {}, {}, {0.3},
    {}, {}, {0.6},
    {}, {}, {0.9}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, SwaWithMultiOutputBranches)
{
  // Smoke test that SWA's per-layer averaging correctly recurses through
  // MultiOutputLayer's branches (MultiOutputLayer::accumulate_swa_average_impl)
  // without crashing or hitting a branch/array size mismatch.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };
  EvaluationConfig clean_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails branch_a_output(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails branch_b_output(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
  std::vector<MultiOutputLayerDetails> multi_output_layer_details = {
    MultiOutputLayerDetails({}, branch_a_output),
    MultiOutputLayerDetails({}, branch_b_output)
  };

  auto options = NeuralNetworkOptions::create({ 1, 2, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(multi_output_layer_details)
    .with_learning_rate(0.05)
    .with_number_of_epoch(40)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.25, 0.1))
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3},
    {0.4}, {0.5}, {0.6},
    {0.7}, {0.8}, {0.9}
  };
  std::vector<std::vector<double>> outputs = {
    {0.1, 0.2}, {0.2, 0.4}, {0.3, 0.6},
    {0.4, 0.8}, {0.5, 1.0}, {0.6, 1.2},
    {0.7, 1.4}, {0.8, 1.6}, {0.9, 1.8}
  };
  std::vector<std::vector<double>> think_inputs = {
    {0.1, 0.2, 0.3},
    {0.4, 0.5, 0.6},
    {0.7, 0.8, 0.9}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto predictions = nn.think(think_inputs);
  ASSERT_EQ(predictions.size(), 3);
  for (const auto& row : predictions)
  {
    for (double v : row)
    {
      EXPECT_TRUE(std::isfinite(v));
      EXPECT_LT(std::abs(v), 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, ShuffleBpttBatchesBehavior)
{
  auto options_no_shuffle = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(5)
    .with_shuffle_bptt_batches(false)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  auto options_shuffle = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(5)
    .with_shuffle_bptt_batches(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  std::vector<std::vector<double>> inputs = {
    {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.9, 1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.3}, {0.7}, {1.1}, {1.5}, {1.9}
  };

  NeuralNetwork nn_no_shuffle(options_no_shuffle);
  nn_no_shuffle.train(inputs, outputs);

  NeuralNetwork nn_shuffle(options_shuffle);
  nn_shuffle.train(inputs, outputs);

  SUCCEED();
}

TEST(NetworkIntegrationTest, LockOptimizationConvergenceTest)
{
  auto options = NeuralNetworkOptions::create({ 2, 3, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(10)
    .with_batch_size(2)
    .with_enable_bptt(false)
    .build();

  std::vector<std::vector<double>> inputs = {
    { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }
  };
  std::vector<std::vector<double>> outputs = {
    { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 }
  };

  NeuralNetwork nn(options);
  // Verify that the network trains successfully with the lock optimisation
  EXPECT_NO_THROW(nn.train(inputs, outputs));
}

TEST(NetworkIntegrationTest, UpdateWeightsTouchesEveryLayerAcrossThreadCounts)
{
  // Regression test for Layers::update_weights (include/neuralnetwork/layers/layers.cpp):
  // gradient calculation/application used to be dispatched per layer onto a
  // Layers-owned thread pool running in parallel with each layer's own
  // internal thread pool, oversubscribing the CPU. That outer dispatch was
  // removed in favour of a plain sequential loop across layers, relying
  // solely on each layer's own internal parallelism. This checks the
  // sequential loop still visits and updates every hidden and output layer,
  // for a range of thread-count settings.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 6, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::Adam, 0.0, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::FF, 6, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::Adam, 0.0, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::FF, 6, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::Adam, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };

  std::vector<std::vector<double>> inputs = {
    { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }
  };
  std::vector<std::vector<double>> outputs = {
    { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 }
  };

  for (const int number_of_threads : { 1, 2, 8 })
  {
    SCOPED_TRACE("number_of_threads=" + std::to_string(number_of_threads));

    auto options = NeuralNetworkOptions::create({ 2, 6, 6, 6, 1 })
      .with_hidden_layers(hidden_layers)
      .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.0))
      .with_learning_rate(0.1)
      .with_number_of_epoch(1)
      .with_batch_size(4)
      .with_shuffle_training_data(false)
      .with_has_bias(true)
      .with_number_of_threads(number_of_threads)
      .build();

    NeuralNetwork nn(options);
    auto& layers = const_cast<Layers&>(nn.get_layers());
    const auto number_of_layers = static_cast<unsigned>(layers.size());

    std::vector<std::vector<double>> w_before;
    std::vector<std::vector<double>> b_before;
    for (unsigned i = 1; i < number_of_layers; ++i)
    {
      w_before.push_back(layers[i].get_w_values());
      b_before.push_back(layers[i].get_b_values());
    }

    nn.train(inputs, outputs);

    for (unsigned i = 1; i < number_of_layers; ++i)
    {
      SCOPED_TRACE("layer=" + std::to_string(i));
      EXPECT_NE(w_before[i - 1], layers[i].get_w_values()) << "Layer " << i << " weights did not change after training.";
      if (!b_before[i - 1].empty())
      {
        EXPECT_NE(b_before[i - 1], layers[i].get_b_values()) << "Layer " << i << " biases did not change after training.";
      }
    }
  }
}

TEST(NetworkIntegrationTest, DeepNetworkConvergesWithExplicitThreadCount)
{
  // Companion to UpdateWeightsTouchesEveryLayerAcrossThreadCounts: reuses the
  // known-good hand-set XOR weights from XorFFConvergence to confirm that
  // training through the now-always-sequential Layers::update_weights loop
  // still converges correctly when number_of_threads is explicitly set above 1.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::Adam, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.0))
    .with_learning_rate(0.1)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .with_number_of_threads(4)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  layers[1].set_w_values({
    10.0, 10.0, 0.0, 0.0,
    10.0, 10.0, 0.0, 0.0
  });
  layers[1].set_b_values({ -5.0, -15.0, 0.0, 0.0 });
  layers[2].set_w_values({ 10.0, -20.0, 0.0, 0.0 });
  layers[2].set_b_values({ -5.0 });

  std::vector<std::vector<double>> inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
  };

  EXPECT_NO_THROW(nn.train(inputs, outputs));

  auto predictions = nn.think(inputs);
  ASSERT_EQ(predictions.size(), 4);
  EXPECT_NEAR(predictions[0][0], 0.0, 0.15);
  EXPECT_NEAR(predictions[1][0], 1.0, 0.15);
  EXPECT_NEAR(predictions[2][0], 1.0, 0.15);
  EXPECT_NEAR(predictions[3][0], 0.0, 0.15);
}

TEST(NetworkIntegrationTest, ThinkSequenceInputValidation)
{
  // 1. Test standard/non-BPTT network sequence validation
  auto options_no_bptt = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_enable_bptt(false)
    .build();
  
  NeuralNetwork nn_no_bptt(options_no_bptt);
  
  // Valid single-step size (matches topology first layer: 2)
  EXPECT_NO_THROW({
    auto out = nn_no_bptt.think(std::vector<double>{ 0.1, 0.2 });
    // Expect output size matches output layer topology (1)
    EXPECT_EQ(out.size(), 1);
  });

  // Invalid sizes should be caught and return empty vector
  auto out_invalid_1 = nn_no_bptt.think(std::vector<double>{ 0.1 });
  EXPECT_TRUE(out_invalid_1.empty());

  auto out_invalid_2 = nn_no_bptt.think(std::vector<double>{ 0.1, 0.2, 0.3, 0.4 });
  EXPECT_TRUE(out_invalid_2.empty());

  // 2. Test BPTT network sequence validation
  auto options_bptt = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  NeuralNetwork nn_bptt(options_bptt);

  // Single-step size (2) should be valid
  auto out_bptt_single = nn_bptt.think(std::vector<double>{ 0.1, 0.2 });
  EXPECT_EQ(out_bptt_single.size(), 1);

  // Multi-step sequence size (multiple of 2, e.g. 6) should be valid
  auto out_bptt_seq = nn_bptt.think(std::vector<double>{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
  EXPECT_EQ(out_bptt_seq.size(), 1);

  // Non-multiple size (e.g. 5) should be caught and return empty vector
  auto out_bptt_invalid = nn_bptt.think(std::vector<double>{ 0.1, 0.2, 0.3, 0.4, 0.5 });
  EXPECT_TRUE(out_bptt_invalid.empty());
}

TEST(NetworkIntegrationTest, BPTTForecastMetricsActualHistoryCorrectness)
{
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(5)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  // Sequential data
  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}, {1.1}
  };

  NeuralNetwork nn(options);
  // Training will populate the training helper and execute calculate_forward_feed_for_forecast_metrics
  EXPECT_NO_THROW(nn.train(inputs, outputs));

  // Directly calculate forecast metrics to verify it completes successfully
  std::vector<NeuralNetworkHelperMetrics> metrics;
  EXPECT_NO_THROW({
    metrics = nn.calculate_forecast_metric_all_layers(ErrorCalculation::type::mse);
  });

  ASSERT_FALSE(metrics.empty());
  EXPECT_GE(metrics[0].error(), 0.0);
}

TEST(NetworkIntegrationTest, ForecastMetricsDefaultInSample)
{
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(5)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}, {1.1}
  };

  NeuralNetwork nn(options);
  nn.train(inputs, outputs);

  // Default call (should evaluate in-sample, i.e., training indexes)
  auto metrics_default = nn.calculate_forecast_metrics({ ErrorCalculation::type::mse });

  // Explicit in_sample = true
  auto metrics_in_sample = nn.calculate_forecast_metrics({ ErrorCalculation::type::mse }, true);

  // Explicit in_sample = false (should evaluate out-of-sample final check indexes)
  auto metrics_out_of_sample = nn.calculate_forecast_metrics({ ErrorCalculation::type::mse }, false);

  ASSERT_FALSE(metrics_default.empty());
  ASSERT_FALSE(metrics_in_sample.empty());
  ASSERT_FALSE(metrics_out_of_sample.empty());

  // Verify default is identical to explicit in_sample = true
  EXPECT_NEAR(metrics_default[0].error(), metrics_in_sample[0].error(), 1e-9);

  // Out of sample error can be different (or we just assert the methods run successfully)
  EXPECT_GE(metrics_default[0].error(), 0.0);
  EXPECT_GE(metrics_out_of_sample[0].error(), 0.0);
}

TEST(NetworkIntegrationTest, BPTTForecastMetricsCacheReuseRepeatable)
{
  // calculate_forecast_metrics_all_layers_impl reuses a thread_local GradientsAndOutputs
  // cache across calls, only clearing what a forward-only pass actually needs (via
  // reset_for_inference()) instead of a full zero(). With BPTT enabled, the output layer's
  // _rnn_outputs gets populated on every call; if reset_for_inference() failed to clear a
  // stale sequence from a previous call, a reused cache row could silently report the
  // previous call's prediction instead of a freshly computed one.
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(5)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  std::vector<std::vector<double>> inputs = {
    {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}, {1.1}
  };

  NeuralNetwork nn(options);
  nn.train(inputs, outputs);

  auto first_in_sample = nn.calculate_forecast_metrics({ ErrorCalculation::type::mse }, true);
  ASSERT_FALSE(first_in_sample.empty());

  // Interleave an out-of-sample call (different index set / prediction_size), forcing the
  // cache to shrink and/or grow and exercising reset_for_inference() on the reused rows.
  auto out_of_sample = nn.calculate_forecast_metrics({ ErrorCalculation::type::mse }, false);
  ASSERT_FALSE(out_of_sample.empty());

  // Repeating the exact same in-sample call must reproduce bit-identical results.
  auto second_in_sample = nn.calculate_forecast_metrics({ ErrorCalculation::type::mse }, true);
  ASSERT_FALSE(second_in_sample.empty());

  EXPECT_DOUBLE_EQ(first_in_sample[0].error(), second_in_sample[0].error());

  // A third repeat, again interleaved, confirms the cache stays consistent under reuse.
  auto out_of_sample_2 = nn.calculate_forecast_metrics({ ErrorCalculation::type::mse }, false);
  ASSERT_FALSE(out_of_sample_2.empty());
  EXPECT_DOUBLE_EQ(out_of_sample[0].error(), out_of_sample_2[0].error());
}

TEST(NetworkIntegrationTest, ShuffleSingleStepsBehavior)
{
  auto options_no_shuffle = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(5)
    .with_shuffle_bptt_batches(false)
    .with_enable_bptt(false)
    .build();

  auto options_shuffle = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(5)
    .with_shuffle_bptt_batches(true)
    .with_enable_bptt(false)
    .build();

  std::vector<std::vector<double>> inputs = {
    {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.9, 1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.3}, {0.7}, {1.1}, {1.5}, {1.9}
  };

  NeuralNetwork nn_no_shuffle(options_no_shuffle);
  nn_no_shuffle.train(inputs, outputs);

  NeuralNetwork nn_shuffle(options_shuffle);
  nn_shuffle.train(inputs, outputs);

  SUCCEED();
}

TEST(NetworkIntegrationTest, CalculateForecastMetricsComprehensiveTest)
{
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_learning_rate(0.01)
    .with_number_of_epoch(5)
    .build();

  std::vector<std::vector<double>> inputs = {
    {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.9, 1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.3}, {0.7}, {1.1}, {1.5}, {1.9}
  };

  NeuralNetwork nn(options);

  // 1. Check before training (should return default error metric)
  NeuralNetworkHelperMetrics metric_before = nn.calculate_forecast_metric(ErrorCalculation::type::mse);
  EXPECT_EQ(metric_before.error_type(), ErrorCalculation::type::mse);

  auto all_metrics_before = nn.calculate_forecast_metric_all_layers(ErrorCalculation::type::mae);
  EXPECT_TRUE(all_metrics_before.empty() || all_metrics_before[0].error_type() == ErrorCalculation::type::mae);

  // 2. Train network
  nn.train(inputs, outputs);

  // 3. Test single forecast metric (out of sample)
  NeuralNetworkHelperMetrics single_metric = nn.calculate_forecast_metric(ErrorCalculation::type::mse);
  EXPECT_EQ(single_metric.error_type(), ErrorCalculation::type::mse);
  EXPECT_GE(single_metric.error(), 0.0);

  // 4. Test single forecast metric all layers
  std::vector<NeuralNetworkHelperMetrics> layer_single_metrics = nn.calculate_forecast_metric_all_layers(ErrorCalculation::type::mae);
  ASSERT_FALSE(layer_single_metrics.empty());
  EXPECT_EQ(layer_single_metrics[0].error_type(), ErrorCalculation::type::mae);
  EXPECT_GE(layer_single_metrics[0].error(), 0.0);

  // 5. Test multiple forecast metrics (in-sample vs out-of-sample)
  std::vector<ErrorCalculation::type> error_types = { ErrorCalculation::type::mse, ErrorCalculation::type::mae };
  auto metrics_in = nn.calculate_forecast_metrics(error_types, true);
  auto metrics_out = nn.calculate_forecast_metrics(error_types, false);

  ASSERT_EQ(metrics_in.size(), 2u);
  ASSERT_EQ(metrics_out.size(), 2u);
  EXPECT_EQ(metrics_in[0].error_type(), ErrorCalculation::type::mse);
  EXPECT_EQ(metrics_in[1].error_type(), ErrorCalculation::type::mae);
  EXPECT_GE(metrics_in[0].error(), 0.0);
  EXPECT_GE(metrics_out[0].error(), 0.0);

  // 6. Test forecast metrics all layers
  auto all_layers_in = nn.calculate_forecast_metrics_all_layers(error_types, true);
  auto all_layers_out = nn.calculate_forecast_metrics_all_layers(error_types, false);
  ASSERT_FALSE(all_layers_in.empty());
  ASSERT_FALSE(all_layers_out.empty());
  ASSERT_EQ(all_layers_in[0].size(), 2u);
  EXPECT_EQ(all_layers_in[0][0].error_type(), ErrorCalculation::type::mse);
  EXPECT_EQ(all_layers_in[0][1].error_type(), ErrorCalculation::type::mae);

  // 7. Test NeuralNetworkHelper wrapper methods
  NeuralNetworkHelper helper(nn, 0.01, 5, inputs, outputs);
  auto helper_single = helper.calculate_forecast_metric(ErrorCalculation::type::mse);
  ASSERT_FALSE(helper_single.empty());
  EXPECT_EQ(helper_single[0].error_type(), ErrorCalculation::type::mse);

  auto helper_multi = helper.calculate_forecast_metrics(error_types, true);
  ASSERT_FALSE(helper_multi.empty());
  ASSERT_EQ(helper_multi[0].size(), 2u);
}

TEST(NetworkIntegrationTest, BpttBatchShufflePreservesPairingIntegrity)
{
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .with_shuffle_bptt_batches(true)
    .build();

  NeuralNetwork nn(options);

  // 12 distinct steps forming 4 BPTT sequence batches (each tick=3, input_dim=2, output_dim=1)
  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> outputs;
  for (size_t i = 0; i < 12; ++i)
  {
    inputs.push_back({ static_cast<double>(i + 1), static_cast<double>((i + 1) * 10) });
    outputs.push_back({ static_cast<double>((i + 1) * 100) });
  }

  std::vector<std::vector<double>> bptt_in;
  std::vector<std::vector<double>> bptt_out;
  nn.create_bptt_batches(inputs, outputs, bptt_in, bptt_out);

  ASSERT_EQ(bptt_in.size(), 4u);
  ASSERT_EQ(bptt_out.size(), 4u);

  // Perform 50 shuffle passes and verify 1-to-1 input/output sequence alignment
  for (int pass = 0; pass < 50; ++pass)
  {
    nn.create_bptt_batches(inputs, outputs, bptt_in, bptt_out);
    for (size_t seq = 0; seq < bptt_in.size(); ++seq)
    {
      const auto& seq_in = bptt_in[seq];
      const auto& seq_out = bptt_out[seq];
      ASSERT_EQ(seq_in.size(), 6u); // 3 ticks * 2 inputs
      ASSERT_EQ(seq_out.size(), 3u); // 3 ticks * 1 output

      for (size_t t = 0; t < 3; ++t)
      {
        double in_val1 = seq_in[t * 2];
        double in_val2 = seq_in[t * 2 + 1];
        double out_val = seq_out[t];

        EXPECT_DOUBLE_EQ(in_val2, in_val1 * 10.0);
        EXPECT_DOUBLE_EQ(out_val, in_val1 * 100.0);
      }
    }
  }
}

TEST(NetworkIntegrationTest, BpttSuperviseLastStepOnlyShapeAndValue)
{
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .with_shuffle_bptt_batches(false)
    .with_bptt_supervise_last_step_only(true)
    .build();

  NeuralNetwork nn(options);

  // 12 distinct steps forming 4 BPTT sequence blocks (each tick=3, input_dim=2, output_dim=1)
  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> outputs;
  for (size_t i = 0; i < 12; ++i)
  {
    inputs.push_back({ static_cast<double>(i + 1), static_cast<double>((i + 1) * 10) });
    outputs.push_back({ static_cast<double>((i + 1) * 100) });
  }

  std::vector<std::vector<double>> bptt_in;
  std::vector<std::vector<double>> bptt_out;
  nn.create_bptt_batches(inputs, outputs, bptt_in, bptt_out);

  ASSERT_EQ(bptt_in.size(), 4u);
  ASSERT_EQ(bptt_out.size(), 4u);

  for (size_t seq = 0; seq < bptt_in.size(); ++seq)
  {
    const auto& seq_in = bptt_in[seq];
    const auto& seq_out = bptt_out[seq];

    // Input side is unaffected by the option: still the full 3-tick sequence.
    ASSERT_EQ(seq_in.size(), 6u); // 3 ticks * 2 inputs

    // Output side now holds only the LAST tick's target, not all 3 ticks.
    ASSERT_EQ(seq_out.size(), 1u); // 1 output, not 3 ticks * 1 output

    // Block `seq` covers steps [seq*3, seq*3+2]; the retained value must be
    // the LAST step's own output, not the first step's or an average.
    const size_t last_step_index = seq * 3 + 2;
    const double expected_last_output = static_cast<double>((last_step_index + 1) * 100);
    EXPECT_DOUBLE_EQ(seq_out[0], expected_last_output);
  }
}

TEST(NetworkIntegrationTest, BpttBatchShuffleDistributionUniformity)
{
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(2)
    .with_shuffle_bptt_batches(true)
    .build();

  NeuralNetwork nn(options);

  // 20 steps forming 10 sequence batches
  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> outputs;
  for (size_t i = 0; i < 20; ++i)
  {
    inputs.push_back({ static_cast<double>(i + 1) });
    outputs.push_back({ static_cast<double>((i + 1) * 2) });
  }

  std::vector<std::vector<double>> bptt_in;
  std::vector<std::vector<double>> bptt_out;
  nn.create_bptt_batches(inputs, outputs, bptt_in, bptt_out);

  ASSERT_EQ(bptt_in.size(), 10u);

  // Track position frequencies of first sequence element across 500 iterations
  std::vector<int> position_counts(10, 0);
  const int iterations = 500;
  for (int iter = 0; iter < iterations; ++iter)
  {
    nn.create_bptt_batches(inputs, outputs, bptt_in, bptt_out);
    for (size_t pos = 0; pos < bptt_in.size(); ++pos)
    {
      if (bptt_in[pos][0] == 1.0)
      {
        position_counts[pos]++;
        break;
      }
    }
  }

  // Every position should be hit (statistically expected ~50 times)
  for (size_t pos = 0; pos < position_counts.size(); ++pos)
  {
    EXPECT_GT(position_counts[pos], 5) << "Position " << pos << " was hit too few times, indicating non-uniform shuffle distribution.";
  }
}

TEST(NetworkIntegrationTest, SingleStepShufflePreservesPairingIntegrity)
{
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_enable_bptt(false)
    .with_shuffle_bptt_batches(true)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    {1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0}, {4.0, 40.0}, {5.0, 50.0}
  };
  std::vector<std::vector<double>> outputs = {
    {100.0}, {200.0}, {300.0}, {400.0}, {500.0}
  };

  std::vector<std::vector<double>> bptt_in;
  std::vector<std::vector<double>> bptt_out;
  nn.create_bptt_batches(inputs, outputs, bptt_in, bptt_out);

  for (int pass = 0; pass < 50; ++pass)
  {
    nn.create_bptt_batches(inputs, outputs, bptt_in, bptt_out);
    ASSERT_EQ(bptt_in.size(), 5u);
    ASSERT_EQ(bptt_out.size(), 5u);

    for (size_t i = 0; i < bptt_in.size(); ++i)
    {
      EXPECT_DOUBLE_EQ(bptt_in[i][1], bptt_in[i][0] * 10.0);
      EXPECT_DOUBLE_EQ(bptt_out[i][0], bptt_in[i][0] * 100.0);
    }
  }
}

TEST(NetworkIntegrationTest, ThinkPerformanceSingleInferenceThroughput)
{
  auto options = NeuralNetworkOptions::create({ 4, 16, 8, 2 })
    .with_has_bias(true)
    .build();

  NeuralNetwork nn(options);
  std::vector<double> input = { 0.1, 0.2, 0.3, 0.4 };

  // Run initial call to warm up thread-local caches
  auto initial_result = nn.think(input);
  ASSERT_EQ(initial_result.size(), 2u);

  // Perform 50,000 single-sample inference calls
  for (int i = 0; i < 50000; ++i)
  {
    auto result = nn.think(input);
    EXPECT_EQ(result.size(), 2u);
    EXPECT_DOUBLE_EQ(result[0], initial_result[0]);
    EXPECT_DOUBLE_EQ(result[1], initial_result[1]);
  }
}

TEST(NetworkIntegrationTest, ThinkPerformanceBatchedInferenceThroughput)
{
  auto options = NeuralNetworkOptions::create({ 4, 16, 8, 2 })
    .with_has_bias(true)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> batch_inputs = {
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.6, 0.7, 0.8 },
    { 0.9, 1.0, 1.1, 1.2 }
  };

  // Run initial call to warm up thread-local caches
  auto initial_results = nn.think(batch_inputs);
  ASSERT_EQ(initial_results.size(), 3u);

  // Perform 10,000 batch inference calls
  for (int i = 0; i < 10000; ++i)
  {
    auto results = nn.think(batch_inputs);
    EXPECT_EQ(results.size(), 3u);
    for (size_t b = 0; b < 3; ++b)
    {
      EXPECT_DOUBLE_EQ(results[b][0], initial_results[b][0]);
      EXPECT_DOUBLE_EQ(results[b][1], initial_results[b][1]);
    }
  }
}

TEST(NetworkIntegrationTest, ThinkPerformanceRecurrentInferenceThroughput)
{
  auto options = NeuralNetworkOptions::create({ 2, 8, 1 })
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(4)
    .build();

  NeuralNetwork nn(options);
  std::vector<double> sequence_input = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

  auto initial_result = nn.think(sequence_input);
  ASSERT_EQ(initial_result.size(), 1u);

  for (int i = 0; i < 10000; ++i)
  {
    auto result = nn.think(sequence_input);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result[0], initial_result[0]);
  }
}

TEST(NetworkIntegrationTest, ThinkEmptyInputsHandling)
{
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 }).build();
  NeuralNetwork nn(options);

  const std::vector<double> empty_single;
  const auto single_res = nn.think(empty_single);
  EXPECT_TRUE(single_res.empty());

  const std::vector<std::vector<double>> empty_batch;
  const auto batch_res = nn.think(empty_batch);
  EXPECT_TRUE(batch_res.empty());
}

TEST(NetworkIntegrationTest, ThinkInvalidTopologySizeHandling)
{
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 }).build();
  NeuralNetwork nn(options);

  // Input size 3 does not match input layer topology size 2
  const std::vector<double> invalid_input = { 0.5, 0.5, 0.5 };
  const auto res = nn.think(invalid_input);
  EXPECT_TRUE(res.empty());
}

TEST(NetworkIntegrationTest, ThinkBatchVersusSingleConsistency)
{
  auto options = NeuralNetworkOptions::create({ 2, 8, 2 }).build();
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> batch_inputs = {
    { 0.1, 0.2 },
    { 0.3, 0.4 },
    { 0.5, 0.6 },
    { 0.7, 0.8 }
  };

  const auto batch_outputs = nn.think(batch_inputs);
  ASSERT_EQ(batch_outputs.size(), 4u);

  for (size_t i = 0; i < batch_inputs.size(); ++i)
  {
    const auto single_out = nn.think(batch_inputs[i]);
    ASSERT_EQ(single_out.size(), 2u);
    EXPECT_NEAR(batch_outputs[i][0], single_out[0], 1e-12);
    EXPECT_NEAR(batch_outputs[i][1], single_out[1], 1e-12);
  }
}

TEST(NetworkIntegrationTest, ThinkConcurrentMultiThreadedInference)
{
  auto options = NeuralNetworkOptions::create({ 3, 16, 2 }).build();
  NeuralNetwork nn(options);

  const std::vector<double> test_input = { 0.2, 0.4, 0.6 };
  const auto expected_out = nn.think(test_input);
  ASSERT_EQ(expected_out.size(), 2u);

  std::atomic<bool> success{ true };
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int t = 0; t < 8; ++t)
  {
    threads.emplace_back([&nn, &test_input, &expected_out, &success]()
    {
      for (int i = 0; i < 1000; ++i)
      {
        const auto out = nn.think(test_input);
        if (out.size() != 2u || out[0] != expected_out[0] || out[1] != expected_out[1])
        {
          success.store(false, std::memory_order_relaxed);
          break;
        }
      }
    });
  }

  for (auto& th : threads)
  {
    if (th.joinable())
    {
      th.join();
    }
  }

  EXPECT_TRUE(success.load());
}

namespace {
  // Shared topology for the seed-determinism tests below: a GRU hidden layer
  // (exercises recurrent + gate weight-init seeding) with dropout enabled
  // (exercises dropout seeding), plus shuffling enabled (exercises shuffle-
  // order seeding) - so a passing test here is real end-to-end coverage of
  // item 6a's goal, not just one mechanism in isolation.
  NeuralNetworkOptions make_seed_test_options(std::optional<uint32_t> seed)
  {
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Gru, 4, activation(activation::method::tanh, 0.0), 0.3, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    };
    return NeuralNetworkOptions::create({ 2, 4, 1 })
      .with_hidden_layers(hidden_layers)
      .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
      .with_learning_rate(0.05)
      .with_number_of_epoch(5)
      .with_shuffle_training_data(true)
      .with_has_bias(true)
      .with_seed(seed)
      .build();
  }

  std::vector<std::vector<double>> get_all_layer_weights(const NeuralNetwork& nn)
  {
    std::vector<std::vector<double>> all_weights;
    const auto& layers = nn.get_layers();
    for (unsigned i = 0; i < layers.size(); ++i)
    {
      all_weights.push_back(layers[i].get_w_values());
      all_weights.push_back(layers[i].get_b_values());
    }
    return all_weights;
  }
}

TEST(NetworkIntegrationTest, SeedProducesIdenticalInitialWeights)
{
  NeuralNetwork nn1(make_seed_test_options(std::optional<uint32_t>(2026)));
  NeuralNetwork nn2(make_seed_test_options(std::optional<uint32_t>(2026)));

  EXPECT_EQ(get_all_layer_weights(nn1), get_all_layer_weights(nn2));
}

TEST(NetworkIntegrationTest, SeedProducesIdenticalTrainingOutcome)
{
  std::vector<std::vector<double>> inputs = {
    {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
    {0.2, 0.7}, {0.9, 0.1}, {0.4, 0.4}, {0.6, 0.3}
  };
  std::vector<std::vector<double>> outputs = {
    {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}
  };

  NeuralNetwork nn1(make_seed_test_options(std::optional<uint32_t>(4242)));
  NeuralNetwork nn2(make_seed_test_options(std::optional<uint32_t>(4242)));

  nn1.train(inputs, outputs);
  nn2.train(inputs, outputs);

  EXPECT_EQ(get_all_layer_weights(nn1), get_all_layer_weights(nn2));

  auto predictions1 = nn1.think(inputs);
  auto predictions2 = nn2.think(inputs);
  ASSERT_EQ(predictions1.size(), predictions2.size());
  for (size_t i = 0; i < predictions1.size(); ++i)
  {
    ASSERT_EQ(predictions1[i].size(), predictions2[i].size());
    for (size_t j = 0; j < predictions1[i].size(); ++j)
    {
      EXPECT_DOUBLE_EQ(predictions1[i][j], predictions2[i][j]);
    }
  }
}

TEST(NetworkIntegrationTest, DifferentSeedsProduceDifferentInitialWeights)
{
  NeuralNetwork nn1(make_seed_test_options(std::optional<uint32_t>(111)));
  NeuralNetwork nn2(make_seed_test_options(std::optional<uint32_t>(222)));

  EXPECT_NE(get_all_layer_weights(nn1), get_all_layer_weights(nn2));
}

TEST(NetworkIntegrationTest, UnseededRunsAreNotForciblyIdentical)
{
  // Sanity check that seeding is opt-in: two unseeded networks are not
  // artificially pinned to each other (this would almost never coincidentally
  // match given how many independent random draws feed into construction).
  NeuralNetwork nn1(make_seed_test_options(std::nullopt));
  NeuralNetwork nn2(make_seed_test_options(std::nullopt));

  EXPECT_NE(get_all_layer_weights(nn1), get_all_layer_weights(nn2));
}

TEST(NetworkIntegrationTest, SeedRoundTripsThroughSerialization)
{
  NeuralNetwork nn(make_seed_test_options(std::optional<uint32_t>(31337)));
  std::vector<std::vector<double>> inputs = { {0.5, 0.5} };
  std::vector<std::vector<double>> outputs = { {1.0} };
  nn.train(inputs, outputs);

  std::string test_path = "test_seed_option_serializer.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);
  ASSERT_TRUE(loaded_nn->options().seed().has_value());
  EXPECT_EQ(loaded_nn->options().seed().value(), 31337u);

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, UnsetSeedRoundTripsThroughSerializationAsNullopt)
{
  NeuralNetwork nn(make_seed_test_options(std::nullopt));
  std::vector<std::vector<double>> inputs = { {0.5, 0.5} };
  std::vector<std::vector<double>> outputs = { {1.0} };
  nn.train(inputs, outputs);

  std::string test_path = "test_no_seed_option_serializer.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);
  EXPECT_FALSE(loaded_nn->options().seed().has_value());

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, FloatingPointWeightsSerializationPrecision)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::Adam, 0.9))
    .with_learning_rate(0.02)
    .with_number_of_epoch(1)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.5, 0.5} };
  std::vector<std::vector<double>> outputs = { {1.0} };
  nn.train(inputs, outputs);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  auto& fflayer = static_cast<FFLayer&>(layers[1]);

  // Set weights containing fractional parts that could trigger 64-bit overflow if scaled by 10^19
  std::vector<double> special_weights = {
    -0.93365601966497314,
    0.98588064269909437,
    0.92233720368547756,
    0.92233720368547758
  };
  fflayer.set_w_values(special_weights);

  std::string test_path = "test_float_weights_precision_serializer.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  const auto& loaded_layers = loaded_nn->get_layers();
  const auto& loaded_fflayer = static_cast<const FFLayer&>(loaded_layers[1]);
  const auto& loaded_weights = loaded_fflayer.get_w_values();

  ASSERT_EQ(loaded_weights.size(), special_weights.size());
  for (size_t i = 0; i < special_weights.size(); ++i)
  {
    EXPECT_NEAR(loaded_weights[i], special_weights[i], 1e-9);
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, SelfAttentionSerializationRoundTripWithResidualAndMomentum)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(
      Layer::Architecture::SelfAttention,
      4,
      activation(activation::method::relu, 0.01),
      0.0,
      0.01,
      OptimiserType::AdamW,
      0.88,
      true,
      0, 0, 0,
      2,
      8,
      0, 0)
  };

  auto options = NeuralNetworkOptions::create({ 4, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.01, OptimiserType::AdamW, 0.88))
    .with_residual_layer_jump(1)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(8)
    .with_learning_rate(0.01)
    .with_number_of_epoch(1)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 } };
  std::vector<std::vector<double>> targets = { { 0.5, 0.5 } };
  nn.train(inputs, targets);

  auto pred_before = nn.think(inputs);

  std::string test_path = "test_selfattention_serializer_roundtrip.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  const auto& loaded_layers = loaded_nn->get_layers();
  ASSERT_GE(loaded_layers.size(), 2u);
  const auto* sa_layer = dynamic_cast<const SelfAttentionLayer*>(&loaded_layers[1]);
  ASSERT_NE(sa_layer, nullptr);

  EXPECT_EQ(sa_layer->get_residual_layer_number(), 0);
  EXPECT_NEAR(sa_layer->get_momentum(), 0.88, 1e-9);
  EXPECT_TRUE(sa_layer->get_use_layer_normalisation());
  EXPECT_EQ(sa_layer->get_number_of_heads(), 2u);
  EXPECT_EQ(sa_layer->get_feed_forward_hidden_size(), 8u);

  auto pred_after = loaded_nn->think(inputs);
  ASSERT_EQ(pred_before.size(), pred_after.size());
  for (size_t b = 0; b < pred_before.size(); ++b)
  {
    ASSERT_EQ(pred_before[b].size(), pred_after[b].size());
    for (size_t o = 0; o < pred_before[b].size(); ++o)
    {
      EXPECT_NEAR(pred_before[b][o], pred_after[b][o], 1e-9);
    }
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, QuickGeluSerializerSaveLoad)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::quickGelu, 1.702), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  auto options = NeuralNetworkOptions::create({ 4, 8, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(
      2,
      activation(activation::method::quickGelu, 1.5),
      ErrorCalculation::type::mse,
      { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 },
      0.0,
      OptimiserType::Adam,
      0.9))
    .with_learning_rate(0.01)
    .with_batch_size(1)
    .with_number_of_epoch(5)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { { 0.5, -0.2, 0.8, -0.1 }, { -0.3, 0.4, -0.6, 0.2 } };
  std::vector<std::vector<double>> targets = { { 0.1, 0.9 }, { 0.8, 0.2 } };
  nn.train(inputs, targets);

  auto pred_before = nn.think(inputs);

  std::string test_path = "test_quickgelu_serializer_roundtrip.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  const auto& loaded_layers = loaded_nn->get_layers();
  ASSERT_EQ(loaded_layers.size(), 3u);
  EXPECT_EQ(loaded_layers[1].get_activation().get_method(), activation::method::quickGelu);
  EXPECT_NEAR(loaded_layers[1].get_activation().get_alpha(), 1.702, 1e-6);

  EXPECT_EQ(loaded_layers[2].get_activation(0).get_method(), activation::method::quickGelu);
  EXPECT_NEAR(loaded_layers[2].get_activation(0).get_alpha(), 1.5, 1e-6);

  auto pred_after = loaded_nn->think(inputs);
  ASSERT_EQ(pred_before.size(), pred_after.size());
  for (size_t b = 0; b < pred_before.size(); ++b)
  {
    ASSERT_EQ(pred_before[b].size(), pred_after[b].size());
    for (size_t o = 0; o < pred_before[b].size(); ++o)
    {
      EXPECT_NEAR(pred_before[b][o], pred_after[b][o], 1e-9);
    }
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, LabelSmoothingCrossEntropyTrainingAndSerializationRoundtrip)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  EvaluationConfig eval_cfg(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.1, { 0.5 }, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(eval_cfg.label_smoothing(), 0.1);

  auto options = NeuralNetworkOptions::create({ 4, 8, 3 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(
      3,
      activation(activation::method::softmax, 0.0, 1.0),
      ErrorCalculation::type::cross_entropy,
      eval_cfg,
      0.0,
      OptimiserType::Adam,
      0.9))
    .with_learning_rate(0.01)
    .with_batch_size(1)
    .with_number_of_epoch(5)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    { 1.0, 0.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0, 0.0 }
  };
  std::vector<std::vector<double>> targets = {
    { 1.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 }
  };

  nn.train(inputs, targets);
  auto pred_before = nn.think(inputs);

  std::string test_path = "test_label_smoothing_serializer_roundtrip.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  auto pred_after = loaded_nn->think(inputs);

  const auto& loaded_layers = loaded_nn->get_layers();
  ASSERT_EQ(loaded_layers.size(), 3u);
  const auto& loaded_output_details = loaded_nn->options().output_layer_details();
  ASSERT_EQ(loaded_output_details.size(), 1u);
  EXPECT_NEAR(loaded_output_details[0].get_error_evaluation_config().label_smoothing(), 0.1, 1e-9);

  const auto& w1_before = nn.get_layers()[1].get_w_values();
  const auto& w1_after = loaded_nn->get_layers()[1].get_w_values();
  ASSERT_EQ(w1_before.size(), w1_after.size());
  for (size_t i = 0; i < w1_before.size(); ++i)
  {
    EXPECT_NEAR(w1_before[i], w1_after[i], 1e-9);
  }

  const auto& w2_before = nn.get_layers()[2].get_w_values();
  const auto& w2_after = loaded_nn->get_layers()[2].get_w_values();
  ASSERT_EQ(w2_before.size(), w2_after.size());
  for (size_t i = 0; i < w2_before.size(); ++i)
  {
    EXPECT_NEAR(w2_before[i], w2_after[i], 1e-9);
  }

  const auto& b1_before = nn.get_layers()[1].get_b_values();
  const auto& b1_after = loaded_nn->get_layers()[1].get_b_values();
  ASSERT_EQ(b1_before.size(), b1_after.size());
  for (size_t i = 0; i < b1_before.size(); ++i)
  {
    EXPECT_NEAR(b1_before[i], b1_after[i], 1e-9);
  }

  const auto& b2_before = nn.get_layers()[2].get_b_values();
  const auto& b2_after = loaded_nn->get_layers()[2].get_b_values();
  ASSERT_EQ(b2_before.size(), b2_after.size());
  for (size_t i = 0; i < b2_before.size(); ++i)
  {
    EXPECT_NEAR(b2_before[i], b2_after[i], 1e-9);
  }

  ASSERT_EQ(pred_before.size(), pred_after.size());
  for (size_t b = 0; b < pred_before.size(); ++b)
  {
    ASSERT_EQ(pred_before[b].size(), pred_after[b].size());
    for (size_t o = 0; o < pred_before[b].size(); ++o)
    {
      EXPECT_NEAR(pred_before[b][o], pred_after[b][o], 1e-9);
    }
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, LabelSmoothingBCETraining)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  EvaluationConfig eval_cfg(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.1, { 0.5 }, 0.0, 0.0);

  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(
      1,
      activation(activation::method::sigmoid, 0.0),
      ErrorCalculation::type::bce_loss,
      eval_cfg,
      0.0,
      OptimiserType::Adam,
      0.9))
    .with_learning_rate(0.05)
    .with_batch_size(2)
    .with_number_of_epoch(50)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { { 0.0, 0.0 }, { 1.0, 1.0 } };
  std::vector<std::vector<double>> targets = { { 0.0 }, { 1.0 } };

  EXPECT_NO_THROW(nn.train(inputs, targets));
  auto pred = nn.think(inputs);
  EXPECT_EQ(pred.size(), 2u);
}

TEST(NetworkIntegrationTest, XorFFConvergenceRAdam)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::RAdam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.0, OptimiserType::RAdam, 0.9))
    .with_learning_rate(0.1)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .build();

  NeuralNetwork nn(options);

  auto& layers = const_cast<Layers&>(nn.get_layers());
  layers[1].set_w_values({
    10.0, 10.0, 0.0, 0.0,
    10.0, 10.0, 0.0, 0.0
  });
  layers[1].set_b_values({ -5.0, -15.0, 0.0, 0.0 });
  layers[2].set_w_values({ 10.0, -20.0, 0.0, 0.0 });
  layers[2].set_b_values({ -5.0 });

  std::vector<std::vector<double>> inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
  };
  std::vector<std::vector<double>> outputs = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
  };

  nn.train(inputs, outputs);

  auto predictions = nn.think(inputs);
  ASSERT_EQ(predictions.size(), 4);
  EXPECT_NEAR(predictions[0][0], 0.0, 0.15);
  EXPECT_NEAR(predictions[1][0], 1.0, 0.15);
  EXPECT_NEAR(predictions[2][0], 1.0, 0.15);
  EXPECT_NEAR(predictions[3][0], 0.0, 0.15);
}

TEST(NetworkIntegrationTest, RAdamTrainingAndSerializationRoundtrip)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::RAdam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };
  auto options = NeuralNetworkOptions::create({ 4, 8, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(3, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0 }, 0.01, OptimiserType::RAdam, 0.9))
    .with_learning_rate(0.01)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    {0.1, 0.2, 0.3, 0.4},
    {0.9, 0.8, 0.7, 0.6},
    {0.5, 0.5, 0.1, 0.2}
  };
  std::vector<std::vector<double>> targets = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}
  };

  nn.train(inputs, targets);

  auto preds_before = nn.think(inputs);
  ASSERT_EQ(preds_before.size(), 3);

  // Save to file and reload
  std::string test_path = "test_radam_roundtrip.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  auto preds_after = loaded_nn->think(inputs);
  ASSERT_EQ(preds_after.size(), 3);

  for (size_t i = 0; i < preds_before.size(); ++i)
  {
    for (size_t j = 0; j < preds_before[i].size(); ++j)
    {
      EXPECT_NEAR(preds_before[i][j], preds_after[i][j], 1e-9);
    }
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, CosineAnnealingWarmRestartsTraining)
{
  std::map<int, double> captured_rates;
  std::mutex mutex;

  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(30)
    .with_cosine_annealing_warm_restarts(true, 10, 1.0, 0.01, 1.0)
    .with_progress_callback([&](NeuralNetworkHelper& helper)
    {
      std::lock_guard<std::mutex> lock(mutex);
      captured_rates[static_cast<int>(helper.epoch())] = helper.learning_rate();
      return true;
    })
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0} };
  std::vector<std::vector<double>> outputs = { {0.0}, {1.0}, {1.0}, {0.0} };

  nn.train(inputs, outputs);

  ASSERT_FALSE(captured_rates.empty());
  for (const auto& [epoch, rate] : captured_rates)
  {
    if (epoch < 30)
    {
      double expected = options.cosine_annealing_warm_restarts().calculate_learning_rate(epoch, 0.1);
      EXPECT_NEAR(rate, expected, 1e-4) << "Mismatch at epoch " << epoch;
    }
  }
}

TEST(NetworkIntegrationTest, CosineAnnealingWarmRestartsSerializerSaveLoad)
{
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_learning_rate(0.08)
    .with_number_of_epoch(20)
    .with_cosine_annealing_warm_restarts(true, 12, 1.5, 0.002, 0.85)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {0.2, 0.3}, {0.5, 0.6}, {0.8, 0.1} };
  std::vector<std::vector<double>> outputs = { {0.1}, {0.7}, {0.4} };

  nn.train(inputs, outputs);

  auto preds_before = nn.think(inputs);

  std::string test_path = "test_cosine_annealing_serializer.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  const auto& loaded_ca = loaded_nn->options().cosine_annealing_warm_restarts();
  EXPECT_TRUE(loaded_ca.enabled());
  EXPECT_EQ(loaded_ca.first_cycle_epochs(), 12);
  EXPECT_DOUBLE_EQ(loaded_ca.cycle_multiplier(), 1.5);
  EXPECT_DOUBLE_EQ(loaded_ca.minimum_learning_rate(), 0.002);
  EXPECT_DOUBLE_EQ(loaded_ca.restart_decay(), 0.85);

  auto preds_after = loaded_nn->think(inputs);
  ASSERT_EQ(preds_after.size(), preds_before.size());
  for (size_t i = 0; i < preds_before.size(); ++i)
  {
    for (size_t j = 0; j < preds_before[i].size(); ++j)
    {
      EXPECT_NEAR(preds_before[i][j], preds_after[i][j], 1e-9);
    }
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, MultiQuantileRegressionTrainingAndSerializationRoundtrip)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  const std::vector<double> quantiles = { 0.1, 0.5, 0.9 };
  EvaluationConfig eval_cfg(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, quantiles, 0.0, 0.0);
  EXPECT_EQ(eval_cfg.quantiles().size(), 3u);
  EXPECT_DOUBLE_EQ(eval_cfg.quantiles()[0], 0.1);
  EXPECT_DOUBLE_EQ(eval_cfg.quantiles()[1], 0.5);
  EXPECT_DOUBLE_EQ(eval_cfg.quantiles()[2], 0.9);

  auto options = NeuralNetworkOptions::create({ 2, 8, 3 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(
      3,
      activation(activation::method::linear, 0.0),
      ErrorCalculation::type::quantile_loss,
      eval_cfg,
      0.0,
      OptimiserType::Adam,
      0.9))
    .with_learning_rate(0.01)
    .with_number_of_epoch(50)
    .with_batch_size(4)
    .with_shuffle_training_data(true)
    .with_seed(42)
    .build();

  NeuralNetwork nn(options);

  // Generate synthetic linear data with noise: target is replicated 3 times for 3 quantile heads
  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> targets;
  for (int i = 0; i < 40; ++i)
  {
    const double x1 = static_cast<double>(i % 10) / 10.0;
    const double x2 = static_cast<double>((i * 3) % 10) / 10.0;
    const double noise = ((i % 5) - 2) * 0.05;
    const double y = 0.5 * x1 + 0.3 * x2 + noise;
    inputs.push_back({ x1, x2 });
    targets.push_back({ y, y, y });
  }

  nn.train(inputs, targets);

  auto preds_before = nn.think(inputs);

  std::string test_path = "test_quantile_loss_serializer.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  const auto& loaded_output_layers = loaded_nn->options().output_layer_details();
  ASSERT_EQ(loaded_output_layers.size(), size_t(1));
  EXPECT_EQ(loaded_output_layers[0].get_output_error_calculation_type(), ErrorCalculation::type::quantile_loss);
  const auto& loaded_quantiles = loaded_output_layers[0].get_error_evaluation_config().quantiles();
  ASSERT_EQ(loaded_quantiles.size(), size_t(3));
  EXPECT_DOUBLE_EQ(loaded_quantiles[0], 0.1);
  EXPECT_DOUBLE_EQ(loaded_quantiles[1], 0.5);
  EXPECT_DOUBLE_EQ(loaded_quantiles[2], 0.9);

  auto preds_after = loaded_nn->think(inputs);
  ASSERT_EQ(preds_after.size(), preds_before.size());
  for (size_t i = 0; i < preds_before.size(); ++i)
  {
    for (size_t j = 0; j < preds_before[i].size(); ++j)
    {
      EXPECT_NEAR(preds_before[i][j], preds_after[i][j], 1e-9);
    }
  }

  std::remove(test_path.c_str());
}

TEST(NetworkIntegrationTest, SingleQuantileMedianTrainingConvergence)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  auto options = NeuralNetworkOptions::create({ 1, 4, 1 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::quantile_loss, { 0.5 }, OptimiserType::Adam, 0.9)
    .with_learning_rate(0.02)
    .with_number_of_epoch(80)
    .with_batch_size(4)
    .with_shuffle_training_data(true)
    .with_seed(100)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> targets;
  for (int i = 0; i < 20; ++i)
  {
    const double x = static_cast<double>(i) / 20.0;
    inputs.push_back({ x });
    targets.push_back({ 2.0 * x + 0.1 });
  }

  const auto initial_preds = nn.think(inputs);
  const double loss_before = ErrorCalculation::calculate_quantile_loss(targets, initial_preds, options.output_layer_details()[0].get_error_evaluation_config());
  nn.train(inputs, targets);
  const auto trained_preds = nn.think(inputs);
  const double loss_after = ErrorCalculation::calculate_quantile_loss(targets, trained_preds, options.output_layer_details()[0].get_error_evaluation_config());

  EXPECT_LT(loss_after, loss_before);
}

TEST(NetworkIntegrationTest, SharpeRatioLossTrainingConvergence)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  // Train network to predict trading positions from features using Sharpe ratio loss
  auto options = NeuralNetworkOptions::create({ 1, 4, 1 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(1, activation(activation::method::tanh, 0.0), ErrorCalculation::type::sharpe_ratio_loss, 0.001, 0.0, OptimiserType::Adam, 0.9)
    .with_learning_rate(0.02)
    .with_number_of_epoch(100)
    .with_batch_size(4)
    .with_shuffle_training_data(false)
    .with_seed(42)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> returns;
  for (int i = 0; i < 24; ++i)
  {
    const double x = (i % 2 == 0) ? 1.0 : -1.0;
    inputs.push_back({ x });
    returns.push_back({ (x > 0.0) ? 0.05 : -0.05 });
  }

  const auto initial_preds = nn.think(inputs);
  const double loss_before = ErrorCalculation::calculate_sharpe_ratio_loss(returns, initial_preds, options.output_layer_details()[0].get_error_evaluation_config());
  nn.train(inputs, returns);
  const auto trained_preds = nn.think(inputs);
  const double loss_after = ErrorCalculation::calculate_sharpe_ratio_loss(returns, trained_preds, options.output_layer_details()[0].get_error_evaluation_config());

  EXPECT_LT(loss_after, loss_before);
}

TEST(NetworkIntegrationTest, SortinoRatioLossTrainingConvergence)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  auto options = NeuralNetworkOptions::create({ 1, 4, 1 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(1, activation(activation::method::tanh, 0.0), ErrorCalculation::type::sortino_ratio_loss, 0.001, 0.01, OptimiserType::Adam, 0.9)
    .with_learning_rate(0.02)
    .with_number_of_epoch(100)
    .with_batch_size(4)
    .with_shuffle_training_data(false)
    .with_seed(42)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> returns;
  for (int i = 0; i < 24; ++i)
  {
    const double x = (i % 2 == 0) ? 1.0 : -1.0;
    inputs.push_back({ x });
    returns.push_back({ (x > 0.0) ? 0.04 : -0.04 });
  }

  const auto initial_preds = nn.think(inputs);
  const double loss_before = ErrorCalculation::calculate_sortino_ratio_loss(returns, initial_preds, options.output_layer_details()[0].get_error_evaluation_config());
  nn.train(inputs, returns);
  const auto trained_preds = nn.think(inputs);
  const double loss_after = ErrorCalculation::calculate_sortino_ratio_loss(returns, trained_preds, options.output_layer_details()[0].get_error_evaluation_config());

  EXPECT_LT(loss_after, loss_before);
}

TEST(NetworkIntegrationTest, SharpeAndSortinoSerializationRoundTrip)
{
  MYODDWEB_PROFILE_FUNCTION("NetworkIntegrationTest");
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(1, activation(activation::method::tanh, 0.0), ErrorCalculation::type::sortino_ratio_loss, 0.002, 0.015, OptimiserType::Adam, 0.9)
    .with_learning_rate(0.01)
    .with_number_of_epoch(1)
    .with_seed(123)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> sample_inputs = { { 0.5, -0.5 }, { -0.2, 0.8 } };
  std::vector<std::vector<double>> sample_targets = { { 0.05 }, { -0.03 } };
  nn.train(sample_inputs, sample_targets);

  const std::string test_path = "test_sharpe_sortino_roundtrip.json";
  NeuralNetworkSerializer::save(nn, test_path);

  auto restored_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  std::remove(test_path.c_str());
  ASSERT_NE(restored_nn, nullptr);

  const auto& restored_opts = restored_nn->options();
  ASSERT_EQ(restored_opts.output_layer_details().size(), 1u);
  const auto& out_cfg = restored_opts.output_layer_details()[0].get_error_evaluation_config();

  EXPECT_EQ(restored_opts.output_layer_details()[0].get_output_error_calculation_type(), ErrorCalculation::type::sortino_ratio_loss);
  EXPECT_DOUBLE_EQ(out_cfg.transaction_cost_penalty(), 0.002);
  EXPECT_DOUBLE_EQ(out_cfg.sortino_target_return(), 0.015);

  auto orig_preds = nn.think(sample_inputs);
  auto rest_preds = restored_nn->think(sample_inputs);

  ASSERT_EQ(orig_preds.size(), rest_preds.size());
  for (size_t i = 0; i < orig_preds.size(); ++i)
  {
    ASSERT_EQ(orig_preds[i].size(), rest_preds[i].size());
    for (size_t j = 0; j < orig_preds[i].size(); ++j)
    {
      EXPECT_NEAR(orig_preds[i][j], rest_preds[i][j], 1e-12);
    }
  }
}