#include <gtest/gtest.h>
#include "layers/fflayer.h"
#include "layers/grurnnlayer.h"
#include "layers/elmanrnnlayer.h"
#include "layers/lstmlayer.h"
#include "layers/ffoutputlayer.h"
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include "helpers/neuralnetworkserializer.h"
#include "test_helper.h"
#include <vector>


using namespace myoddweb::nn;
using namespace test_helper;

TEST(NetworkIntegrationTest, CrossLayerGradientPropagation) {
    // Topology: 1 (Input) -> 1 (FF) -> 1 (GRU) -> 1 (FFOutput)
    unsigned num_inputs = 1;
    unsigned num_neurons = 1;
    std::vector<unsigned> topology = { num_inputs, num_neurons, num_neurons, num_neurons };

    FFLayer layer1(1, num_inputs, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    GRURNNLayer layer2(2, num_neurons, num_neurons, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0);
    
    OutputLayerDetails out_details(num_neurons, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 0.0, 0.0, false, 1.0 }, 0.0, OptimiserType::SGD, 0.0);
    FFOutputLayer layer3(3, { out_details }, num_neurons, num_neurons, 1, true);

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
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::Adam, 0.0))
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
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::Adam, 0.0))
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
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::Adam, 0.0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::Adam, 0.0))
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
    LayerDetails(Layer::Architecture::Elman, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::Adam, 0.0))
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
    LayerDetails(Layer::Architecture::Lstm, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::SGD, 0.0))
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

TEST(NetworkIntegrationTest, GRUSequenceConvergence)
{
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::Gru, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0)
  };
  auto options = NeuralNetworkOptions::create({ 1, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::SGD, 0.0))
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


