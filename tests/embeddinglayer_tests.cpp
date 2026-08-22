#include <gtest/gtest.h>
#include "layers/embeddinglayer.h"
#include "layers/layer.h"
#include "neuralnetwork.h"
#include "helpers/neuralnetworkserializer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <numeric>
#include <memory>

using namespace myoddweb::nn;
using namespace test_helper;

namespace
{
EmbeddingLayer make_embedding_layer(
  unsigned num_inputs,
  unsigned vocabulary_size,
  unsigned embedding_dimension,
  const activation& activation_method = activation(activation::method::linear, 0.0),
  double dropout = 0.0,
  double weight_decay = 0.0,
  OptimiserType optimiser_type = OptimiserType::Adam,
  double momentum = 0.0,
  int residual_layer_number = -1,
  ResidualProjector* residual_projector = nullptr)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  return EmbeddingLayer(
    1,
    num_inputs,
    vocabulary_size,
    embedding_dimension,
    weight_decay,
    Layer::Role::Hidden,
    activation_method,
    optimiser_type,
    residual_layer_number,
    dropout,
    residual_projector,
    1,
    momentum,
    std::nullopt);
}
} // namespace

TEST(EmbeddingLayerTest, ConstructionAndProperties)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  const unsigned num_inputs = 3;
  const unsigned vocab_size = 10;
  const unsigned embed_dim = 4;
  EmbeddingLayer layer = make_embedding_layer(num_inputs, vocab_size, embed_dim);

  EXPECT_EQ(layer.get_layer_index(), 1u);
  EXPECT_EQ(layer.get_number_input_neurons(), num_inputs);
  EXPECT_EQ(layer.get_number_neurons(), num_inputs * embed_dim);
  EXPECT_EQ(layer.get_vocabulary_size(), vocab_size);
  EXPECT_EQ(layer.get_embedding_dimension(), embed_dim);
  EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::Embedding);
  EXPECT_EQ(layer.get_w_values().size(), vocab_size * embed_dim);
  EXPECT_TRUE(layer.get_b_values().empty());
  EXPECT_FALSE(layer.has_bias());
}

TEST(EmbeddingLayerTest, ValidationThrowsOnInvalidConfiguration)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  // 1. vocab_size == 0
  EXPECT_THROW(
    EmbeddingLayer(
      1, 2, 0, 4, 0.0, Layer::Role::Hidden,
      activation(activation::method::linear, 0.0),
      OptimiserType::Adam, -1, 0.0, nullptr, 1, 0.0, std::nullopt),
    std::invalid_argument);

  // 2. embed_dim == 0
  EXPECT_THROW(
    EmbeddingLayer(
      1, 2, 10, 0, 0.0, Layer::Role::Hidden,
      activation(activation::method::linear, 0.0),
      OptimiserType::Adam, -1, 0.0, nullptr, 1, 0.0, std::nullopt),
    std::invalid_argument);

  // 3. layer_size != num_inputs * embed_dim through Layer::create_hidden_layer
  LayerDetails bad_ld(
    Layer::Architecture::Embedding, 7, activation(activation::method::linear, 0.0),
    0.0, 0.0, OptimiserType::Adam, 0.0, false,
    0, 0, 0, 0, 0,
    10, 4);
  EXPECT_THROW(
    Layer::create_hidden_layer(1, 2, bad_ld, 1, false, Layer::Architecture::None, -1, nullptr, std::nullopt),
    std::runtime_error);
}

TEST(EmbeddingLayerTest, ForwardPassHandComputed)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  const unsigned num_inputs = 2;
  const unsigned vocab_size = 4;
  const unsigned embed_dim = 3;

  EmbeddingLayer layer = make_embedding_layer(num_inputs, vocab_size, embed_dim);

  // Set known weights:
  // Cat 0: [ 0.1,  0.2,  0.3]
  // Cat 1: [ 1.1,  1.2,  1.3]
  // Cat 2: [ 2.1,  2.2,  2.3]
  // Cat 3: [ 3.1,  3.2,  3.3]
  std::vector<double> custom_weights = {
    0.1, 0.2, 0.3,
    1.1, 1.2, 1.3,
    2.1, 2.2, 2.3,
    3.1, 3.2, 3.3
  };
  layer.set_w_values(custom_weights);

  std::vector<unsigned> topology = { num_inputs, num_inputs * embed_dim };
  auto batch_go = create_batch_gradients_and_outputs(topology, 2);
  auto batch_hs = create_batch_hidden_states(topology, 2, 1, 1);
  MockLayer previous_layer(0, num_inputs);

  // Batch 0: indices [0.0, 2.0] -> expected [0.1, 0.2, 0.3, 2.1, 2.2, 2.3]
  // Batch 1: indices [3.0, 1.0] -> expected [3.1, 3.2, 3.3, 1.1, 1.2, 1.3]
  batch_go[0].set_outputs(0, { 0.0, 2.0 });
  batch_go[1].set_outputs(0, { 3.0, 1.0 });

  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 2, false);

  const auto out0 = batch_go[0].get_outputs(1);
  const auto out1 = batch_go[1].get_outputs(1);

  ASSERT_EQ(out0.size(), 6u);
  EXPECT_DOUBLE_EQ(out0[0], 0.1);
  EXPECT_DOUBLE_EQ(out0[1], 0.2);
  EXPECT_DOUBLE_EQ(out0[2], 0.3);
  EXPECT_DOUBLE_EQ(out0[3], 2.1);
  EXPECT_DOUBLE_EQ(out0[4], 2.2);
  EXPECT_DOUBLE_EQ(out0[5], 2.3);

  ASSERT_EQ(out1.size(), 6u);
  EXPECT_DOUBLE_EQ(out1[0], 3.1);
  EXPECT_DOUBLE_EQ(out1[1], 3.2);
  EXPECT_DOUBLE_EQ(out1[2], 3.3);
  EXPECT_DOUBLE_EQ(out1[3], 1.1);
  EXPECT_DOUBLE_EQ(out1[4], 1.2);
  EXPECT_DOUBLE_EQ(out1[5], 1.3);
}

TEST(EmbeddingLayerTest, ClampingOutOfBoundsIndices)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  const unsigned num_inputs = 2;
  const unsigned vocab_size = 4;
  const unsigned embed_dim = 2;

  EmbeddingLayer layer = make_embedding_layer(num_inputs, vocab_size, embed_dim);

  std::vector<double> custom_weights = {
    0.1, 0.2,  // index 0
    1.1, 1.2,  // index 1
    2.1, 2.2,  // index 2
    3.1, 3.2   // index 3
  };
  layer.set_w_values(custom_weights);

  std::vector<unsigned> topology = { num_inputs, num_inputs * embed_dim };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
  MockLayer previous_layer(0, num_inputs);

  // Negative index -5.0 clamped to 0, excessive index 99.0 clamped to 3 (vocab_size - 1)
  batch_go[0].set_outputs(0, { -5.0, 99.0 });

  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

  const auto out = batch_go[0].get_outputs(1);
  ASSERT_EQ(out.size(), 4u);
  EXPECT_DOUBLE_EQ(out[0], 0.1);
  EXPECT_DOUBLE_EQ(out[1], 0.2);
  EXPECT_DOUBLE_EQ(out[2], 3.1);
  EXPECT_DOUBLE_EQ(out[3], 3.2);
}

TEST(EmbeddingLayerTest, WeightGradientsMatchNumericalGradient)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  const unsigned num_inputs = 2;
  const unsigned vocab_size = 5;
  const unsigned embed_dim = 3;

  EmbeddingLayer layer = make_embedding_layer(num_inputs, vocab_size, embed_dim);

  std::vector<double> init_weights(vocab_size * embed_dim);
  for (size_t i = 0; i < init_weights.size(); ++i)
  {
    init_weights[i] = 0.1 * static_cast<double>(i + 1);
  }
  layer.set_w_values(init_weights);

  std::vector<unsigned> topology = { num_inputs, num_inputs * embed_dim };
  auto batch_go = create_batch_gradients_and_outputs(topology, 1);
  auto batch_hs = create_batch_hidden_states(topology, 1, 1, 1);
  MockLayer previous_layer(0, num_inputs);

  std::vector<double> input_features = { 1.0, 3.0 };
  batch_go[0].set_outputs(0, input_features);

  // Forward pass
  layer.calculate_forward_feed(batch_go, previous_layer, {}, batch_hs, 1, false);

  // Set incoming delta gradients from the next layer
  std::vector<double> downstream_gradients = { 0.5, -0.3, 0.2, -0.4, 0.6, -0.1 };
  batch_go[0].set_gradients(1, downstream_gradients);

  // Calculate and store analytical gradients
  layer.calculate_and_store_gradients(batch_go, batch_hs, previous_layer, 1, 1);

  const auto& stored_grads = layer.get_w_grads();
  ASSERT_EQ(stored_grads.size(), vocab_size * embed_dim);

  // Numerical gradient check via finite differences
  const double eps = 1e-6;
  for (size_t w_idx = 0; w_idx < init_weights.size(); ++w_idx)
  {
    // Compute loss at w + eps
    auto w_plus = init_weights;
    w_plus[w_idx] += eps;
    layer.set_w_values(w_plus);

    auto batch_go_plus = create_batch_gradients_and_outputs(topology, 1);
    batch_go_plus[0].set_outputs(0, input_features);
    layer.calculate_forward_feed(batch_go_plus, previous_layer, {}, batch_hs, 1, false);
    const auto out_plus = batch_go_plus[0].get_outputs(1);
    double loss_plus = 0.0;
    for (size_t k = 0; k < downstream_gradients.size(); ++k)
    {
      loss_plus += downstream_gradients[k] * out_plus[k];
    }

    // Compute loss at w - eps
    auto w_minus = init_weights;
    w_minus[w_idx] -= eps;
    layer.set_w_values(w_minus);

    auto batch_go_minus = create_batch_gradients_and_outputs(topology, 1);
    batch_go_minus[0].set_outputs(0, input_features);
    layer.calculate_forward_feed(batch_go_minus, previous_layer, {}, batch_hs, 1, false);
    const auto out_minus = batch_go_minus[0].get_outputs(1);
    double loss_minus = 0.0;
    for (size_t k = 0; k < downstream_gradients.size(); ++k)
    {
      loss_minus += downstream_gradients[k] * out_minus[k];
    }

    double numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
    EXPECT_NEAR(stored_grads[w_idx], numerical_grad, 1e-5)
      << "Weight gradient mismatch at index " << w_idx;
  }
}

TEST(EmbeddingLayerTest, MultiThreadedGradientsAccumulationConsistency)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  const unsigned num_inputs = 4;
  const unsigned vocab_size = 20;
  const unsigned embed_dim = 8;
  const size_t batch_size = 16;

  EmbeddingLayer layer1 = make_embedding_layer(num_inputs, vocab_size, embed_dim);
  EmbeddingLayer layer2 = make_embedding_layer(num_inputs, vocab_size, embed_dim);

  std::vector<double> init_weights(vocab_size * embed_dim, 0.0);
  for (size_t i = 0; i < init_weights.size(); ++i)
  {
    init_weights[i] = static_cast<double>(i) * 0.01 - 0.5;
  }
  layer1.set_w_values(init_weights);
  layer2.set_w_values(init_weights);

  std::vector<unsigned> topology = { num_inputs, num_inputs * embed_dim };
  auto batch_go1 = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_go2 = create_batch_gradients_and_outputs(topology, batch_size);
  auto batch_hs = create_batch_hidden_states(topology, batch_size, 1, 1);
  MockLayer previous_layer(0, num_inputs);

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> inputs = {
      static_cast<double>((b * 3 + 1) % vocab_size),
      static_cast<double>((b * 7 + 2) % vocab_size),
      static_cast<double>((b * 5 + 4) % vocab_size),
      static_cast<double>((b * 2 + 0) % vocab_size)
    };
    batch_go1[b].set_outputs(0, inputs);
    batch_go2[b].set_outputs(0, inputs);

    std::vector<double> deltas(num_inputs * embed_dim);
    for (size_t d = 0; d < deltas.size(); ++d)
    {
      deltas[d] = 0.1 * static_cast<double>((b + d) % 5);
    }
    batch_go1[b].set_gradients(1, deltas);
    batch_go2[b].set_gradients(1, deltas);
  }

  // Single-threaded accumulation
  layer1.set_number_of_threads(1);
  layer1.calculate_and_store_gradients(batch_go1, batch_hs, previous_layer, batch_size, 1);

  // Multi-threaded accumulation
  layer2.set_number_of_threads(4);
  layer2.calculate_and_store_gradients(batch_go2, batch_hs, previous_layer, batch_size, 1);

  const auto& grads1 = layer1.get_w_grads();
  const auto& grads2 = layer2.get_w_grads();
  ASSERT_EQ(grads1.size(), grads2.size());
  for (size_t i = 0; i < grads1.size(); ++i)
  {
    EXPECT_NEAR(grads1[i], grads2[i], 1e-12);
  }
}

TEST(EmbeddingLayerTest, SerializerSaveLoadRoundTrip)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  const unsigned num_inputs = 2;
  const unsigned vocab_size = 8;
  const unsigned embed_dim = 4;
  const unsigned embedding_layer_size = num_inputs * embed_dim;

  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(
      Layer::Architecture::Embedding,
      embedding_layer_size,
      activation(activation::method::linear, 0.0),
      0.0,
      0.01,
      OptimiserType::AdamW,
      0.9,
      false,
      0, 0, 0, 0, 0,
      vocab_size,
      embed_dim),
    LayerDetails(
      Layer::Architecture::FF,
      8,
      activation(activation::method::relu, 0.0),
      0.0,
      0.01,
      OptimiserType::AdamW,
      0.9,
      false,
      0, 0, 0, 0, 0,
      0, 0)
  };

  auto options = NeuralNetworkOptions::create({ num_inputs, embedding_layer_size, 8, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(
      1,
      activation(activation::method::sigmoid, 0.0),
      ErrorCalculation::type::mse,
      { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0 },
      0.01,
      OptimiserType::AdamW,
      0.9))
    .with_learning_rate(0.05)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> train_inputs = {
    { 1.0, 4.0 },
    { 2.0, 6.0 },
    { 0.0, 7.0 }
  };
  std::vector<std::vector<double>> train_targets = {
    { 0.2 },
    { 0.8 },
    { 0.5 }
  };

  nn.train(train_inputs, train_targets);

  auto pred_before = nn.think(train_inputs);

  std::string test_path = "test_embedding_layer_serializer.json";
  std::remove(test_path.c_str());
  NeuralNetworkSerializer::save(nn, test_path);

  auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(test_path));
  ASSERT_NE(loaded_nn, nullptr);

  const auto& loaded_layers = loaded_nn->get_layers();
  ASSERT_GE(loaded_layers.size(), 3u);

  const auto* emb_layer = dynamic_cast<const EmbeddingLayer*>(&loaded_layers[1]);
  ASSERT_NE(emb_layer, nullptr);
  EXPECT_EQ(emb_layer->get_vocabulary_size(), vocab_size);
  EXPECT_EQ(emb_layer->get_embedding_dimension(), embed_dim);
  EXPECT_EQ(emb_layer->get_layer_architecture(), Layer::Architecture::Embedding);

  // Compare predictions
  auto pred_after = loaded_nn->think(train_inputs);
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

TEST(EmbeddingLayerTest, EndToEndTrainingConvergence)
{
  MYODDWEB_PROFILE_FUNCTION("EmbeddingLayerTests");
  // Simple categorical mapping: 4 categories mapped to target scalars
  // Category 0 -> 0.1
  // Category 1 -> 0.4
  // Category 2 -> 0.7
  // Category 3 -> 0.9
  const unsigned num_inputs = 1;
  const unsigned vocab_size = 4;
  const unsigned embed_dim = 4;

  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(
      Layer::Architecture::Embedding,
      embed_dim,
      activation(activation::method::linear, 0.0),
      0.0,
      0.0,
      OptimiserType::SGD,
      0.0,
      false,
      0, 0, 0, 0, 0,
      vocab_size,
      embed_dim)
  };

  auto options = NeuralNetworkOptions::create({ num_inputs, num_inputs * embed_dim, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(
      1,
      activation(activation::method::linear, 0.0),
      ErrorCalculation::type::mse,
      { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0 },
      0.0,
      OptimiserType::SGD,
      0.0))
    .with_learning_rate(0.2)
    .with_batch_size(1)
    .with_number_of_epoch(500)
    .with_shuffle_training_data(true)
    .with_seed(1234)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = {
    { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }
  };
  std::vector<std::vector<double>> targets = {
    { 0.1 }, { 0.4 }, { 0.7 }, { 0.9 }
  };

  nn.train(inputs, targets);

  auto predictions = nn.think(inputs);
  ASSERT_EQ(predictions.size(), 4u);
  for (size_t i = 0; i < 4; ++i)
  {
    EXPECT_NEAR(predictions[i][0], targets[i][0], 0.05)
      << "Convergence failed for category " << i;
  }
}
