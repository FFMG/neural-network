#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <vector>

#include "../include/neuralnetwork/common/lookaheaddetails.h"
#include "../include/neuralnetwork/common/simd_utils.h"
#include "../include/neuralnetwork/helpers/neuralnetworkserializer.h"
#include "../include/neuralnetwork/layers/attentionpoollayer.h"
#include "../include/neuralnetwork/layers/elmanrnnlayer.h"
#include "../include/neuralnetwork/layers/embeddinglayer.h"
#include "../include/neuralnetwork/layers/fflayer.h"
#include "../include/neuralnetwork/layers/grurnnlayer.h"
#include "../include/neuralnetwork/layers/lstmlayer.h"
#include "../include/neuralnetwork/layers/multioutputlayer.h"
#include "../include/neuralnetwork/layers/residualprojector.h"
#include "../include/neuralnetwork/layers/selfattentionlayer.h"
#include "../include/neuralnetwork/layers/tcnlayer.h"
#include "../include/neuralnetwork/neuralnetwork.h"
#include "../include/neuralnetwork/neuralnetworkoptions.h"

using namespace myoddweb::nn;

namespace
{
std::string get_unique_temp_filename(const std::string& prefix)
{
  static size_t counter = 0;
  return (std::filesystem::temp_directory_path() / (prefix + "_" + std::to_string(++counter) + ".json")).string();
}
} // namespace

// ============================================================================
// 1. LookaheadDetails Unit Tests
// ============================================================================

TEST(LookaheadDetailsTest, ConstructorAndGetters)
{
  LookaheadDetails details(true, 10, 0.6);
  EXPECT_TRUE(details.enabled());
  EXPECT_EQ(details.synchronisation_period(), 10);
  EXPECT_DOUBLE_EQ(details.slow_weights_step_size(), 0.6);
}

TEST(LookaheadDetailsTest, CopySemantics)
{
  LookaheadDetails original(true, 8, 0.7);
  LookaheadDetails copy_constructed(original);

  EXPECT_TRUE(copy_constructed.enabled());
  EXPECT_EQ(copy_constructed.synchronisation_period(), 8);
  EXPECT_DOUBLE_EQ(copy_constructed.slow_weights_step_size(), 0.7);

  LookaheadDetails copy_assigned(false, 5, 0.5);
  copy_assigned = original;

  EXPECT_TRUE(copy_assigned.enabled());
  EXPECT_EQ(copy_assigned.synchronisation_period(), 8);
  EXPECT_DOUBLE_EQ(copy_assigned.slow_weights_step_size(), 0.7);
}

TEST(LookaheadDetailsTest, MoveSemantics)
{
  LookaheadDetails source(true, 12, 0.8);
  LookaheadDetails moved_constructed(std::move(source));

  EXPECT_TRUE(moved_constructed.enabled());
  EXPECT_EQ(moved_constructed.synchronisation_period(), 12);
  EXPECT_DOUBLE_EQ(moved_constructed.slow_weights_step_size(), 0.8);
  EXPECT_FALSE(source.enabled());

  LookaheadDetails another_source(true, 15, 0.4);
  LookaheadDetails move_assigned(false, 5, 0.5);
  move_assigned = std::move(another_source);

  EXPECT_TRUE(move_assigned.enabled());
  EXPECT_EQ(move_assigned.synchronisation_period(), 15);
  EXPECT_DOUBLE_EQ(move_assigned.slow_weights_step_size(), 0.4);
  EXPECT_FALSE(another_source.enabled());
}

TEST(LookaheadDetailsTest, NeuralNetworkOptionsIntegrationAndValidation)
{
  // Valid configuration
  auto opt = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_lookahead(LookaheadDetails(true, 6, 0.5))
    .build();

  EXPECT_TRUE(opt.lookahead().enabled());
  EXPECT_EQ(opt.lookahead().synchronisation_period(), 6);
  EXPECT_DOUBLE_EQ(opt.lookahead().slow_weights_step_size(), 0.5);

  // Fluent overload
  auto opt_fluent = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_lookahead(true, 8, 0.6)
    .build();

  EXPECT_TRUE(opt_fluent.lookahead().enabled());
  EXPECT_EQ(opt_fluent.lookahead().synchronisation_period(), 8);
  EXPECT_DOUBLE_EQ(opt_fluent.lookahead().slow_weights_step_size(), 0.6);

  // Invalid period <= 0
  EXPECT_THROW(
    NeuralNetworkOptions::create({ 2, 4, 1 })
      .with_lookahead(true, 0, 0.5)
      .build(),
    std::runtime_error
  );

  // Invalid step size <= 0.0
  EXPECT_THROW(
    NeuralNetworkOptions::create({ 2, 4, 1 })
      .with_lookahead(true, 5, 0.0)
      .build(),
    std::runtime_error
  );

  // Invalid step size > 1.0
  EXPECT_THROW(
    NeuralNetworkOptions::create({ 2, 4, 1 })
      .with_lookahead(true, 5, 1.1)
      .build(),
    std::runtime_error
  );
}

// ============================================================================
// 2. SIMD & Scalar Lookahead Step Kernels
// ============================================================================

TEST(LookaheadSimdTest, HandComputedScalarMath)
{
  std::vector<double> slow = { 1.0, 2.0, 3.0, 4.0 };
  std::vector<double> fast = { 2.0, 0.0, 5.0, 2.0 };
  const double alpha = 0.5;

  simd::scalar_lookahead_step(slow.data(), fast.data(), alpha, slow.size());

  // phi = phi + 0.5 * (theta - phi) = [1.5, 1.0, 4.0, 3.0]
  // theta = phi
  EXPECT_NEAR(slow[0], 1.5, 1e-12);
  EXPECT_NEAR(slow[1], 1.0, 1e-12);
  EXPECT_NEAR(slow[2], 4.0, 1e-12);
  EXPECT_NEAR(slow[3], 3.0, 1e-12);

  EXPECT_NEAR(fast[0], 1.5, 1e-12);
  EXPECT_NEAR(fast[1], 1.0, 1e-12);
  EXPECT_NEAR(fast[2], 4.0, 1e-12);
  EXPECT_NEAR(fast[3], 3.0, 1e-12);
}

TEST(LookaheadSimdTest, SimdVsScalarEquivalenceAcrossSizes)
{
  const std::vector<size_t> test_sizes = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 127, 128, 255, 256 };
  const double alpha = 0.6;

  for (size_t n : test_sizes)
  {
    std::vector<double> slow_scalar(n);
    std::vector<double> fast_scalar(n);
    std::vector<double> slow_simd(n);
    std::vector<double> fast_simd(n);

    for (size_t i = 0; i < n; ++i)
    {
      slow_scalar[i] = std::sin(static_cast<double>(i) * 0.3);
      fast_scalar[i] = std::cos(static_cast<double>(i) * 0.5);
      slow_simd[i] = slow_scalar[i];
      fast_simd[i] = fast_scalar[i];
    }

    simd::scalar_lookahead_step(slow_scalar.data(), fast_scalar.data(), alpha, n);
    simd::lookahead_step(slow_simd.data(), fast_simd.data(), alpha, n);

    for (size_t i = 0; i < n; ++i)
    {
      EXPECT_NEAR(slow_simd[i], slow_scalar[i], 1e-12) << "Mismatch at index " << i << " for size " << n;
      EXPECT_NEAR(fast_simd[i], fast_scalar[i], 1e-12) << "Mismatch at index " << i << " for size " << n;
      EXPECT_NEAR(slow_simd[i], fast_simd[i], 1e-12) << "Fast not synchronized to slow at index " << i;
    }
  }
}

// ============================================================================
// 3. Layer-Level Lookahead Interpolation Tests
// ============================================================================

TEST(LookaheadLayerInterpolationTest, FFLayerInterpolation)
{
  const auto opt = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.0, OptimiserType::None, 0.0))
    .build();

  Layers slow_layers(opt);
  Layers fast_layers(opt);

  slow_layers[1].set_w_values({ 1.0, 2.0, 3.0, 4.0 });
  slow_layers[1].set_b_values({ 10.0, 20.0 });

  fast_layers[1].set_w_values({ 3.0, 6.0, 7.0, 8.0 });
  fast_layers[1].set_b_values({ 30.0, 40.0 });

  const double alpha = 0.5;
  slow_layers.update_lookahead_slow_weights(fast_layers, alpha);

  // phi = 1.0 + 0.5 * (3.0 - 1.0) = 2.0
  EXPECT_NEAR(slow_layers[1].get_w_values()[0], 2.0, 1e-10);
  EXPECT_NEAR(slow_layers[1].get_w_values()[1], 4.0, 1e-10);
  EXPECT_NEAR(slow_layers[1].get_w_values()[2], 5.0, 1e-10);
  EXPECT_NEAR(slow_layers[1].get_w_values()[3], 6.0, 1e-10);

  EXPECT_NEAR(fast_layers[1].get_w_values()[0], 2.0, 1e-10);
  EXPECT_NEAR(fast_layers[1].get_w_values()[1], 4.0, 1e-10);

  EXPECT_NEAR(slow_layers[1].get_b_values()[0], 20.0, 1e-10);
  EXPECT_NEAR(slow_layers[1].get_b_values()[1], 30.0, 1e-10);
  EXPECT_NEAR(fast_layers[1].get_b_values()[0], 20.0, 1e-10);
  EXPECT_NEAR(fast_layers[1].get_b_values()[1], 30.0, 1e-10);
}

TEST(LookaheadLayerInterpolationTest, FFLayerRefreshesTransposeCacheOnFastLayer)
{
  const auto opt = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.0, OptimiserType::None, 0.0))
    .build();

  Layers slow_layers(opt);
  Layers fast_layers(opt);

  slow_layers[1].set_w_values({ 1.0, 2.0, 3.0, 4.0 });
  fast_layers[1].set_w_values({ 3.0, 6.0, 7.0, 8.0 });

  // Populate the fast layer's transpose cache with the pre-sync weights, the
  // same way apply_stored_gradients() would before a lookahead sync happens.
  auto& fast_ff = static_cast<FFLayer&>(fast_layers[1]);
  fast_ff.cache_recurrent_weights();

  const double alpha = 0.5;
  slow_layers.update_lookahead_slow_weights(fast_layers, alpha);

  // phi = 1.0 + 0.5 * (3.0 - 1.0) = 2.0, so the fast weights are rewritten
  // in place to the new slow values.
  ASSERT_EQ(fast_ff.get_w_values()[0], 2.0);

  // The cached transpose must be rebuilt against the new weights, not left
  // stale from the pre-sync cache_recurrent_weights() call above.
  std::vector<double> expected_w_values_t(4);
  simd::transpose(fast_ff.get_w_values().data(), expected_w_values_t.data(), 2, 2);
  ASSERT_EQ(fast_ff.get_w_values_T().size(), expected_w_values_t.size());
  for (size_t i = 0; i < expected_w_values_t.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(fast_ff.get_w_values_T()[i], expected_w_values_t[i]);
  }
}

TEST(LookaheadLayerInterpolationTest, ElmanRNNLayerInterpolation)
{
  const auto opt = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(5)
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::Elman, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.0, OptimiserType::None, 0.0))
    .build();

  Layers slow_layers(opt);
  Layers fast_layers(opt);

  slow_layers[1].set_w_values({ 1.0, 1.0, 1.0, 1.0 });
  fast_layers[1].set_w_values({ 5.0, 5.0, 5.0, 5.0 });

  const double alpha = 0.25;
  slow_layers.update_lookahead_slow_weights(fast_layers, alpha);

  // 1.0 + 0.25 * (5.0 - 1.0) = 2.0
  EXPECT_NEAR(slow_layers[1].get_w_values()[0], 2.0, 1e-10);
  EXPECT_NEAR(fast_layers[1].get_w_values()[0], 2.0, 1e-10);
}

TEST(LookaheadLayerInterpolationTest, GRURNNLayerInterpolation)
{
  const auto opt = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(5)
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::Gru, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.0, OptimiserType::None, 0.0))
    .build();

  Layers slow_layers(opt);
  Layers fast_layers(opt);

  slow_layers[1].set_w_values({ 2.0, 2.0, 2.0, 2.0 });
  fast_layers[1].set_w_values({ 6.0, 6.0, 6.0, 6.0 });

  const double alpha = 0.5;
  slow_layers.update_lookahead_slow_weights(fast_layers, alpha);

  EXPECT_NEAR(slow_layers[1].get_w_values()[0], 4.0, 1e-10);
  EXPECT_NEAR(fast_layers[1].get_w_values()[0], 4.0, 1e-10);
}

TEST(LookaheadLayerInterpolationTest, LSTMLayerInterpolation)
{
  const auto opt = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(5)
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::Lstm, 2, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.0, OptimiserType::None, 0.0))
    .build();

  Layers slow_layers(opt);
  Layers fast_layers(opt);

  slow_layers[1].set_w_values({ 1.0, 1.0, 1.0, 1.0 });
  fast_layers[1].set_w_values({ 11.0, 11.0, 11.0, 11.0 });

  const double alpha = 0.3;
  slow_layers.update_lookahead_slow_weights(fast_layers, alpha);

  EXPECT_NEAR(slow_layers[1].get_w_values()[0], 4.0, 1e-10);
  EXPECT_NEAR(fast_layers[1].get_w_values()[0], 4.0, 1e-10);
}

TEST(LookaheadLayerInterpolationTest, EmbeddingLayerInterpolation)
{
  const auto opt = NeuralNetworkOptions::create({ 1, 4, 1 })
    .with_hidden_layers({
      LayerDetails(
        Layer::Architecture::Embedding,
        4,
        activation(activation::method::linear, 0.0),
        0.0,
        0.0,
        OptimiserType::None,
        0.0,
        false,
        0, 0, 0, 0, 0,
        10, // vocabulary_size
        4   // embedding_dimension
      )
    })
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.0, OptimiserType::None, 0.0))
    .build();

  Layers slow_layers(opt);
  Layers fast_layers(opt);

  slow_layers[1].set_w_values({ 1.0, 1.0, 1.0, 1.0 });
  fast_layers[1].set_w_values({ 5.0, 5.0, 5.0, 5.0 });

  const double alpha = 0.5;
  slow_layers.update_lookahead_slow_weights(fast_layers, alpha);

  EXPECT_NEAR(slow_layers[1].get_w_values()[0], 3.0, 1e-10);
  EXPECT_NEAR(fast_layers[1].get_w_values()[0], 3.0, 1e-10);
}

TEST(LookaheadLayerInterpolationTest, ResidualProjectorInterpolation)
{
  ResidualProjector slow_proj(16, 8, activation(activation::method::relu, 0.0), 0.0, 42);
  ResidualProjector fast_proj(16, 8, activation(activation::method::relu, 0.0), 0.0, 42);

  const auto orig_w0 = slow_proj.get_w_values()[0];
  fast_proj.update_weight(0, 0, 4.0);

  const double alpha = 0.5;
  slow_proj.update_lookahead_slow_weights(fast_proj, alpha);

  EXPECT_NEAR(slow_proj.get_w_values()[0], orig_w0 + 2.0, 1e-10);
  EXPECT_NEAR(fast_proj.get_w_values()[0], orig_w0 + 2.0, 1e-10);
}

TEST(LookaheadDetailsTest, TrainGuardsAgainstUnvalidatedSynchronisationPeriod)
{
  // NeuralNetworkOptions::validate() (invoked by build()) rejects a
  // synchronisation period <= 0, but NeuralNetwork's primary constructor
  // does not require build() to have been called. train() must still guard
  // against this to avoid a division by zero.
  const auto opt = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_lookahead(true, 0, 0.5);

  NeuralNetwork nn(opt);
  const std::vector<std::vector<double>> inputs = { { 0.0, 0.0 } };
  const std::vector<std::vector<double>> targets = { { 0.0 } };

  EXPECT_THROW(nn.train(inputs, targets), std::runtime_error);
}

// ============================================================================
// 4. End-to-End Training Integration Tests
// ============================================================================

TEST(LookaheadTrainingTest, XorConvergenceWithAdamW)
{
  const std::vector<std::vector<double>> inputs = {
    { 0.0, 0.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 },
    { 1.0, 1.0 }
  };
  const std::vector<std::vector<double>> targets = {
    { 0.0 },
    { 1.0 },
    { 1.0 },
    { 0.0 }
  };

  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::AdamW, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(
      1,
      activation(activation::method::sigmoid, 1.0),
      ErrorCalculation::type::mse,
      { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } },
      0.0,
      OptimiserType::AdamW,
      0.0
    ))
    .with_lookahead(true, 5, 0.5)
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

  nn.train(inputs, targets);

  const auto out00 = nn.think(inputs[0])[0];
  const auto out01 = nn.think(inputs[1])[0];
  const auto out10 = nn.think(inputs[2])[0];
  const auto out11 = nn.think(inputs[3])[0];

  EXPECT_NEAR(out00, 0.0, 0.15);
  EXPECT_NEAR(out01, 1.0, 0.15);
  EXPECT_NEAR(out10, 1.0, 0.15);
  EXPECT_NEAR(out11, 0.0, 0.15);
}

TEST(LookaheadTrainingTest, XorConvergenceWithRAdam)
{
  const std::vector<std::vector<double>> inputs = {
    { 0.0, 0.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 },
    { 1.0, 1.0 }
  };
  const std::vector<std::vector<double>> targets = {
    { 0.0 },
    { 1.0 },
    { 1.0 },
    { 0.0 }
  };

  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::RAdam, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(
      1,
      activation(activation::method::sigmoid, 1.0),
      ErrorCalculation::type::mse,
      { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } },
      0.0,
      OptimiserType::RAdam,
      0.0
    ))
    .with_lookahead(true, 5, 0.5)
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

  nn.train(inputs, targets);

  const auto out00 = nn.think(inputs[0])[0];
  const auto out01 = nn.think(inputs[1])[0];
  const auto out10 = nn.think(inputs[2])[0];
  const auto out11 = nn.think(inputs[3])[0];

  EXPECT_NEAR(out00, 0.0, 0.15);
  EXPECT_NEAR(out01, 1.0, 0.15);
  EXPECT_NEAR(out10, 1.0, 0.15);
  EXPECT_NEAR(out11, 0.0, 0.15);
}

TEST(LookaheadTrainingTest, CoexistenceWithStochasticWeightAveraging)
{
  const std::vector<std::vector<double>> inputs = {
    { 0.0, 0.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 },
    { 1.0, 1.0 }
  };
  const std::vector<std::vector<double>> targets = {
    { 0.0 },
    { 1.0 },
    { 1.0 },
    { 0.0 }
  };

  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(true)
    .with_has_bias(true)
    .with_hidden_layers({
      LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::AdamW, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
    })
    .with_output_layer_details(OutputLayerDetails(
      1,
      activation(activation::method::sigmoid, 1.0),
      ErrorCalculation::type::mse,
      { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } },
      0.0,
      OptimiserType::AdamW,
      0.0
    ))
    .with_lookahead(true, 5, 0.5)
    .with_stochastic_weight_averaging(true, 0.5, 0.05)
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

  nn.train(inputs, targets);

  const auto out00 = nn.think(inputs[0])[0];
  const auto out01 = nn.think(inputs[1])[0];
  const auto out10 = nn.think(inputs[2])[0];
  const auto out11 = nn.think(inputs[3])[0];

  EXPECT_NEAR(out00, 0.0, 0.2);
  EXPECT_NEAR(out01, 1.0, 0.2);
  EXPECT_NEAR(out10, 1.0, 0.2);
  EXPECT_NEAR(out11, 0.0, 0.2);
}

// ============================================================================
// 5. Model Serialization Round-Trip Tests
// ============================================================================

TEST(LookaheadSerializerTest, SaveAndLoadOptionsRoundTrip)
{
  const std::string file_path = get_unique_temp_filename("lookahead_roundtrip");

  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_learning_rate(0.05)
    .with_number_of_epoch(5)
    .with_lookahead(true, 8, 0.65)
    .build();

  NeuralNetwork nn(options);
  const std::vector<std::vector<double>> dummy_in = { { 0.1, 0.2 } };
  const std::vector<std::vector<double>> dummy_out = { { 0.5 } };
  nn.train(dummy_in, dummy_out);

  NeuralNetworkSerializer::save(nn, file_path);

  std::unique_ptr<NeuralNetwork> loaded_nn(NeuralNetworkSerializer::load(file_path));
  ASSERT_NE(loaded_nn, nullptr);

  const auto& loaded_opt = loaded_nn->options();
  EXPECT_TRUE(loaded_opt.lookahead().enabled());
  EXPECT_EQ(loaded_opt.lookahead().synchronisation_period(), 8);
  EXPECT_DOUBLE_EQ(loaded_opt.lookahead().slow_weights_step_size(), 0.65);

  std::filesystem::remove(file_path);
}
