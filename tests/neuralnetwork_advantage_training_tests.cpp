#include <gtest/gtest.h>
#include "helpers/neuralnetworkserializer.h"
#include "layers/fflayer.h"
#include "layers/multioutputlayer.h"
#include "layers/multioutputlayerdetails.h"
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace myoddweb::nn;

namespace
{
NeuralNetworkOptions build_policy_options(OptimiserType optimiser, double learning_rate, int batch_size, std::optional<uint32_t> seed)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, optimiser, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    optimiser,
    0.9);

  return NeuralNetworkOptions::create({ 3, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(learning_rate)
    .with_batch_size(batch_size)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(seed)
    .with_clip_threshold(1000.0)
    .build();
}

NeuralNetworkOptions build_multi_output_policy_options()
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails branch_a_output(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, eval_config, 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails branch_b_output(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, eval_config, 0.0, OptimiserType::SGD, 0.0);
  std::vector<MultiOutputLayerDetails> multi_output_layer_details =
  {
    MultiOutputLayerDetails({}, branch_a_output),
    MultiOutputLayerDetails({}, branch_b_output)
  };

  return NeuralNetworkOptions::create({ 2, 2, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(multi_output_layer_details)
    .with_learning_rate(0.05)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_has_bias(true)
    .build();
}

std::vector<double> collect_layer_weights(const NeuralNetwork& nn, unsigned layer_index)
{
  const auto& layer = nn.get_layer(layer_index);
  const auto number_input_neurons = layer.get_number_input_neurons();
  const auto number_output_neurons = layer.get_number_output_neurons();

  std::vector<double> weights;
  for (unsigned input_index = 0; input_index < number_input_neurons; ++input_index)
  {
    for (unsigned output_index = 0; output_index < number_output_neurons; ++output_index)
    {
      weights.push_back(layer.get_weight_value(input_index, output_index));
    }
  }
  return weights;
}

std::vector<double> collect_output_layer_weights(const NeuralNetwork& nn)
{
  return collect_layer_weights(nn, 2);
}
} // namespace

TEST(NeuralNetworkAdvantageTrainingTest, PositiveAdvantageIncreasesTakenActionProbability)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.5, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, advantages);
  const auto after = nn.think(inputs[0]);

  EXPECT_GT(after[0], before[0]);
}

TEST(NeuralNetworkAdvantageTrainingTest, NegativeAdvantageDecreasesTakenActionProbability)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.5, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> advantages = { -1.0 };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, advantages);
  const auto after = nn.think(inputs[0]);

  EXPECT_LT(after[0], before[0]);
}

TEST(NeuralNetworkAdvantageTrainingTest, ZeroAdvantageLeavesOutputWeightsUnchanged)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.5, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 }, { -0.3, 0.5, 0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 }, { 0.0, 1.0 } };
  std::vector<double> advantages = { 0.0, 0.0 };

  const auto before = collect_output_layer_weights(nn);
  nn.train_with_advantages(inputs, action_targets, advantages);
  const auto after = collect_output_layer_weights(nn);

  ASSERT_EQ(before.size(), after.size());
  for (size_t i = 0; i < before.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(before[i], after[i]);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, AdvantageMagnitudeScalesOutputWeightDeltaLinearly)
{
  auto options_a = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  auto options_b = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn_a(options_a);
  NeuralNetwork nn_b(options_b);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };

  const auto before_a = collect_output_layer_weights(nn_a);
  const auto before_b = collect_output_layer_weights(nn_b);
  ASSERT_EQ(before_a, before_b);

  nn_a.train_with_advantages(inputs, action_targets, { 1.0 });
  nn_b.train_with_advantages(inputs, action_targets, { 2.0 });

  const auto after_a = collect_output_layer_weights(nn_a);
  const auto after_b = collect_output_layer_weights(nn_b);

  ASSERT_EQ(before_a.size(), after_a.size());
  for (size_t i = 0; i < before_a.size(); ++i)
  {
    const double delta_a = after_a[i] - before_a[i];
    const double delta_b = after_b[i] - before_b[i];
    EXPECT_NEAR(delta_b, 2.0 * delta_a, 1e-9);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, HandlesMoreExamplesThanConfiguredBatchSizeByChunking)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.05, 2, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs =
  {
    { 0.1, 0.2, 0.3 },
    { -0.2, 0.1, 0.0 },
    { 0.4, -0.3, 0.2 },
    { -0.1, -0.2, 0.3 },
    { 0.3, 0.3, -0.1 }
  };
  std::vector<std::vector<double>> action_targets =
  {
    { 1.0, 0.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 }
  };
  std::vector<double> advantages = { 1.0, -1.0, 0.5, -0.5, 1.0 };

  ASSERT_NO_THROW(nn.train_with_advantages(inputs, action_targets, advantages));

  const auto result = nn.think(inputs[0]);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_NEAR(result[0] + result[1], 1.0, 1e-9);
  EXPECT_GE(result[0], 0.0);
  EXPECT_GE(result[1], 0.0);
}

TEST(NeuralNetworkAdvantageTrainingTest, MismatchedInputSizesThrows)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> too_few_advantages = {};

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, too_few_advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, MultiOutputLayerHeadThrows)
{
  auto options = build_multi_output_policy_options();
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ZeroAdvantageLeavesHiddenWeightsUnchanged)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.5, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 }, { -0.3, 0.5, 0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 }, { 0.0, 1.0 } };
  std::vector<double> advantages = { 0.0, 0.0 };

  const auto before = collect_layer_weights(nn, 1);
  nn.train_with_advantages(inputs, action_targets, advantages);
  const auto after = collect_layer_weights(nn, 1);

  ASSERT_EQ(before.size(), after.size());
  for (size_t i = 0; i < before.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(before[i], after[i]);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, AdvantageMagnitudeScalesHiddenWeightDeltaLinearly)
{
  auto options_a = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  auto options_b = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn_a(options_a);
  NeuralNetwork nn_b(options_b);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };

  const auto before_a = collect_layer_weights(nn_a, 1);
  const auto before_b = collect_layer_weights(nn_b, 1);
  ASSERT_EQ(before_a, before_b);

  nn_a.train_with_advantages(inputs, action_targets, { 1.0 });
  nn_b.train_with_advantages(inputs, action_targets, { 2.0 });

  const auto after_a = collect_layer_weights(nn_a, 1);
  const auto after_b = collect_layer_weights(nn_b, 1);

  ASSERT_EQ(before_a.size(), after_a.size());
  for (size_t i = 0; i < before_a.size(); ++i)
  {
    const double delta_a = after_a[i] - before_a[i];
    const double delta_b = after_b[i] - before_b[i];
    EXPECT_NEAR(delta_b, 2.0 * delta_a, 1e-9);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, EmptyInputsLeavesWeightsUnchanged)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.5, 4, 42u);
  NeuralNetwork nn(options);

  const auto before = collect_output_layer_weights(nn);
  ASSERT_NO_THROW(nn.train_with_advantages({}, {}, {}));
  const auto after = collect_output_layer_weights(nn);

  ASSERT_EQ(before.size(), after.size());
  for (size_t i = 0; i < before.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(before[i], after[i]);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, SerializerSavesAndLoadsAdvantageTrainedNetwork)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.2, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 }, { -0.3, 0.5, 0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 }, { 0.0, 1.0 } };
  std::vector<double> advantages = { 1.5, 0.8 };

  nn.train_with_advantages(inputs, action_targets, advantages);

  const auto output_before_save = nn.think(inputs[0]);
  const auto weights_layer1_before = collect_layer_weights(nn, 1);
  const auto weights_layer2_before = collect_layer_weights(nn, 2);

  const std::string test_file_path = "test_advantage_trained_network.json";
  NeuralNetworkSerializer::save(nn, test_file_path);

  std::unique_ptr<NeuralNetwork> loaded_nn(NeuralNetworkSerializer::load(test_file_path));
  std::remove(test_file_path.c_str());

  ASSERT_NE(loaded_nn, nullptr);

  const auto output_after_load = loaded_nn->think(inputs[0]);
  ASSERT_EQ(output_before_save.size(), output_after_load.size());
  for (size_t i = 0; i < output_before_save.size(); ++i)
  {
    EXPECT_NEAR(output_after_load[i], output_before_save[i], 1e-9);
  }

  const auto weights_layer1_after = collect_layer_weights(*loaded_nn, 1);
  const auto weights_layer2_after = collect_layer_weights(*loaded_nn, 2);

  ASSERT_EQ(weights_layer1_before.size(), weights_layer1_after.size());
  for (size_t i = 0; i < weights_layer1_before.size(); ++i)
  {
    EXPECT_NEAR(weights_layer1_after[i], weights_layer1_before[i], 1e-9);
  }

  ASSERT_EQ(weights_layer2_before.size(), weights_layer2_after.size());
  for (size_t i = 0; i < weights_layer2_before.size(); ++i)
  {
    EXPECT_NEAR(weights_layer2_after[i], weights_layer2_before[i], 1e-9);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, WorksWithAdamOptimiser)
{
  auto options = build_policy_options(OptimiserType::Adam, 0.05, 2, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, advantages);
  const auto after = nn.think(inputs[0]);

  EXPECT_GT(after[0], before[0]);
}

TEST(NeuralNetworkAdvantageTrainingTest, MismatchedInputDimensionsThrows)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, MismatchedActionTargetDimensionsThrows)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, NonFiniteAdvantageThrows)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };

  EXPECT_THROW(
    nn.train_with_advantages(inputs, action_targets, { std::numeric_limits<double>::quiet_NaN() }),
    std::runtime_error);

  EXPECT_THROW(
    nn.train_with_advantages(inputs, action_targets, { std::numeric_limits<double>::infinity() }),
    std::runtime_error);

  EXPECT_THROW(
    nn.train_with_advantages(inputs, action_targets, { -std::numeric_limits<double>::infinity() }),
    std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ContinuousActionRegressionPolicyWithMSE)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    1,
    activation(activation::method::linear, 0.0),
    ErrorCalculation::type::mse,
    eval_config,
    0.0,
    OptimiserType::SGD,
    0.9);

  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.1)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.5, -0.2 } };
  const auto initial_prediction = nn.think(inputs[0])[0];
  const double target = initial_prediction + 1.0;
  std::vector<std::vector<double>> targets = { { target } };

  nn.train_with_advantages(inputs, targets, { 1.0 });
  const auto after_positive = nn.think(inputs[0])[0];
  EXPECT_GT(after_positive, initial_prediction);
}

TEST(NeuralNetworkAdvantageTrainingTest, RecurrentLSTMPolicyNetwork)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::Lstm, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::Adam,
    0.9);

  auto options = NeuralNetworkOptions::create({ 2, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.05)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(2)
    .with_seed(42u)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.5, 0.3 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  const auto before_weights = collect_layer_weights(nn, 1);
  ASSERT_NO_THROW(nn.train_with_advantages(inputs, action_targets, advantages));
  const auto after_weights = collect_layer_weights(nn, 1);

  ASSERT_EQ(before_weights.size(), after_weights.size());
  bool any_weight_changed = false;
  for (size_t i = 0; i < before_weights.size(); ++i)
  {
    if (std::abs(after_weights[i] - before_weights[i]) > 1e-12)
    {
      any_weight_changed = true;
      break;
    }
  }
  EXPECT_TRUE(any_weight_changed);
}

TEST(NeuralNetworkAdvantageTrainingTest, ExtremeAdvantageScalingNumericalStability)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.01, 1, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.5, -0.2, 0.3 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };

  std::vector<double> extreme_positive_adv = { 100.0 };
  ASSERT_NO_THROW(nn.train_with_advantages(inputs, action_targets, extreme_positive_adv));

  const auto output_positive = nn.think(inputs[0]);
  EXPECT_TRUE(std::isfinite(output_positive[0]));
  EXPECT_TRUE(std::isfinite(output_positive[1]));
  EXPECT_GE(output_positive[0], 0.0);
  EXPECT_LE(output_positive[0], 1.0);

  std::vector<double> extreme_negative_adv = { -100.0 };
  ASSERT_NO_THROW(nn.train_with_advantages(inputs, action_targets, extreme_negative_adv));

  const auto output_negative = nn.think(inputs[0]);
  EXPECT_TRUE(std::isfinite(output_negative[0]));
  EXPECT_TRUE(std::isfinite(output_negative[1]));
  EXPECT_GE(output_negative[0], 0.0);
  EXPECT_LE(output_negative[0], 1.0);
}

