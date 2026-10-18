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

NeuralNetworkOptions build_three_action_policy_options(double entropy_coefficient)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    3,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::SGD,
    0.9);

  return NeuralNetworkOptions::create({ 3, 4, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.5)
    .with_batch_size(4)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .with_clip_threshold(1000.0)
    .build()
    .with_entropy_coefficient(entropy_coefficient);
}

std::vector<double> collect_output_neuron_weights(const NeuralNetwork& nn, unsigned output_index)
{
  const auto& layer = nn.get_layer(2);
  std::vector<double> weights;
  for (unsigned input_index = 0; input_index < layer.get_number_input_neurons(); ++input_index)
  {
    weights.push_back(layer.get_weight_value(input_index, output_index));
  }
  return weights;
}

void expect_same_weights(const std::vector<double>& expected, const std::vector<double>& actual, double tolerance = 1e-12)
{
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], actual[i], tolerance);
  }
}

bool any_weight_changed(const std::vector<double>& before, const std::vector<double>& after)
{
  for (size_t i = 0; i < before.size() && i < after.size(); ++i)
  {
    if (before[i] != after[i])
    {
      return true;
    }
  }
  return false;
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

TEST(NeuralNetworkAdvantageTrainingTest, NonFiniteInputThrows)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  std::vector<std::vector<double>> nan_inputs = { { 0.2, std::numeric_limits<double>::quiet_NaN(), 0.4 } };
  EXPECT_THROW(nn.train_with_advantages(nan_inputs, action_targets, advantages), std::runtime_error);

  std::vector<std::vector<double>> inf_inputs = { { 0.2, std::numeric_limits<double>::infinity(), 0.4 } };
  EXPECT_THROW(nn.train_with_advantages(inf_inputs, action_targets, advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, NonFiniteActionTargetThrows)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<double> advantages = { 1.0 };

  std::vector<std::vector<double>> nan_targets = { { std::numeric_limits<double>::quiet_NaN(), 0.0 } };
  EXPECT_THROW(nn.train_with_advantages(inputs, nan_targets, advantages), std::runtime_error);

  std::vector<std::vector<double>> inf_targets = { { 1.0, std::numeric_limits<double>::infinity() } };
  EXPECT_THROW(nn.train_with_advantages(inputs, inf_targets, advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, NegativeActionTargetWithSoftmaxThrows)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> negative_targets = { { -0.5, 1.5 } };
  std::vector<double> advantages = { 1.0 };

  EXPECT_THROW(nn.train_with_advantages(inputs, negative_targets, advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ZeroSumActionTargetWithSoftmaxThrows)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> zero_targets = { { 0.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  EXPECT_THROW(nn.train_with_advantages(inputs, zero_targets, advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, SoftmaxTemperatureScalingInPolicyNetwork)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 2.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::SGD,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.2)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, { 1.0 });
  const auto after = nn.think(inputs[0]);

  EXPECT_GT(after[0], before[0]);
}

TEST(NeuralNetworkAdvantageTrainingTest, LabelSmoothingWithAdvantageTraining)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.2, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::SGD,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.2)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };

  const auto before = nn.think(inputs[0]);
  ASSERT_NO_THROW(nn.train_with_advantages(inputs, action_targets, { 1.0 }));
  const auto after = nn.think(inputs[0]);

  EXPECT_GT(after[0], before[0]);
}

TEST(NeuralNetworkAdvantageTrainingTest, FFTanhHiddenLayerWithAdamWAndSoftmax)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.01, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 4, 2 })
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

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, advantages);
  const auto after = nn.think(inputs[0]);

  EXPECT_GT(after[0], before[0]);
  EXPECT_LT(after[1], before[1]);
  EXPECT_NEAR(after[0] + after[1], 1.0, 1e-6);
}

TEST(NeuralNetworkAdvantageTrainingTest, FFTanhOutputLayerWithAdamWAndMSE)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.01, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    1,
    activation(activation::method::tanh, 0.0),
    ErrorCalculation::type::mse,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 4, 1 })
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

  std::vector<std::vector<double>> inputs = { { 0.3, -0.2, 0.5 } };
  std::vector<std::vector<double>> action_targets = { { 0.8 } };
  std::vector<double> advantages = { 1.0 };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, advantages);
  const auto after = nn.think(inputs[0]);

  EXPECT_GT(after[0], before[0]);
  EXPECT_LT(std::abs(after[0] - action_targets[0][0]), std::abs(before[0] - action_targets[0][0]));
}

TEST(NeuralNetworkAdvantageTrainingTest, AdamWDecoupledWeightDecayWithAdvantageTraining)
{
  const double weight_decay = 0.1;
  const double learning_rate = 0.05;

  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 2, activation(activation::method::tanh, 0.0), 0.0, weight_decay, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    1,
    activation(activation::method::tanh, 0.0),
    ErrorCalculation::type::mse,
    eval_config,
    weight_decay,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(learning_rate)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .build();

  NeuralNetwork nn(options);

  const auto initial_hidden_weights = collect_layer_weights(nn, 1);
  const auto initial_output_weights = collect_layer_weights(nn, 2);
  const double initial_bias = nn.get_layer(1).get_bias_value(0);

  std::vector<std::vector<double>> inputs = { { 0.5, -0.5 } };
  std::vector<std::vector<double>> targets = { { 0.5 } };
  std::vector<double> zero_advantages = { 0.0 };

  nn.train_with_advantages(inputs, targets, zero_advantages);

  const auto updated_hidden_weights = collect_layer_weights(nn, 1);
  const auto updated_output_weights = collect_layer_weights(nn, 2);
  const double updated_bias = nn.get_layer(1).get_bias_value(0);

  const double expected_decay_factor = 1.0 - learning_rate * weight_decay;
  for (size_t i = 0; i < initial_hidden_weights.size(); ++i)
  {
    EXPECT_NEAR(updated_hidden_weights[i], initial_hidden_weights[i] * expected_decay_factor, 1e-6);
  }
  for (size_t i = 0; i < initial_output_weights.size(); ++i)
  {
    EXPECT_NEAR(updated_output_weights[i], initial_output_weights[i] * expected_decay_factor, 1e-6);
  }
  EXPECT_DOUBLE_EQ(updated_bias, initial_bias);
}

TEST(NeuralNetworkAdvantageTrainingTest, TanhOutputTargetOutOfRangeThrows)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    1,
    activation(activation::method::tanh, 0.0),
    ErrorCalculation::type::mse,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 4, 1 })
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

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<double> advantages = { 1.0 };

  std::vector<std::vector<double>> out_of_range_high = { { 1.05 } };
  EXPECT_THROW(nn.train_with_advantages(inputs, out_of_range_high, advantages), std::runtime_error);

  std::vector<std::vector<double>> out_of_range_low = { { -1.05 } };
  EXPECT_THROW(nn.train_with_advantages(inputs, out_of_range_low, advantages), std::runtime_error);

  std::vector<std::vector<double>> in_range = { { 0.85 } };
  EXPECT_NO_THROW(nn.train_with_advantages(inputs, in_range, advantages));
}

TEST(NeuralNetworkAdvantageTrainingTest, ZeroEntropyCoefficientMatchesPreExistingZeroAdvantageBehaviour)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.5, 4, 42u).with_entropy_coefficient(0.0);
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

TEST(NeuralNetworkAdvantageTrainingTest, PositiveEntropyCoefficientMovesProbabilitiesTowardUniformWithZeroAdvantage)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.5, 4, 42u).with_entropy_coefficient(0.2);
  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> zero_advantage = { 0.0 };

  const auto before = nn.think(inputs[0]);
  ASSERT_NE(before[0], 0.5);
  nn.train_with_advantages(inputs, action_targets, zero_advantage);
  const auto after = nn.think(inputs[0]);

  EXPECT_LT(std::abs(after[0] - 0.5), std::abs(before[0] - 0.5));
  EXPECT_NEAR(after[0] + after[1], 1.0, 1e-9);
}

TEST(NeuralNetworkAdvantageTrainingTest, EntropyCoefficientMagnitudeScalesWeightDeltaLinearly)
{
  auto options_a = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u).with_entropy_coefficient(0.1);
  auto options_b = build_policy_options(OptimiserType::SGD, 0.1, 4, 42u).with_entropy_coefficient(0.2);
  NeuralNetwork nn_a(options_a);
  NeuralNetwork nn_b(options_b);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> zero_advantage = { 0.0 };

  const auto before_a = collect_output_layer_weights(nn_a);
  const auto before_b = collect_output_layer_weights(nn_b);
  ASSERT_EQ(before_a, before_b);

  nn_a.train_with_advantages(inputs, action_targets, zero_advantage);
  nn_b.train_with_advantages(inputs, action_targets, zero_advantage);

  const auto after_a = collect_output_layer_weights(nn_a);
  const auto after_b = collect_output_layer_weights(nn_b);

  ASSERT_EQ(before_a.size(), after_a.size());
  bool any_nonzero_delta = false;
  for (size_t i = 0; i < before_a.size(); ++i)
  {
    const double delta_a = after_a[i] - before_a[i];
    const double delta_b = after_b[i] - before_b[i];
    EXPECT_NEAR(delta_b, 2.0 * delta_a, 1e-9);
    if (std::abs(delta_a) > 1e-12)
    {
      any_nonzero_delta = true;
    }
  }
  EXPECT_TRUE(any_nonzero_delta);
}

TEST(NeuralNetworkAdvantageTrainingTest, EntropyCoefficientHasNoEffectOnNonSoftmaxOutput)
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

  auto options_no_entropy = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.1)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .with_entropy_coefficient(0.0)
    .build();

  auto options_with_entropy = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.1)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .with_entropy_coefficient(5.0)
    .build();

  NeuralNetwork nn_no_entropy(options_no_entropy);
  NeuralNetwork nn_with_entropy(options_with_entropy);

  std::vector<std::vector<double>> inputs = { { 0.5, -0.2 } };
  std::vector<std::vector<double>> targets = { { 0.8 } };
  std::vector<double> advantages = { 1.0 };

  nn_no_entropy.train_with_advantages(inputs, targets, advantages);
  nn_with_entropy.train_with_advantages(inputs, targets, advantages);

  const auto output_no_entropy = nn_no_entropy.think(inputs[0]);
  const auto output_with_entropy = nn_with_entropy.think(inputs[0]);

  ASSERT_EQ(output_no_entropy.size(), output_with_entropy.size());
  for (size_t i = 0; i < output_no_entropy.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(output_no_entropy[i], output_with_entropy[i]);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, FFTanhWithNegativeAdvantageDecreasesProbability)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.01, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 4, 2 })
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

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  std::vector<double> negative_advantages = { -1.0 };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, negative_advantages);
  const auto after = nn.think(inputs[0]);

  EXPECT_LT(after[0], before[0]);
  EXPECT_GT(after[1], before[1]);
}

TEST(NeuralNetworkAdvantageTrainingTest, SerialisationPreservesEntropyCoefficient)
{
  auto options = build_policy_options(OptimiserType::SGD, 0.2, 4, 42u).with_entropy_coefficient(0.15);
  NeuralNetwork nn(options);

  const std::string test_file_path = "test_entropy_coeff_network.json";
  NeuralNetworkSerializer::save(nn, test_file_path);

  std::unique_ptr<NeuralNetwork> loaded_nn(NeuralNetworkSerializer::load(test_file_path));
  std::remove(test_file_path.c_str());

  ASSERT_NE(loaded_nn, nullptr);
  EXPECT_DOUBLE_EQ(loaded_nn->options().entropy_coefficient(), 0.15);
}

TEST(NeuralNetworkAdvantageTrainingTest, EntropyCoefficientNegativeThrows)
{
  EXPECT_THROW(
    build_policy_options(OptimiserType::SGD, 0.2, 4, 42u).with_entropy_coefficient(-0.1).build(),
    std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, EntropyCoefficientNonFiniteThrows)
{
  EXPECT_THROW(
    build_policy_options(OptimiserType::SGD, 0.2, 4, 42u).with_entropy_coefficient(std::numeric_limits<double>::quiet_NaN()).build(),
    std::runtime_error);
  EXPECT_THROW(
    build_policy_options(OptimiserType::SGD, 0.2, 4, 42u).with_entropy_coefficient(std::numeric_limits<double>::infinity()).build(),
    std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, EntropyBonusAppliedToRecurrentLayerBPTT)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::Elman, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::SGD,
    0.0);

  auto options = NeuralNetworkOptions::create({ 2, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.1)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(2)
    .with_entropy_coefficient(0.2)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.5, -0.2, 0.1, 0.3 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0, 1.0 } };
  std::vector<double> advantages = { 1.0 };

  EXPECT_NO_THROW(nn.train_with_advantages(inputs, action_targets, advantages));
}

TEST(NeuralNetworkAdvantageTrainingTest, EntropyBonusZeroCoefficientIsIdenticalToDisabled)
{
  auto options_default = build_policy_options(OptimiserType::SGD, 0.1, 1, 42u);
  auto options_zero = build_policy_options(OptimiserType::SGD, 0.1, 1, 42u).with_entropy_coefficient(0.0);

  NeuralNetwork nn_default(options_default);
  NeuralNetwork nn_zero(options_zero);

  const std::vector<std::vector<double>> inputs = { { 0.3, -0.2, 0.5 } };
  const std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  const std::vector<double> advantages = { 0.75 };

  nn_default.train_with_advantages(inputs, action_targets, advantages);
  nn_zero.train_with_advantages(inputs, action_targets, advantages);

  const auto out_default = nn_default.think(inputs[0]);
  const auto out_zero = nn_zero.think(inputs[0]);

  ASSERT_EQ(out_default.size(), out_zero.size());
  for (size_t i = 0; i < out_default.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(out_default[i], out_zero[i]);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, EntropyBonusRecurrentSingleTimestepSafe)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::Elman, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::SGD,
    0.0);

  auto options = NeuralNetworkOptions::create({ 2, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.1)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(1)
    .with_entropy_coefficient(0.1)
    .build();

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.4, -0.6 } };
  std::vector<std::vector<double>> action_targets = { { 0.0, 1.0 } };
  std::vector<double> advantages = { 1.5 };

  EXPECT_NO_THROW(nn.train_with_advantages(inputs, action_targets, advantages));
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskEmptyListMatchesUnmaskedOverload)
{
  NeuralNetwork unmasked(build_three_action_policy_options(0.0));
  NeuralNetwork masked(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 }, { -0.3, 0.5, 0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0 } };
  std::vector<double> advantages = { -0.7, 1.3 };
  std::vector<std::vector<double>> no_masks;

  unmasked.train_with_advantages(inputs, action_targets, advantages);
  masked.train_with_advantages(inputs, action_targets, advantages, no_masks);

  expect_same_weights(collect_output_layer_weights(unmasked), collect_output_layer_weights(masked));
  expect_same_weights(collect_layer_weights(unmasked, 1), collect_layer_weights(masked, 1));
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskAllLegalMatchesUnmasked)
{
  NeuralNetwork unmasked(build_three_action_policy_options(0.0));
  NeuralNetwork masked(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 }, { -0.3, 0.5, 0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };
  std::vector<double> advantages = { -0.7, 1.3 };
  std::vector<std::vector<double>> all_legal = { { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } };

  unmasked.train_with_advantages(inputs, action_targets, advantages);
  masked.train_with_advantages(inputs, action_targets, advantages, all_legal);

  const auto unmasked_weights = collect_output_layer_weights(unmasked);
  const auto masked_weights = collect_output_layer_weights(masked);
  ASSERT_EQ(unmasked_weights.size(), masked_weights.size());
  for (size_t i = 0; i < unmasked_weights.size(); ++i)
  {
    EXPECT_NEAR(unmasked_weights[i], masked_weights[i], 1e-12);
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskIllegalActionOutputWeightsUnchanged)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { -1.0 };
  std::vector<std::vector<double>> masks = { { 1.0, 1.0, 0.0 } };

  const auto illegal_before = collect_output_neuron_weights(nn, 2);
  const auto legal_before = collect_output_neuron_weights(nn, 1);
  nn.train_with_advantages(inputs, action_targets, advantages, masks);

  expect_same_weights(illegal_before, collect_output_neuron_weights(nn, 2));
  EXPECT_TRUE(any_weight_changed(legal_before, collect_output_neuron_weights(nn, 1)));
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskUnmaskedControlDoesMoveIllegalActionWeights)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { -1.0 };

  const auto before = collect_output_neuron_weights(nn, 2);
  nn.train_with_advantages(inputs, action_targets, advantages);

  EXPECT_TRUE(any_weight_changed(before, collect_output_neuron_weights(nn, 2)));
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskNegativeAdvantageShiftsLegalProbabilityAwayFromTakenAction)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { -1.0 };
  std::vector<std::vector<double>> masks = { { 1.0, 1.0, 0.0 } };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, advantages, masks);
  const auto after = nn.think(inputs[0]);

  EXPECT_LT(after[0] / (after[0] + after[1]), before[0] / (before[0] + before[1]));
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskEntropyLeavesIllegalActionOutputWeightsUnchanged)
{
  NeuralNetwork nn(build_three_action_policy_options(0.5));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> zero_advantage = { 0.0 };
  std::vector<std::vector<double>> masks = { { 1.0, 1.0, 0.0 } };

  const auto illegal_before = collect_output_neuron_weights(nn, 2);
  const auto legal_before = collect_output_neuron_weights(nn, 0);
  nn.train_with_advantages(inputs, action_targets, zero_advantage, masks);

  expect_same_weights(illegal_before, collect_output_neuron_weights(nn, 2));
  EXPECT_TRUE(any_weight_changed(legal_before, collect_output_neuron_weights(nn, 0)));
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskCountMismatchThrows)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 }, { -0.3, 0.5, 0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };
  std::vector<double> advantages = { 1.0, 1.0 };
  std::vector<std::vector<double>> one_mask = { { 1.0, 1.0, 1.0 } };

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages, one_mask), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskSizeMismatchThrows)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };
  std::vector<std::vector<double>> short_mask = { { 1.0, 1.0 } };

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages, short_mask), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskNonBinaryValueThrows)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };
  std::vector<std::vector<double>> half_mask = { { 1.0, 0.5, 1.0 } };

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages, half_mask), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskTargetOnIllegalActionThrows)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 0.0, 0.0, 1.0 } };
  std::vector<double> advantages = { 1.0 };
  std::vector<std::vector<double>> masks = { { 1.0, 1.0, 0.0 } };

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages, masks), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskOnNonSoftmaxOutputThrows)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };
  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails linear_output(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, eval_config, 0.0, OptimiserType::SGD, 0.9);
  auto linear_options = NeuralNetworkOptions::create({ 3, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(linear_output)
    .with_learning_rate(0.1)
    .with_batch_size(4)
    .with_number_of_epoch(1)
    .with_has_bias(true)
    .with_seed(42u)
    .build();
  NeuralNetwork linear_nn(linear_options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 0.5, 0.0 } };
  std::vector<double> advantages = { 1.0 };
  std::vector<std::vector<double>> masks = { { 1.0, 0.0 } };

  EXPECT_THROW(linear_nn.train_with_advantages(inputs, action_targets, advantages, masks), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskPositiveAdvantageShiftsLegalProbabilityTowardsTakenAction)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };
  std::vector<std::vector<double>> masks = { { 1.0, 1.0, 0.0 } };

  const auto illegal_before = collect_output_neuron_weights(nn, 2);
  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, advantages, masks);
  const auto after = nn.think(inputs[0]);

  EXPECT_GT(after[0] / (after[0] + after[1]), before[0] / (before[0] + before[1]));
  expect_same_weights(illegal_before, collect_output_neuron_weights(nn, 2));
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskSingleLegalActionLeavesWeightsUnchanged)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { 1.5 };
  std::vector<std::vector<double>> masks = { { 1.0, 0.0, 0.0 } };

  const auto before = collect_output_layer_weights(nn);
  nn.train_with_advantages(inputs, action_targets, advantages, masks);
  const auto after = collect_output_layer_weights(nn);

  expect_same_weights(before, after);
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskBatchedMultipleSamples)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 }, { -0.3, 0.5, 0.1 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0 } };
  std::vector<double> advantages = { 1.0, -1.0 };
  std::vector<std::vector<double>> masks = { { 1.0, 1.0, 0.0 }, { 1.0, 0.0, 1.0 } };

  const auto before = collect_output_layer_weights(nn);
  EXPECT_NO_THROW(nn.train_with_advantages(inputs, action_targets, advantages, masks));
  const auto after = collect_output_layer_weights(nn);

  EXPECT_TRUE(any_weight_changed(before, after));
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskAllIllegalThrows)
{
  NeuralNetwork nn(build_three_action_policy_options(0.0));

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { 1.0 };
  std::vector<std::vector<double>> masks = { { 0.0, 0.0, 0.0 } };

  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages, masks), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, ActionMaskMultiBatchRemainderAndRepeatedCallsReuseBufferCleanly)
{
  // Batch size 2 with 5 samples (batches of 2, 2, 1) followed by a second call of 1 sample.
  // Verifies that _training_masked_targets_buffer reuses capacity without stale data leakage
  // and produces identical results to training each mini-batch in its own call.
  auto options_single_call = build_three_action_policy_options(0.15);
  options_single_call = NeuralNetworkOptions(options_single_call)
    .with_batch_size(2)
    .with_entropy_coefficient(0.15);

  NeuralNetwork nn_batched(options_single_call);
  NeuralNetwork nn_stepwise(options_single_call);

  std::vector<std::vector<double>> inputs =
  {
    { 0.2, -0.1, 0.4 },
    { -0.3, 0.5, 0.1 },
    { 0.1, 0.2, -0.4 },
    { -0.5, -0.2, 0.3 },
    { 0.4, -0.3, 0.2 }
  };
  std::vector<std::vector<double>> action_targets =
  {
    { 1.0, 0.0, 0.0 },
    { 0.0, 0.0, 1.0 },
    { 0.0, 1.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 }
  };
  std::vector<double> advantages = { 1.2, -0.8, 0.5, -1.1, 0.9 };
  std::vector<std::vector<double>> masks =
  {
    { 1.0, 1.0, 0.0 },
    { 1.0, 0.0, 1.0 },
    { 0.0, 1.0, 1.0 },
    { 1.0, 1.0, 1.0 },
    { 1.0, 1.0, 0.0 }
  };

  nn_batched.train_with_advantages(inputs, action_targets, advantages, masks);

  // Feed the same 5 samples in chunks of 2, 2, 1 to nn_stepwise
  nn_stepwise.train_with_advantages(
    { inputs[0], inputs[1] },
    { action_targets[0], action_targets[1] },
    { advantages[0], advantages[1] },
    { masks[0], masks[1] });
  nn_stepwise.train_with_advantages(
    { inputs[2], inputs[3] },
    { action_targets[2], action_targets[3] },
    { advantages[2], advantages[3] },
    { masks[2], masks[3] });
  nn_stepwise.train_with_advantages(
    { inputs[4] },
    { action_targets[4] },
    { advantages[4] },
    { masks[4] });

  expect_same_weights(collect_output_layer_weights(nn_stepwise), collect_output_layer_weights(nn_batched), 1e-12);
  expect_same_weights(collect_layer_weights(nn_stepwise, 1), collect_layer_weights(nn_batched, 1), 1e-12);

  // Follow up with another single-sample call on both networks to ensure shrunk buffer re-expands/reuses cleanly
  nn_batched.train_with_advantages({ inputs[0] }, { action_targets[0] }, { advantages[0] }, { masks[0] });
  nn_stepwise.train_with_advantages({ inputs[0] }, { action_targets[0] }, { advantages[0] }, { masks[0] });

  expect_same_weights(collect_output_layer_weights(nn_stepwise), collect_output_layer_weights(nn_batched), 1e-12);
}

TEST(NeuralNetworkAdvantageTrainingTest, LargeActionSpaceExceedingStackCacheMatchesUnmaskedAndMaskedExpectations)
{
  // 72 actions (> 64 stack cache threshold in add_entropy_bonus_to_gradients and add_masked_entropy_bonus_to_gradients)
  const unsigned num_actions = 72;
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    num_actions,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::SGD,
    0.9);

  auto options = NeuralNetworkOptions::create({ 4, 8, num_actions })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.2)
    .with_batch_size(2)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(99u)
    .with_clip_threshold(1000.0)
    .build()
    .with_entropy_coefficient(0.25);

  NeuralNetwork nn_unmasked(options);
  NeuralNetwork nn_all_legal(options);

  std::vector<double> target_row(num_actions, 0.0);
  target_row[5] = 1.0;
  std::vector<double> all_legal_row(num_actions, 1.0);

  std::vector<std::vector<double>> inputs = { { 0.3, -0.2, 0.5, 0.1 } };
  std::vector<std::vector<double>> targets = { target_row };
  std::vector<double> advantages = { 1.0 };
  std::vector<std::vector<double>> all_legal_masks = { all_legal_row };

  nn_unmasked.train_with_advantages(inputs, targets, advantages);
  nn_all_legal.train_with_advantages(inputs, targets, advantages, all_legal_masks);

  expect_same_weights(collect_output_layer_weights(nn_unmasked), collect_output_layer_weights(nn_all_legal), 1e-12);

  // Also verify masked illegal action weight invariance for >64 actions
  NeuralNetwork nn_masked(options);
  std::vector<double> partial_mask(num_actions, 1.0);
  partial_mask[70] = 0.0;
  const auto illegal_before = collect_output_neuron_weights(nn_masked, 70);
  nn_masked.train_with_advantages(inputs, targets, advantages, { partial_mask });
  expect_same_weights(illegal_before, collect_output_neuron_weights(nn_masked, 70), 1e-12);
}

TEST(NeuralNetworkAdvantageTrainingTest, CompoundOutputLayerMixedSoftmaxAndTanhHeadsValidationAndTraining)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  std::vector<OutputLayerDetails> compound_outputs =
  {
    OutputLayerDetails(2, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, eval_config, 0.0, OptimiserType::SGD, 0.9),
    OutputLayerDetails(1, activation(activation::method::tanh, 0.0), ErrorCalculation::type::mse, eval_config, 0.0, OptimiserType::SGD, 0.9)
  };

  auto options = NeuralNetworkOptions::create({ 3, 4, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(compound_outputs)
    .with_learning_rate(0.1)
    .with_batch_size(2)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(42u)
    .build()
    .with_entropy_coefficient(0.1);

  NeuralNetwork nn(options);

  std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 }, { -0.3, 0.5, 0.1 } };
  std::vector<std::vector<double>> valid_targets = { { 1.0, 0.0, 0.5 }, { 0.0, 1.0, -0.5 } };
  std::vector<double> advantages = { 1.0, 0.8 };

  const auto before_0 = nn.think(inputs[0]);
  EXPECT_NO_THROW(nn.train_with_advantages(inputs, valid_targets, advantages));
  const auto after_0 = nn.think(inputs[0]);
  EXPECT_GT(after_0[0], before_0[0]);

  // Out-of-range tanh target in the second head must be caught by the pre-computed activation validation
  std::vector<std::vector<double>> invalid_tanh_targets = { { 1.0, 0.0, 1.5 }, { 0.0, 1.0, -0.5 } };
  EXPECT_THROW(nn.train_with_advantages(inputs, invalid_tanh_targets, advantages), std::runtime_error);
}

TEST(NeuralNetworkAdvantageTrainingTest, GrnAndSelfAttentionPoolPolicyTrunkAdvantageTraining)
{
  const unsigned input_size = 4;
  const unsigned hidden_size = 4;
  const unsigned num_actions = 3;

  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails::create_grn(hidden_size, 6, activation(activation::method::elu, 1.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, true),
    LayerDetails(Layer::Architecture::SelfAttention, hidden_size, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, true, 0, 0, 0, 2, 6, 0, 0),
    LayerDetails(Layer::Architecture::Gru, hidden_size, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::AttentionPool, hidden_size, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 3, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    num_actions,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ input_size, hidden_size, hidden_size, hidden_size, hidden_size, num_actions })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.05)
    .with_batch_size(2)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(2)
    .with_seed(42u)
    .build()
    .with_entropy_coefficient(0.05);

  NeuralNetwork nn(options);

  std::vector<double> state_a = { 0.2, -0.4, 0.5, 0.1 };
  std::vector<double> state_b = { -0.3, 0.3, -0.2, 0.4 };

  std::vector<std::vector<double>> inputs = { state_a, state_b };
  std::vector<std::vector<double>> action_targets = { { 1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 } };
  std::vector<double> advantages = { 1.5, 1.2 };
  std::vector<std::vector<double>> masks = { { 1.0, 1.0, 0.0 }, { 1.0, 1.0, 1.0 } };

  const auto before_a = nn.think(state_a);
  ASSERT_EQ(before_a.size(), num_actions);

  // Action masks with BPTT must throw, whereas unmasked advantage training succeeds
  EXPECT_THROW(nn.train_with_advantages(inputs, action_targets, advantages, masks), std::runtime_error);
  EXPECT_NO_THROW(nn.train_with_advantages(inputs, action_targets, advantages));

  const auto after_a = nn.think(state_a);
  ASSERT_EQ(after_a.size(), num_actions);
  for (unsigned a = 0; a < num_actions; ++a)
  {
    EXPECT_TRUE(std::isfinite(after_a[a]));
  }
  EXPECT_GT(after_a[0], before_a[0]);
}

TEST(NeuralNetworkAdvantageTrainingTest, SingleSampleThinkReuseAndAdvantageScalingConsistency)
{
  auto options = build_policy_options(OptimiserType::AdamW, 0.05, 4, 42u)
    .with_entropy_coefficient(0.05);
  NeuralNetwork nn(options);

  const std::vector<std::vector<double>> states =
  {
    { 0.25, -0.15, 0.45 },
    { -0.35, 0.55, -0.10 },
    { 0.10, 0.20, -0.30 },
    { -0.20, -0.40, 0.60 }
  };

  // Verify that single-sample think(input, output_buf) matches batched think(states) across repeated rollouts
  const auto batch_preds = nn.think(states);
  ASSERT_EQ(batch_preds.size(), states.size());

  std::vector<double> single_pred;
  for (size_t i = 0; i < states.size(); ++i)
  {
    nn.think(states[i], single_pred);
    ASSERT_EQ(single_pred.size(), batch_preds[i].size());
    for (size_t j = 0; j < single_pred.size(); ++j)
    {
      EXPECT_NEAR(single_pred[j], batch_preds[i][j], 1e-12);
    }
  }

  // Run multiple advantage-training updates with mixed positive, zero, unit, and negative advantages
  const std::vector<std::vector<double>> targets =
  {
    { 1.0, 0.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 },
    { 0.0, 1.0 }
  };
  const std::vector<double> advantages = { 1.5, 0.0, 1.0, -0.75 };

  for (int step = 0; step < 5; ++step)
  {
    nn.train_with_advantages(states, targets, advantages);
  }

  const auto updated_batch_preds = nn.think(states);
  for (size_t i = 0; i < states.size(); ++i)
  {
    nn.think(states[i], single_pred);
    for (size_t j = 0; j < single_pred.size(); ++j)
    {
      EXPECT_TRUE(std::isfinite(single_pred[j]));
      EXPECT_NEAR(single_pred[j], updated_batch_preds[i][j], 1e-12);
    }
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, LazyTransposedWeightsUpdatedOnDemandDuringAdvantageTraining)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 5, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    3,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 5, 4, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.05)
    .with_batch_size(2)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(123u)
    .build()
    .with_entropy_coefficient(0.02);

  NeuralNetwork nn(options);

  const auto* layer1_ff = dynamic_cast<const FFLayer*>(&nn.get_layer(1));
  const auto* layer2_ff = dynamic_cast<const FFLayer*>(&nn.get_layer(2));
  const auto* layer3_ff = dynamic_cast<const FFLayer*>(&nn.get_layer(3));
  ASSERT_NE(layer1_ff, nullptr);
  ASSERT_NE(layer2_ff, nullptr);
  ASSERT_NE(layer3_ff, nullptr);

  const std::vector<std::vector<double>> inputs =
  {
    { 0.4, -0.2, 0.7 },
    { -0.5, 0.6, -0.1 }
  };
  const std::vector<std::vector<double>> targets =
  {
    { 1.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 }
  };
  const std::vector<double> advantages = { 1.5, 1.2 };

  const auto before_probs = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, targets, advantages);
  const auto after_probs = nn.think(inputs[0]);
  EXPECT_GT(after_probs[0], before_probs[0]);

  // Verify lazy get_w_values_T() matches exact transpose of get_w_values() on all FFLayers after training
  const FFLayer* ff_layers[3] = { layer1_ff, layer2_ff, layer3_ff };
  for (int idx = 0; idx < 3; ++idx)
  {
    const FFLayer* ff = ff_layers[idx];
    const size_t num_in = ff->get_number_input_neurons();
    const size_t num_out = ff->get_number_neurons();
    const auto& w = ff->get_w_values();
    const auto& w_T = ff->get_w_values_T();
    ASSERT_EQ(w.size(), num_in * num_out);
    ASSERT_EQ(w_T.size(), num_in * num_out);
    for (size_t i = 0; i < num_in; ++i)
    {
      for (size_t j = 0; j < num_out; ++j)
      {
        EXPECT_DOUBLE_EQ(w_T[j * num_in + i], w[i * num_out + j]);
      }
    }
  }
}

TEST(NeuralNetworkAdvantageTrainingTest, TcnGrnSelfAttentionFollowedByFFLayerAdvantageTraining)
{
  const unsigned input_size = 4;
  const unsigned hidden_size = 4;
  const unsigned num_actions = 3;

  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::Tcn, hidden_size, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 2, 1, 0, 0, 0, 0),
    LayerDetails::create_grn(hidden_size, 6, activation(activation::method::elu, 1.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, true),
    LayerDetails(Layer::Architecture::SelfAttention, hidden_size, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, true, 0, 0, 0, 2, 6, 0, 0),
    LayerDetails(Layer::Architecture::Gru, hidden_size, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::AttentionPool, hidden_size, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 3, 0, 0, 0, 0, 0, 0),
    LayerDetails(Layer::Architecture::FF, hidden_size, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    num_actions,
    activation(activation::method::softmax, 0.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ input_size, hidden_size, hidden_size, hidden_size, hidden_size, hidden_size, hidden_size, num_actions })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.05)
    .with_batch_size(2)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(2)
    .with_seed(77u)
    .build()
    .with_entropy_coefficient(0.02);

  NeuralNetwork nn(options);

  const std::vector<std::vector<double>> inputs =
  {
    { 0.3, -0.2, 0.5, 0.1 },
    { -0.4, 0.4, -0.1, 0.3 },
    { 0.2, 0.1, -0.3, 0.6 },
    { -0.1, -0.3, 0.4, -0.2 }
  };
  const std::vector<std::vector<double>> targets =
  {
    { 1.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 }
  };
  const std::vector<double> advantages = { 1.0, 1.2, 0.9, 1.4 };

  const auto initial_probs = nn.think(inputs[1]);
  ASSERT_EQ(initial_probs.size(), num_actions);

  for (int step = 0; step < 3; ++step)
  {
    EXPECT_NO_THROW(nn.train_with_advantages(inputs, targets, advantages));
  }

  const auto updated_probs = nn.think(inputs[1]);
  ASSERT_EQ(updated_probs.size(), num_actions);
  for (unsigned a = 0; a < num_actions; ++a)
  {
    EXPECT_TRUE(std::isfinite(updated_probs[a]));
  }
  EXPECT_GT(updated_probs[0], initial_probs[0]);
}


