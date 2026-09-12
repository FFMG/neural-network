#include <gtest/gtest.h>
#include "helpers/neuralnetworkserializer.h"
#include "layers/fflayer.h"
#include "layers/multioutputlayer.h"
#include "layers/multioutputlayerdetails.h"
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>

using namespace myoddweb::nn;

namespace
{
double calculate_entropy(const std::vector<double>& probs)
{
  double entropy = 0.0;
  for (double p : probs)
  {
    if (p > 1e-15)
    {
      entropy -= p * std::log(p);
    }
  }
  return entropy;
}

} // namespace

TEST(TemperatureAndOptimiserTest, ActivationTemperatureClampingAndValidation)
{
  // Negative, zero, and sub-epsilon temperatures must be clamped to 1e-6
  activation act_neg(activation::method::softmax, 0.0, -2.0, -0.5);
  EXPECT_DOUBLE_EQ(act_neg.get_temperature(), 1e-6);
  EXPECT_DOUBLE_EQ(act_neg.get_inference_temperature(), 1e-6);

  activation act_zero(activation::method::softmax, 0.0, 0.0, 0.0);
  EXPECT_DOUBLE_EQ(act_zero.get_temperature(), 1e-6);
  EXPECT_DOUBLE_EQ(act_zero.get_inference_temperature(), 1e-6);

  const double nan_val = std::numeric_limits<double>::quiet_NaN();
  const double inf_val = std::numeric_limits<double>::infinity();
  activation act_nonfinite(activation::method::softmax, 0.0, nan_val, inf_val);
  EXPECT_DOUBLE_EQ(act_nonfinite.get_temperature(), 1e-6);
  EXPECT_DOUBLE_EQ(act_nonfinite.get_inference_temperature(), 1e-6);

  // Setters must also clamp safely
  activation act(activation::method::softmax, 0.0, 1.0, 1.0);
  act.set_temperature(-5.0);
  EXPECT_DOUBLE_EQ(act.get_temperature(), 1e-6);

  act.set_inference_temperature(0.0);
  EXPECT_DOUBLE_EQ(act.get_inference_temperature(), 1e-6);

  act.set_temperature(nan_val);
  EXPECT_DOUBLE_EQ(act.get_temperature(), 1e-6);

  act.set_inference_temperature(inf_val);
  EXPECT_DOUBLE_EQ(act.get_inference_temperature(), 1e-6);

  act.set_temperature(1.5);
  EXPECT_DOUBLE_EQ(act.get_temperature(), 1.5);

  act.set_inference_temperature(0.3);
  EXPECT_DOUBLE_EQ(act.get_inference_temperature(), 0.3);
}

TEST(TemperatureAndOptimiserTest, SoftmaxInferenceTemperatureControlsSharpness)
{
  const std::vector<double> base_logits = { 2.0, 1.0, 0.0 };

  // Standard temperature T = 1.0
  activation act_standard(activation::method::softmax, 0.0, 1.0, 1.0);
  std::vector<double> out_standard = base_logits;
  act_standard.activate(out_standard.data(), out_standard.data() + out_standard.size(), false);

  // Low temperature T = 0.1 (sharp, high confidence, low entropy)
  activation act_low(activation::method::softmax, 0.0, 1.0, 0.1);
  std::vector<double> out_low = base_logits;
  act_low.activate(out_low.data(), out_low.data() + out_low.size(), false);

  // High temperature T = 5.0 (flat, low confidence, high entropy)
  activation act_high(activation::method::softmax, 0.0, 1.0, 5.0);
  std::vector<double> out_high = base_logits;
  act_high.activate(out_high.data(), out_high.data() + out_high.size(), false);

  // Verification
  EXPECT_GT(out_low[0], out_standard[0]);
  EXPECT_GT(out_standard[0], out_high[0]);

  // Lower temperature produces lower entropy (sharper distribution)
  const double entropy_low = calculate_entropy(out_low);
  const double entropy_standard = calculate_entropy(out_standard);
  const double entropy_high = calculate_entropy(out_high);

  EXPECT_LT(entropy_low, entropy_standard);
  EXPECT_LT(entropy_standard, entropy_high);

  // Probabilities must all sum to 1.0
  double sum_low = out_low[0] + out_low[1] + out_low[2];
  double sum_standard = out_standard[0] + out_standard[1] + out_standard[2];
  double sum_high = out_high[0] + out_high[1] + out_high[2];

  EXPECT_NEAR(sum_low, 1.0, 1e-12);
  EXPECT_NEAR(sum_standard, 1.0, 1e-12);
  EXPECT_NEAR(sum_high, 1.0, 1e-12);
}

TEST(TemperatureAndOptimiserTest, SoftmaxDualTemperatureTrainingVsInference)
{
  // Training temperature 2.0 (high entropy exploration)
  // Inference temperature 0.2 (greedy exploitation)
  activation act(activation::method::softmax, 0.0, 2.0, 0.2);

  std::vector<double> train_run = { 1.5, 0.5 };
  act.activate(train_run.data(), train_run.data() + train_run.size(), true);

  std::vector<double> infer_run = { 1.5, 0.5 };
  act.activate(infer_run.data(), infer_run.data() + infer_run.size(), false);

  // Under training mode (T = 2.0), probabilities are flatter
  // Under inference mode (T = 0.2), winning class is significantly sharper
  EXPECT_GT(infer_run[0], train_run[0]);
  EXPECT_LT(infer_run[1], train_run[1]);

  EXPECT_NEAR(train_run[0] + train_run[1], 1.0, 1e-12);
  EXPECT_NEAR(infer_run[0] + infer_run[1], 1.0, 1e-12);

  // Exact math:
  // train: diff = 1.0 / 2.0 = 0.5 -> P(0) = 1 / (1 + exp(-0.5))
  const double expected_train_0 = 1.0 / (1.0 + std::exp(-0.5));
  EXPECT_NEAR(train_run[0], expected_train_0, 1e-12);

  // infer: diff = 1.0 / 0.2 = 5.0 -> P(0) = 1 / (1 + exp(-5.0))
  const double expected_infer_0 = 1.0 / (1.0 + std::exp(-5.0));
  EXPECT_NEAR(infer_run[0], expected_infer_0, 1e-12);
}

TEST(TemperatureAndOptimiserTest, NeuralNetworkInferenceTemperatureRuntimeAdjustment)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 1.0, 1.0),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.05)
    .with_batch_size(1)
    .with_has_bias(true)
    .with_seed(42u)
    .build();

  NeuralNetwork nn(options);

  const std::vector<double> input = { 0.5, -0.3, 0.8 };
  const auto out_initial = nn.think(input);

  EXPECT_DOUBLE_EQ(nn.get_temperature(), 1.0);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(), 1.0);

  // Adjust inference temperature to 0.1 at runtime
  nn.set_inference_temperature(0.1);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(), 0.1);

  const auto out_sharp = nn.think(input);

  // The leading class probability should be much closer to 1.0
  const size_t leading_idx = (out_initial[0] > out_initial[1]) ? 0 : 1;
  EXPECT_GT(out_sharp[leading_idx], out_initial[leading_idx]);

  // Adjust inference temperature to 10.0 at runtime
  nn.set_inference_temperature(10.0);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(), 10.0);

  const auto out_flat = nn.think(input);

  // The distribution should be closer to 0.5, 0.5
  EXPECT_NEAR(out_flat[0], 0.5, 0.15);
  EXPECT_NEAR(out_flat[1], 0.5, 0.15);

  // Training temperature should have remained unchanged at 1.0
  EXPECT_DOUBLE_EQ(nn.get_temperature(), 1.0);

  // Adjust training temperature to 2.5
  nn.set_temperature(2.5);
  EXPECT_DOUBLE_EQ(nn.get_temperature(), 2.5);
}

TEST(TemperatureAndOptimiserTest, MultiOutputLayerIndependentTemperatures)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails branch_a(2, activation(activation::method::softmax, 0.0, 1.0, 0.2), ErrorCalculation::type::cross_entropy, eval_config, 0.0, OptimiserType::AdamW, 0.9);
  OutputLayerDetails branch_b(2, activation(activation::method::softmax, 0.0, 2.0, 3.0), ErrorCalculation::type::cross_entropy, eval_config, 0.0, OptimiserType::AdamW, 0.9);

  std::vector<MultiOutputLayerDetails> multi_output =
  {
    MultiOutputLayerDetails({}, branch_a),
    MultiOutputLayerDetails({}, branch_b)
  };

  auto options = NeuralNetworkOptions::create({ 3, 4, 4 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(multi_output)
    .with_learning_rate(0.05)
    .with_batch_size(1)
    .with_has_bias(true)
    .with_seed(42u)
    .build();

  NeuralNetwork nn(options);

  EXPECT_DOUBLE_EQ(nn.get_temperature(0), 1.0);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(0), 0.2);

  EXPECT_DOUBLE_EQ(nn.get_temperature(1), 2.0);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(1), 3.0);

  // Modify branch 0 inference temperature independently
  nn.set_inference_temperature(0, 0.05);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(0), 0.05);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(1), 3.0);

  // Modify branch 1 training temperature independently
  nn.set_temperature(1, 0.8);
  EXPECT_DOUBLE_EQ(nn.get_temperature(0), 1.0);
  EXPECT_DOUBLE_EQ(nn.get_temperature(1), 0.8);
}

TEST(TemperatureAndOptimiserTest, RLAdvantageTrainingWithExplorationVsExploitationTemperature)
{
  // Setup RL policy network:
  // High training temperature T_train = 2.0 for exploration during policy gradient updates
  // Low inference temperature T_infer = 0.2 for greedy exploitation at inference
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.01, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    2,
    activation(activation::method::softmax, 0.0, 2.0, 0.2),
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

  EXPECT_DOUBLE_EQ(nn.get_temperature(), 2.0);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(), 0.2);

  const std::vector<std::vector<double>> inputs = { { 0.2, -0.1, 0.4 } };
  const std::vector<std::vector<double>> action_targets = { { 1.0, 0.0 } };
  const std::vector<double> advantages = { 1.0 };

  const auto before = nn.think(inputs[0]);
  nn.train_with_advantages(inputs, action_targets, advantages);
  const auto after = nn.think(inputs[0]);

  // Selected action probability should increase
  EXPECT_GT(after[0], before[0]);
  EXPECT_LT(after[1], before[1]);

  // Temperatures must be preserved
  EXPECT_DOUBLE_EQ(nn.get_temperature(), 2.0);
  EXPECT_DOUBLE_EQ(nn.get_inference_temperature(), 0.2);
}

TEST(TemperatureAndOptimiserTest, ExtremeInferenceTemperatureStability)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::AdamW, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  };

  EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
  OutputLayerDetails output_layer_details(
    3,
    activation(activation::method::softmax, 0.0, 1.0, 1e-6),
    ErrorCalculation::type::cross_entropy,
    eval_config,
    0.0,
    OptimiserType::AdamW,
    0.9);

  auto options = NeuralNetworkOptions::create({ 3, 4, 3 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.05)
    .with_batch_size(1)
    .with_has_bias(true)
    .with_seed(42u)
    .build();

  NeuralNetwork nn(options);

  const std::vector<double> input = { 0.5, -0.2, 0.1 };

  // 1. Extreme low temperature (1e-6): acts as argmax, strictly finite
  const auto out_tiny = nn.think(input);
  for (double val : out_tiny)
  {
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_GE(val, 0.0);
    EXPECT_LE(val, 1.0);
  }
  double sum_tiny = out_tiny[0] + out_tiny[1] + out_tiny[2];
  EXPECT_NEAR(sum_tiny, 1.0, 1e-12);

  // 2. Extreme high temperature (100.0): near-uniform, strictly finite
  nn.set_inference_temperature(100.0);
  const auto out_huge = nn.think(input);
  for (double val : out_huge)
  {
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_NEAR(val, 1.0 / 3.0, 0.05);
  }
  double sum_huge = out_huge[0] + out_huge[1] + out_huge[2];
  EXPECT_NEAR(sum_huge, 1.0, 1e-12);
}

TEST(TemperatureAndOptimiserTest, JsonSerializationRoundTripAllOptimisersAndTemperatures)
{
  const std::vector<std::pair<std::string, OptimiserType>> optimisers =
  {
    { "AdamW", OptimiserType::AdamW },
    { "Adam", OptimiserType::Adam },
    { "SGD", OptimiserType::SGD },
    { "Nadam", OptimiserType::Nadam },
    { "NadamW", OptimiserType::NadamW },
    { "Lion", OptimiserType::Lion },
    { "RAdam", OptimiserType::RAdam }
  };

  for (const auto& [opt_str, opt_type] : optimisers)
  {
    const double train_temp = 1.8;
    const double infer_temp = 0.35;

    std::vector<LayerDetails> hidden_layers =
    {
      LayerDetails(Layer::Architecture::FF, 3, activation(activation::method::tanh, 0.0), 0.0, 0.01, opt_type, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
    };

    EvaluationConfig eval_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0);
    OutputLayerDetails output_layer_details(
      2,
      activation(activation::method::softmax, 0.0, train_temp, infer_temp),
      ErrorCalculation::type::cross_entropy,
      eval_config,
      0.01,
      opt_type,
      0.9);

    auto options = NeuralNetworkOptions::create({ 2, 3, 2 })
      .with_hidden_layers(hidden_layers)
      .with_output_layer_details(output_layer_details)
      .with_learning_rate(0.05)
      .with_batch_size(1)
      .with_has_bias(true)
      .with_seed(42u)
      .build();

    NeuralNetwork nn(options);

    EXPECT_DOUBLE_EQ(nn.get_temperature(), train_temp) << "Mismatch for optimiser " << opt_str;
    EXPECT_DOUBLE_EQ(nn.get_inference_temperature(), infer_temp) << "Mismatch for optimiser " << opt_str;

    // Verify inference
    const std::vector<double> input = { 0.4, -0.6 };
    const auto output_before = nn.think(input);
    EXPECT_EQ(output_before.size(), 2u);
    EXPECT_NEAR(output_before[0] + output_before[1], 1.0, 1e-12);

    // Save and reload
    const std::string test_saved_file_path = "test_temp_network_saved.json";
    NeuralNetworkSerializer::save(nn, test_saved_file_path);
    std::unique_ptr<NeuralNetwork> nn_reloaded(NeuralNetworkSerializer::load(test_saved_file_path));
    std::remove(test_saved_file_path.c_str());

    ASSERT_NE(nn_reloaded, nullptr) << "Failed to reload network with optimiser " << opt_str;

    EXPECT_DOUBLE_EQ(nn_reloaded->get_temperature(), train_temp) << "Mismatch after reload for " << opt_str;
    EXPECT_DOUBLE_EQ(nn_reloaded->get_inference_temperature(), infer_temp) << "Mismatch after reload for " << opt_str;
    EXPECT_EQ(nn_reloaded->options().output_layer_details()[0].get_optimiser_type(), opt_type) << "Mismatch for " << opt_str;
    EXPECT_EQ(nn_reloaded->options().hidden_layers()[0].get_optimiser_type(), opt_type) << "Mismatch for " << opt_str;

    const auto output_after = nn_reloaded->think(input);
    EXPECT_NEAR(output_after[0], output_before[0], 1e-12);
    EXPECT_NEAR(output_after[1], output_before[1], 1e-12);
  }
}
