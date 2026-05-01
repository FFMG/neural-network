#include <gtest/gtest.h>
#include "../src/neuralnetwork/neuralnetwork.h"
#include "../src/neuralnetwork/neuralnetworkoptions.h"
#include "../src/neuralnetwork/logger.h"
#include <vector>
#include <map>
#include <cmath>
#include <mutex>

#ifndef M_PI
# define M_PI   3.141592653589793238462643383279502884
#endif

namespace {
  // Simple training data for testing
  void get_simple_test_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs) {
    inputs = { {0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8} };
    outputs = { {0.3}, {0.7}, {1.1}, {1.5} };
  }

  // Helper to capture learning rates during training
  struct LrCapture {
    std::map<int, double> rates;
    std::mutex mutex;

    bool callback(NeuralNetworkHelper& helper) {
      std::lock_guard<std::mutex> lock(mutex);
      rates[static_cast<int>(helper.epoch())] = helper.learning_rate();
      return true;
    }

    std::map<int, double> get_rates() {
      std::lock_guard<std::mutex> lock(mutex);
      return rates;
    }
  };

  // Helper method to create a standard neural network for testing
  NeuralNetwork create_test_nn(NeuralNetworkOptions& options) {
    return NeuralNetwork(options);
  }
}

class LearningRateTest : public ::testing::Test {
protected:
  void SetUp() override {
    Logger::set_level(Logger::LogLevel::None);
  }
};

TEST_F(LearningRateTest, WarmupLinearAppliedCorrectly) {
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double start_lr = 0.0;
  double target_lr = 0.1;
  double warmup_target = 0.5; // 50% of epochs
  int epochs = 100; // Increased epochs to give more breathing room for async callback

  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(target_lr)
    .with_learning_rate_warmup(start_lr, warmup_target)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h) { return capture.callback(h); })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  auto captured_rates = capture.get_rates();
  ASSERT_FALSE(captured_rates.empty());

  for (auto const& [epoch, rate] : captured_rates) {
    double completed_percent = static_cast<double>(epoch) / epochs;
    if (completed_percent < warmup_target) {
      double ratio = completed_percent / warmup_target;
      double expected = start_lr + (target_lr - start_lr) * ratio;
      EXPECT_NEAR(rate, expected, 1e-7) << "Linear Warmup fail at epoch " << epoch;
    } else {
      EXPECT_NEAR(rate, target_lr, 1e-7) << "Post-warmup fail at epoch " << epoch;
    }
  }
}

TEST_F(LearningRateTest, WarmupGeometricAppliedCorrectly) {
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double start_lr = 0.01;
  double target_lr = 0.1;
  double warmup_target = 0.5;
  int epochs = 100;

  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(target_lr)
    .with_learning_rate_warmup(start_lr, warmup_target)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h) { return capture.callback(h); })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  auto captured_rates = capture.get_rates();
  ASSERT_FALSE(captured_rates.empty());

  for (auto const& [epoch, rate] : captured_rates) {
    double completed_percent = static_cast<double>(epoch) / epochs;
    if (completed_percent < warmup_target) {
      double ratio = completed_percent / warmup_target;
      double expected = start_lr * std::pow(target_lr / start_lr, ratio);
      EXPECT_NEAR(rate, expected, 1e-7) << "Geometric Warmup fail at epoch " << epoch;
    } else {
      EXPECT_NEAR(rate, target_lr, 1e-7) << "Post-warmup fail at epoch " << epoch;
    }
  }
}

TEST_F(LearningRateTest, ExponentialDecayAppliedCorrectly) {
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double target_lr = 0.1;
  double decay_rate_opt = 0.5; // End at 50% of target
  int epochs = 100;

  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(target_lr)
    .with_learning_rate_decay_rate(decay_rate_opt)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h) { return capture.callback(h); })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  double lr_decay_rate = std::log(1.0 / decay_rate_opt) / epochs;
  auto captured_rates = capture.get_rates();
  ASSERT_FALSE(captured_rates.empty());

  for (auto const& [epoch, rate] : captured_rates) {
    double expected = target_lr * std::exp(-lr_decay_rate * epoch);
    EXPECT_NEAR(rate, expected, 1e-7) << "Exponential Decay fail at epoch " << epoch;
  }
}

TEST_F(LearningRateTest, BoostAppliedCorrectly) {
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double target_lr = 0.1;
  double restart_rate = 0.5; // Every 50% of progress
  double restart_boost = 0.2; // Total 20% boost
  int epochs = 100;

  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(target_lr)
    .with_learning_rate_boost_rate(restart_rate, restart_boost)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h) { return capture.callback(h); })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  int boost_interval = static_cast<int>(std::round(restart_rate * epochs));
  int total_boosts = epochs / boost_interval;
  double per_boost_ratio = restart_boost / total_boosts;

  auto captured_rates = capture.get_rates();
  ASSERT_FALSE(captured_rates.empty());

  for (auto const& [epoch, rate] : captured_rates) {
    int cycle_pos = epoch % boost_interval;
    double progress = static_cast<double>(cycle_pos) / boost_interval;
    double cosine_multiplier = (1.0 - std::cos(progress * M_PI)) / 2.0;
    double current_boost = per_boost_ratio * cosine_multiplier;
    double completed_cycles = epoch / boost_interval;
    double cumulative_boost = completed_cycles * per_boost_ratio;
    double expected = target_lr * (1.0 + cumulative_boost + current_boost);
    EXPECT_NEAR(rate, expected, 1e-7) << "Boost fail at epoch " << epoch;
  }
}

TEST_F(LearningRateTest, AdaptiveLearningRateDoesNotChangeBeforeHistoryFull) {
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double target_lr = 0.1;
  int epochs = 50; // Increased slightly

  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(target_lr)
    .with_adaptive_learning_rates(true)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h) { return capture.callback(h); })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  auto captured_rates = capture.get_rates();
  ASSERT_FALSE(captured_rates.empty());

  for (auto const& [epoch, rate] : captured_rates) {
    EXPECT_NEAR(rate, target_lr, 1e-7) << "Adaptive LR changed unexpectedly at epoch " << epoch;
  }
}

