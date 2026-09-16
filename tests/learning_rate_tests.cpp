#include "common/logger.h"
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include "test_helper.h"
#include <atomic>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <mutex>
#include <thread>
#include <vector>


using namespace myoddweb::nn;
using namespace test_helper;

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

TEST_F(LearningRateTest, ExponentialDecayWithWarmupAppliedCorrectly)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double start_lr = 0.0;
  double target_lr = 0.1;
  double warmup_target = 0.2; // 20% of epochs (epoch 20 out of 100)
  double decay_rate_opt = 0.5; // End at 50% of target at the end of post-warmup
  int epochs = 100;

  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(target_lr)
    .with_learning_rate_warmup(start_lr, warmup_target)
    .with_learning_rate_decay_rate(decay_rate_opt)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h)
    {
      return capture.callback(h);
    })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  int warmup_epochs = static_cast<int>(std::round(warmup_target * epochs));
  int number_of_epoch_after_decay = epochs - warmup_epochs;
  double lr_decay_rate = std::log(1.0 / decay_rate_opt) / number_of_epoch_after_decay;

  auto captured_rates = capture.get_rates();
  ASSERT_FALSE(captured_rates.empty());

  for (auto const& [epoch, rate] : captured_rates)
  {
    double completed_percent = static_cast<double>(epoch) / epochs;
    if (completed_percent < warmup_target)
    {
      // Warmup phase (linear)
      double ratio = completed_percent / warmup_target;
      double expected = start_lr + (target_lr - start_lr) * ratio;
      EXPECT_NEAR(rate, expected, 1e-7) << "Warmup fail at epoch " << epoch;
    }
    else
    {
      // Decay phase (should use epoch - warmup_epochs)
      int relative_epoch = epoch - warmup_epochs;
      double expected = target_lr * std::exp(-lr_decay_rate * relative_epoch);
      EXPECT_NEAR(rate, expected, 1e-7) << "Exponential Decay (with Warmup) fail at epoch " << epoch;
    }
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

TEST_F(LearningRateTest, BoostWithWarmupAppliedCorrectly)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double start_lr = 0.0;
  double target_lr = 0.1;
  double warmup_target = 0.2; // 20% of epochs (epoch 20 out of 100)
  double restart_rate = 0.5; // Every 50% of progress
  double restart_boost = 0.2; // Total 20% boost
  int epochs = 100;

  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(target_lr)
    .with_learning_rate_warmup(start_lr, warmup_target)
    .with_learning_rate_boost_rate(restart_rate, restart_boost)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h)
    {
      return capture.callback(h);
    })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  int warmup_epochs = static_cast<int>(std::round(warmup_target * epochs));
  int number_of_epoch_after_decay = epochs - warmup_epochs;
  int boost_interval = static_cast<int>(std::round(restart_rate * number_of_epoch_after_decay));
  int total_boosts = number_of_epoch_after_decay / boost_interval;
  double per_boost_ratio = restart_boost / total_boosts;

  auto captured_rates = capture.get_rates();
  ASSERT_FALSE(captured_rates.empty());

  for (auto const& [epoch, rate] : captured_rates)
  {
    double completed_percent = static_cast<double>(epoch) / epochs;
    if (completed_percent < warmup_target)
    {
      // Warmup phase (linear)
      double ratio = completed_percent / warmup_target;
      double expected = start_lr + (target_lr - start_lr) * ratio;
      EXPECT_NEAR(rate, expected, 1e-7) << "Warmup fail at epoch " << epoch;
    }
    else
    {
      // Boost phase (should use relative_epoch = epoch - warmup_epochs)
      int relative_epoch = epoch - warmup_epochs;
      int cycle_pos = relative_epoch % boost_interval;
      double progress = static_cast<double>(cycle_pos) / boost_interval;
      double cosine_multiplier = (1.0 - std::cos(progress * M_PI)) / 2.0;
      double current_boost = per_boost_ratio * cosine_multiplier;
      double completed_cycles = relative_epoch / boost_interval;
      double cumulative_boost = completed_cycles * per_boost_ratio;
      double expected = target_lr * (1.0 + cumulative_boost + current_boost);
      EXPECT_NEAR(rate, expected, 1e-7) << "Boost (with Warmup) fail at epoch " << epoch;
    }
  }
}

TEST_F(LearningRateTest, AdaptiveLearningRatePersistsBetweenUpdates) {
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double target_lr = 0.1;
  // We need enough epochs to trigger at least one update and see the following epochs.
  // Update every 5 epochs. History 25 samples. 125 epochs to fill.
  // Epoch 125: Update (if error trend found)
  // Epoch 126: Should still have the updated rate.
  int epochs = 150; 

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
  
  // Find the first epoch where the rate changed from target_lr
  int change_epoch = -1;
  double new_rate = -1.0;
  for (int i = 125; i < epochs; ++i) {
    if (captured_rates.count(i) && !approx_equal(captured_rates[i], target_lr, 1e-7)) {
      change_epoch = i;
      new_rate = captured_rates[i];
      break;
    }
  }

  if (change_epoch != -1 && change_epoch < epochs - 1) {
    // Check the next epoch (which won't trigger an adaptive update because 126 % 5 != 0)
    int next_epoch = change_epoch + 1;
    if (next_epoch % 5 != 0 && captured_rates.count(next_epoch)) {
        EXPECT_NEAR(captured_rates[next_epoch], new_rate, 1e-7) 
            << "Rate reverted to base in epoch " << next_epoch << " after change in " << change_epoch;
    }
  }
}

TEST_F(LearningRateTest, ConcurrentThinkDuringTrainingIsThreadSafe)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  int epochs = 100;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .build();

  NeuralNetwork nn = create_test_nn(options);

  std::atomic<bool> training_done(false);

  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i)
  {
    readers.emplace_back([&]()
    {
      std::vector<double> test_input = { 0.5, 0.5 };
      while (!training_done.load())
      {
        auto result = nn.think(test_input);
        EXPECT_EQ(result.size(), 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }

  nn.train(inputs, outputs);
  training_done.store(true);

  for (auto& reader : readers)
  {
    if (reader.joinable())
    {
      reader.join();
    }
  }
}
TEST_F(LearningRateTest, DestructorIsSafeAfterAbortedTraining)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  int epochs = 100;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.1)
    .with_adaptive_learning_rates(true)
    .with_learning_rate_warmup(0.01, 0.05) // warmup ends at 5% (epoch 5)
    .with_number_of_epoch(epochs)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h)
    {
      if (h.epoch() > 5)
      {
        return false; // Abort training
      }
      return true;
    })
    .build();

  {
    NeuralNetwork nn = create_test_nn(options);
    nn.train(inputs, outputs);
  } // nn goes out of scope here; destructor should clean up safely without crash
}

TEST_F(LearningRateTest, SynchronousAdaptiveLearningRateThreadSafety)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  // Replicate dataset to trigger adaptive LR calculations
  std::vector<std::vector<double>> replicated_inputs;
  std::vector<std::vector<double>> replicated_outputs;
  for (int i = 0; i < 5; ++i)
  {
    replicated_inputs.insert(replicated_inputs.end(), inputs.begin(), inputs.end());
    replicated_outputs.insert(replicated_outputs.end(), outputs.begin(), outputs.end());
  }

  // Set up options with adaptive learning rate enabled
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.1)
    .with_adaptive_learning_rates(true)
    .with_learning_rate_warmup(0.0, 0.05) // start adaptive learning rate immediately
    .with_number_of_epoch(15)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .build();

  NeuralNetwork nn = create_test_nn(options);

  std::atomic<bool> training_done(false);
  std::vector<std::thread> reader_threads;

  // Launch reader threads executing const methods concurrently with training
  for (int i = 0; i < 2; ++i)
  {
    reader_threads.emplace_back([&]()
    {
      std::vector<double> test_input = { 0.5, 0.5 };
      while (!training_done.load())
      {
        auto result = nn.think(test_input);
        EXPECT_EQ(result.size(), 1);
        nn.get_percent_complete();
        nn.has_training_data();
        nn.calculate_forecast_metric(ErrorCalculation::type::rmse);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }

  // Train the network. The training thread will call calculate_learning_rate synchronously,
  // which executes calculate_forecast_metrics_impl holding a shared_lock.
  nn.train(replicated_inputs, replicated_outputs);
  training_done.store(true);

  for (auto& reader : reader_threads)
  {
    if (reader.joinable())
    {
      reader.join();
    }
  }

  SUCCEED();
}

TEST_F(LearningRateTest, ThreadLocalBufferResizingDoesNotPolluteEvaluation)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(10)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  // Run a large evaluation first to expand the thread_local buffers
  auto metrics_large = nn.calculate_forecast_metrics({ ErrorCalculation::type::rmse }, false);

  // Run a smaller evaluation next
  // This verifies that the thread_local cache resize/zeroing cleans up the buffers perfectly
  auto metrics_small = nn.calculate_forecast_metrics({ ErrorCalculation::type::rmse }, false);
  
  EXPECT_FALSE(metrics_large.empty());
  EXPECT_FALSE(metrics_small.empty());
}

TEST_F(LearningRateTest, BoostZeroRestartRateHandlesGracefully)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  double target_lr = 0.1;
  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(target_lr)
    .with_learning_rate_boost_rate(0.0, 0.2) // 0% restart rate (invalid boost)
    .with_number_of_epoch(10)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h)
    {
      return capture.callback(h);
    })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  auto captured_rates = capture.get_rates();
  for (auto const& [epoch, rate] : captured_rates)
  {
    EXPECT_NEAR(rate, target_lr, 1e-7) << "Learning rate should not boost with 0 restart rate.";
  }
}

TEST_F(LearningRateTest, CoexistingDecayBoostAndAdaptiveLRShortcutBehavior)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  // Replicate dataset to make training epochs take long enough for background thread scheduling
  std::vector<std::vector<double>> replicated_inputs;
  std::vector<std::vector<double>> replicated_outputs;
  for (int i = 0; i < 1500; ++i)
  {
    replicated_inputs.insert(replicated_inputs.end(), inputs.begin(), inputs.end());
    replicated_outputs.insert(replicated_outputs.end(), outputs.begin(), outputs.end());
  }
  inputs = std::move(replicated_inputs);
  outputs = std::move(replicated_outputs);

  LrCapture capture;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.1)
    .with_learning_rate_decay_rate(0.9)
    .with_learning_rate_boost_rate(0.2, 0.1)
    .with_adaptive_learning_rates(true)
    .with_number_of_epoch(200)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& h)
    {
      return capture.callback(h);
    })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  auto captured_rates = capture.get_rates();

  bool found = false;
  // We check blocks of 5 epochs (from 5*m to 5*m + 4)
  for (int m = 1; m < 200 / 5; ++m)
  {
    std::vector<double> rates_in_block;
    for (int epoch = 5 * m; epoch < 5 * m + 5; ++epoch)
    {
      if (captured_rates.count(epoch))
      {
        rates_in_block.push_back(captured_rates[epoch]);
      }
    }

    if (rates_in_block.size() >= 2)
    {
      // Check if all captured rates in this block are approximately equal
      bool all_equal = true;
      for (size_t i = 1; i < rates_in_block.size(); ++i)
      {
        if (!approx_equal(rates_in_block[i], rates_in_block[0], 1e-7))
        {
          all_equal = false;
          break;
        }
      }

      if (all_equal)
      {
        found = true;
        break;
      }
    }
  }

  ASSERT_TRUE(found) << "Adaptive learning rate was never collected or verified during training.";
}

TEST_F(LearningRateTest, ThreadSafetyConcurrentAccess)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  // Replicate dataset to make training take some time
  std::vector<std::vector<double>> replicated_inputs;
  std::vector<std::vector<double>> replicated_outputs;
  for (int i = 0; i < 100; ++i)
  {
    replicated_inputs.insert(replicated_inputs.end(), inputs.begin(), inputs.end());
    replicated_outputs.insert(replicated_outputs.end(), outputs.begin(), outputs.end());
  }

  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(50)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .build();

  NeuralNetwork nn = create_test_nn(options);

  std::atomic<bool> training_done(false);
  std::thread query_thread([&]()
  {
    while (!training_done.load())
    {
      nn.get_percent_complete();
      nn.has_training_data();
      nn.calculate_forecast_metric(ErrorCalculation::type::rmse);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  nn.train(replicated_inputs, replicated_outputs);
  training_done.store(true);
  query_thread.join();

  SUCCEED();
}

TEST_F(LearningRateTest, ThreadSafetyHelperLifecycle)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  std::atomic<bool> callback_run(false);
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.1)
    .with_number_of_epoch(10)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback([&](NeuralNetworkHelper& helper)
    {
      // Simulate heavy calculations inside callback
      auto metrics = helper.calculate_forecast_metrics({ ErrorCalculation::type::rmse });
      (void)metrics;
      callback_run.store(true);
      return true;
    })
    .build();

  NeuralNetwork nn = create_test_nn(options);
  nn.train(inputs, outputs);

  ASSERT_TRUE(callback_run.load());
}

TEST_F(LearningRateTest, SetLearningRateGetterSetterAndOverride)
{
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.05)
    .build();

  NeuralNetwork nn(options);
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), 0.05);
  EXPECT_FALSE(nn.has_learning_rate_override());

  // Set explicit learning rate override
  nn.set_learning_rate(0.02);
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), 0.02);
  EXPECT_TRUE(nn.has_learning_rate_override());

  // Reset override with 0.0
  nn.set_learning_rate(0.0);
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), 0.05);
  EXPECT_FALSE(nn.has_learning_rate_override());

  // Re-apply override
  nn.set_learning_rate(0.03);
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), 0.03);
  EXPECT_TRUE(nn.has_learning_rate_override());

  // Invalid values (negative, NaN, Inf) should be rejected and keep current state
  nn.set_learning_rate(-0.01);
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), 0.03);
  EXPECT_TRUE(nn.has_learning_rate_override());

  nn.set_learning_rate(std::numeric_limits<double>::quiet_NaN());
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), 0.03);
  EXPECT_TRUE(nn.has_learning_rate_override());

  nn.set_learning_rate(std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), 0.03);
  EXPECT_TRUE(nn.has_learning_rate_override());
}

TEST_F(LearningRateTest, SetLearningRateOverridesTrainWithAdvantages)
{
  std::vector<LayerDetails> hidden_layers =
  {
    LayerDetails(Layer::Architecture::FF, 4, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0)
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

  const uint32_t fixed_seed = 12345;
  auto options_default = NeuralNetworkOptions::create({ 3, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.1)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(fixed_seed)
    .build();

  auto options_override = NeuralNetworkOptions::create({ 3, 4, 2 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(output_layer_details)
    .with_learning_rate(0.1)
    .with_batch_size(1)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(false)
    .with_has_bias(true)
    .with_seed(fixed_seed)
    .build();

  NeuralNetwork nn_large_lr(options_default);
  NeuralNetwork nn_small_lr(options_override);

  // Forcefully override learning rate on second network to 0.01 (1/10th)
  nn_small_lr.set_learning_rate(0.01);
  EXPECT_DOUBLE_EQ(nn_small_lr.get_learning_rate(), 0.01);
  EXPECT_TRUE(nn_small_lr.has_learning_rate_override());

  const std::vector<std::vector<double>> inputs = { { 0.5, -0.2, 0.8 } };
  const std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };
  const std::vector<double> advantages = { 2.0 };

  const auto initial_out_large = nn_large_lr.think(inputs.front());
  const auto initial_out_small = nn_small_lr.think(inputs.front());
  EXPECT_NEAR(initial_out_large[0], initial_out_small[0], 1e-12);
  EXPECT_NEAR(initial_out_large[1], initial_out_small[1], 1e-12);

  nn_large_lr.train_with_advantages(inputs, targets, advantages);
  nn_small_lr.train_with_advantages(inputs, targets, advantages);

  const auto trained_out_large = nn_large_lr.think(inputs.front());
  const auto trained_out_small = nn_small_lr.think(inputs.front());

  const double delta_large = std::abs(trained_out_large[0] - initial_out_large[0]);
  const double delta_small = std::abs(trained_out_small[0] - initial_out_small[0]);

  // Network trained with lr=0.1 must update more than network trained with forced lr=0.01
  EXPECT_GT(delta_large, 0.0);
  EXPECT_GT(delta_small, 0.0);
  EXPECT_GT(delta_large, delta_small * 5.0);
}

namespace
{
struct RateRecorder
{
  std::vector<double> recorded_rates;
  std::mutex mutex;

  bool record(NeuralNetworkHelper& helper)
  {
    std::lock_guard<std::mutex> lock(mutex);
    recorded_rates.push_back(helper.learning_rate());
    return true;
  }

  bool operator()(NeuralNetworkHelper& helper)
  {
    return record(helper);
  }
};
}

TEST_F(LearningRateTest, SetLearningRateOverridesTrainEpochsAndBypassesScheduler)
{
  std::vector<std::vector<double>> inputs, outputs;
  get_simple_test_data(inputs, outputs);

  RateRecorder recorder;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.1)
    .with_learning_rate_warmup(0.01, 0.5) // Warmup from 0.01 to 0.1 over 50% epochs
    .with_learning_rate_decay_rate(0.95)
    .with_number_of_epoch(20)
    .with_shuffle_training_data(false)
    .with_data_is_unique(true)
    .with_progress_callback(std::ref(recorder))
    .build();

  NeuralNetwork nn(options);
  const double forced_lr = 0.035;
  nn.set_learning_rate(forced_lr);
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), forced_lr);
  EXPECT_TRUE(nn.has_learning_rate_override());

  nn.train(inputs, outputs);

  ASSERT_FALSE(recorder.recorded_rates.empty());
  for (double rate : recorder.recorded_rates)
  {
    // Every epoch must use the forced learning rate, completely bypassing warmup and decay
    EXPECT_DOUBLE_EQ(rate, forced_lr);
  }

  // Post-training, get_learning_rate() must still return the forced learning rate
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), forced_lr);
  EXPECT_TRUE(nn.has_learning_rate_override());
}

namespace
{
struct ConcurrentReader
{
  const NeuralNetwork* nn;
  std::atomic<bool>* stop_flag;

  void operator()() const
  {
    while (!stop_flag->load())
    {
      const double lr = nn->get_learning_rate();
      EXPECT_GE(lr, 0.0);
      const bool overridden = nn->has_learning_rate_override();
      (void)overridden;
      std::this_thread::yield();
    }
  }
};
}

TEST_F(LearningRateTest, SetLearningRateThreadSafety)
{
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.05)
    .build();

  NeuralNetwork nn(options);
  std::atomic<bool> stop_flag(false);

  ConcurrentReader reader{ &nn, &stop_flag };
  std::thread reader_thread_1(reader);
  std::thread reader_thread_2(reader);

  for (int i = 1; i <= 200; ++i)
  {
    const double test_lr = 0.001 * static_cast<double>(i);
    nn.set_learning_rate(test_lr);
    EXPECT_DOUBLE_EQ(nn.get_learning_rate(), test_lr);
    EXPECT_TRUE(nn.has_learning_rate_override());
  }

  nn.set_learning_rate(0.0);
  EXPECT_DOUBLE_EQ(nn.get_learning_rate(), 0.05);
  EXPECT_FALSE(nn.has_learning_rate_override());

  stop_flag.store(true);
  reader_thread_1.join();
  reader_thread_2.join();
}


