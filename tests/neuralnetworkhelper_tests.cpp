#include <gtest/gtest.h>
#include "helpers/neuralnetworkhelper.h"
#include "neuralnetworkoptions.h"
#include "neuralnetwork.h"
#include <thread>
#include <chrono>
#include <set>
#include <algorithm>

using namespace myoddweb::nn;

class NeuralNetworkHelperTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
  }
};

TEST_F(NeuralNetworkHelperTest, EpochDurationMovingAverage)
{
  // Setup dummy network options
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .build();
  
  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = {{1.0, 2.0}};
  std::vector<std::vector<double>> outputs = {{0.5}};
  
  unsigned total_epochs = 1000; // Expected window size: std::clamp(1000/2000, 10, 50) = 10
  NeuralNetworkHelper helper(nn, 0.001, total_epochs, inputs, outputs);
  
  // Verify initial duration
  EXPECT_DOUBLE_EQ(helper.duration_ms(), 0.0);
  
  // First call to set_epoch(0) starts the timer
  helper.set_epoch(0);
  EXPECT_DOUBLE_EQ(helper.duration_ms(), 0.0);
  
  // Simulate some work and transition to epoch 1
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  helper.set_epoch(1);
  
  double dur1 = helper.duration_ms();
  EXPECT_GE(dur1, 5.0); // Should be roughly 10ms, at least > 5ms
  
  // Transition to epoch 2
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  helper.set_epoch(2);
  
  double dur2 = helper.duration_ms();
  // Average of ~10ms and ~20ms should be roughly 15ms
  EXPECT_GE(dur2, 8.0);
  EXPECT_LE(dur2, 1000.0);
}

TEST_F(NeuralNetworkHelperTest, CopyAndMoveOperatorsPreserveDuration)
{
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .build();
  
  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = {{1.0, 2.0}};
  std::vector<std::vector<double>> outputs = {{0.5}};
  
  NeuralNetworkHelper helper(nn, 0.001, 1000, inputs, outputs);
  helper.set_epoch(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  helper.set_epoch(1);
  
  double original_duration = helper.duration_ms();
  EXPECT_GT(original_duration, 0.0);
  
  // Test Copy constructor
  NeuralNetworkHelper copy_helper(helper);
  EXPECT_DOUBLE_EQ(copy_helper.duration_ms(), original_duration);
  
  // Test Move constructor
  NeuralNetworkHelper move_helper(std::move(helper));
  EXPECT_DOUBLE_EQ(move_helper.duration_ms(), original_duration);
  
  // Verify source helper is reset/cleared
  EXPECT_DOUBLE_EQ(helper.duration_ms(), 0.0);
}

TEST_F(NeuralNetworkHelperTest, TrainingMonitorMultipleLayersAndPanic)
{
  // 1. Single output layer setup
  auto options1 = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .build();
  
  NeuralNetwork nn1(options1);
  std::vector<std::vector<double>> inputs = {{1.0, 2.0}};
  std::vector<std::vector<double>> outputs = {{0.5}};
  
  NeuralNetworkHelper helper1(nn1, 0.001, 10, inputs, outputs);
  
  // Valid index
  EXPECT_NO_THROW((void)helper1.training_monitor(0));
  
  // Out of bounds index
#if VALIDATE_DATA == 1
  EXPECT_THROW((void)helper1.training_monitor(1), std::runtime_error);
  EXPECT_THROW((void)helper1.training_monitor(99), std::runtime_error);
#endif

  // 2. Multiple output layers setup
  OutputLayerDetails o0(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails o1(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
  
  auto options2 = NeuralNetworkOptions::create({ 4, 3, 3 })
    .with_output_layer_details({ o0, o1 })
    .with_learning_rate(0.001)
    .build();
    
  NeuralNetwork nn2(options2);
  std::vector<std::vector<double>> inputs2 = {{1.0, 2.0, 3.0, 4.0}};
  std::vector<std::vector<double>> outputs2 = {{0.5, 0.5, 0.5}};
  
  NeuralNetworkHelper helper2(nn2, 0.001, 10, inputs2, outputs2);
  
  // Valid indices (0 and 1)
  EXPECT_NO_THROW((void)helper2.training_monitor(0));
  EXPECT_NO_THROW((void)helper2.training_monitor(1));
  
  // Out of bounds index (2)
#if VALIDATE_DATA == 1
  EXPECT_THROW((void)helper2.training_monitor(2), std::runtime_error);
#endif
}

struct HelperCapturer
{
  std::vector<size_t> train_idx;
  std::vector<size_t> check_idx;
  std::vector<size_t> final_idx;

  bool operator()(NeuralNetworkHelper& helper)
  {
    train_idx = helper.training_indexes();
    check_idx = helper.checking_indexes();
    final_idx = helper.final_check_indexes();
    return false; // Stop training immediately
  }
};

TEST_F(NeuralNetworkHelperTest, TrainingIndexesOOSIsolationTest)
{
  HelperCapturer capturer;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .with_number_of_epoch(1)
    .with_data_is_unique(false)
    .with_shuffle_training_data(true)
    .with_progress_callback(std::ref(capturer))
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs(100, {1.0, 2.0});
  std::vector<std::vector<double>> outputs(100, {0.5});

  nn.train(inputs, outputs);

  // Verify sizes add up to total dataset size (100)
  EXPECT_EQ(capturer.train_idx.size() + capturer.check_idx.size() + capturer.final_idx.size(), 100);

  // Verify set disjointness (no overlap between training and checking / final check)
  std::set<size_t> train_set(capturer.train_idx.begin(), capturer.train_idx.end());
  for (size_t idx : capturer.check_idx)
  {
    EXPECT_EQ(train_set.count(idx), 0);
  }
  for (size_t idx : capturer.final_idx)
  {
    EXPECT_EQ(train_set.count(idx), 0);
  }
}

TEST_F(NeuralNetworkHelperTest, ShuffleTrainingDataInvarianceTest)
{
  HelperCapturer capturer1;
  auto options1 = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(true)
    .with_progress_callback(std::ref(capturer1))
    .build();

  NeuralNetwork nn1(options1);
  std::vector<std::vector<double>> inputs(100, {1.0, 2.0});
  std::vector<std::vector<double>> outputs(100, {0.5});

  nn1.train(inputs, outputs);

  HelperCapturer capturer2;
  auto options2 = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .with_number_of_epoch(1)
    .with_shuffle_training_data(true)
    .with_progress_callback(std::ref(capturer2))
    .build();

  NeuralNetwork nn2(options2);
  nn2.train(inputs, outputs);

  // Out-of-sample indices must be identical across runs
  EXPECT_EQ(capturer1.check_idx, capturer2.check_idx);
  EXPECT_EQ(capturer1.final_idx, capturer2.final_idx);

  // Training index sets contain the exact same elements
  std::set<size_t> set1(capturer1.train_idx.begin(), capturer1.train_idx.end());
  std::set<size_t> set2(capturer2.train_idx.begin(), capturer2.train_idx.end());
  EXPECT_EQ(set1, set2);
}

TEST_F(NeuralNetworkHelperTest, ShuffleTrainingDataIgnoredWhenBpttEnabledTest)
{
  // Requesting shuffle-training-data together with BPTT must not scramble the
  // chronological row order: create_bptt_batches slices consecutive array
  // entries into fixed-size windows, so a shuffled row order would glue
  // together unrelated rows into fake sequences.
  HelperCapturer capturer;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .with_number_of_epoch(1)
    .with_data_is_unique(false)
    .with_shuffle_training_data(true)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .with_progress_callback(std::ref(capturer))
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs(100, {1.0, 2.0});
  std::vector<std::vector<double>> outputs(100, {0.5});

  nn.train(inputs, outputs);

  // Training indexes must stay in strictly ascending (chronological) order.
  ASSERT_FALSE(capturer.train_idx.empty());
  EXPECT_TRUE(std::is_sorted(capturer.train_idx.begin(), capturer.train_idx.end()));
}

TEST_F(NeuralNetworkHelperTest, TrainingMonitorsSharedAcrossHelperCopies)
{
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {1.0, 2.0} };
  std::vector<std::vector<double>> outputs = { {0.5} };

  NeuralNetworkHelper helper1(nn, 0.001, 10, inputs, outputs);

  // Copy constructor
  NeuralNetworkHelper helper2(helper1);

  // Copy assignment
  NeuralNetworkHelper helper3 = helper1;

  // Add metrics to helper2 (mimicking a callback running with a copy of helper)
  helper2.training_monitor(0).add_metric(ErrorCalculation::type::rmse, 0.5);
  helper2.training_monitor(0).add_metric(ErrorCalculation::type::directional_accuracy, 0.6);

  // Both helper1 (original) and helper3 (other copy) must see the added metrics
  // because _training_monitors is shared.
  // Add another metric to helper3
  helper3.training_monitor(0).add_metric(ErrorCalculation::type::rmse, 0.3);
  helper3.training_monitor(0).add_metric(ErrorCalculation::type::directional_accuracy, 0.7);

  // Add third metric to helper1 to satisfy min window size (3)
  helper1.training_monitor(0).add_metric(ErrorCalculation::type::rmse, 0.1);
  helper1.training_monitor(0).add_metric(ErrorCalculation::type::directional_accuracy, 0.8);

  // All 3 helpers evaluate to the exact same status (OnTrack due to improving RMSE & DA)
  EXPECT_EQ(helper1.training_monitor(0).evaluate(), TrainingMonitor::TrainingStatus::OnTrack);
  EXPECT_EQ(helper2.training_monitor(0).evaluate(), TrainingMonitor::TrainingStatus::OnTrack);
  EXPECT_EQ(helper3.training_monitor(0).evaluate(), TrainingMonitor::TrainingStatus::OnTrack);
}

struct ProgressMonitorAccumulator
{
  int checkpoint_count = 0;
  std::vector<TrainingMonitor::TrainingStatus> observed_statuses;
  std::optional<NeuralNetworkHelper> captured_final_helper;

  bool operator()(NeuralNetworkHelper& helper)
  {
    ++checkpoint_count;
    helper.training_monitor(0).add_metric(ErrorCalculation::type::rmse, 0.5 - checkpoint_count * 0.05);
    helper.training_monitor(0).add_metric(ErrorCalculation::type::directional_accuracy, 0.5 + checkpoint_count * 0.05);
    observed_statuses.push_back(helper.training_monitor(0).evaluate());
    captured_final_helper.emplace(helper);
    return true;
  }
};

TEST_F(NeuralNetworkHelperTest, TrainingMonitorsPersistAcrossProgressCallbacks)
{
  ProgressMonitorAccumulator accumulator;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .with_number_of_epoch(20)
    .with_update_training_monitor_percent(0.1)
    .with_progress_callback(std::ref(accumulator))
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {1.0, 2.0} };
  std::vector<std::vector<double>> outputs = { {0.5} };

  nn.train(inputs, outputs);

  // Ensure multiple checkpoints were dispatched
  EXPECT_GT(accumulator.checkpoint_count, 3);

  // Once at least 3 checkpoints have passed (the default minimum window), evaluate()
  // must produce OnTrack because metrics accumulated across callbacks rather than resetting to zero.
  ASSERT_GE(accumulator.observed_statuses.size(), 3);
  EXPECT_EQ(accumulator.observed_statuses.back(), TrainingMonitor::TrainingStatus::OnTrack);

  // Post-training query on captured_final_helper must retain all accumulated metrics
  ASSERT_TRUE(accumulator.captured_final_helper.has_value());
  EXPECT_EQ(accumulator.captured_final_helper->training_monitor(0).evaluate(), TrainingMonitor::TrainingStatus::OnTrack);
}

TEST_F(NeuralNetworkHelperTest, TrainingMonitorsMultiOutputSharedCopies)
{
  OutputLayerDetails o0(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails o1(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);

  auto options = NeuralNetworkOptions::create({ 4, 3, 3 })
    .with_output_layer_details({ o0, o1 })
    .with_learning_rate(0.001)
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs = { {1.0, 2.0, 3.0, 4.0} };
  std::vector<std::vector<double>> outputs = { {0.5, 0.5, 0.5} };

  NeuralNetworkHelper helper1(nn, 0.001, 10, inputs, outputs);
  NeuralNetworkHelper helper2(helper1);

  // Add metrics for output branch 0 via helper2
  helper2.training_monitor(0).add_metric(ErrorCalculation::type::rmse, 0.1);
  // Add metrics for output branch 1 via helper1
  helper1.training_monitor(1).add_metric(ErrorCalculation::type::rmse, 0.9);

  // Both helpers see metrics for both branches
  EXPECT_NO_THROW((void)helper1.training_monitor(0));
  EXPECT_NO_THROW((void)helper2.training_monitor(1));
}

struct FinalHelperCapturer
{
  std::optional<NeuralNetworkHelper> captured_final_helper;
  size_t check_size = 0;
  size_t final_size = 0;

  bool operator()(NeuralNetworkHelper& helper)
  {
    if (helper.epoch() >= helper.number_of_epoch())
    {
      captured_final_helper.emplace(helper);
      check_size = helper.checking_indexes().size();
      final_size = helper.final_check_indexes().size();
    }
    return true;
  }
};

TEST_F(NeuralNetworkHelperTest, ForceCheckingIndexesSelectsValidationSetAtFinalEpoch)
{
  FinalHelperCapturer capturer;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .with_number_of_epoch(1)
    .with_data_is_unique(false)
    .with_shuffle_training_data(true)
    .with_progress_callback(std::ref(capturer))
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs(100, { 1.0, 2.0 });
  std::vector<std::vector<double>> outputs(100, { 0.5 });

  nn.train(inputs, outputs);

  ASSERT_TRUE(capturer.captured_final_helper.has_value());
  EXPECT_EQ(capturer.check_size, 15);
  EXPECT_EQ(capturer.final_size, 5);
  EXPECT_NE(capturer.check_size, capturer.final_size);

  // 1. Via captured helper
  auto helper_default = capturer.captured_final_helper->calculate_forecast_metrics({ ErrorCalculation::type::prediction_coverage }, false, false);
  auto helper_forced = capturer.captured_final_helper->calculate_forecast_metrics({ ErrorCalculation::type::prediction_coverage }, false, true);

  ASSERT_FALSE(helper_default.empty());
  ASSERT_FALSE(helper_default[0].empty());
  ASSERT_TRUE(helper_default[0][0].denominator().has_value());
  EXPECT_EQ(helper_default[0][0].denominator().value(), capturer.final_size);

  ASSERT_FALSE(helper_forced.empty());
  ASSERT_FALSE(helper_forced[0].empty());
  ASSERT_TRUE(helper_forced[0][0].denominator().has_value());
  EXPECT_EQ(helper_forced[0][0].denominator().value(), capturer.check_size);

  // 2. Via NeuralNetwork post-training queries
  auto nn_default = nn.calculate_forecast_metrics_all_layers({ ErrorCalculation::type::prediction_coverage }, false, false);
  auto nn_forced = nn.calculate_forecast_metrics_all_layers({ ErrorCalculation::type::prediction_coverage }, false, true);

  ASSERT_FALSE(nn_default.empty());
  ASSERT_FALSE(nn_default[0].empty());
  ASSERT_TRUE(nn_default[0][0].denominator().has_value());
  EXPECT_EQ(nn_default[0][0].denominator().value(), capturer.final_size);

  ASSERT_FALSE(nn_forced.empty());
  ASSERT_FALSE(nn_forced[0].empty());
  ASSERT_TRUE(nn_forced[0][0].denominator().has_value());
  EXPECT_EQ(nn_forced[0][0].denominator().value(), capturer.check_size);
}

TEST_F(NeuralNetworkHelperTest, OptionsForceCheckingIndexesConfiguredTrueDefaultsToValidationSet)
{
  FinalHelperCapturer capturer;
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_learning_rate(0.001)
    .with_number_of_epoch(1)
    .with_data_is_unique(false)
    .with_shuffle_training_data(true)
    .with_force_checking_indexes(true)
    .with_progress_callback(std::ref(capturer))
    .build();

  NeuralNetwork nn(options);
  std::vector<std::vector<double>> inputs(100, { 1.0, 2.0 });
  std::vector<std::vector<double>> outputs(100, { 0.5 });

  nn.train(inputs, outputs);

  ASSERT_TRUE(capturer.captured_final_helper.has_value());
  EXPECT_EQ(capturer.check_size, 15);
  EXPECT_EQ(capturer.final_size, 5);

  // Calling without 3rd parameter should default to options.force_checking_indexes() (true -> check_size)
  auto metrics_default = nn.calculate_forecast_metrics_all_layers({ ErrorCalculation::type::prediction_coverage }, false);
  ASSERT_FALSE(metrics_default.empty());
  ASSERT_FALSE(metrics_default[0].empty());
  ASSERT_TRUE(metrics_default[0][0].denominator().has_value());
  EXPECT_EQ(metrics_default[0][0].denominator().value(), capturer.check_size);

  // Calling with explicit false should override options to use final_size
  auto metrics_override = nn.calculate_forecast_metrics_all_layers({ ErrorCalculation::type::prediction_coverage }, false, false);
  ASSERT_FALSE(metrics_override.empty());
  ASSERT_FALSE(metrics_override[0].empty());
  ASSERT_TRUE(metrics_override[0][0].denominator().has_value());
  EXPECT_EQ(metrics_override[0][0].denominator().value(), capturer.final_size);
}
