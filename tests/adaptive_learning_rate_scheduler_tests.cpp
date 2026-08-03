#include <gtest/gtest.h>
#include "helpers/adaptivelearningratescheduler.h"
#include "common/logger.h"
#include <vector>


using namespace myoddweb::nn;
class AdaptiveLearningRateSchedulerTest : public ::testing::Test {
protected:
  void SetUp() override {
    Logger::set_level(Logger::LogLevel::None);
  }
};

TEST_F(AdaptiveLearningRateSchedulerTest, NoChangeBeforeHistoryFull) {
  size_t history_size = 10;
  AdaptiveLearningRateScheduler scheduler(history_size);
  double initial_lr = 0.1;
  
  for (size_t i = 0; i < history_size - 1; ++i) {
    double lr = scheduler.update(1.0, initial_lr, 0, 100);
    EXPECT_DOUBLE_EQ(lr, initial_lr) << "LR changed before history was full at iteration " << i;
  }
}

TEST_F(AdaptiveLearningRateSchedulerTest, DecreasingErrorIncreasesLearningRate) {
  size_t history_size = 10;
  double min_percent_change = 0.01; // 1%
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, 0.0005, min_percent_change, adjustment_rate);
  double initial_lr = 0.1;
  
  // Fill history with decreasing error sequence
  double err = 1.0;
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(err, initial_lr, 0, 100);
    err *= 0.95; // 5% decrease
  }
  scheduler.reset_cool_down();

  double lr = scheduler.update(err, initial_lr, 0, 100);

  EXPECT_GT(lr, initial_lr);
  EXPECT_NEAR(lr, initial_lr * (1.0 + adjustment_rate / 2.0), 1e-7);
}

TEST_F(AdaptiveLearningRateSchedulerTest, IncreasingErrorDecreasesLearningRate) {
  size_t history_size = 10;
  double min_percent_change = 0.01;
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, 0.0005, min_percent_change, adjustment_rate);
  double initial_lr = 0.1;
  
  // Fill history with increasing error sequence
  double err = 1.0;
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(err, initial_lr, 0, 100);
    err *= 1.012; // 1.2% increase
  }
  scheduler.reset_cool_down();

  double lr = scheduler.update(err, initial_lr, 0, 100);

  EXPECT_LT(lr, initial_lr);
  EXPECT_NEAR(lr, initial_lr * (1.0 - adjustment_rate * 1.5), 1e-7);
}

TEST_F(AdaptiveLearningRateSchedulerTest, ExplodingErrorDecreasesLearningRateFast) {
  size_t history_size = 10;
  double min_percent_change = 0.01;
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, 0.0005, min_percent_change, adjustment_rate);
  double initial_lr = 0.1;
  
  // Fill history with exploding error sequence
  double err = 1.0;
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(err, initial_lr, 0, 100);
    err *= 1.10; // 10% increase
  }
  scheduler.reset_cool_down();

  double lr = scheduler.update(err, initial_lr, 0, 100);

  EXPECT_LT(lr, initial_lr);
  EXPECT_NEAR(lr, initial_lr * (1.0 - adjustment_rate * 2.0), 1e-7);
}

TEST_F(AdaptiveLearningRateSchedulerTest, PlateauingErrorDecreasesLearningRateMildly) {
  size_t history_size = 10;
  double min_plateau_percent_change = 0.0005;
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, min_plateau_percent_change, 0.005, adjustment_rate);
  double initial_lr = 0.1;
  
  // Fill history with constant error
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(1.0, initial_lr, 0, 100);
  }
  scheduler.reset_cool_down();

  double lr = scheduler.update(1.0, initial_lr, 10, 100);

  EXPECT_LT(lr, initial_lr);
  EXPECT_NEAR(lr, initial_lr * (1.0 - adjustment_rate / 2.0), 1e-7);
}

TEST_F(AdaptiveLearningRateSchedulerTest, CooldownPreventsImmediateFurtherChanges) {
  size_t history_size = 10;
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, 0.0005, 0.01, adjustment_rate);
  double initial_lr = 0.1;
  
  // Fill history with decreasing error sequence
  double err = 1.0;
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(err, initial_lr, 0, 100);
    err *= 0.90;
  }
  scheduler.reset_cool_down();

  // Trigger a change (Decreasing)
  double lr = scheduler.update(err, initial_lr, 0, 100);
  EXPECT_NE(lr, initial_lr);
  double lr_after_first_change = lr;

  // Subsequent updates should be in cooldown.
  for (int i = 0; i < 50; ++i) {
    err *= 0.90;
    lr = scheduler.update(err, lr, 0, 100);
    EXPECT_DOUBLE_EQ(lr, lr_after_first_change) << "LR changed during cooldown at iteration " << i;
  }
}

TEST_F(AdaptiveLearningRateSchedulerTest, RateIsClamped) {
  size_t history_size = 10;
  AdaptiveLearningRateScheduler scheduler(history_size);
  double initial_lr = 0.1;
  
  // Fill history with decreasing error sequence
  double err = 1.0;
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(err, initial_lr, 0, 100);
    err *= 0.90;
  }
  scheduler.reset_cool_down();

  double lr = initial_lr;
  for (int i = 0; i < 1000; ++i) {
    err *= 0.90;
    lr = scheduler.update(err, lr, 0, 100);
  }

  EXPECT_LE(lr, 0.2000000000001);
  EXPECT_GE(lr, 1e-6);
}

TEST_F(AdaptiveLearningRateSchedulerTest, ExtendedEpochTrainingDoesNotOverflow) {
  size_t history_size = 10;
  AdaptiveLearningRateScheduler scheduler(history_size);
  double initial_lr = 0.1;

  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(1.0, initial_lr, 0, 100);
  }
  scheduler.reset_cool_down();

  // Epoch exceeding total_epochs (e.g. epoch 150 of 100)
  double lr = scheduler.update(1.0, initial_lr, 150, 100);
  EXPECT_GE(lr, 1e-6);
  EXPECT_LE(lr, 0.2);
}

TEST_F(AdaptiveLearningRateSchedulerTest, NoisyErrorHistoryDecreasingState) {
  size_t history_size = 12;
  double min_percent_change = 0.005; // 0.5%
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, 0.0005, min_percent_change, adjustment_rate);
  double initial_lr = 0.1;

  std::vector<double> noisy_errors = {
    1.0, 0.99, 0.98, 0.985, 0.97, 0.965, 0.95, 0.955, 0.94, 0.93, 0.925, 0.91, 0.90
  };

  double lr = initial_lr;
  for (double err : noisy_errors) {
    lr = scheduler.update(err, lr, 10, 100);
  }

  EXPECT_GT(lr, initial_lr);
}
