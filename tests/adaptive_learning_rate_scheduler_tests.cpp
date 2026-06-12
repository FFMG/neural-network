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
  
  // Fill history with constant error
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(1.0, initial_lr, 0, 100);
  }

  // Now provide decreasing error. 
  // State: Decreasing if change < 0 and change <= (-2 * _min_percent_change)
  // -2 * 1% = -2% change per step.
  double error = 1.0;
  double lr = initial_lr;
  for (int i = 0; i < 5; ++i) {
    error *= 0.95; // 5% decrease, which is > 2%
    lr = scheduler.update(error, lr, 0, 100);
  }

  // Adjustment: current_learning_rate * (1.0 + (_adjustment_rate / 2.0))
  // 0.1 * (1.0 + 0.05) = 0.105
  EXPECT_GT(lr, initial_lr);
  EXPECT_NEAR(lr, initial_lr * (1.0 + adjustment_rate / 2.0), 1e-7);
}

TEST_F(AdaptiveLearningRateSchedulerTest, IncreasingErrorDecreasesLearningRate) {
  size_t history_size = 10;
  double min_percent_change = 0.01;
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, 0.0005, min_percent_change, adjustment_rate);
  double initial_lr = 0.1;
  
  // Fill history
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(1.0, initial_lr, 0, 100);
  }

  // Increasing error. State: Increasing if change > 0 and change >= _min_percent_change
  // And change < 2 * _min_percent_change to avoid Exploding state.
  double error = 1.0;
  double lr = initial_lr;
  for (int i = 0; i < 5; ++i) {
    error *= 1.012; // 1.2% increase, which is between 1% and 2%
    lr = scheduler.update(error, lr, 0, 100);
  }

  // Adjustment: current_learning_rate * (1.0 - _adjustment_rate * 1.5)
  // 0.1 * (1.0 - 0.15) = 0.085
  EXPECT_LT(lr, initial_lr);
  EXPECT_NEAR(lr, initial_lr * (1.0 - adjustment_rate * 1.5), 1e-7);
}

TEST_F(AdaptiveLearningRateSchedulerTest, ExplodingErrorDecreasesLearningRateFast) {
  size_t history_size = 10;
  double min_percent_change = 0.01;
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, 0.0005, min_percent_change, adjustment_rate);
  double initial_lr = 0.1;
  
  // Fill history
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(1.0, initial_lr, 0, 100);
  }

  // Exploding error. State: Exploding if change > 0 and change >= (2*_min_percent_change)
  double error = 1.0;
  double lr = initial_lr;
  for (int i = 0; i < 5; ++i) {
    error *= 1.10; // 10% increase, which is > 2%
    lr = scheduler.update(error, lr, 0, 100);
  }

  // Adjustment: current_learning_rate * (1.0 - _adjustment_rate * 2.0)
  // 0.1 * (1.0 - 0.2) = 0.08
  EXPECT_LT(lr, initial_lr);
  EXPECT_NEAR(lr, initial_lr * (1.0 - adjustment_rate * 2.0), 1e-7);
}

TEST_F(AdaptiveLearningRateSchedulerTest, PlateauingErrorDecreasesLearningRateMildly) {
  size_t history_size = 10;
  double min_plateau_percent_change = 0.0005;
  AdaptiveLearningRateScheduler scheduler(history_size, min_plateau_percent_change);
  double initial_lr = 0.1;
  
  // Fill history
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(1.0, initial_lr, 0, 100);
  }

  // Plateauing error. State: Plateauing if fabs(change) <= _min_plateau_percent_change
  double lr = initial_lr;
  int epoch = 10;
  int total_epochs = 100;
  for (int i = 0; i < 5; ++i) {
    lr = scheduler.update(1.0, lr, epoch, total_epochs);
  }

  // Adjustment: current_learning_rate * (1.0 - static_cast<double>(epoch) / number_of_epoch)
  // 0.1 * (1.0 - 10/100) = 0.09
  EXPECT_LT(lr, initial_lr);
  EXPECT_NEAR(lr, initial_lr * (1.0 - static_cast<double>(epoch) / total_epochs), 1e-7);
}

TEST_F(AdaptiveLearningRateSchedulerTest, CooldownPreventsImmediateFurtherChanges) {
  size_t history_size = 10;
  double adjustment_rate = 0.1;
  AdaptiveLearningRateScheduler scheduler(history_size, 0.0005, 0.01, adjustment_rate);
  double initial_lr = 0.1;
  
  // Fill history
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(1.0, initial_lr, 0, 100);
  }

  // Trigger a change (Decreasing)
  double error = 1.0;
  double lr = initial_lr;
  bool changed = false;
  for (int i = 0; i < 10; ++i) {
    error *= 0.90; // Large decrease
    lr = scheduler.update(error, lr, 0, 100);
    if (lr != initial_lr) {
      changed = true;
      break;
    }
  }
  
  ASSERT_TRUE(changed) << "Failed to trigger a learning rate change";
  double lr_after_first_change = lr;

  // Subsequent updates should be in cooldown.
  // CoolDownDecreasing = 10 * _history_size = 100 iterations.
  for (int i = 0; i < 50; ++i) {
    error *= 0.90;
    lr = scheduler.update(error, lr, 0, 100);
    EXPECT_DOUBLE_EQ(lr, lr_after_first_change) << "LR changed during cooldown at iteration " << i;
  }
}

TEST_F(AdaptiveLearningRateSchedulerTest, RateIsClamped) {
  size_t history_size = 10;
  AdaptiveLearningRateScheduler scheduler(history_size);
  double initial_lr = 0.1;
  
  // Fill history
  for (size_t i = 0; i < history_size; ++i) {
    scheduler.update(1.0, initial_lr, 0, 100);
  }

  // Max LR is set to clamp(2*current, current, 0.99) = 0.2
  // Let's force many increases
  double error = 1.0;
  double lr = initial_lr;
  for (int i = 0; i < 1000; ++i) {
    error *= 0.90;
    lr = scheduler.update(error, lr, 0, 100);
  }

  EXPECT_LE(lr, 0.2000000000001);
  EXPECT_GE(lr, 1e-6);
}
