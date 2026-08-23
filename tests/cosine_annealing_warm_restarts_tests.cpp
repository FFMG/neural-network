#include <gtest/gtest.h>
#include "../include/neuralnetwork/common/cosineannealingwarmrestartsdetails.h"
#include "../include/neuralnetwork/neuralnetworkoptions.h"

using namespace myoddweb::nn;

TEST(CosineAnnealingWarmRestartsTest, ConstructorAndGetters)
{
  CosineAnnealingWarmRestartsDetails ca(true, 20, 2.0, 0.001, 0.8);
  EXPECT_TRUE(ca.enabled());
  EXPECT_EQ(ca.first_cycle_epochs(), 20);
  EXPECT_DOUBLE_EQ(ca.cycle_multiplier(), 2.0);
  EXPECT_DOUBLE_EQ(ca.minimum_learning_rate(), 0.001);
  EXPECT_DOUBLE_EQ(ca.restart_decay(), 0.8);
}

TEST(CosineAnnealingWarmRestartsTest, CopySemantics)
{
  CosineAnnealingWarmRestartsDetails original(true, 15, 1.5, 0.0005, 0.9);
  CosineAnnealingWarmRestartsDetails copy_constructed(original);

  EXPECT_TRUE(copy_constructed.enabled());
  EXPECT_EQ(copy_constructed.first_cycle_epochs(), 15);
  EXPECT_DOUBLE_EQ(copy_constructed.cycle_multiplier(), 1.5);
  EXPECT_DOUBLE_EQ(copy_constructed.minimum_learning_rate(), 0.0005);
  EXPECT_DOUBLE_EQ(copy_constructed.restart_decay(), 0.9);

  CosineAnnealingWarmRestartsDetails copy_assigned(false, 5, 1.0, 0.0, 1.0);
  copy_assigned = original;

  EXPECT_TRUE(copy_assigned.enabled());
  EXPECT_EQ(copy_assigned.first_cycle_epochs(), 15);
  EXPECT_DOUBLE_EQ(copy_assigned.cycle_multiplier(), 1.5);
  EXPECT_DOUBLE_EQ(copy_assigned.minimum_learning_rate(), 0.0005);
  EXPECT_DOUBLE_EQ(copy_assigned.restart_decay(), 0.9);
}

TEST(CosineAnnealingWarmRestartsTest, MoveSemantics)
{
  CosineAnnealingWarmRestartsDetails source(true, 25, 2.0, 0.002, 0.75);
  CosineAnnealingWarmRestartsDetails moved_constructed(std::move(source));

  EXPECT_TRUE(moved_constructed.enabled());
  EXPECT_EQ(moved_constructed.first_cycle_epochs(), 25);
  EXPECT_DOUBLE_EQ(moved_constructed.cycle_multiplier(), 2.0);
  EXPECT_DOUBLE_EQ(moved_constructed.minimum_learning_rate(), 0.002);
  EXPECT_DOUBLE_EQ(moved_constructed.restart_decay(), 0.75);
  EXPECT_FALSE(source.enabled());

  CosineAnnealingWarmRestartsDetails another_source(true, 30, 1.2, 0.003, 0.85);
  CosineAnnealingWarmRestartsDetails move_assigned(false, 1, 1.0, 0.0, 1.0);
  move_assigned = std::move(another_source);

  EXPECT_TRUE(move_assigned.enabled());
  EXPECT_EQ(move_assigned.first_cycle_epochs(), 30);
  EXPECT_DOUBLE_EQ(move_assigned.cycle_multiplier(), 1.2);
  EXPECT_DOUBLE_EQ(move_assigned.minimum_learning_rate(), 0.003);
  EXPECT_DOUBLE_EQ(move_assigned.restart_decay(), 0.85);
  EXPECT_FALSE(another_source.enabled());
}

TEST(CosineAnnealingWarmRestartsTest, DisabledReturnsBaseLearningRate)
{
  CosineAnnealingWarmRestartsDetails disabled(false, 10, 1.0, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(disabled.calculate_learning_rate(0, 0.1), 0.1);
  EXPECT_DOUBLE_EQ(disabled.calculate_learning_rate(5, 0.1), 0.1);
  EXPECT_DOUBLE_EQ(disabled.calculate_learning_rate(10, 0.1), 0.1);
  EXPECT_DOUBLE_EQ(disabled.calculate_learning_rate(-1, 0.1), 0.1);
}

TEST(CosineAnnealingWarmRestartsTest, ConstantPeriodCosineProgression)
{
  // T_0 = 10, T_mult = 1.0, eta_min = 0.0, gamma = 1.0, base_lr = 0.1
  CosineAnnealingWarmRestartsDetails ca(true, 10, 1.0, 0.0, 1.0);
  constexpr double base_lr = 0.1;

  // Cycle 0: t=0 -> 0.1, t=5 (midpoint) -> 0.05
  EXPECT_NEAR(ca.calculate_learning_rate(0, base_lr), 0.1, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(5, base_lr), 0.05, 1e-12);

  // Restart at t=10 (Cycle 1 start): t=10 -> 0.1, t=15 -> 0.05
  EXPECT_NEAR(ca.calculate_learning_rate(10, base_lr), 0.1, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(15, base_lr), 0.05, 1e-12);

  // Restart at t=20 (Cycle 2 start): t=20 -> 0.1, t=25 -> 0.05
  EXPECT_NEAR(ca.calculate_learning_rate(20, base_lr), 0.1, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(25, base_lr), 0.05, 1e-12);

  // Verify full analytical formula for each epoch in [0..29]
  constexpr double pi = 3.14159265358979323846;
  for (int t = 0; t < 30; ++t)
  {
    int t_cur = t % 10;
    double expected = 0.5 * base_lr * (1.0 + std::cos((static_cast<double>(t_cur) / 10.0) * pi));
    EXPECT_NEAR(ca.calculate_learning_rate(t, base_lr), expected, 1e-12);
  }
}

TEST(CosineAnnealingWarmRestartsTest, RestartDecayGamma)
{
  // T_0 = 10, T_mult = 1.0, eta_min = 0.0, gamma = 0.5, base_lr = 0.1
  CosineAnnealingWarmRestartsDetails ca(true, 10, 1.0, 0.0, 0.5);
  constexpr double base_lr = 0.1;

  // Cycle 0: peak = 0.1, midpoint = 0.05
  EXPECT_NEAR(ca.calculate_learning_rate(0, base_lr), 0.1, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(5, base_lr), 0.05, 1e-12);

  // Cycle 1: peak = 0.05, midpoint = 0.025
  EXPECT_NEAR(ca.calculate_learning_rate(10, base_lr), 0.05, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(15, base_lr), 0.025, 1e-12);

  // Cycle 2: peak = 0.025, midpoint = 0.0125
  EXPECT_NEAR(ca.calculate_learning_rate(20, base_lr), 0.025, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(25, base_lr), 0.0125, 1e-12);
}

TEST(CosineAnnealingWarmRestartsTest, GeometricPeriodGrowth)
{
  // T_0 = 10, T_mult = 2.0, eta_min = 0.0, gamma = 1.0, base_lr = 0.1
  // Cycle 0: T_0 = 10 -> epochs [0..9]
  // Cycle 1: T_1 = 20 -> epochs [10..29]
  // Cycle 2: T_2 = 40 -> epochs [30..69]
  CosineAnnealingWarmRestartsDetails ca(true, 10, 2.0, 0.0, 1.0);
  constexpr double base_lr = 0.1;

  // Cycle 0
  EXPECT_NEAR(ca.calculate_learning_rate(0, base_lr), 0.1, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(5, base_lr), 0.05, 1e-12);

  // Cycle 1 (starts at t=10, length=20, midpoint at t=10+10=20)
  EXPECT_NEAR(ca.calculate_learning_rate(10, base_lr), 0.1, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(20, base_lr), 0.05, 1e-12);

  // Cycle 2 (starts at t=30, length=40, midpoint at t=30+20=50)
  EXPECT_NEAR(ca.calculate_learning_rate(30, base_lr), 0.1, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(50, base_lr), 0.05, 1e-12);
}

TEST(CosineAnnealingWarmRestartsTest, NonZeroMinimumLearningRate)
{
  // T_0 = 10, T_mult = 1.0, eta_min = 0.02, gamma = 1.0, base_lr = 0.1
  CosineAnnealingWarmRestartsDetails ca(true, 10, 1.0, 0.02, 1.0);
  constexpr double base_lr = 0.1;

  // Peak at t=0 -> 0.1
  EXPECT_NEAR(ca.calculate_learning_rate(0, base_lr), 0.1, 1e-12);
  // Midpoint at t=5 -> 0.02 + 0.5 * (0.1 - 0.02) = 0.06
  EXPECT_NEAR(ca.calculate_learning_rate(5, base_lr), 0.06, 1e-12);
}

TEST(CosineAnnealingWarmRestartsTest, RestartDecayNeverGoesBelowMinimumLearningRate)
{
  // T_0 = 10, T_mult = 1.0, eta_min = 0.05, gamma = 0.5, base_lr = 0.1
  // By cycle 2 the undamped peak (0.1 * 0.5^2 = 0.025) would fall below eta_min,
  // which must not push the schedule below the configured floor.
  CosineAnnealingWarmRestartsDetails ca(true, 10, 1.0, 0.05, 0.5);
  constexpr double base_lr = 0.1;

  for (int t = 0; t < 100; ++t)
  {
    EXPECT_GE(ca.calculate_learning_rate(t, base_lr), 0.05 - 1e-12);
  }
}

TEST(CosineAnnealingWarmRestartsTest, GeometricGrowthDoesNotStallWithSmallFirstCycle)
{
  // T_0 = 1, T_mult = 1.4: naive std::round(1 * 1.4) == 1, which would stall
  // cycle growth forever, so t_cur/cur_t_i would always be 0/1 (i.e. the peak,
  // 0.1) at every epoch. With growth working, cycles lengthen to 1, 2, 3, ...
  // so relative_epoch=5 lands two-thirds of the way through a length-3 cycle.
  CosineAnnealingWarmRestartsDetails ca(true, 1, 1.4, 0.0, 1.0);
  constexpr double base_lr = 0.1;

  EXPECT_NEAR(ca.calculate_learning_rate(0, base_lr), 0.1, 1e-12);
  EXPECT_NEAR(ca.calculate_learning_rate(5, base_lr), 0.025, 1e-9);
}

TEST(CosineAnnealingWarmRestartsTest, OptionsWithCosineAnnealingObject)
{
  CosineAnnealingWarmRestartsDetails ca(true, 15, 1.5, 0.001, 0.95);
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_cosine_annealing_warm_restarts(ca)
    .build();

  EXPECT_TRUE(options.cosine_annealing_warm_restarts().enabled());
  EXPECT_EQ(options.cosine_annealing_warm_restarts().first_cycle_epochs(), 15);
  EXPECT_DOUBLE_EQ(options.cosine_annealing_warm_restarts().cycle_multiplier(), 1.5);
  EXPECT_DOUBLE_EQ(options.cosine_annealing_warm_restarts().minimum_learning_rate(), 0.001);
  EXPECT_DOUBLE_EQ(options.cosine_annealing_warm_restarts().restart_decay(), 0.95);
}

TEST(CosineAnnealingWarmRestartsTest, OptionsWithCosineAnnealingHelperFunction)
{
  auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
    .with_cosine_annealing_warm_restarts(true, 12, 2.0, 0.005, 0.8)
    .build();

  EXPECT_TRUE(options.cosine_annealing_warm_restarts().enabled());
  EXPECT_EQ(options.cosine_annealing_warm_restarts().first_cycle_epochs(), 12);
  EXPECT_DOUBLE_EQ(options.cosine_annealing_warm_restarts().cycle_multiplier(), 2.0);
  EXPECT_DOUBLE_EQ(options.cosine_annealing_warm_restarts().minimum_learning_rate(), 0.005);
  EXPECT_DOUBLE_EQ(options.cosine_annealing_warm_restarts().restart_decay(), 0.8);
}

TEST(CosineAnnealingWarmRestartsTest, ValidationRejectsInvalidFirstCycleEpochs)
{
  EXPECT_THROW(
    {
      auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
        .with_cosine_annealing_warm_restarts(true, 0, 1.0, 0.0, 1.0)
        .build();
    },
    std::runtime_error
  );
}

TEST(CosineAnnealingWarmRestartsTest, ValidationRejectsInvalidCycleMultiplier)
{
  EXPECT_THROW(
    {
      auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
        .with_cosine_annealing_warm_restarts(true, 10, 0.5, 0.0, 1.0)
        .build();
    },
    std::runtime_error
  );
}

TEST(CosineAnnealingWarmRestartsTest, ValidationRejectsInvalidMinimumLearningRate)
{
  EXPECT_THROW(
    {
      auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
        .with_cosine_annealing_warm_restarts(true, 10, 1.0, -0.01, 1.0)
        .build();
    },
    std::runtime_error
  );
}

TEST(CosineAnnealingWarmRestartsTest, ValidationRejectsInvalidRestartDecay)
{
  EXPECT_THROW(
    {
      auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
        .with_cosine_annealing_warm_restarts(true, 10, 1.0, 0.0, 0.0)
        .build();
    },
    std::runtime_error
  );

  EXPECT_THROW(
    {
      auto options = NeuralNetworkOptions::create({ 2, 4, 1 })
        .with_cosine_annealing_warm_restarts(true, 10, 1.0, 0.0, 1.5)
        .build();
    },
    std::runtime_error
  );
}
