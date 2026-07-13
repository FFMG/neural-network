#include <gtest/gtest.h>
#include "helpers/neuralnetworkhelper.h"
#include "neuralnetworkoptions.h"
#include "neuralnetwork.h"
#include <thread>
#include <chrono>

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
  EXPECT_LE(dur2, 35.0);
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
