#include <gtest/gtest.h>
#include "common/stochasticweightaveragingdetails.h"
#include "neuralnetworkoptions.h"
#include "layers/outputlayerdetails.h"

using namespace myoddweb::nn;

TEST(StochasticWeightAveragingDetailsTest, ConstructorAndGetters)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  StochasticWeightAveragingDetails details(true, 0.65, 0.05);

  EXPECT_TRUE(details.enabled());
  EXPECT_TRUE(details.swa_enabled());
  EXPECT_NEAR(details.start_percent(), 0.65, 1e-9);
  EXPECT_NEAR(details.swa_start_percent(), 0.65, 1e-9);
  EXPECT_NEAR(details.update_percent(), 0.05, 1e-9);
  EXPECT_NEAR(details.swa_update_percent(), 0.05, 1e-9);
}

TEST(StochasticWeightAveragingDetailsTest, CopyConstructor)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  StochasticWeightAveragingDetails original(true, 0.8, 0.04);
  StochasticWeightAveragingDetails copy(original);

  EXPECT_EQ(copy.enabled(), original.enabled());
  EXPECT_NEAR(copy.start_percent(), original.start_percent(), 1e-9);
  EXPECT_NEAR(copy.update_percent(), original.update_percent(), 1e-9);
}

TEST(StochasticWeightAveragingDetailsTest, MoveConstructor)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  StochasticWeightAveragingDetails original(true, 0.8, 0.04);
  StochasticWeightAveragingDetails moved(std::move(original));

  EXPECT_TRUE(moved.enabled());
  EXPECT_NEAR(moved.start_percent(), 0.8, 1e-9);
  EXPECT_NEAR(moved.update_percent(), 0.04, 1e-9);

  // Moved-from object state should be reset
  EXPECT_FALSE(original.enabled());
  EXPECT_NEAR(original.start_percent(), 0.0, 1e-9);
  EXPECT_NEAR(original.update_percent(), 0.0, 1e-9);
}

TEST(StochasticWeightAveragingDetailsTest, CopyAssignment)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  StochasticWeightAveragingDetails original(true, 0.7, 0.03);
  StochasticWeightAveragingDetails target(false, 0.0, 0.0);

  target = original;

  EXPECT_TRUE(target.enabled());
  EXPECT_NEAR(target.start_percent(), 0.7, 1e-9);
  EXPECT_NEAR(target.update_percent(), 0.03, 1e-9);

  // Self-assignment safety check
  target = *&target;
  EXPECT_TRUE(target.enabled());
  EXPECT_NEAR(target.start_percent(), 0.7, 1e-9);
  EXPECT_NEAR(target.update_percent(), 0.03, 1e-9);
}

TEST(StochasticWeightAveragingDetailsTest, MoveAssignment)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  StochasticWeightAveragingDetails original(true, 0.7, 0.03);
  StochasticWeightAveragingDetails target(false, 0.0, 0.0);

  target = std::move(original);

  EXPECT_TRUE(target.enabled());
  EXPECT_NEAR(target.start_percent(), 0.7, 1e-9);
  EXPECT_NEAR(target.update_percent(), 0.03, 1e-9);

  EXPECT_FALSE(original.enabled());
  EXPECT_NEAR(original.start_percent(), 0.0, 1e-9);
  EXPECT_NEAR(original.update_percent(), 0.0, 1e-9);
}

TEST(StochasticWeightAveragingDetailsTest, OptionsWithSwaObject)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.05, OptimiserType::SGD, 0.99))
    .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.6, 0.05))
    .build();

  EXPECT_TRUE(options.stochastic_weight_averaging().enabled());
  EXPECT_NEAR(options.stochastic_weight_averaging().start_percent(), 0.6, 1e-9);
  EXPECT_NEAR(options.stochastic_weight_averaging().update_percent(), 0.05, 1e-9);
}

TEST(StochasticWeightAveragingDetailsTest, OptionsWithSwaHelperFunction)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  auto options = NeuralNetworkOptions::create({ 2, 2, 1 })
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.05, OptimiserType::SGD, 0.99))
    .with_stochastic_weight_averaging(true, 0.7, 0.02)
    .build();

  EXPECT_TRUE(options.stochastic_weight_averaging().enabled());
  EXPECT_NEAR(options.stochastic_weight_averaging().start_percent(), 0.7, 1e-9);
  EXPECT_NEAR(options.stochastic_weight_averaging().update_percent(), 0.02, 1e-9);
}

TEST(StochasticWeightAveragingDetailsTest, ValidationRejectsInvalidStartPercent)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  // start_percent < 0.0
  EXPECT_ANY_THROW(
    NeuralNetworkOptions::create({ 2, 2, 1 })
      .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.05, OptimiserType::SGD, 0.99))
      .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, -0.1, 0.05))
      .build()
  );

  // start_percent >= 1.0
  EXPECT_ANY_THROW(
    NeuralNetworkOptions::create({ 2, 2, 1 })
      .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.05, OptimiserType::SGD, 0.99))
      .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 1.0, 0.05))
      .build()
  );
}

TEST(StochasticWeightAveragingDetailsTest, ValidationRejectsInvalidUpdatePercent)
{
  MYODDWEB_PROFILE_FUNCTION("StochasticWeightAveragingDetailsTest");
  // update_percent <= 0.0
  EXPECT_ANY_THROW(
    NeuralNetworkOptions::create({ 2, 2, 1 })
      .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.05, OptimiserType::SGD, 0.99))
      .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.5, 0.0))
      .build()
  );

  // update_percent > 1.0
  EXPECT_ANY_THROW(
    NeuralNetworkOptions::create({ 2, 2, 1 })
      .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 } }, 0.05, OptimiserType::SGD, 0.99))
      .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.5, 1.5))
      .build()
  );
}
