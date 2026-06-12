#include <gtest/gtest.h>
#include "layers/outputlayerdetails.h"


using namespace myoddweb::nn;
TEST(OutputLayerDetailsTest, ConstructorAndGetters) {
  unsigned layer_size = 10;
  activation act(activation::method::softmax, 0.1, 1.2, 0.8);
  ErrorCalculation::type err_type = ErrorCalculation::type::cross_entropy;
  EvaluationConfig config;
  double weight_decay = 0.01;
  OptimiserType optimiser = OptimiserType::Adam;
  double momentum = 0.9;

  OutputLayerDetails details(layer_size, act, err_type, config, weight_decay, optimiser, momentum);

  EXPECT_EQ(details.get_size(), layer_size);
  EXPECT_EQ(details.get_activation().get_method(), activation::method::softmax);
  EXPECT_NEAR(details.get_activation().get_alpha(), 0.1, 1e-9);
  EXPECT_NEAR(details.get_activation().get_temperature(), 1.2, 1e-9);
  EXPECT_NEAR(details.get_activation().get_inference_temperature(), 0.8, 1e-9);
  EXPECT_EQ(details.get_output_error_calculation_type(), err_type);
  EXPECT_NEAR(details.get_weight_decay(), weight_decay, 1e-9);
  EXPECT_EQ(details.get_optimiser_type(), optimiser);
  EXPECT_NEAR(details.get_momentum(), momentum, 1e-9);
}

TEST(OutputLayerDetailsTest, CopyConstructor) {
  OutputLayerDetails details(5, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails copy = details;

  EXPECT_EQ(copy.get_size(), details.get_size());
  EXPECT_EQ(copy.get_activation().get_method(), details.get_activation().get_method());
  EXPECT_EQ(copy.get_output_error_calculation_type(), details.get_output_error_calculation_type());
}

TEST(OutputLayerDetailsTest, MoveConstructor) {
  OutputLayerDetails details(5, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails moved = std::move(details);

  EXPECT_EQ(moved.get_size(), 5);
  // Source should be in a valid but unspecified state, or as per implementation (size set to 0)
  EXPECT_EQ(details.get_size(), 0); 
}

TEST(OutputLayerDetailsTest, AssignmentOperator) {
  OutputLayerDetails details(5, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails other(10, activation(activation::method::relu, 0.0), ErrorCalculation::type::rmse, EvaluationConfig(), 0.1, OptimiserType::Adam, 0.9);
  
  other = details;

  EXPECT_EQ(other.get_size(), 5);
  EXPECT_EQ(other.get_activation().get_method(), activation::method::linear);
}

TEST(OutputLayerDetailsTest, MoveAssignmentOperator) {
  OutputLayerDetails details(5, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
  OutputLayerDetails other(10, activation(activation::method::relu, 0.0), ErrorCalculation::type::rmse, EvaluationConfig(), 0.1, OptimiserType::Adam, 0.9);
  
  other = std::move(details);

  EXPECT_EQ(other.get_size(), 5);
  EXPECT_EQ(details.get_size(), 0);
}

TEST(OutputLayerDetailsTest, InvalidWeightDecay) {
  EXPECT_ANY_THROW(OutputLayerDetails(5, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), -0.1, OptimiserType::SGD, 0.0));
}
