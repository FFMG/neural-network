#include <gtest/gtest.h>
#include "layers/layer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <algorithm>


using namespace myoddweb::nn;
using namespace test_helper;

TEST(LayerTest, ArchitectureToString) {
    EXPECT_EQ(Layer::architecture_to_string(Layer::Architecture::FF), "FF");
    EXPECT_EQ(Layer::architecture_to_string(Layer::Architecture::Elman), "Elman");
    EXPECT_EQ(Layer::architecture_to_string(Layer::Architecture::Gru), "Gru");
    EXPECT_EQ(Layer::architecture_to_string(Layer::Architecture::Lstm), "Lstm");
    EXPECT_EQ(Layer::architecture_to_string(Layer::Architecture::MultiOutput), "MultiOutput");
    EXPECT_EQ(Layer::architecture_to_string(Layer::Architecture::None), "None");
}

TEST(LayerTest, ArchitectureFromString) {
    EXPECT_EQ(Layer::architecture_from_string("FF"), Layer::Architecture::FF);
    EXPECT_EQ(Layer::architecture_from_string("ff"), Layer::Architecture::FF);
    EXPECT_EQ(Layer::architecture_from_string("Elman"), Layer::Architecture::Elman);
    EXPECT_EQ(Layer::architecture_from_string("gru"), Layer::Architecture::Gru);
    EXPECT_EQ(Layer::architecture_from_string("lstm"), Layer::Architecture::Lstm);
    EXPECT_EQ(Layer::architecture_from_string("multioutput"), Layer::Architecture::MultiOutput);
    EXPECT_EQ(Layer::architecture_from_string("none"), Layer::Architecture::None);
}

TEST(LayerTest, CreateWDecays) {
    auto decays = Layer::create_w_decays(2, 3, 0.01);
    EXPECT_EQ(decays.size(), 6);
    for (double d : decays) {
        EXPECT_DOUBLE_EQ(d, 0.01);
    }
}

TEST(LayerTest, HelperMethods) {
    MockLayer layer(5, 10);
    EXPECT_EQ(layer.get_layer_index(), 5);
    EXPECT_EQ(layer.get_number_neurons(), 10);
    EXPECT_EQ(layer.get_layer_role(), Layer::Role::Input);
    EXPECT_EQ(layer.get_optimiser_type(), OptimiserType::None);
    EXPECT_FALSE(layer.has_bias());
    EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);
}

TEST(LayerTest, SettersAndGetters) {
    MockLayer layer(0, 2);
    layer.set_w_values({ 1.0, 2.0 });
    EXPECT_EQ(layer.get_w_values().size(), 2);
    EXPECT_DOUBLE_EQ(layer.get_w_values()[0], 1.0);

    layer.set_b_values({ 0.5, 0.6 });
    EXPECT_TRUE(layer.has_bias());
    EXPECT_DOUBLE_EQ(layer.get_bias_value(0), 0.5);
    EXPECT_DOUBLE_EQ(layer.get_bias_value(1), 0.6);
}

TEST(LayerTest, ResetOptimizerState) {
    MockLayer layer(0, 1);
    layer.set_w_velocities({ 1.0 });
    layer.set_w_m1({ 2.0 });
    layer.set_w_m2({ 3.0 });
    
    layer.reset_optimizer_state();
    
    EXPECT_DOUBLE_EQ(layer.get_w_velocities()[0], 0.0);
    EXPECT_DOUBLE_EQ(layer.get_w_m1()[0], 0.0);
    EXPECT_DOUBLE_EQ(layer.get_w_m2()[0], 0.0);
}

TEST(LayerTest, OptimiserTypeToString) {
    EXPECT_EQ(optimiser_type_to_string(OptimiserType::Adam), "Adam");
    EXPECT_EQ(optimiser_type_to_string(OptimiserType::SGD), "SGD");
    EXPECT_EQ(optimiser_type_to_string(OptimiserType::None), "None");
}

TEST(LayerTest, StringToOptimiserType) {
    EXPECT_EQ(string_to_optimiser_type("Adam"), OptimiserType::Adam);
    EXPECT_EQ(string_to_optimiser_type("adamw"), OptimiserType::AdamW);
    EXPECT_EQ(string_to_optimiser_type("sgd"), OptimiserType::SGD);
}

TEST(LayerTest, CalculateErrorDeltasBCENonSigmoid) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 0.0 };
    std::vector<double> given = { 0.8, 0.4 };
    // For non-sigmoid, dL/da = (a-y)/(a*(1-a))
    // Neuron 0: (0.8 - 1.0) / (0.8 * 0.2) = -0.2 / 0.16 = -1.25
    // Neuron 1: (0.4 - 0.0) / (0.4 * 0.6) = 0.4 / 0.24 = 1.666666666...
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::bce_loss, EvaluationConfig(), activation::method::linear, 0, 1);
    EXPECT_NEAR(deltas[0], -1.25 * 0.5, 1e-9); // 0.5 because of inv_num_neurons
    EXPECT_NEAR(deltas[1], (0.4/0.24) * 0.5, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasCENonSoftmax) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 0.0 };
    std::vector<double> given = { 0.8, 0.2 };
    // For non-softmax, dL/da = -y/a
    // Neuron 0: -1.0 / 0.8 = -1.25
    // Neuron 1: -0.0 / 0.2 = 0.0
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::cross_entropy, EvaluationConfig(), activation::method::linear, 0, 1);
    EXPECT_NEAR(deltas[0], -1.25, 1e-9);
    EXPECT_NEAR(deltas[1], 0.0, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasMSE) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 0.0 };
    std::vector<double> given = { 0.8, 0.4 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::mse, EvaluationConfig(), activation::method::linear, 0, 1);
    EXPECT_NEAR(deltas[0], -0.1, 1e-9);
    EXPECT_NEAR(deltas[1], 0.2, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasRMSE) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 0.0 };
    std::vector<double> given = { 0.8, 0.4 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::rmse, EvaluationConfig(), activation::method::linear, 0, 1);
    double rmse = std::sqrt(0.1);
    EXPECT_NEAR(deltas[0], -0.2 * 0.5 / rmse, 1e-9);
    EXPECT_NEAR(deltas[1], 0.4 * 0.5 / rmse, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasBCE) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 0.0 };
    std::vector<double> given = { 0.8, 0.4 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::bce_loss, EvaluationConfig(), activation::method::sigmoid, 0, 1);
    EXPECT_NEAR(deltas[0], -0.1, 1e-9);
    EXPECT_NEAR(deltas[1], 0.2, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasCE) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 0.0 };
    std::vector<double> given = { 0.8, 0.2 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::cross_entropy, EvaluationConfig(), activation::method::softmax, 0, 1);
    EXPECT_NEAR(deltas[0], -0.2, 1e-9);
    EXPECT_NEAR(deltas[1], 0.2, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasMulticlassSoftmax) {
    unsigned num_classes = 3;
    MockLayer layer(0, num_classes);
    layer.get_activation_helper().set_bounds(activation(activation::method::softmax, 0.0, 2.0), 0, num_classes);
    
    std::vector<double> deltas(num_classes, 0.0);
    std::vector<double> targets = { 1.0, 0.0, 0.0 };
    std::vector<double> given = { 0.7, 0.2, 0.1 };
    
    EvaluationConfig config;
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::cross_entropy, config, activation::method::softmax, 0, 2);
    
    EXPECT_NEAR(deltas[0], -0.15, 1e-9);
    EXPECT_NEAR(deltas[1], 0.10, 1e-9);
    EXPECT_NEAR(deltas[2], 0.05, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasHuber) {
    MockLayer layer(0, 1);
    std::vector<double> deltas(1, 0.0);
    std::vector<double> targets = { 1.0 };
    EvaluationConfig config;
    
    std::vector<double> given_small = { 1.2 };
    layer.calculate_error_deltas(deltas, targets, given_small, ErrorCalculation::type::huber_loss, config, activation::method::linear, 0, 0);
    EXPECT_NEAR(deltas[0], 0.2, 1e-9);

    std::vector<double> given_large = { 3.0 };
    layer.calculate_error_deltas(deltas, targets, given_large, ErrorCalculation::type::huber_loss, config, activation::method::linear, 0, 0);
    EXPECT_NEAR(deltas[0], 1.0, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasLogCosh) {
    MockLayer layer(0, 1);
    std::vector<double> deltas(1, 0.0);
    std::vector<double> targets = { 1.0 };
    std::vector<double> given = { 1.5 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::log_cosh, EvaluationConfig(), activation::method::linear, 0, 0);
    EXPECT_NEAR(deltas[0], std::tanh(0.5), 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasRobustness) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 1.0 };
    std::vector<double> given = { 1.0, 1.0 };

    EXPECT_NO_THROW(layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::mse, EvaluationConfig(), activation::method::linear, 0, 1));

#if VALIDATE_DATA == 1
    EXPECT_ANY_THROW(layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::mse, EvaluationConfig(), activation::method::linear, 1, 0));
    EXPECT_ANY_THROW(layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::mse, EvaluationConfig(), activation::method::linear, 0, 2));
#endif
}

TEST(LayerTest, DropoutConsistency) {
    unsigned num_neurons = 100;
    double dropout_rate = 0.5;
    MockLayer layer(1, num_neurons);
    
    for (unsigned i = 0; i < num_neurons; ++i) {
        EXPECT_FALSE(layer.get_neuron(i).is_dropout());
    }

    auto neurons = MockLayer::create_neurons_exposed(dropout_rate, num_neurons);
    EXPECT_EQ(neurons.size(), num_neurons);
    for (const auto& n : neurons) {
        if (dropout_rate > 0.0) {
            EXPECT_TRUE(n.is_dropout());
            EXPECT_DOUBLE_EQ(n.get_dropout_rate(), dropout_rate);
        } else {
            EXPECT_FALSE(n.is_dropout());
        }
    }
}

TEST(LayerTest, DropoutStatisticalVerification) {
    const unsigned num_neurons = 1000;
    const double dropout_rate = 0.3;
    auto neurons = MockLayer::create_neurons_exposed(dropout_rate, num_neurons);
    
    unsigned dropped = 0;
    for (unsigned i = 0; i < num_neurons; ++i) {
        if (neurons[i].must_randomly_drop()) {
            dropped++;
        }
    }
    
    double actual_rate = static_cast<double>(dropped) / num_neurons;
    EXPECT_NEAR(actual_rate, dropout_rate, 0.05); // 5% tolerance for 10k samples
}
