#include <gtest/gtest.h>
#include "../src/neuralnetwork/layer.h"
#include "test_helper.h"
#include <vector>

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
    std::vector<double> w = { 0.1, 0.2, 0.3, 0.4 }; // MockLayer has 0 input neurons in constructor? 
    // Wait, MockLayer(unsigned layer_index, unsigned num_neurons) :
    // Layer(layer_index, Layer::Role::Input, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0, num_neurons, {}, false, 0.0, nullptr, 1, 0.0) {}
    // Number of inputs is 0.
    
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
    // RMSE = sqrt( ((0.8-1)^2 + (0.4-0)^2) / 2 ) = sqrt( (0.04 + 0.16) / 2 ) = sqrt(0.1) approx 0.316227766
    // Factor = inv_num_neurons / RMSE = 0.5 / sqrt(0.1) = 0.5 / 0.316227766 approx 1.58113883
    // delta_0 = (0.8 - 1.0) * 1.58113883 = -0.316227766
    // delta_1 = (0.4 - 0.0) * 1.58113883 = 0.632455532
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
    // BCE Delta = (y - t) / N = (0.8 - 1.0) / 2 = -0.1
    //                         = (0.4 - 0.0) / 2 = 0.2
    EXPECT_NEAR(deltas[0], -0.1, 1e-9);
    EXPECT_NEAR(deltas[1], 0.2, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasCE) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 0.0 };
    std::vector<double> given = { 0.8, 0.2 }; // Must sum to 1 for CE? Actually implementation just does y-t
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::cross_entropy, EvaluationConfig(), activation::method::softmax, 0, 1);
    // CE Delta = (y - t) = (0.8 - 1.0) = -0.2
    //                   = (0.2 - 0.0) = 0.2
    EXPECT_NEAR(deltas[0], -0.2, 1e-9);
    EXPECT_NEAR(deltas[1], 0.2, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasMulticlassSoftmax) {
    // Test 3 classes with Softmax and Cross Entropy
    // Given outputs (y) = [0.7, 0.2, 0.1]
    // Target outputs (t) = [1.0, 0.0, 0.0] (Class 0 is correct)
    // Temperature (T) = 2.0
    // dL/dz = (y - t) / T
    
    // We can't override get_activation as it's not virtual. 
    // Instead, we use a MockLayer initialized with the correct activation.
    unsigned num_classes = 3;
    MockLayer layer(0, num_classes);
    layer.get_activation_helper().set_bounds(activation(activation::method::softmax, 0.0, 2.0), 0, num_classes);
    
    std::vector<double> deltas(num_classes, 0.0);
    std::vector<double> targets = { 1.0, 0.0, 0.0 };
    std::vector<double> given = { 0.7, 0.2, 0.1 };
    
    EvaluationConfig config; // cross_entropy_lambda = 1.0, direction_penalty = false
    
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::cross_entropy, config, activation::method::softmax, 0, 2);
    
    // Expected grads:
    // grad_0 = (0.7 - 1.0) / 2.0 = -0.15
    // grad_1 = (0.2 - 0.0) / 2.0 = 0.10
    // grad_2 = (0.1 - 0.0) / 2.0 = 0.05
    EXPECT_NEAR(deltas[0], -0.15, 1e-9);
    EXPECT_NEAR(deltas[1], 0.10, 1e-9);
    EXPECT_NEAR(deltas[2], 0.05, 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasHuber) {
    MockLayer layer(0, 1);
    std::vector<double> deltas(1, 0.0);
    std::vector<double> targets = { 1.0 };
    EvaluationConfig config; // Default delta is likely 1.0
    
    // Case 1: Error < Delta (0.2 < 1.0) -> behaves like MSE (but with 1/N? let's check)
    // Actually Huber in code: grad = error * inv_num_neurons
    std::vector<double> given_small = { 1.2 };
    layer.calculate_error_deltas(deltas, targets, given_small, ErrorCalculation::type::huber_loss, config, activation::method::linear, 0, 0);
    EXPECT_NEAR(deltas[0], 0.2, 1e-9);

    // Case 2: Error > Delta (2.0 > 1.0) -> behaves like MAE (grad = delta * inv_num_neurons)
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
    // log(cosh(x))' = tanh(x)
    // grad = tanh(y - t) * inv_num_neurons = tanh(0.5) * 1.0
    EXPECT_NEAR(deltas[0], std::tanh(0.5), 1e-9);
}

TEST(LayerTest, CalculateErrorDeltasRobustness) {
    MockLayer layer(0, 2);
    std::vector<double> deltas(2, 0.0);
    std::vector<double> targets = { 1.0, 1.0 };
    std::vector<double> given = { 1.0, 1.0 };

    // Valid range [0, 1]
    EXPECT_NO_THROW(layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::mse, EvaluationConfig(), activation::method::linear, 0, 1));

    // Invalid range: end < start
    // In VALIDATE_DATA mode, this should panic (throw).
#if VALIDATE_DATA == 1
    EXPECT_ANY_THROW(layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::mse, EvaluationConfig(), activation::method::linear, 1, 0));
    
    // Invalid range: end >= get_number_neurons()
    EXPECT_ANY_THROW(layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::mse, EvaluationConfig(), activation::method::linear, 0, 2));
#endif
}

TEST(LayerTest, CalculateErrorDeltasOutOfSync) {
    // Manually create a layer where _neurons is smaller than reported output size
    class OutOfSyncLayer : public MockLayer {
    public:
        OutOfSyncLayer() : MockLayer(0, 5, 0) {
            _neurons.clear(); // Force out of sync
        }
    };

    OutOfSyncLayer layer;
    std::vector<double> deltas(5, 0.0);
    std::vector<double> targets(5, 1.0);
    std::vector<double> given(5, 1.0);

#if VALIDATE_DATA == 1
    // This should throw because end_neuron (4) >= _neurons.size() (0)
    EXPECT_ANY_THROW(layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::mse, EvaluationConfig(), activation::method::linear, 0, 4));
#endif
}
