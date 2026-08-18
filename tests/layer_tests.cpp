#include "layers/layer.h"
#include "layers/layers.h"
#include "layers/layerdetails.h"
#include "common/activation.h"
#include "neuralnetworkoptions.h"
#include "common/tempbuffer.h"
#include "test_helper.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>


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
    EXPECT_EQ(optimiser_type_to_string(OptimiserType::Lion), "Lion");
    EXPECT_EQ(optimiser_type_to_string(OptimiserType::None), "None");
}

TEST(LayerTest, StringToOptimiserType) {
    EXPECT_EQ(string_to_optimiser_type("Adam"), OptimiserType::Adam);
    EXPECT_EQ(string_to_optimiser_type("adamw"), OptimiserType::AdamW);
    EXPECT_EQ(string_to_optimiser_type("lion"), OptimiserType::Lion);
    EXPECT_EQ(string_to_optimiser_type("Lion"), OptimiserType::Lion);
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

TEST(LayerTest, CalculateErrorDeltasHuberDirectionPenalty)
{
  MockLayer layer(0, 1);
  std::vector<double> deltas(1, 0.0);

  EvaluationConfig config_penalty(0.01, 0.15, 1.0, 0.5, true, 1.0);
  EvaluationConfig config_no_penalty(0.01, 0.15, 1.0, 0.5, false, 1.0);

  // Case 1: Sign mismatch, target is -1.0, output is 0.5
  // use_direction_penalty is true
  {
    std::vector<double> targets = { -1.0 };
    std::vector<double> given = { 0.5 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::huber_loss, config_penalty, activation::method::linear, 0, 0);
    EXPECT_NEAR(deltas[0], 1.25, 1e-9);
  }

  // Case 2: No sign mismatch, target is 1.0, output is 0.5
  // use_direction_penalty is true
  {
    std::vector<double> targets = { 1.0 };
    std::vector<double> given = { 0.5 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::huber_loss, config_penalty, activation::method::linear, 0, 0);
    EXPECT_NEAR(deltas[0], -0.5, 1e-9);
  }

  // Case 3: Sign mismatch but target within neutral tolerance, target = 0.005, output = -0.5
  // use_direction_penalty is true
  {
    std::vector<double> targets = { 0.005 };
    std::vector<double> given = { -0.5 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::huber_loss, config_penalty, activation::method::linear, 0, 0);
    EXPECT_NEAR(deltas[0], -0.505, 1e-9);
  }

  // Case 4: Sign mismatch but use_direction_penalty is false, target = -1.0, output = 0.5
  {
    std::vector<double> targets = { -1.0 };
    std::vector<double> given = { 0.5 };
    layer.calculate_error_deltas(deltas, targets, given, ErrorCalculation::type::huber_loss, config_no_penalty, activation::method::linear, 0, 0);
    EXPECT_NEAR(deltas[0], 1.0, 1e-9);
  }
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

    auto neurons = MockLayer::create_neurons_exposed(dropout_rate, num_neurons, std::nullopt);
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
    const unsigned num_neurons = 5000;
    const double dropout_rate = 0.3;
    auto neurons = MockLayer::create_neurons_exposed(dropout_rate, num_neurons, std::nullopt);

    unsigned dropped = 0;
    for (unsigned i = 0; i < num_neurons; ++i) {
        if (neurons[i].must_randomly_drop(i)) {
            dropped++;
        }
    }
    
    double actual_rate = static_cast<double>(dropped) / num_neurons;
    EXPECT_NEAR(actual_rate, dropout_rate, 0.06); // Relaxed tolerance and larger sample size to prevent flakiness
}

TEST(LayerTest, SetNumberOfThreads) {
    MockLayer layer(1, 10);
    EXPECT_NO_THROW(layer.set_number_of_threads(4));
    
    FFLayer ff_layer(1, 2, 3, 0.0, Layer::Role::Hidden, activation(activation::method::relu, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
    EXPECT_NO_THROW(ff_layer.set_number_of_threads(2));
}

TEST(LayerTest, LayersSetNumberOfThreads) {
    std::vector<unsigned> topology = {3, 2, 1};
    auto options = NeuralNetworkOptions::create(topology)
      .with_number_of_threads(4)
      .build();
    Layers layers(options);

    EXPECT_NO_THROW(layers.set_number_of_threads(1));
}

TEST(LayerTest, CreateHiddenLayerRejectsLayerNormOnFFAndElman) {
    activation act(activation::method::tanh, 0.0, 1.0);

    LayerDetails ff_with_ln(Layer::Architecture::FF, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0);
    EXPECT_THROW(Layer::create_hidden_layer(1, 3, ff_with_ln, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);

    LayerDetails elman_with_ln(Layer::Architecture::Elman, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0);
    EXPECT_THROW(Layer::create_hidden_layer(1, 3, elman_with_ln, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);

    // Same architectures without the flag are unaffected.
    LayerDetails ff_without_ln(Layer::Architecture::FF, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0);
    EXPECT_NO_THROW(Layer::create_hidden_layer(1, 3, ff_without_ln, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerAcceptsLayerNormOnGruAndLstm) {
    activation act(activation::method::tanh, 0.0, 1.0);

    LayerDetails gru_with_ln(Layer::Architecture::Gru, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0);
    EXPECT_NO_THROW(Layer::create_hidden_layer(1, 3, gru_with_ln, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt));

    LayerDetails lstm_with_ln(Layer::Architecture::Lstm, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0);
    EXPECT_NO_THROW(Layer::create_hidden_layer(1, 3, lstm_with_ln, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerRejectsAttentionPoolOnNonRecurrentPrevious) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::AttentionPool, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 8, 0, 0, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);
    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::FF, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsAttentionPoolOnElman) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::AttentionPool, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 8, 0, 0, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Elman, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsAttentionPoolSizeMismatch) {
    activation act(activation::method::linear, 0.0);
    // previous (Gru) layer has 4 neurons, but the AttentionPool layer's own size is 5.
    LayerDetails ld(Layer::Architecture::AttentionPool, 5, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 8, 0, 0, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Gru, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsAttentionPoolWithLayerNorm) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::AttentionPool, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, true, 8, 0, 0, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Gru, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsAttentionPoolZeroHiddenSize) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::AttentionPool, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Lstm, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsAttentionPoolWithResidual) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::AttentionPool, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 8, 0, 0, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Gru, 0, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerAcceptsAttentionPoolAfterGruOrLstm) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::AttentionPool, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 8, 0, 0, 0, 0);

    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Gru, -1, nullptr, std::nullopt));
    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Lstm, -1, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerRejectsTcnZeroKernelSize) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::Tcn, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 1, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsTcnZeroDilation) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::Tcn, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 0, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsTcnWithLayerNorm) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::Tcn, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 3, 1, 0, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerAcceptsTcnAsFirstHiddenLayer) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::Tcn, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 1, 0, 0);

    // Unlike AttentionPool, Tcn has no previous-architecture restriction: it
    // may be the very first hidden layer (previous_layer_architecture == None)
    // or follow any architecture.
    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerAcceptsTcnAfterLstm) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::Tcn, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 1, 0, 0);

    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Lstm, -1, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerAcceptsTcnWithResidual) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::Tcn, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 1, 0, 0);

    // Unlike AttentionPool, Tcn accepts the external residual mechanism.
    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::FF, 0, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerAcceptsTcnChangingChannelWidth) {
    activation act(activation::method::linear, 0.0);
    // Tcn's own size (6) need not match the preceding layer's size (4) -
    // unlike AttentionPool, which enforces size preservation.
    LayerDetails ld(Layer::Architecture::Tcn, 6, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 1, 0, 0);

    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::FF, -1, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerRejectsSelfAttentionZeroNumberOfHeads) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::SelfAttention, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 8);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsSelfAttentionHeadsNotDividingSize) {
    activation act(activation::method::linear, 0.0);
    // size=5 is not evenly divisible by number_of_heads=2.
    LayerDetails ld(Layer::Architecture::SelfAttention, 5, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 2, 8);

    EXPECT_THROW(Layer::create_hidden_layer(2, 5, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsSelfAttentionZeroFeedForwardHiddenSize) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::SelfAttention, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 2, 0);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerRejectsSelfAttentionSizeMismatch) {
    activation act(activation::method::linear, 0.0);
    // previous layer has 4 neurons, but SelfAttention's own size is 6.
    LayerDetails ld(Layer::Architecture::SelfAttention, 6, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 2, 8);

    EXPECT_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::FF, -1, nullptr, std::nullopt), std::runtime_error);
}

TEST(LayerTest, CreateHiddenLayerAcceptsSelfAttentionAsFirstHiddenLayer) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::SelfAttention, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 2, 8);

    // Unlike AttentionPool, SelfAttention has no previous-architecture
    // restriction: it may be the very first hidden layer.
    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::None, -1, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerAcceptsSelfAttentionWithLayerNormalisation) {
    activation act(activation::method::linear, 0.0);
    // Deliberate divergence from AttentionPool, which panics on this flag.
    LayerDetails ld(Layer::Architecture::SelfAttention, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 2, 8);

    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::Lstm, -1, nullptr, std::nullopt));
}

TEST(LayerTest, CreateHiddenLayerAcceptsSelfAttentionWithResidual) {
    activation act(activation::method::linear, 0.0);
    LayerDetails ld(Layer::Architecture::SelfAttention, 4, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 2, 8);

    EXPECT_NO_THROW(Layer::create_hidden_layer(2, 4, ld, 1, true, Layer::Architecture::FF, 0, nullptr, std::nullopt));
}

TEST(LayerTest, TempBufferCorrectness)
{
  // 1. Basic properties and zero initialization
  {
    TempBuffer<double, 100> buf(10, true);
    EXPECT_EQ(buf.size(), 10);
    EXPECT_FALSE(buf.empty());
    EXPECT_NE(buf.data(), nullptr);
    for (size_t i = 0; i < 10; ++i)
    {
      EXPECT_DOUBLE_EQ(buf.data()[i], 0.0);
    }
  }

  // 2. Sizing and vector access
  {
    TempBuffer<double, 101> buf1(100);
    EXPECT_EQ(buf1.size(), 100);
    EXPECT_EQ(buf1.vec().size(), 100);
  }
  {
    TempBuffer<double, 101> buf2(200);
    EXPECT_EQ(buf2.size(), 200);
    EXPECT_EQ(buf2.vec().size(), 200);
  }
  {
    TempBuffer<double, 101> buf3(50);
    EXPECT_EQ(buf3.size(), 50);
    EXPECT_EQ(buf3.vec().size(), 50);
  }

  // 3. Buffer isolation
  {
    TempBuffer<double, 102> buf_tag1(10);
    TempBuffer<double, 103> buf_tag2(10);
    EXPECT_NE(buf_tag1.data(), buf_tag2.data());
  }

  // 4. Large size buffer
  {
    TempBuffer<double, 104> large_buf(1500000);
    EXPECT_EQ(large_buf.size(), 1500000);
    TempBuffer<double, 104> small_buf(10);
    EXPECT_NE(large_buf.data(), small_buf.data());
  }

  // 5. Re-assignment
  {
    TempBuffer<double, 105> buf(5);
    buf.assign(15, 3.14);
    EXPECT_EQ(buf.size(), 15);
    for (size_t i = 0; i < 15; ++i)
    {
      EXPECT_DOUBLE_EQ(buf.data()[i], 3.14);
    }
  }
}

TEST(LayerTest, LayersUpdateWeightsWorkloadThreshold) {
  auto options = NeuralNetworkOptions::create({ 2, 3, 2 });
  Layers layers(options);

  std::vector<std::vector<double>> inputs = { { 0.5, 0.2 }, { 0.1, -0.4 } };
  std::vector<std::vector<double>> outputs = { { 0.1, 0.9 }, { 0.8, 0.2 } };

  auto inputs_it = inputs.cbegin();
  auto outputs_it = outputs.cbegin();

  // Small batch size (batch_size = 2) triggers workload thresholding path cleanly during train()
  layers.train(options, 0.01, inputs_it, outputs_it, 2);
  
  EXPECT_EQ(layers.size(), 3);
}

TEST(LayerTest, LayersTrainRepeatedBatchBufferReuse) {
  auto options = NeuralNetworkOptions::create({ 3, 5, 2 });
  Layers layers(options);

  std::vector<std::vector<double>> inputs = {
    { 0.1, 0.2, 0.3 },
    { 0.4, 0.5, 0.6 },
    { 0.7, 0.8, 0.9 },
    { 0.2, 0.4, 0.6 }
  };
  std::vector<std::vector<double>> outputs = {
    { 1.0, 0.0 },
    { 0.0, 1.0 },
    { 0.5, 0.5 },
    { 0.2, 0.8 }
  };

  // Test batch_size = 1
  auto in_it1 = inputs.cbegin();
  auto out_it1 = outputs.cbegin();
  layers.train(options, 0.05, in_it1, out_it1, 1);

  // Test expanding batch_size = 4 (buffer grows and zeroes reused/new elements)
  auto in_it4 = inputs.cbegin();
  auto out_it4 = outputs.cbegin();
  layers.train(options, 0.05, in_it4, out_it4, 4);

  // Test shrinking batch_size = 2 (reuses existing buffer without reallocation)
  auto in_it2 = inputs.cbegin();
  auto out_it2 = outputs.cbegin();
  layers.train(options, 0.05, in_it2, out_it2, 2);

  // Verify layer weight values remain valid and finite after all batch iterations
  for (unsigned i = 1; i < layers.size(); ++i)
  {
    for (double w : layers[i].get_w_values())
    {
      EXPECT_TRUE(std::isfinite(w));
    }
  }
}

TEST(LayerTest, LayersTrainParallelWeightUpdateCorrectness) {
  auto options = NeuralNetworkOptions::create({ 4, 16, 8, 2 })
    .with_number_of_threads(4)
    .build();

  Layers layers_parallel(options);
  Layers layers_single(options);

  // Set matching initial weights for both
  for (unsigned i = 1; i < layers_parallel.size(); ++i)
  {
    layers_parallel[i].set_w_values(layers_single[i].get_w_values());
    layers_parallel[i].set_b_values(layers_single[i].get_b_values());
  }

  std::vector<std::vector<double>> inputs(16, std::vector<double>{ 0.1, 0.2, 0.3, 0.4 });
  std::vector<std::vector<double>> outputs(16, std::vector<double>{ 0.8, 0.2 });

  auto in_it_p = inputs.cbegin();
  auto out_it_p = outputs.cbegin();
  layers_parallel.train(options, 0.01, in_it_p, out_it_p, 16);

  auto in_it_s = inputs.cbegin();
  auto out_it_s = outputs.cbegin();
  layers_single.train(options, 0.01, in_it_s, out_it_s, 16);

  // Parallel and single-threaded training must produce identical weights
  for (unsigned i = 1; i < layers_parallel.size(); ++i)
  {
    const auto& w_p = layers_parallel[i].get_w_values();
    const auto& w_s = layers_single[i].get_w_values();
    ASSERT_EQ(w_p.size(), w_s.size());
    for (size_t j = 0; j < w_p.size(); ++j)
    {
      EXPECT_NEAR(w_p[j], w_s[j], 1e-12);
    }
  }
}

TEST(LayerTest, LayersTrainGradientClippingDisabledFastPath) {
  auto options = NeuralNetworkOptions::create({ 2, 4, 2 })
    .with_clip_threshold(std::numeric_limits<double>::infinity())
    .build();

  Layers layers(options);

  std::vector<std::vector<double>> inputs = { { 0.1, 0.9 }, { 0.8, 0.2 } };
  std::vector<std::vector<double>> outputs = { { 1.0, 0.0 }, { 0.0, 1.0 } };

  auto in_it = inputs.cbegin();
  auto out_it = outputs.cbegin();
  layers.train(options, 0.01, in_it, out_it, 2);

  for (unsigned i = 1; i < layers.size(); ++i)
  {
    for (double w : layers[i].get_w_values())
    {
      EXPECT_TRUE(std::isfinite(w));
    }
  }
}

TEST(LayerTest, LayersTrainRecurrentSequenceBackprop) {
  auto options = NeuralNetworkOptions::create({ 2, 4, 2 })
    .with_hidden_layers({ LayerDetails(Layer::Architecture::Elman, 4, activation(activation::method::sigmoid, 0.01), 0.0, 0.05, OptimiserType::SGD, 0.99, false, 0, 0, 0, 0, 0) })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(3)
    .build();

  Layers layers(options);

  std::vector<std::vector<double>> sequence_inputs = {
    { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }
  };
  std::vector<std::vector<double>> sequence_outputs = {
    { 0.9, 0.1 }
  };

  auto in_it = sequence_inputs.cbegin();
  auto out_it = sequence_outputs.cbegin();
  layers.train(options, 0.01, in_it, out_it, 1);

  for (unsigned i = 1; i < layers.size(); ++i)
  {
    for (double w : layers[i].get_w_values())
    {
      EXPECT_TRUE(std::isfinite(w));
    }
  }
}

TEST(LayerTest, LayerTypeIdentificationVirtuals) {
  auto elman_options = NeuralNetworkOptions::create({ 2, 4, 2 })
    .with_hidden_layers({ LayerDetails(Layer::Architecture::Elman, 4, activation(activation::method::sigmoid, 0.01), 0.0, 0.05, OptimiserType::SGD, 0.99, false, 0, 0, 0, 0, 0) })
    .build();
  Layers elman_layers(elman_options);

  auto ff_options = NeuralNetworkOptions::create({ 2, 4, 2 }).build();
  Layers ff_layers(ff_options);

  EXPECT_TRUE(ff_layers[1].is_ff_layer());
  EXPECT_TRUE(ff_layers[2].is_ff_layer());
  EXPECT_FALSE(elman_layers[1].is_ff_layer());
  EXPECT_FALSE(ff_layers[1].is_multi_output());
}


TEST(LayerTest, LayersGetTotalWeightsCaching) {
  auto options = NeuralNetworkOptions::create({ 3, 16, 2 }).build();
  Layers layers(options);

  size_t expected_weights = (3 * 16 + 16) + (16 * 2 + 2);
  EXPECT_EQ(layers.get_total_weights(), expected_weights);
  EXPECT_EQ(layers.get_total_weights(), expected_weights);
}

TEST(LayerTest, LayersAccessorsAndForwardFeedOptimizedPaths) {
  auto options = NeuralNetworkOptions::create({ 2, 4, 3 }).build();
  Layers layers(options);

  EXPECT_EQ(layers.input_layer().get_number_neurons(), 2U);
  EXPECT_EQ(layers.hidden_layer(1).get_number_neurons(), 4U);
  EXPECT_EQ(layers.output_layer().get_number_neurons(), 3U);

  std::vector<std::vector<double>> inputs = { { 0.5, 0.2 }, { 0.1, 0.9 } };
  std::vector<std::vector<double>> outputs = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } };

  auto in_it = inputs.cbegin();
  auto out_it = outputs.cbegin();
  layers.train(options, 0.05, in_it, out_it, 2);

  for (unsigned i = 1; i < layers.size(); ++i)
  {
    for (double w : layers[i].get_w_values())
    {
      EXPECT_TRUE(std::isfinite(w));
    }
  }
}

TEST(LayerTest, FFLayerCalculateAndStoreGradientsMathematicalSoundness) {
  const unsigned num_inputs = 3;
  const unsigned num_outputs = 4;
  const size_t batch_size = 4;
  const size_t num_time_steps = 2;

  std::vector<unsigned> topology = { num_inputs, num_outputs };
  auto options = NeuralNetworkOptions::create(topology).build();

  FFLayer layer(1, num_inputs, num_outputs, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::SGD, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  std::vector<GradientsAndOutputs> batch_gradients_and_outputs;
  batch_gradients_and_outputs.reserve(batch_size);
  for (size_t b = 0; b < batch_size; ++b)
  {
    batch_gradients_and_outputs.emplace_back(topology);
  }

  std::vector<HiddenStates> hidden_states;
  hidden_states.reserve(batch_size);
  for (size_t b = 0; b < batch_size; ++b)
  {
    hidden_states.emplace_back(topology);
    hidden_states[b].assign(1, num_time_steps, HiddenState(), 1);
  }

  // Populate deterministic inputs and gradients for each sample and timestep
  std::vector<std::vector<double>> inputs_data(batch_size * num_time_steps, std::vector<double>(num_inputs, 0.0));
  std::vector<std::vector<double>> grads_data(batch_size * num_time_steps, std::vector<double>(num_outputs, 0.0));

  for (size_t b = 0; b < batch_size; ++b)
  {
    std::vector<double> rnn_inputs(num_time_steps * num_inputs);
    std::vector<double> rnn_grads(num_time_steps * num_outputs);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const size_t idx = b * num_time_steps + t;
      for (size_t i = 0; i < num_inputs; ++i)
      {
        const double x_val = static_cast<double>(idx * 10 + i + 1) * 0.1;
        inputs_data[idx][i] = x_val;
        rnn_inputs[t * num_inputs + i] = x_val;
      }
      for (size_t j = 0; j < num_outputs; ++j)
      {
        const double g_val = static_cast<double>(idx * 5 + j + 2) * 0.05;
        grads_data[idx][j] = g_val;
        rnn_grads[t * num_outputs + j] = g_val;
      }
    }
    batch_gradients_and_outputs[b].set_rnn_outputs(0, rnn_inputs);
    batch_gradients_and_outputs[b].set_rnn_gradients(1, rnn_grads);
  }

  // Execute optimized gradient calculation
  MockLayer prev_layer(0, num_inputs);
  layer.calculate_and_store_gradients(batch_gradients_and_outputs, hidden_states, prev_layer, batch_size, 1);

  // Compute reference gradients via mathematical definition: dW_ij = (1/B) * sum_{b,t} (x_i * g_j)
  std::vector<double> expected_w_grads(num_inputs * num_outputs, 0.0);
  std::vector<double> expected_b_grads(num_outputs, 0.0);

  const double inv_batch = 1.0 / static_cast<double>(batch_size);

  for (size_t b = 0; b < batch_size; ++b)
  {
    for (size_t t = 0; t < num_time_steps; ++t)
    {
      const size_t idx = b * num_time_steps + t;
      for (size_t i = 0; i < num_inputs; ++i)
      {
        for (size_t j = 0; j < num_outputs; ++j)
        {
          expected_w_grads[i * num_outputs + j] += inputs_data[idx][i] * grads_data[idx][j];
        }
      }
      for (size_t j = 0; j < num_outputs; ++j)
      {
        expected_b_grads[j] += grads_data[idx][j];
      }
    }
  }

  for (size_t k = 0; k < expected_w_grads.size(); ++k)
  {
    expected_w_grads[k] *= inv_batch;
  }
  for (size_t j = 0; j < expected_b_grads.size(); ++j)
  {
    expected_b_grads[j] *= inv_batch;
  }

  // Verify mathematical equivalence to double precision floating-point tolerance
  const auto& actual_w_grads = layer.get_w_grads();
  const auto& actual_b_grads = layer.get_b_grads();

  ASSERT_EQ(actual_w_grads.size(), expected_w_grads.size());
  for (size_t k = 0; k < expected_w_grads.size(); ++k)
  {
    EXPECT_NEAR(actual_w_grads[k], expected_w_grads[k], 1e-14);
  }

  ASSERT_EQ(actual_b_grads.size(), expected_b_grads.size());
  for (size_t j = 0; j < expected_b_grads.size(); ++j)
  {
    EXPECT_NEAR(actual_b_grads[j], expected_b_grads[j], 1e-14);
  }
}

TEST(LayerTest, LayersTrainCoverageAndConsistencyAcrossBatchSizes) {
  auto options = NeuralNetworkOptions::create({ 4, 8, 3 }).build();
  Layers layers(options);

  std::vector<std::vector<double>> inputs = {
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.6, 0.7, 0.8 },
    { 0.9, 0.1, 0.2, 0.3 },
    { 0.4, 0.5, 0.6, 0.7 }
  };
  std::vector<std::vector<double>> outputs = {
    { 1.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 },
    { 0.5, 0.5, 0.0 }
  };

  // Test single sample batch
  auto in_it1 = inputs.cbegin();
  auto out_it1 = outputs.cbegin();
  layers.train(options, 0.01, in_it1, out_it1, 1);

  // Test multi-sample batch
  auto in_it4 = inputs.cbegin();
  auto out_it4 = outputs.cbegin();
  layers.train(options, 0.01, in_it4, out_it4, 4);

  // Verify numerical stability
  for (unsigned i = 1; i < layers.size(); ++i)
  {
    for (double w : layers[i].get_w_values())
    {
      EXPECT_TRUE(std::isfinite(w));
    }
  }
}

TEST(LayerTest, LayersTrainPerformanceFixCoverage)
{
  MYODDWEB_PROFILE_FUNCTION("LayerTest");
  auto options = NeuralNetworkOptions::create({ 8, 16, 12, 4 });
  Layers layers(options);

  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> outputs;
  for (size_t i = 0; i < 16; ++i)
  {
    std::vector<double> in_vec(8, 0.1 * static_cast<double>(i + 1));
    std::vector<double> out_vec(4, 0.05 * static_cast<double>(i + 1));
    inputs.push_back(in_vec);
    outputs.push_back(out_vec);
  }

  // Ensure train runs without copying bottleneck or invalid state
  auto in_it = inputs.cbegin();
  auto out_it = outputs.cbegin();
  for (int iter = 0; iter < 10; ++iter)
  {
    in_it = inputs.cbegin();
    out_it = outputs.cbegin();
    layers.train(options, 0.01, in_it, out_it, 16);
  }

  for (unsigned i = 1; i < layers.size(); ++i)
  {
    for (double w : layers[i].get_w_values())
    {
      EXPECT_TRUE(std::isfinite(w));
    }
  }
}

TEST(LayerTest, LayersTrainMathematicalSoundnessMultiLayerRecurrentGradientFlow)
{
  MYODDWEB_PROFILE_FUNCTION("LayerTest");
  // Build a multi-layer network with a recurrent layer preceding an output layer
  LayerDetails ff_detail(Layer::Architecture::FF, 6, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0);
  LayerDetails elman_detail(Layer::Architecture::Elman, 6, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0);

  auto options = NeuralNetworkOptions::create({ 4, 6, 6, 2 })
    .with_hidden_layers({ ff_detail, elman_detail })
    .with_enable_bptt(true)
    .with_bptt_max_ticks(2)
    .with_learning_rate(0.05);

  Layers layers(options);

  const auto w1_before = layers[1].get_w_values();

  std::vector<std::vector<double>> inputs = {
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.6, 0.7, 0.8 }
  };
  std::vector<std::vector<double>> outputs = {
    { 1.0, 0.0 },
    { 0.0, 1.0 }
  };

  auto in_it = inputs.cbegin();
  auto out_it = outputs.cbegin();
  layers.train(options, 0.05, in_it, out_it, 2);

  // Check mathematical soundness: weights of preceding layer (Layer 1) must be updated by backprop
  const auto w1_after = layers[1].get_w_values();
  ASSERT_EQ(w1_before.size(), w1_after.size());

  bool weights_changed = false;
  for (size_t i = 0; i < w1_before.size(); ++i)
  {
    EXPECT_TRUE(std::isfinite(w1_after[i]));
    if (std::abs(w1_after[i] - w1_before[i]) > 1e-12)
    {
      weights_changed = true;
    }
  }
  EXPECT_TRUE(weights_changed);
}


TEST(LayerTest, LayersTrainAsymmetricLayerSizesNoBufferOverflow)
{
  MYODDWEB_PROFILE_FUNCTION("LayerTest");
  // Test topology where N_prev (52) > N_this (24), matching Florent's scenario
  const std::vector<Layer::Architecture> architectures = {
    Layer::Architecture::Gru,
    Layer::Architecture::Elman,
    Layer::Architecture::Lstm
  };

  for (auto arch : architectures)
  {
    LayerDetails hidden_detail(arch, 24, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0);
    auto options = NeuralNetworkOptions::create({ 52, 24, 10 })
      .with_hidden_layers({ hidden_detail })
      .with_enable_bptt(true)
      .with_bptt_max_ticks(2)
      .with_learning_rate(0.01);

    Layers layers(options);

    std::vector<std::vector<double>> inputs = {
      std::vector<double>(52, 0.1),
      std::vector<double>(52, 0.2)
    };
    std::vector<std::vector<double>> outputs = {
      std::vector<double>(10, 0.5),
      std::vector<double>(10, 0.8)
    };

    auto in_it = inputs.cbegin();
    auto out_it = outputs.cbegin();
    // Must execute train without crashing or overflowing buffer
    EXPECT_NO_THROW(layers.train(options, 0.01, in_it, out_it, 2));

    for (unsigned i = 1; i < layers.size(); ++i)
    {
      for (double w : layers[i].get_w_values())
      {
        EXPECT_TRUE(std::isfinite(w));
      }
    }
  }
}

TEST(LayerTest, ResidualProjectorProjectBatchIntoEquivalence)
{
  MYODDWEB_PROFILE_FUNCTION("LayerTest");
  ResidualProjector projector(4, 6, activation(activation::method::relu, 0.0), 0.0, std::nullopt);

  std::vector<std::vector<double>> inputs = {
    { 0.1, 0.2, 0.3, 0.4 },
    { 0.5, 0.6, 0.7, 0.8 }
  };

  std::vector<const double*> raw_inputs = {
    inputs[0].data(),
    inputs[1].data()
  };

  auto expected = projector.project_batch(raw_inputs);

  std::vector<std::vector<double>> actual;
  projector.project_batch_into(raw_inputs, actual);

  ASSERT_EQ(expected.size(), actual.size());
  for (size_t b = 0; b < expected.size(); ++b)
  {
    ASSERT_EQ(expected[b].size(), actual[b].size());
    for (size_t i = 0; i < expected[b].size(); ++i)
    {
      EXPECT_DOUBLE_EQ(expected[b][i], actual[b][i]);
    }
  }
}

TEST(LayerTest, LayersForwardFeedResidualLeakDoesNotContaminateNonResidualLayer)
{
  MYODDWEB_PROFILE_FUNCTION("LayerTest");
  // Regression test: Layers::calculate_forward_feed reuses a thread_local
  // scratch buffer for residual projections. Layer 2 (the output layer) has
  // no residual connection of its own, but it immediately follows layer 1,
  // which does. Both layers share the same width here, so if the thread_local
  // buffer were not cleared for layers without a residual projector, layer 2
  // would silently pick up layer 1's leftover residual-projected values.
  const unsigned width = 4;

  auto output_layer = OutputLayerDetails(
    width,
    activation(activation::method::linear, 0.0),
    ErrorCalculation::type::mse,
    { 0.0, 0.0, 1.0, 0.0, false, 1.0 },
    0.0,
    OptimiserType::None,
    0.0);

  LayerDetails hidden_detail(Layer::Architecture::FF, width, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0);

  auto options = NeuralNetworkOptions::create({ 2, width, width })
    .with_hidden_layers({ hidden_detail })
    .with_output_layer_details(output_layer)
    .with_residual_layer_jump(1);

  Layers layers(options);

  // Zero every real weight so the only possible nonzero contribution to the
  // output layer's pre-activation sum is a leaked residual value.
  layers[1].set_w_values(std::vector<double>(2 * width, 0.0));
  layers[1].set_b_values({ 1.0, 2.0, 3.0, 4.0 });
  layers[2].set_w_values(std::vector<double>(width * width, 0.0));
  const std::vector<double> output_bias = { 10.0, 20.0, 30.0, 40.0 };
  layers[2].set_b_values(output_bias);

  const auto result = layers.think(options, std::vector<double>{ 0.5, -0.25 });

  ASSERT_EQ(result.size(), width);
  for (size_t i = 0; i < width; ++i)
  {
    EXPECT_DOUBLE_EQ(result[i], output_bias[i]);
  }
}








