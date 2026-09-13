#include <gtest/gtest.h>
#include "layers/ffoutputlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>


using namespace myoddweb::nn;
using namespace test_helper;

class FFOutputLayerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(FFOutputLayerTest, ConstructorAndClone) {
    unsigned num_inputs = 4;
    unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::Adam, 0.9)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);

    EXPECT_EQ(layer.get_layer_index(), 1);
    EXPECT_EQ(layer.get_number_input_neurons(), num_inputs);
    EXPECT_EQ(layer.get_number_output_neurons(), num_outputs);
    EXPECT_EQ(layer.get_layer_architecture(), Layer::Architecture::FF);
    EXPECT_EQ(layer.get_layer_role(), Layer::Role::Output);
    EXPECT_EQ(layer.get_pre_activation_multiplier(), 1);

    std::unique_ptr<Layer> cloned(layer.clone());
    EXPECT_EQ(cloned->get_layer_index(), 1);
    EXPECT_EQ(cloned->get_number_input_neurons(), num_inputs);
    EXPECT_EQ(cloned->get_number_output_neurons(), num_outputs);
    EXPECT_EQ(cloned->get_pre_activation_multiplier(), 1);
}

TEST_F(FFOutputLayerTest, CalculateOutputGradientsMSE) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_hs[0].at(1, 0).set_hidden_state_values({ 0.8, 0.4 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.8, 0.4 });
    batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0 });

    std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], -0.1, 1e-9);
    EXPECT_NEAR(grads[1], 0.2, 1e-9);
}

TEST_F(FFOutputLayerTest, CalculateOutputGradientsBCE) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 1;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::bce_loss, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_hs[0].at(1, 0).set_hidden_state_values({ 0.5 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.0 });
    batch_hs[0].at(1, 0).set_cell_state_values({ 1.0 });

    std::vector<std::vector<double>> targets = { { 1.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], -0.5, 1e-9);
}

TEST_F(FFOutputLayerTest, CalculateOutputGradientsCE) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::softmax, 0.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_hs[0].at(1, 0).set_hidden_state_values({ 0.5, 0.5 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.0, 0.0 });
    batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0 });

    std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], -0.5, 1e-9);
    EXPECT_NEAR(grads[1], 0.5, 1e-9);
}

TEST_F(FFOutputLayerTest, SharpeRatioLossDoesNotCoupleTransactionCostAcrossExamples) {
    // Regression test for the batch-statistics builder coupling transaction cost across unrelated
    // training examples: example 0 has a large position swing between its two time steps
    // (1.0 -> -1.0), example 1's position never moves (1.0 -> 1.0, so its own cost is always 0).
    // With the fix, example 1's first step return is 1.0*0.03 = 0.03 exactly; the old flattened
    // computation would have wrongly charged it a transaction cost against example 0's last
    // position, giving 0.028 instead and a different (wrong) gradient everywhere.
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    const double c = 0.001;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::sharpe_ratio_loss,
          EvaluationConfig(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, c, 0.0), 0.0, OptimiserType::None, 0.0)
    };

    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);

    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 2);

    // Example 0: positions 1.0 -> -1.0 (large swing, incurs its own transaction cost at t=1).
    batch_hs[0].at(1, 0).set_hidden_state_values({ 1.0 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 1.0 });
    batch_hs[0].at(1, 0).set_cell_state_values({ 1.0 });
    batch_hs[0].at(1, 1).set_hidden_state_values({ -1.0 });
    batch_hs[0].at(1, 1).set_pre_activation_sums({ -1.0 });
    batch_hs[0].at(1, 1).set_cell_state_values({ 1.0 });

    // Example 1: positions 1.0 -> 1.0 (no swing at all, own transaction cost is always 0).
    batch_hs[1].at(1, 0).set_hidden_state_values({ 1.0 });
    batch_hs[1].at(1, 0).set_pre_activation_sums({ 1.0 });
    batch_hs[1].at(1, 0).set_cell_state_values({ 1.0 });
    batch_hs[1].at(1, 1).set_hidden_state_values({ 1.0 });
    batch_hs[1].at(1, 1).set_pre_activation_sums({ 1.0 });
    batch_hs[1].at(1, 1).set_cell_state_values({ 1.0 });

    std::vector<std::vector<double>> targets = { { 0.02, -0.01 }, { 0.03, 0.01 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 2);

    const auto& grads_0 = batch_go[0].get_rnn_gradients(1);
    const auto& grads_1 = batch_go[1].get_rnn_gradients(1);
    ASSERT_EQ(grads_0.size(), 2u);
    ASSERT_EQ(grads_1.size(), 2u);

    // Hand-derived from the correct per-example return sequences [0.02, 0.008] and [0.03, 0.01]
    // (batch mean 0.017, sigma over the pooled 4 returns): see the finite-difference-verified
    // formula in Layer::calculate_sharpe_ratio_loss_error_deltas.
    EXPECT_NEAR(grads_0[0], -0.10730054771020339, 1e-6);
    EXPECT_NEAR(grads_0[1], 0.7659038626677686, 1e-6);
    EXPECT_NEAR(grads_1[0], 1.598408043170035, 1e-6);
    EXPECT_NEAR(grads_1[1], -0.7252036579521893, 1e-6);
}

TEST_F(FFOutputLayerTest, DropoutStatisticalVerification) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 5000;
    double dropout_rate = 0.5;
    
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    std::vector<Neuron> neurons;
    for (unsigned i = 0; i < num_outputs; ++i) {
        neurons.emplace_back(i, Neuron::Type::Dropout, dropout_rate, std::nullopt);
    }

    FFOutputLayer layer(
        1, details, num_inputs, num_outputs, neurons,
        std::vector<double>(num_inputs * num_outputs, 1.0),
        std::vector<double>(num_inputs * num_outputs, 0.0),
        {}, {}, {}, {}, std::vector<double>(num_inputs * num_outputs, 0.0),
        std::vector<double>(num_outputs, 0.0),
        std::vector<double>(num_outputs, 0.0),
        {}, {}, {}, {}, std::vector<double>(num_outputs, 0.0),
        1
    );

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_outputs(1);
    int dropped_count = 0;
    int kept_count = 0;
    for (double out : outputs) {
        if (out == 0.0) dropped_count++;
        else if (approx_equal(out, 1.0 / (1.0 - dropout_rate))) kept_count++;
    }

    EXPECT_EQ(dropped_count + kept_count, (int)num_outputs);
    EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.08);
}

TEST_F(FFOutputLayerTest, DropoutNotInference) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;

    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    std::vector<Neuron> neurons;
    for (unsigned i = 0; i < num_outputs; ++i) {
        neurons.emplace_back(i, Neuron::Type::Dropout, dropout_rate, std::nullopt);
    }

    FFOutputLayer layer(
        1, details, num_inputs, num_outputs, neurons,
        std::vector<double>(num_inputs * num_outputs, 1.0),
        std::vector<double>(num_inputs * num_outputs, 0.0),
        {}, {}, {}, {}, std::vector<double>(num_inputs * num_outputs, 0.0),
        std::vector<double>(num_outputs, 0.0),
        std::vector<double>(num_outputs, 0.0),
        {}, {}, {}, {}, std::vector<double>(num_outputs, 0.0),
        1
    );

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    const auto& outputs = batch_go[0].get_outputs(1);
    for (double out : outputs) {
        EXPECT_NEAR(out, 1.0, 1e-9);
    }
}

TEST_F(FFOutputLayerTest, DropoutConsistencyVerification) {
    // 1 neuron with 100% dropout
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    double dropout_rate = 1.0;

    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    std::vector<Neuron> neurons;
    neurons.emplace_back(0, Neuron::Type::Dropout, dropout_rate, std::nullopt);

    FFOutputLayer layer(
        1, details, num_inputs, num_outputs, neurons,
        { 1.0 }, { 0.0 }, {}, {}, {}, {}, { 0.0 },
        { 0.0 }, { 0.0 }, {}, {}, {}, {}, { 0.0 },
        1
    );

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    // Forward pass: should drop (output 0.0)
    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);
    EXPECT_NEAR(batch_go[0].get_outputs(1)[0], 0.0, 1e-9);

    // Backward pass: gradient should also be 0.0
    std::vector<std::vector<double>> targets = { { 1.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    // The gradient should be 0.0 because the neuron was dropped.
    EXPECT_NEAR(batch_go[0].get_gradients(1)[0], 0.0, 1e-9);
}

TEST_F(FFOutputLayerTest, DropoutWithTanhActivationDerivative) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 200;
    double dropout_rate = 0.5;

    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::tanh, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    std::vector<Neuron> neurons;
    for (unsigned i = 0; i < num_outputs; ++i) {
        neurons.emplace_back(i, Neuron::Type::Dropout, dropout_rate, std::nullopt);
    }

    FFOutputLayer layer(
        1, details, num_inputs, num_outputs, neurons,
        std::vector<double>(num_inputs * num_outputs, 1.0),
        std::vector<double>(num_inputs * num_outputs, 0.0),
        {}, {}, {}, {}, std::vector<double>(num_inputs * num_outputs, 0.0),
        std::vector<double>(num_outputs, 0.0),
        std::vector<double>(num_outputs, 0.0),
        {}, {}, {}, {}, std::vector<double>(num_outputs, 0.0),
        1
    );

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    std::vector<std::vector<double>> targets = { std::vector<double>(num_outputs, 0.0) };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    const auto& grads = batch_go[0].get_gradients(1);
    const double tanh_val = std::tanh(1.0);
    const double scale = 1.0 / (1.0 - dropout_rate);
    // dE/dy = (y - target) / N = (scale * tanh_val) / N for a kept neuron;
    // dE/dz = dE/dy * dy/dz = dE/dy * (scale * tanh'(z)).
    const double expected_kept_grad = (scale * tanh_val / static_cast<double>(num_outputs)) * (1.0 - tanh_val * tanh_val) * scale;

    int kept_count = 0;
    int dropped_count = 0;
    for (size_t j = 0; j < num_outputs; ++j)
    {
        if (grads[j] == 0.0)
        {
            dropped_count++;
        }
        else
        {
            kept_count++;
            EXPECT_GT(grads[j], 0.0);
            EXPECT_NEAR(grads[j], expected_kept_grad, 1e-9);
        }
    }

    EXPECT_GT(kept_count, 0);
    EXPECT_GT(dropped_count, 0);
    EXPECT_EQ(kept_count + dropped_count, static_cast<int>(num_outputs));
}

TEST_F(FFOutputLayerTest, MultiHeadOutput) {
    unsigned num_inputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0),
        OutputLayerDetails(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::bce_loss, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    unsigned num_outputs = 2;
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_hs[0].at(1, 0).set_hidden_state_values({ 0.8, 0.5 });
    batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.8, 0.0 });
    batch_hs[0].at(1, 0).set_cell_state_values({ 1.0, 1.0 });

    std::vector<std::vector<double>> targets = { { 1.0, 1.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    const auto grads = batch_go[0].get_gradients(1);
    EXPECT_NEAR(grads[0], -0.2, 1e-9);
    EXPECT_NEAR(grads[1], -0.5, 1e-9);
}

TEST_F(FFOutputLayerTest, CalculateOutputMetrics) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 1;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);

    std::vector<std::vector<double>> predictions = { { 0.8 }, { 0.4 } };
    std::vector<std::vector<double>> targets = { { 1.0 }, { 0.0 } };

    auto metrics = layer.calculate_output_metrics({ ErrorCalculation::type::mse }, targets, predictions);

    EXPECT_EQ(metrics.size(), 1); 
    EXPECT_EQ(metrics[0].size(), 1);

    EXPECT_NEAR((double)metrics[0][0].error(), 0.1, 1e-9);
}

TEST_F(FFOutputLayerTest, AllActivationTypes) {
    std::vector<activation::method> methods = {
        activation::method::linear,
        activation::method::sigmoid,
        activation::method::relu,
        activation::method::tanh,
        activation::method::leakyRelu,
        activation::method::PRelu,
        activation::method::selu,
        activation::method::swish,
        activation::method::mish,
        activation::method::gelu,
        activation::method::elu,
        activation::method::softmax
    };

    unsigned num_inputs = 1;
    unsigned num_outputs = 1;

    for (auto m : methods) {
        ErrorCalculation::type err = (m == activation::method::softmax) ? ErrorCalculation::type::cross_entropy : ErrorCalculation::type::mse;
        std::vector<OutputLayerDetails> details = {
            OutputLayerDetails(num_outputs, activation(m, 0.1), err, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
        };
        FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
        
        std::vector<unsigned> topology = { num_inputs, num_outputs };
        auto batch_go = create_batch_gradients_and_outputs(topology, 1);
        auto batch_hs = create_batch_hidden_states(topology, 1, 1);
        
        batch_hs[0].at(1, 0).set_hidden_state_values({ 0.5 });
        batch_hs[0].at(1, 0).set_pre_activation_sums({ 0.5 });
        batch_hs[0].at(1, 0).set_cell_state_values({ 1.0 });
        
        std::vector<std::vector<double>> targets = { { 1.0 } };
        EXPECT_NO_THROW(layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1));
        
        double grad = batch_go[0].get_gradients(1)[0];
        EXPECT_TRUE(std::isfinite(grad));
    }
}

TEST_F(FFOutputLayerTest, GetMomentum) {
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.1),
        OutputLayerDetails(3, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.5)
    };
    FFOutputLayer layer(1, details, 1, 5, 1, true, std::nullopt);

    EXPECT_DOUBLE_EQ(layer.get_momentum(0), 0.1);
    EXPECT_DOUBLE_EQ(layer.get_momentum(1), 0.1);
    EXPECT_DOUBLE_EQ(layer.get_momentum(2), 0.5);
    EXPECT_DOUBLE_EQ(layer.get_momentum(4), 0.5);
}

TEST_F(FFOutputLayerTest, ApplyStoredGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);

    layer.set_w_values({ 1.0 });
    layer.set_w_grads({ 0.1 });
    layer.apply_stored_gradients(0.1, 1.0); 

    EXPECT_NEAR(layer.get_w_values()[0], 0.99, 1e-9);
    EXPECT_DOUBLE_EQ(layer.get_w_grads()[0], 0.0);
}

TEST_F(FFOutputLayerTest, LearningRateRobustness) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);

    std::vector<double> learning_rates = { 0.0, 0.0001, 0.01, 0.5, 1.0, 2.0 };
    
    for (double lr : learning_rates) {
        double initial_w = 1.0;
        double initial_b = 0.5;
        layer.set_w_values({ initial_w });
        layer.set_b_values({ initial_b });
        
        double w_grad = 0.1;
        double b_grad = 0.05;
        layer.set_w_grads({ w_grad });
        layer.set_b_grads({ b_grad });

        layer.apply_stored_gradients(lr, 1.0);

        double expected_w = initial_w - lr * w_grad;
        double expected_b = initial_b - lr * b_grad;
        
        EXPECT_NEAR(layer.get_w_values()[0], expected_w, 1e-9);
        EXPECT_NEAR(layer.get_b_values()[0], expected_b, 1e-9);
    }
}

TEST_F(FFOutputLayerTest, SequentialGradients) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 2); 

    batch_go[0].set_rnn_outputs(0, { 1.0, 2.0 });
    batch_go[0].set_rnn_gradients(1, { 0.5, 0.3 });

    layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 0);

    EXPECT_NEAR(layer.get_w_grads()[0], 1.1, 1e-9);
    EXPECT_NEAR(layer.get_b_grads()[0], 0.8, 1e-9);
}

TEST_F(FFOutputLayerTest, ForwardFeed) {
    unsigned num_inputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0),
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    unsigned num_outputs = 2;
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);

    layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 });
    layer.set_b_values({ 0.0, 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    batch_go[0].set_outputs(0, { 0.5, -0.2 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);

    EXPECT_NEAR(batch_go[0].get_output(1, 0), 0.5, 1e-9);
    EXPECT_NEAR(batch_go[0].get_output(1, 1), -0.2, 1e-9);
}

TEST_F(FFOutputLayerTest, IterativeSoftmaxTraining) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    
    layer.set_w_values({ 0.1, -0.1, -0.1, 0.1 });
    layer.set_b_values({ 0.0, 0.0 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);

    std::vector<double> input_vals = { 1.0, 0.5 };
    batch_go[0].set_outputs(0, input_vals);
    std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };

    double initial_error = 0.0;

    for (int iter = 0; iter < 5; ++iter) {
        layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

        auto out_span = batch_go[0].get_outputs(1);
        std::vector<std::vector<double>> predictions = { std::vector<double>(out_span.begin(), out_span.end()) };
        auto metrics = layer.calculate_output_metrics({ ErrorCalculation::type::cross_entropy }, targets, predictions);
        double current_error = (double)metrics[0][0].error();

        if (iter == 0) {
            initial_error = current_error;
        }

        layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);
        layer.calculate_and_store_gradients(batch_go, batch_hs, prev_layer, 1, 0);

        layer.apply_stored_gradients(0.5, 1.0); 
    }

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, false);
    auto final_out_span = batch_go[0].get_outputs(1);
    std::vector<double> final_outputs(final_out_span.begin(), final_out_span.end());

    EXPECT_GT(final_outputs[0], final_outputs[1]);

    std::vector<std::vector<double>> final_predictions = { final_outputs };
    auto final_metrics = layer.calculate_output_metrics({ ErrorCalculation::type::cross_entropy }, targets, final_predictions);
    EXPECT_LT(final_metrics[0][0].error(), initial_error);
}

TEST_F(FFOutputLayerTest, StateAndMemoryAllocationOptimizationVerification) {
    unsigned num_inputs = 2;
    unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::relu, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);

    layer.set_w_values({ 0.1, 0.2, 0.3, 0.4 });
    layer.set_b_values({ 0.05, 0.15 });

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 2);
    auto batch_hs = create_batch_hidden_states(topology, 2, 3); // 3 steps

    // Batch 0: [[1.0, 0.5], [-0.5, 1.0], [0.0, 0.0]]
    // Batch 1: [[0.5, -0.5], [1.0, 1.0], [-1.0, 0.5]]
    batch_go[0].set_rnn_outputs(0, { 1.0, 0.5, -0.5, 1.0, 0.0, 0.0 });
    batch_go[1].set_rnn_outputs(0, { 0.5, -0.5, 1.0, 1.0, -1.0, 0.5 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 2, false);

    // Verify outputs
    const auto& outputs_0 = batch_go[0].get_rnn_outputs(1);
    const auto& outputs_1 = batch_go[1].get_rnn_outputs(1);

    ASSERT_EQ(outputs_0.size(), 6);
    ASSERT_EQ(outputs_1.size(), 6);

    // t=0, Batch 0:
    // pre_act[0] = 1.0 * 0.1 + 0.5 * 0.3 + 0.05 = 0.3 -> relu -> 0.3
    // pre_act[1] = 1.0 * 0.2 + 0.5 * 0.4 + 0.15 = 0.55 -> relu -> 0.55
    EXPECT_NEAR(outputs_0[0], 0.3, 1e-9);
    EXPECT_NEAR(outputs_0[1], 0.55, 1e-9);

    // Verify non-zero/retention propagate correctly
    EXPECT_NEAR(outputs_0[4], 0.05, 1e-9); // relu(0.0 * 0.1 + 0.0 * 0.3 + 0.05) = 0.05
    EXPECT_NEAR(outputs_0[5], 0.15, 1e-9); // relu(0.0 * 0.2 + 0.0 * 0.4 + 0.15) = 0.15
}

TEST_F(FFOutputLayerTest, TransposedWeightsCacheUpdateOnApplyGradients)
{
    const unsigned num_inputs = 2;
    const unsigned num_outputs = 3;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, false, std::nullopt);
    layer.set_w_values({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    const auto& w_t_initial = layer.get_w_values_T();
    ASSERT_EQ(w_t_initial.size(), 6);
    // W is [2 x 3]:
    // row 0: 1, 2, 3
    // row 1: 4, 5, 6
    // W_T is [3 x 2]:
    // row 0: 1, 4
    // row 1: 2, 5
    // row 2: 3, 6
    EXPECT_NEAR(w_t_initial[0], 1.0, 1e-9);
    EXPECT_NEAR(w_t_initial[1], 4.0, 1e-9);
    EXPECT_NEAR(w_t_initial[2], 2.0, 1e-9);
    EXPECT_NEAR(w_t_initial[3], 5.0, 1e-9);
    EXPECT_NEAR(w_t_initial[4], 3.0, 1e-9);
    EXPECT_NEAR(w_t_initial[5], 6.0, 1e-9);

    // Apply gradient update
    layer.set_w_grads({ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });
    layer.apply_stored_gradients(1.0, 1.0);

    const auto& w_updated = layer.get_w_values();
    EXPECT_NEAR(w_updated[0], 0.5, 1e-9);
    EXPECT_NEAR(w_updated[1], 1.5, 1e-9);
    EXPECT_NEAR(w_updated[2], 2.5, 1e-9);
    EXPECT_NEAR(w_updated[3], 3.5, 1e-9);
    EXPECT_NEAR(w_updated[4], 4.5, 1e-9);
    EXPECT_NEAR(w_updated[5], 5.5, 1e-9);

    const auto& w_t_updated = layer.get_w_values_T();
    EXPECT_NEAR(w_t_updated[0], 0.5, 1e-9);
    EXPECT_NEAR(w_t_updated[1], 3.5, 1e-9);
    EXPECT_NEAR(w_t_updated[2], 1.5, 1e-9);
    EXPECT_NEAR(w_t_updated[3], 4.5, 1e-9);
    EXPECT_NEAR(w_t_updated[4], 2.5, 1e-9);
    EXPECT_NEAR(w_t_updated[5], 5.5, 1e-9);
}

TEST_F(FFOutputLayerTest, DropoutMultiTimestepGradientFlow)
{
    const unsigned num_inputs = 1;
    const unsigned num_outputs = 50;
    const size_t num_time_steps = 3;
    const double dropout_rate = 0.5;

    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::tanh, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    std::vector<Neuron> neurons;
    for (unsigned i = 0; i < num_outputs; ++i)
    {
        neurons.emplace_back(i, Neuron::Type::Dropout, dropout_rate, std::nullopt);
    }

    FFOutputLayer layer(
        1, details, num_inputs, num_outputs, neurons,
        std::vector<double>(num_inputs * num_outputs, 1.0),
        std::vector<double>(num_inputs * num_outputs, 0.0),
        {}, {}, {}, {}, std::vector<double>(num_inputs * num_outputs, 0.0),
        std::vector<double>(num_outputs, 0.0),
        std::vector<double>(num_outputs, 0.0),
        {}, {}, {}, {}, std::vector<double>(num_outputs, 0.0),
        1
    );

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, num_time_steps);

    std::vector<double> rnn_inputs(num_time_steps * num_inputs, 1.0);
    batch_go[0].set_rnn_outputs(0, rnn_inputs.data(), rnn_inputs.size());

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    std::vector<std::vector<double>> targets = { std::vector<double>(num_time_steps * num_outputs, 0.0) };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);

    const auto& rnn_grads = batch_go[0].get_rnn_gradients(1);
    ASSERT_EQ(rnn_grads.size(), num_time_steps * num_outputs);

    const double tanh_val = std::tanh(1.0);
    const double scale = 1.0 / (1.0 - dropout_rate);
    const double expected_kept_grad = (scale * tanh_val / static_cast<double>(num_outputs)) * (1.0 - tanh_val * tanh_val) * scale;

    for (size_t t = 0; t < num_time_steps; ++t)
    {
        int kept_count = 0;
        int dropped_count = 0;
        for (size_t j = 0; j < num_outputs; ++j)
        {
            const double grad = rnn_grads[t * num_outputs + j];
            if (grad == 0.0)
            {
                dropped_count++;
            }
            else
            {
                kept_count++;
                EXPECT_NEAR(grad, expected_kept_grad, 1e-9);
            }
        }
        EXPECT_GT(kept_count, 0);
        EXPECT_GT(dropped_count, 0);
        EXPECT_EQ(kept_count + dropped_count, static_cast<int>(num_outputs));
    }
}

TEST_F(FFOutputLayerTest, MultithreadedForwardWithResidualConsolidationEquivalence)
{
    const unsigned num_inputs = 16;
    const unsigned num_outputs = 64;
    const unsigned batch_size = 200;
    const unsigned num_time_steps = 8;
    const unsigned num_threads = 4;

    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    FFOutputLayer layer_st(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    FFOutputLayer layer_mt(1, details, num_inputs, num_outputs, num_threads, true, std::nullopt);

    std::vector<double> w_vals(num_inputs * num_outputs);
    for (size_t i = 0; i < w_vals.size(); ++i)
    {
        w_vals[i] = std::sin(static_cast<double>(i + 1)) * 0.1;
    }
    std::vector<double> b_vals(num_outputs);
    for (size_t i = 0; i < b_vals.size(); ++i)
    {
        b_vals[i] = std::cos(static_cast<double>(i + 1)) * 0.1;
    }
    layer_st.set_w_values(w_vals);
    layer_st.set_b_values(b_vals);
    layer_mt.set_w_values(w_vals);
    layer_mt.set_b_values(b_vals);

    MockLayer prev_layer(0, num_inputs);
    std::vector<unsigned> topology = { num_inputs, num_outputs };

    auto batch_go_baseline = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_baseline = create_batch_hidden_states(topology, batch_size, num_time_steps);
    auto batch_go_residual_st = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_residual_st = create_batch_hidden_states(topology, batch_size, num_time_steps);
    auto batch_go_residual_mt = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs_residual_mt = create_batch_hidden_states(topology, batch_size, num_time_steps);

    std::vector<std::vector<double>> residual_values(batch_size, std::vector<double>(num_outputs));
    for (size_t b = 0; b < batch_size; ++b)
    {
        std::vector<double> in_seq(num_time_steps * num_inputs);
        for (size_t k = 0; k < in_seq.size(); ++k)
        {
            in_seq[k] = std::sin(static_cast<double>(b * num_time_steps + k)) * 0.1;
        }
        batch_go_baseline[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
        batch_go_residual_st[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());
        batch_go_residual_mt[b].set_rnn_outputs(0, in_seq.data(), in_seq.size());

        for (size_t j = 0; j < num_outputs; ++j)
        {
            residual_values[b][j] = std::cos(static_cast<double>(b + j + 1)) * 0.5;
        }
    }

    // effective_batch_size * N_prev * N_this and batch_size * num_time_steps *
    // N_this are both comfortably over the GEMM / post-GEMM multithreading
    // thresholds here, so layer_mt genuinely exercises the chunked dispatch.
    layer_st.calculate_forward_feed(batch_go_baseline, prev_layer, {}, batch_hs_baseline, batch_size, false);
    layer_st.calculate_forward_feed(batch_go_residual_st, prev_layer, residual_values, batch_hs_residual_st, batch_size, false);
    layer_mt.calculate_forward_feed(batch_go_residual_mt, prev_layer, residual_values, batch_hs_residual_mt, batch_size, false);

    for (size_t b = 0; b < batch_size; ++b)
    {
        const auto& out_baseline = batch_go_baseline[b].get_rnn_outputs(1);
        const auto& out_res_st = batch_go_residual_st[b].get_rnn_outputs(1);
        const auto& out_res_mt = batch_go_residual_mt[b].get_rnn_outputs(1);
        ASSERT_EQ(out_baseline.size(), static_cast<size_t>(num_time_steps) * num_outputs);
        ASSERT_EQ(out_res_st.size(), out_baseline.size());
        ASSERT_EQ(out_res_mt.size(), out_baseline.size());

        for (size_t k = 0; k < out_res_st.size(); ++k)
        {
            EXPECT_NEAR(out_res_st[k], out_res_mt[k], 1e-12) << "batch " << b << " index " << k;
        }

        for (size_t t = 0; t + 1 < num_time_steps; ++t)
        {
            for (size_t j = 0; j < num_outputs; ++j)
            {
                EXPECT_NEAR(out_res_st[t * num_outputs + j], out_baseline[t * num_outputs + j], 1e-12)
                    << "batch " << b << " t " << t << " j " << j;
            }
        }
        for (size_t j = 0; j < num_outputs; ++j)
        {
            const size_t last = (static_cast<size_t>(num_time_steps) - 1) * num_outputs + j;
            EXPECT_NEAR(out_res_st[last], out_baseline[last] + residual_values[b][j], 1e-9)
                << "batch " << b << " j " << j;
        }
    }
}

TEST_F(FFOutputLayerTest, SequenceMultiTimestepOutputGradients)
{
    const unsigned num_inputs = 2;
    const unsigned num_outputs = 2;
    const size_t num_time_steps = 3;
    const size_t batch_size = 2;

    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0),
        OutputLayerDetails(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 });
    layer.set_b_values({ 0.0, 0.0 });

    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t t = 0; t < num_time_steps; ++t)
        {
            batch_hs[b].at(1)[t].set_pre_activation_sums({ 0.5, 0.5 });
            batch_hs[b].at(1)[t].set_hidden_state_values({ 0.5, 0.62245933120185456 }); // sig(0.5) ~ 0.622459
        }
    }

    std::vector<std::vector<double>> targets(batch_size, std::vector<double>(num_time_steps * num_outputs, 1.0));
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, batch_size);

    for (size_t b = 0; b < batch_size; ++b)
    {
        const auto rnn_grads = batch_go[b].get_rnn_gradients(1);
        ASSERT_EQ(rnn_grads.size(), num_time_steps * num_outputs);

        const auto std_grads = batch_go[b].get_gradients(1);
        ASSERT_EQ(std_grads.size(), num_outputs);

        // Standard grads must match the last timestep
        EXPECT_NEAR(std_grads[0], rnn_grads[(num_time_steps - 1) * num_outputs], 1e-9);
        EXPECT_NEAR(std_grads[1], rnn_grads[(num_time_steps - 1) * num_outputs + 1], 1e-9);

        for (size_t t = 0; t < num_time_steps; ++t)
        {
            // Linear head MSE delta: given - target = 0.5 - 1.0 = -0.5
            EXPECT_NEAR(rnn_grads[t * num_outputs], -0.5, 1e-6);
            EXPECT_TRUE(std::isfinite(rnn_grads[t * num_outputs + 1]));
        }
    }
}

TEST_F(FFOutputLayerTest, OutputGradientsSingleTargetBroadcastToLastStep)
{
    const unsigned num_inputs = 2;
    const unsigned num_outputs = 2;
    const size_t num_time_steps = 3;
    const size_t batch_size = 1;

    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    layer.set_w_values({ 1.0, 0.0, 0.0, 1.0 });
    layer.set_b_values({ 0.0, 0.0 });

    std::vector<unsigned> topology = { num_inputs, num_outputs };
    auto batch_go = create_batch_gradients_and_outputs(topology, batch_size);
    auto batch_hs = create_batch_hidden_states(topology, batch_size, num_time_steps);

    for (size_t t = 0; t < num_time_steps; ++t)
    {
        batch_hs[0].at(1)[t].set_pre_activation_sums({ 0.8, 0.4 });
        batch_hs[0].at(1)[t].set_hidden_state_values({ 0.8, 0.4 });
    }

    // Only one target provided (size == num_outputs, not num_time_steps * num_outputs)
    std::vector<std::vector<double>> targets = { { 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, batch_size);

    const auto rnn_grads = batch_go[0].get_rnn_gradients(1);
    ASSERT_EQ(rnn_grads.size(), num_time_steps * num_outputs);

    // Timestep 0 and 1 have no target provided, so deltas are 0
    EXPECT_NEAR(rnn_grads[0], 0.0, 1e-9);
    EXPECT_NEAR(rnn_grads[1], 0.0, 1e-9);
    EXPECT_NEAR(rnn_grads[2], 0.0, 1e-9);
    EXPECT_NEAR(rnn_grads[3], 0.0, 1e-9);

    // Timestep 2 (last) has target [1.0, 0.0], so MSE delta = (given - target) / num_neurons = [0.8 - 1.0, 0.4 - 0.0] / 2 = [-0.1, 0.2]
    EXPECT_NEAR(rnn_grads[4], -0.1, 1e-9);
    EXPECT_NEAR(rnn_grads[5], 0.2, 1e-9);

    const auto std_grads = batch_go[0].get_gradients(1);
    ASSERT_EQ(std_grads.size(), num_outputs);
    EXPECT_NEAR(std_grads[0], -0.1, 1e-9);
    EXPECT_NEAR(std_grads[1], 0.2, 1e-9);
}

TEST_F(FFOutputLayerTest, OutputLayerAppliesMomentumWithSGD)
{
    const unsigned num_inputs = 1;
    const unsigned num_outputs = 2;
    const double momentum = 0.9;
    const double learning_rate = 0.1;

    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, momentum)
    };

    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    layer.set_w_values({ 1.0, 1.0 });
    layer.set_w_velocities({ 0.5, 0.5 });
    layer.set_w_grads({ 1.0, 1.0 });
    layer.set_b_values({ 0.0, 0.0 });
    layer.set_b_velocities({ 0.0, 0.0 });
    layer.set_b_grads({ 0.0, 0.0 });

    layer.apply_stored_gradients(learning_rate, 1.0);

    // With momentum = 0.9 and grad = 1.0:
    // v_new = 0.9 * 0.5 + 1.0 = 1.45
    // w_new = 1.0 - 0.1 * 1.45 = 0.855
    const auto& w_vals = layer.get_w_values();
    EXPECT_NEAR(w_vals[0], 0.855, 1e-9);
    EXPECT_NEAR(w_vals[1], 0.855, 1e-9);

    const auto& w_vel = layer.get_w_velocities();
    EXPECT_NEAR(w_vel[0], 1.45, 1e-9);
    EXPECT_NEAR(w_vel[1], 1.45, 1e-9);
}

TEST_F(FFOutputLayerTest, OutputLayerApplyStoredGradientsFastPathEquivalence)
{
    const unsigned num_inputs = 2;
    const unsigned num_outputs = 2;
    const double momentum = 0.8;
    const double learning_rate = 0.05;

    // Single-head configuration (exercises fast path)
    std::vector<OutputLayerDetails> single_details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::AdamW, momentum)
    };
    FFOutputLayer layer_single(1, single_details, num_inputs, num_outputs, 1, true, std::nullopt);
    layer_single.set_w_values({ 0.5, -0.3, 0.2, 0.8 });
    layer_single.set_w_grads({ 0.1, -0.2, 0.3, -0.1 });
    layer_single.set_b_values({ 0.05, -0.05 });
    layer_single.set_b_grads({ 0.02, -0.01 });

    // Multi-head configuration with matching optimizer settings (also routes through fast path)
    std::vector<OutputLayerDetails> multi_details = {
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::AdamW, momentum),
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::AdamW, momentum)
    };
    FFOutputLayer layer_multi(1, multi_details, num_inputs, num_outputs, 1, true, std::nullopt);
    layer_multi.set_w_values({ 0.5, -0.3, 0.2, 0.8 });
    layer_multi.set_w_grads({ 0.1, -0.2, 0.3, -0.1 });
    layer_multi.set_b_values({ 0.05, -0.05 });
    layer_multi.set_b_grads({ 0.02, -0.01 });

    layer_single.apply_stored_gradients(learning_rate, 1.0);
    layer_multi.apply_stored_gradients(learning_rate, 1.0);

    const auto& w_single = layer_single.get_w_values();
    const auto& w_multi = layer_multi.get_w_values();
    ASSERT_EQ(w_single.size(), w_multi.size());
    for (size_t i = 0; i < w_single.size(); ++i)
    {
        EXPECT_NEAR(w_single[i], w_multi[i], 1e-12);
    }

    const auto& b_single = layer_single.get_b_values();
    const auto& b_multi = layer_multi.get_b_values();
    ASSERT_EQ(b_single.size(), b_multi.size());
    for (size_t i = 0; i < b_single.size(); ++i)
    {
        EXPECT_NEAR(b_single[i], b_multi[i], 1e-12);
    }
}

TEST_F(FFOutputLayerTest, OutputLayerMetricsSharpeSortinoNoCrossBatchLeakage)
{
    const unsigned num_inputs = 1;
    const unsigned num_outputs = 1;
    const double penalty = 0.05;

    EvaluationConfig cfg(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-8, 0.0, { 0.5 }, penalty, 0.0);
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::sharpe_ratio_loss, cfg, 0.0, OptimiserType::None, 0.0)
    };

    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, false, std::nullopt);

    // Two independent batch examples of length 2 each.
    // Batch 0: pred = [1.0, 1.0], target = [0.1, 0.1]
    // Batch 1: pred = [-1.0, -1.0], target = [0.1, 0.1]
    // If transaction costs leaked across batch items, step 0 of batch 1 would compare against step 1 of batch 0 (| -1.0 - 1.0 | = 2.0).
    // Without leakage, step 0 of batch 1 compares against initial 0.0 (| -1.0 - 0.0 | = 1.0).
    std::vector<std::vector<double>> predictions = {
        { 1.0, 1.0 },
        { -1.0, -1.0 }
    };
    std::vector<std::vector<double>> targets = {
        { 0.1, 0.1 },
        { 0.1, 0.1 }
    };

    const auto metrics = layer.calculate_output_metrics({ ErrorCalculation::type::sharpe_ratio_loss }, targets, predictions);
    ASSERT_EQ(metrics.size(), 1);
    ASSERT_EQ(metrics[0].size(), 1);

    // Verify metric is finite and valid
    EXPECT_TRUE(std::isfinite(metrics[0][0].error()));

    // Calculate expected return sequence independently per batch item:
    // Batch 0:
    //   t=0: pos=1.0, prev=0.0, cost=0.0, ret = 1.0 * 0.1 = 0.1
    //   t=1: pos=1.0, prev=1.0, cost=0.05 * 0 = 0.0, ret = 0.1
    // Batch 1:
    //   t=0: pos=-1.0, prev=0.0, cost=0.05 * 0 = 0.0 (t=0 has no prev step cost in calculate_portfolio_returns), ret = -0.1
    //   t=1: pos=-1.0, prev=-1.0, cost=0.0, ret = -0.1
    // Pooled returns: [0.1, 0.1, -0.1, -0.1]
    const std::vector<double> expected_pooled = { 0.1, 0.1, -0.1, -0.1 };
    const auto expected_stats = ErrorCalculation::calculate_sharpe_batch_stats(expected_pooled, cfg.epsilon());
    const double expected_loss = (expected_stats.sigma > 0.0) ? (-expected_stats.mean / expected_stats.sigma) : 0.0;

    EXPECT_NEAR(metrics[0][0].error(), expected_loss, 1e-9);
}

TEST_F(FFOutputLayerTest, OutputLayerEmptyHiddenStatesDefensive)
{
    const unsigned num_inputs = 2;
    const unsigned num_outputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };

    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true, std::nullopt);
    std::vector<GradientsAndOutputs> batch_go;
    std::vector<std::vector<double>> targets;
    std::vector<HiddenStates> empty_hs;

    // batch_size = 0 should return cleanly without throwing
    EXPECT_NO_THROW(layer.calculate_output_gradients(batch_go, targets.begin(), empty_hs, 0));

    // empty batch_hidden_states should return cleanly without throwing
    EXPECT_NO_THROW(layer.calculate_output_gradients(batch_go, targets.begin(), empty_hs, 1));
}

