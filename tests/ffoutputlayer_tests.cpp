#include <gtest/gtest.h>
#include "../src/neuralnetwork/ffoutputlayer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>

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
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

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
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
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
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
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
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
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

TEST_F(FFOutputLayerTest, DropoutStatisticalVerification) {
    unsigned num_inputs = 1;
    unsigned num_outputs = 1000;
    double dropout_rate = 0.5;
    
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(num_outputs, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0)
    };

    std::vector<Neuron> neurons;
    for (unsigned i = 0; i < num_outputs; ++i) {
        neurons.emplace_back(i, Neuron::Type::Dropout, dropout_rate);
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
    EXPECT_NEAR(dropped_count, num_outputs * dropout_rate, num_outputs * 0.05);
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
        neurons.emplace_back(i, Neuron::Type::Dropout, dropout_rate);
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
    neurons.emplace_back(0, Neuron::Type::Dropout, dropout_rate);

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

TEST_F(FFOutputLayerTest, MultiHeadOutput) {
    unsigned num_inputs = 2;
    std::vector<OutputLayerDetails> details = {
        OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0),
        OutputLayerDetails(1, activation(activation::method::sigmoid, 0.0), ErrorCalculation::type::bce_loss, EvaluationConfig(), 0.0, OptimiserType::None, 0.0)
    };
    unsigned num_outputs = 2;
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
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
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

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
        FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
        
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
    FFOutputLayer layer(1, details, 1, 5, 1, true);

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
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

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
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

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
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

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
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);

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
    
    FFOutputLayer layer(1, details, num_inputs, num_outputs, 1, true);
    
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
