#include <gtest/gtest.h>
#include "layers/layerdetails.h"
#include "layers/outputlayerdetails.h"
#include "layers/multioutputlayerdetails.h"
#include "layers/multioutputlayer.h"
#include "common/activation.h"
#include "helpers/errorcalculation.h"
#include "common/optimiser.h"
#include "common/evaluationconfig.h"
#include "test_helper.h"
#include <vector>
#include <cmath>
#include <stdexcept>


using namespace myoddweb::nn;
using namespace test_helper;

class LayerDetailsTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(LayerDetailsTest, LayerDetailsMethods) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    activation act(activation::method::relu, 0.1, 1.0);
    LayerDetails details(Layer::Architecture::FF, 10, act, 0.2, 0.001, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0);

    EXPECT_EQ(details.get_layer_architecture(), Layer::Architecture::FF);
    EXPECT_EQ(details.get_size(), 10);
    EXPECT_EQ(details.get_activation().get_method(), activation::method::relu);
    EXPECT_EQ(details.get_activation().get_alpha(), 0.1);
    EXPECT_EQ(details.get_dropout(), 0.2);
    EXPECT_EQ(details.get_weight_decay(), 0.001);
    EXPECT_EQ(details.get_optimiser_type(), OptimiserType::Adam);
    EXPECT_EQ(details.get_momentum(), 0.9);

    // Copy constructor
    LayerDetails copy(details);
    EXPECT_EQ(copy.get_size(), 10);
    EXPECT_EQ(copy.get_dropout(), 0.2);

    // Move constructor
    LayerDetails moved(std::move(copy));
    EXPECT_EQ(moved.get_size(), 10);
    EXPECT_EQ(copy.get_size(), 0); // moved-from state size is 0

    // Copy assignment
    LayerDetails assigned(Layer::Architecture::None, 0, activation(activation::method::linear, 0.0), 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    assigned = details;
    EXPECT_EQ(assigned.get_size(), 10);

    // Move assignment
    LayerDetails move_assigned(Layer::Architecture::None, 0, activation(activation::method::linear, 0.0), 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned.get_size(), 10);
    EXPECT_EQ(assigned.get_size(), 0);
}

TEST_F(LayerDetailsTest, LayerDetailsUseLayerNormalisationFlag) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    activation act(activation::method::tanh, 0.0, 1.0);

    LayerDetails default_details(Layer::Architecture::Gru, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0);
    EXPECT_FALSE(default_details.get_use_layer_normalisation());

    LayerDetails enabled_details(Layer::Architecture::Gru, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, true, 0, 0, 0, 0, 0, 0, 0);
    EXPECT_TRUE(enabled_details.get_use_layer_normalisation());

    // Copy constructor
    LayerDetails copy(enabled_details);
    EXPECT_TRUE(copy.get_use_layer_normalisation());

    // Move constructor
    LayerDetails moved(std::move(copy));
    EXPECT_TRUE(moved.get_use_layer_normalisation());
    EXPECT_FALSE(copy.get_use_layer_normalisation()); // moved-from state resets to false

    // Copy assignment
    LayerDetails assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    assigned = enabled_details;
    EXPECT_TRUE(assigned.get_use_layer_normalisation());

    // Move assignment
    LayerDetails move_assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    move_assigned = std::move(assigned);
    EXPECT_TRUE(move_assigned.get_use_layer_normalisation());
    EXPECT_FALSE(assigned.get_use_layer_normalisation()); // moved-from state resets to false
}

TEST_F(LayerDetailsTest, LayerDetailsAttentionHiddenSizeField) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    activation act(activation::method::tanh, 0.0, 1.0);

    LayerDetails default_details(Layer::Architecture::FF, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0);
    EXPECT_EQ(default_details.get_attention_hidden_size(), 0u);

    LayerDetails pooled_details(Layer::Architecture::AttentionPool, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 16, 0, 0, 0, 0, 0, 0);
    EXPECT_EQ(pooled_details.get_attention_hidden_size(), 16u);

    // Copy constructor
    LayerDetails copy(pooled_details);
    EXPECT_EQ(copy.get_attention_hidden_size(), 16u);

    // Move constructor
    LayerDetails moved(std::move(copy));
    EXPECT_EQ(moved.get_attention_hidden_size(), 16u);
    EXPECT_EQ(copy.get_attention_hidden_size(), 0u); // moved-from state resets to 0

    // Copy assignment
    LayerDetails assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    assigned = pooled_details;
    EXPECT_EQ(assigned.get_attention_hidden_size(), 16u);

    // Move assignment
    LayerDetails move_assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned.get_attention_hidden_size(), 16u);
    EXPECT_EQ(assigned.get_attention_hidden_size(), 0u); // moved-from state resets to 0
}

TEST_F(LayerDetailsTest, LayerDetailsKernelSizeAndDilationFields) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    activation act(activation::method::tanh, 0.0, 1.0);

    LayerDetails default_details(Layer::Architecture::FF, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0);
    EXPECT_EQ(default_details.get_kernel_size(), 0u);
    EXPECT_EQ(default_details.get_dilation(), 0u);

    LayerDetails tcn_details(Layer::Architecture::Tcn, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 3, 2, 0, 0, 0, 0);
    EXPECT_EQ(tcn_details.get_kernel_size(), 3u);
    EXPECT_EQ(tcn_details.get_dilation(), 2u);

    // Copy constructor
    LayerDetails copy(tcn_details);
    EXPECT_EQ(copy.get_kernel_size(), 3u);
    EXPECT_EQ(copy.get_dilation(), 2u);

    // Move constructor
    LayerDetails moved(std::move(copy));
    EXPECT_EQ(moved.get_kernel_size(), 3u);
    EXPECT_EQ(moved.get_dilation(), 2u);
    EXPECT_EQ(copy.get_kernel_size(), 0u); // moved-from state resets to 0
    EXPECT_EQ(copy.get_dilation(), 0u);

    // Copy assignment
    LayerDetails assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    assigned = tcn_details;
    EXPECT_EQ(assigned.get_kernel_size(), 3u);
    EXPECT_EQ(assigned.get_dilation(), 2u);

    // Move assignment
    LayerDetails move_assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned.get_kernel_size(), 3u);
    EXPECT_EQ(move_assigned.get_dilation(), 2u);
    EXPECT_EQ(assigned.get_kernel_size(), 0u); // moved-from state resets to 0
    EXPECT_EQ(assigned.get_dilation(), 0u);
}

TEST_F(LayerDetailsTest, LayerDetailsNumberOfHeadsAndFeedForwardHiddenSizeFields) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    activation act(activation::method::tanh, 0.0, 1.0);

    LayerDetails default_details(Layer::Architecture::FF, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0);
    EXPECT_EQ(default_details.get_number_of_heads(), 0u);
    EXPECT_EQ(default_details.get_feed_forward_hidden_size(), 0u);

    LayerDetails sa_details(Layer::Architecture::SelfAttention, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 4, 64, 0, 0);
    EXPECT_EQ(sa_details.get_number_of_heads(), 4u);
    EXPECT_EQ(sa_details.get_feed_forward_hidden_size(), 64u);

    // Copy constructor
    LayerDetails copy(sa_details);
    EXPECT_EQ(copy.get_number_of_heads(), 4u);
    EXPECT_EQ(copy.get_feed_forward_hidden_size(), 64u);

    // Move constructor
    LayerDetails moved(std::move(copy));
    EXPECT_EQ(moved.get_number_of_heads(), 4u);
    EXPECT_EQ(moved.get_feed_forward_hidden_size(), 64u);
    EXPECT_EQ(copy.get_number_of_heads(), 0u); // moved-from state resets to 0
    EXPECT_EQ(copy.get_feed_forward_hidden_size(), 0u);

    // Copy assignment
    LayerDetails assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    assigned = sa_details;
    EXPECT_EQ(assigned.get_number_of_heads(), 4u);
    EXPECT_EQ(assigned.get_feed_forward_hidden_size(), 64u);

    // Move assignment
    LayerDetails move_assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned.get_number_of_heads(), 4u);
    EXPECT_EQ(move_assigned.get_feed_forward_hidden_size(), 64u);
    EXPECT_EQ(assigned.get_number_of_heads(), 0u); // moved-from state resets to 0
    EXPECT_EQ(assigned.get_feed_forward_hidden_size(), 0u);
}

TEST_F(LayerDetailsTest, LayerDetailsVocabularySizeAndEmbeddingDimensionFields) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    activation act(activation::method::linear, 0.0, 1.0);

    LayerDetails default_details(Layer::Architecture::FF, 8, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0);
    EXPECT_EQ(default_details.get_vocabulary_size(), 0u);
    EXPECT_EQ(default_details.get_embedding_dimension(), 0u);

    LayerDetails emb_details(Layer::Architecture::Embedding, 16, act, 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 100, 16);
    EXPECT_EQ(emb_details.get_vocabulary_size(), 100u);
    EXPECT_EQ(emb_details.get_embedding_dimension(), 16u);

    // Copy constructor
    LayerDetails copy(emb_details);
    EXPECT_EQ(copy.get_vocabulary_size(), 100u);
    EXPECT_EQ(copy.get_embedding_dimension(), 16u);

    // Move constructor
    LayerDetails moved(std::move(copy));
    EXPECT_EQ(moved.get_vocabulary_size(), 100u);
    EXPECT_EQ(moved.get_embedding_dimension(), 16u);
    EXPECT_EQ(copy.get_vocabulary_size(), 0u);
    EXPECT_EQ(copy.get_embedding_dimension(), 0u);

    // Copy assignment
    LayerDetails assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    assigned = emb_details;
    EXPECT_EQ(assigned.get_vocabulary_size(), 100u);
    EXPECT_EQ(assigned.get_embedding_dimension(), 16u);

    // Move assignment
    LayerDetails move_assigned(Layer::Architecture::None, 0, act, 0, 0, OptimiserType::None, 0, false, 0, 0, 0, 0, 0, 0, 0);
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned.get_vocabulary_size(), 100u);
    EXPECT_EQ(move_assigned.get_embedding_dimension(), 16u);
    EXPECT_EQ(assigned.get_vocabulary_size(), 0u);
    EXPECT_EQ(assigned.get_embedding_dimension(), 0u);
}

TEST_F(LayerDetailsTest, OutputLayerDetailsMethods) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    activation act(activation::method::softmax, 0.0, 1.2);
    EvaluationConfig config(0.1, 0.2, 1.0, 0.0, true, 0.5, 1e-10, 0.0, { 0.5 });
    OutputLayerDetails details(5, act, ErrorCalculation::type::cross_entropy, config, 0.01, OptimiserType::Nadam, 0.8);

    EXPECT_EQ(details.get_size(), 5);
    EXPECT_EQ(details.get_activation().get_method(), activation::method::softmax);
    EXPECT_EQ(details.get_activation().get_temperature(), 1.2);
    EXPECT_EQ(details.get_output_error_calculation_type(), ErrorCalculation::type::cross_entropy);
    EXPECT_EQ(details.get_error_evaluation_config().neutral_tolerance(), 0.1);
    EXPECT_EQ(details.get_error_evaluation_config().direction_lambda(), 0.0);
    EXPECT_EQ(details.get_weight_decay(), 0.01);
    EXPECT_EQ(details.get_optimiser_type(), OptimiserType::Nadam);
    EXPECT_EQ(details.get_momentum(), 0.8);

    // Copy
    OutputLayerDetails copy(details);
    EXPECT_EQ(copy.get_size(), 5);
    
    // Move
    OutputLayerDetails moved(std::move(copy));
    EXPECT_EQ(moved.get_size(), 5);
    EXPECT_EQ(copy.get_size(), 0);

    // Assignment
    OutputLayerDetails assigned(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0);
    assigned = details;
    EXPECT_EQ(assigned.get_size(), 5);

    OutputLayerDetails move_assigned(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::None, 0.0);
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned.get_size(), 5);
    EXPECT_EQ(assigned.get_size(), 0);

    // Validation
    EXPECT_THROW(OutputLayerDetails(5, act, ErrorCalculation::type::mse, config, -0.1, OptimiserType::SGD, 0.0), std::runtime_error);
}

TEST_F(LayerDetailsTest, MultiOutputLayerDetailsMethods) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    LayerDetails h1(Layer::Architecture::FF, 10, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    LayerDetails h2(Layer::Architecture::Elman, 5, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    std::vector<LayerDetails> hidden = { h1, h2 };
    
    OutputLayerDetails o(2, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    
    MultiOutputLayerDetails details(hidden, o);
    EXPECT_EQ(details.get_hidden_layers().size(), 2);
    EXPECT_EQ(details.get_hidden_layer(0).get_size(), 10);
    EXPECT_EQ(details.get_hidden_layer(1).get_size(), 5);
    EXPECT_EQ(details.get_output_details().get_size(), 2);

    // Copy
    MultiOutputLayerDetails copy(details);
    EXPECT_EQ(copy.get_hidden_layers().size(), 2);

    // Move
    MultiOutputLayerDetails moved(std::move(copy));
    EXPECT_EQ(moved.get_hidden_layers().size(), 2);
    EXPECT_TRUE(copy.get_hidden_layers().empty());

    // Assignment
    MultiOutputLayerDetails assigned({}, o);
    assigned = details;
    EXPECT_EQ(assigned.get_hidden_layers().size(), 2);

    MultiOutputLayerDetails move_assigned({}, o);
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned.get_hidden_layers().size(), 2);
    EXPECT_TRUE(assigned.get_hidden_layers().empty());
}

TEST_F(LayerDetailsTest, ComplexArchitectureVerification) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    // Branch A: FF(ReLU) -> FF(Tanh) -> Output(Sigmoid, MSE)
    // Branch B: FF(ELU) -> Output(Linear, MSE)
    // Branch C: Output(Softmax, CE)
    
    EvaluationConfig clean_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 });

    LayerDetails hA1(Layer::Architecture::FF, 2, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    LayerDetails hA2(Layer::Architecture::FF, 2, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    OutputLayerDetails oA(1, activation(activation::method::sigmoid, 1.0, 1.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA({hA1, hA2}, oA);

    LayerDetails hB1(Layer::Architecture::FF, 1, activation(activation::method::elu, 0.5, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    OutputLayerDetails oB(1, activation(activation::method::linear, 1.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB({hB1}, oB);

    OutputLayerDetails oC(2, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modC({}, oC);

    std::vector<MultiOutputLayerDetails> details = { modA, modB, modC };
    MultiOutputLayer layer(1, 2, 4, details, 1, true, std::nullopt);
    auto& branches = layer.get_mutable_branches();

    // Setup Weights
    // Branch A H1 (ReLU): W=[[0.1, 0.2], [0.3, 0.4]], B=[0.1, -0.1]
    branches[0].layers[0]->set_w_values({ 0.1, 0.2, 0.3, 0.4 });
    branches[0].layers[0]->set_b_values({ 0.1, -0.1 });
    // Branch A H2 (Tanh): W=[[0.5, 0.6], [0.7, 0.8]], B=[0.0, 0.0]
    branches[0].layers[1]->set_w_values({ 0.5, 0.6, 0.7, 0.8 });
    branches[0].layers[1]->set_b_values({ 0.0, 0.0 });
    // Branch A O (Sigmoid): W=[[0.9, 1.0]], B=[0.1]
    branches[0].layers[2]->set_w_values({ 0.9, 1.0 });
    branches[0].layers[2]->set_b_values({ 0.1 });

    // Branch B H1 (ELU): W=[[0.5], [-0.5]], B=[0.0] (2 input, 1 output)
    branches[1].layers[0]->set_w_values({ 0.5, -0.5 });
    branches[1].layers[0]->set_b_values({ 0.0 });
    // Branch B O (Linear): W=[[1.0]], B=[0.5]
    branches[1].layers[1]->set_w_values({ 1.0 });
    branches[1].layers[1]->set_b_values({ 0.5 });

    // Branch C O (Softmax): W=[[0.1, 0.2], [0.3, 0.4]], B=[0.0, 0.0]
    branches[2].layers[0]->set_w_values({ 0.1, 0.2, 0.3, 0.4 });
    branches[2].layers[0]->set_b_values({ 0.0, 0.0 });

    MockLayer prev_layer(0, 2);
    std::vector<unsigned> topology = { 2, 4 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 1.0, 0.5 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], 0.701036, 1e-5); // Branch A
    EXPECT_NEAR(outputs[1], 0.75, 1e-5);     // Branch B
    EXPECT_NEAR(outputs[2], 0.46257, 1e-5);  // Branch C [0]
    EXPECT_NEAR(outputs[3], 0.53743, 1e-5);  // Branch C [1]

    // Backprop:
    std::vector<std::vector<double>> targets = { { 0.0, 1.0, 1.0, 0.0 } };
    layer.calculate_output_gradients(batch_go, targets.begin(), batch_hs, 1);
    
    layer.backprop_branches(1, 0);

    auto trunk_grads = layer.get_trunk_gradients(1);
    EXPECT_NEAR(trunk_grads[0][0], -0.024027, 2e-3);
    EXPECT_NEAR(trunk_grads[0][1], 0.287379, 2e-3);
}

TEST_F(LayerDetailsTest, RecurrentArchitectureVerification) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    // Branch A: RNN(Tanh) -> Output(Linear, MSE)
    // Branch B: GRU(Sigmoid) -> Output(Linear, MSE)
    
    EvaluationConfig clean_config(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 });

    LayerDetails hA(Layer::Architecture::Elman, 1, activation(activation::method::tanh, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    OutputLayerDetails oA(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modA({hA}, oA);

    LayerDetails hB(Layer::Architecture::Gru, 1, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    OutputLayerDetails oB(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, clean_config, 0.0, OptimiserType::SGD, 0.0);
    MultiOutputLayerDetails modB({hB}, oB);

    MultiOutputLayer layer(1, 1, 2, { modA, modB }, 1, true, std::nullopt);
    auto& branches = layer.get_mutable_branches();

    // Branch A (RNN): W=[0.5], RW=[0.2], B=[0.1]
    branches[0].layers[0]->set_w_values({ 0.5 });
    branches[0].layers[0]->set_rw_values({ 0.2 });
    branches[0].layers[0]->set_b_values({ 0.1 });
    // Branch A O: W=[1.0], B=[0.0]
    branches[0].layers[1]->set_w_values({ 1.0 });
    branches[0].layers[1]->set_b_values({ 0.0 });

    // Branch B (GRU): W=[0.1, 0.2, 0.3], RW=[0.4, 0.5, 0.6], B=[0.0, 0.0, 0.0] (3 gates: z, r, h_hat)
    branches[1].layers[0]->set_w_values({ 0.1, 0.2, 0.3 });
    branches[1].layers[0]->set_rw_values({ 0.4, 0.5, 0.6 });
    branches[1].layers[0]->set_b_values({ 0.0, 0.0, 0.0 });
    // Branch B O: W=[1.0], B=[0.0]
    branches[1].layers[1]->set_w_values({ 1.0 });
    branches[1].layers[1]->set_b_values({ 0.0 });

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 2 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_rnn_outputs(0, { 1.0 }); // Input sequence length 1

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_rnn_outputs(1);
    EXPECT_NEAR(outputs[0], 0.53705, 1e-5);
    EXPECT_NEAR(outputs[1], 0.301570, 1e-5);
}

TEST_F(LayerDetailsTest, ActivationVarietyVerification) {
    MYODDWEB_PROFILE_FUNCTION("LayerDetailsTest");
    // Testing specific activation types: LeakyReLU, SELU, Swish, Mish, Gelu
    
    LayerDetails h1(Layer::Architecture::FF, 1, activation(activation::method::leakyRelu, 0.01), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    LayerDetails h2(Layer::Architecture::FF, 1, activation(activation::method::selu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    LayerDetails h3(Layer::Architecture::FF, 1, activation(activation::method::swish, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    LayerDetails h4(Layer::Architecture::FF, 1, activation(activation::method::mish, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    LayerDetails h5(Layer::Architecture::FF, 1, activation(activation::method::gelu, 0.0), 0.0, 0.0, OptimiserType::SGD, 0.0, false, 0, 0, 0, 0, 0, 0, 0);
    
    OutputLayerDetails o1(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::SGD, 0.0);
    
    MultiOutputLayer layer(1, 1, 5, { MultiOutputLayerDetails({h1}, o1), 
                                     MultiOutputLayerDetails({h2}, o1), 
                                     MultiOutputLayerDetails({h3}, o1), 
                                     MultiOutputLayerDetails({h4}, o1), 
                                     MultiOutputLayerDetails({h5}, o1) }, 1, false, std::nullopt);
    
    auto& branches = layer.get_mutable_branches();
    for(int i=0; i<5; ++i) {
        branches[i].layers[0]->set_w_values({ -1.0 }); // Input -1.0
        branches[i].layers[1]->set_w_values({ 1.0 });
    }

    MockLayer prev_layer(0, 1);
    std::vector<unsigned> topology = { 1, 5 };
    auto batch_go = create_batch_gradients_and_outputs(topology, 1);
    auto batch_hs = create_batch_hidden_states(topology, 1, 1);
    batch_go[0].set_outputs(0, { 1.0 });

    layer.calculate_forward_feed(batch_go, prev_layer, {}, batch_hs, 1, true);

    const auto& outputs = batch_go[0].get_outputs(1);
    EXPECT_NEAR(outputs[0], -0.01, 1e-5);
    EXPECT_NEAR(outputs[1], -1.1113, 1e-3);
    EXPECT_NEAR(outputs[2], -0.26894, 1e-4);
    EXPECT_NEAR(outputs[3], -0.3034, 1e-3);
    EXPECT_NEAR(outputs[4], -0.1587, 1e-2);
}
