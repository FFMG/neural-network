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
