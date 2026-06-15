#include <gtest/gtest.h>
#include "common/gradientsandoutputs.h"
#include <vector>
#include <stdexcept>


using namespace myoddweb::nn;
class GradientsAndOutputsTest : public ::testing::Test {
protected:
    std::vector<unsigned> topology = {3, 5, 2}; // 3 layers
};

TEST_F(GradientsAndOutputsTest, Initialization) {
    GradientsAndOutputs gao(topology);
    
    // Check outputs size
    EXPECT_EQ(gao.get_outputs(0).size(), 3);
    EXPECT_EQ(gao.get_outputs(1).size(), 5);
    EXPECT_EQ(gao.get_outputs(2).size(), 2);

    // Check initial values are zero
    for (unsigned l = 0; l < topology.size(); ++l) {
        auto outputs = gao.get_outputs(l);
        for (double v : outputs) {
            EXPECT_DOUBLE_EQ(v, 0.0);
        }
    }
}

TEST_F(GradientsAndOutputsTest, BiasHandling) {
    GradientsAndOutputs gao(topology);
    
    // get_output(layer, neuron) should return 1.0 if neuron == num_neurons (the bias)
    EXPECT_DOUBLE_EQ(gao.get_output(0, 3), 1.0);
    EXPECT_DOUBLE_EQ(gao.get_output(1, 5), 1.0);
    EXPECT_DOUBLE_EQ(gao.get_output(2, 2), 1.0);
    
    // Standard neurons should still be 0.0
    EXPECT_DOUBLE_EQ(gao.get_output(0, 0), 0.0);
}

TEST_F(GradientsAndOutputsTest, SetAndGetOutputs) {
    GradientsAndOutputs gao(topology);
    std::vector<double> layer1_vals = {0.1, 0.2, 0.3, 0.4, 0.5};
    
    gao.set_outputs(1, layer1_vals);
    
    auto outputs = gao.get_outputs(1);
    ASSERT_EQ(outputs.size(), 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(outputs[i], layer1_vals[i]);
        EXPECT_DOUBLE_EQ(gao.get_output(1, (unsigned)i), layer1_vals[i]);
    }
}

TEST_F(GradientsAndOutputsTest, SetAndGetGradients) {
    GradientsAndOutputs gao(topology);
    std::vector<double> layer2_grads = {0.9, -0.9};
    
    gao.set_gradients(2, layer2_grads);
    
    auto grads = gao.get_gradients(2);
    ASSERT_EQ(grads.size(), 2);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(grads[i], layer2_grads[i]);
    }
    
    // Verify raw pointer access
    double* raw = gao.get_gradients_raw(2);
    EXPECT_DOUBLE_EQ(raw[0], 0.9);
    EXPECT_DOUBLE_EQ(raw[1], -0.9);
}

TEST_F(GradientsAndOutputsTest, OutputBack) {
    GradientsAndOutputs gao(topology);
    std::vector<double> last_layer = {0.77, 0.88};
    gao.set_outputs(2, last_layer);
    
    std::vector<double> back = gao.output_back();
    ASSERT_EQ(back.size(), 2);
    EXPECT_DOUBLE_EQ(back[0], 0.77);
    EXPECT_DOUBLE_EQ(back[1], 0.88);
}

TEST_F(GradientsAndOutputsTest, Zero) {
    GradientsAndOutputs gao(topology);
    gao.set_outputs(0, {1, 2, 3});
    gao.set_gradients(0, {4, 5, 6});
    gao.set_rnn_outputs(10, {7, 8});
    
    gao.zero();
    
    EXPECT_DOUBLE_EQ(gao.get_output(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(gao.get_gradients(0)[0], 0.0);
    EXPECT_TRUE(gao.get_rnn_outputs(10).empty());
}

TEST_F(GradientsAndOutputsTest, RNNState) {
    GradientsAndOutputs gao(topology);
    std::vector<double> rnn_out = {0.5, 0.6};
    std::vector<double> rnn_grad = {0.01, 0.02};
    
    gao.set_rnn_outputs(5, rnn_out);
    gao.set_rnn_gradients(5, rnn_grad);
    
    EXPECT_EQ(gao.get_rnn_outputs(5), rnn_out);
    EXPECT_EQ(gao.get_rnn_gradients(5), rnn_grad);
    
    // Test raw gradient resize logic
    double* raw_grad = gao.get_rnn_gradients_raw(6, 4); // layer 6, size 4
    raw_grad[0] = 1.1;
    raw_grad[3] = 4.4;
    
    auto retrieved = gao.get_rnn_gradients(6);
    ASSERT_EQ(retrieved.size(), 4);
    EXPECT_DOUBLE_EQ(retrieved[0], 1.1);
    EXPECT_DOUBLE_EQ(retrieved[3], 4.4);
}

TEST_F(GradientsAndOutputsTest, CopyAndMove) {
    GradientsAndOutputs gao1(topology);
    gao1.set_outputs(0, {1.1, 1.2, 1.3});
    
    // Copy
    GradientsAndOutputs gao2 = gao1;
    EXPECT_DOUBLE_EQ(gao2.get_output(0, 0), 1.1);
    
    // Move
    GradientsAndOutputs gao3 = std::move(gao1);
    EXPECT_DOUBLE_EQ(gao3.get_output(0, 1), 1.2);
    // Note: gao1 is now in an undefined but valid state (likely empty buffers)
}

TEST_F(GradientsAndOutputsTest, Exceptions) {
    // These tests assume VALIDATE_DATA=1 is defined (which it is in our CMake)
#if VALIDATE_DATA == 1
    GradientsAndOutputs gao(topology);
    
    // Out of bounds layer
    EXPECT_THROW((void)gao.get_gradients(10), std::runtime_error);
    
    // Invalid size for set_outputs
    std::vector<double> too_big = {1, 2, 3, 4}; // layer 0 only has 3
    EXPECT_THROW(gao.set_outputs(0, too_big), std::runtime_error);

    // Invalid layer for raw pointer
    EXPECT_THROW((void)gao.get_outputs_raw(5), std::runtime_error);
#endif
}

TEST_F(GradientsAndOutputsTest, PointerOverloadsAndRawBuffers)
{
    GradientsAndOutputs gao(topology);
    
    // Test set_outputs pointer overload
    std::vector<double> out_vals = {1.2, 3.4, 5.6};
    gao.set_outputs(0, out_vals.data(), out_vals.size());
    EXPECT_DOUBLE_EQ(gao.get_output(0, 0), 1.2);
    EXPECT_DOUBLE_EQ(gao.get_output(0, 1), 3.4);
    EXPECT_DOUBLE_EQ(gao.get_output(0, 2), 5.6);
    
    // Test set_gradients pointer overload
    std::vector<double> grad_vals = {7.8, 9.0};
    gao.set_gradients(2, grad_vals.data(), grad_vals.size());
    EXPECT_DOUBLE_EQ(gao.get_gradients(2)[0], 7.8);
    EXPECT_DOUBLE_EQ(gao.get_gradients(2)[1], 9.0);
    
    // Test get_rnn_outputs_raw resize and pointer usage
    double* rnn_out_ptr = gao.get_rnn_outputs_raw(1, 3);
    rnn_out_ptr[0] = 0.11;
    rnn_out_ptr[1] = 0.22;
    rnn_out_ptr[2] = 0.33;
    
    auto retrieved_rnn_out = gao.get_rnn_outputs(1);
    ASSERT_EQ(retrieved_rnn_out.size(), 3);
    EXPECT_DOUBLE_EQ(retrieved_rnn_out[0], 0.11);
    EXPECT_DOUBLE_EQ(retrieved_rnn_out[1], 0.22);
    EXPECT_DOUBLE_EQ(retrieved_rnn_out[2], 0.33);
    
    // Test set_rnn_outputs pointer overload
    std::vector<double> rnn_out_vals = {0.44, 0.55};
    gao.set_rnn_outputs(2, rnn_out_vals.data(), rnn_out_vals.size());
    EXPECT_EQ(gao.get_rnn_outputs(2), rnn_out_vals);
    
    // Test set_rnn_gradients pointer overload
    std::vector<double> rnn_grad_vals = {0.66, 0.77};
    gao.set_rnn_gradients(2, rnn_grad_vals.data(), rnn_grad_vals.size());
    EXPECT_EQ(gao.get_rnn_gradients(2), rnn_grad_vals);
    
    // Test set_rnn_gate_gradients pointer overload
    std::vector<double> rnn_gate_vals = {0.88, 0.99};
    gao.set_rnn_gate_gradients(2, rnn_gate_vals.data(), rnn_gate_vals.size());
    EXPECT_EQ(gao.get_rnn_gate_gradients(2), rnn_gate_vals);
}
