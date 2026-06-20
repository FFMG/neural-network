#include <gtest/gtest.h>
#include "layers/layer.h"
#include "test_helper.h"
#include <vector>
#include <cmath>


using namespace myoddweb::nn;
using namespace test_helper;

class LayerOptimizerTest : public ::testing::Test {
protected:
    struct MockOptimizerLayer : public MockLayer {
        MockOptimizerLayer(unsigned num_neurons, unsigned num_inputs) : 
            MockLayer(0, num_neurons, num_inputs) {
            // Initialize with custom values if needed
        }
        
        // Expose protected methods for testing
        using Layer::apply_update_to_weight;
        using Layer::apply_update_to_vector;
        using Layer::apply_weight_gradient;
    };
};

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightSGD) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 1.0 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities = { 0.5 };
    std::vector<double> m1, m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.1 };
    
    // SGD Update:
    // grad = final_gradient + decay * value (if not bias)
    // velocity = momentum * prev_velocity + grad
    // value = value - learning_rate * velocity
    
    double input_grad = 0.2;
    double lr = 0.01;
    double clipping = 1.0;
    // double momentum = 0.9; // MockLayer default or we can set it?
    // Let's check Layer constructor: _momentum(momentum)
    // MockLayer(unsigned num_neurons, unsigned num_inputs) calls Layer with momentum 0.0?
    // Wait, test_helper.h: MockLayer(...) : Layer(..., 0.0)
    
    // Let's set momentum explicitly if we can, or just use 0.0
    // Actually, MockLayer in test_helper.h has 0.0 momentum.
    
    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::SGD, 0);
    
    // Momentum = 0.0
    // grad = 0.2 + 0.1 * 1.0 = 0.3
    // velocity = 0.0 * 0.5 + 0.3 = 0.3
    // value = 1.0 - 0.01 * 0.3 = 1.0 - 0.003 = 0.997
    
    EXPECT_NEAR(values[0], 0.997, 1e-9);
    EXPECT_NEAR(grads[0], 0.3, 1e-9);
    EXPECT_NEAR(velocities[0], 0.3, 1e-9);
}

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightAdam) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 1.0 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0 };
    std::vector<double> m2 = { 0.0 };
    std::vector<long long> timesteps = { 0 };
    std::vector<double> decays = { 0.0 };
    
    double input_grad = 0.1;
    double lr = 0.001;
    double clipping = 1.0;
    
    // Adam Update (first step, t=1):
    // beta1=0.0 (momentum), beta2=0.999
    // m1 = 0.0 * 0.0 + (1-0.0) * 0.1 = 0.1
    // m2 = 0.999 * 0.0 + (1-0.999) * (0.1^2) = 0.001 * 0.01 = 0.00001
    // m_hat = 0.1 / (1 - 0^1) = 0.1
    // v_hat = 0.00001 / (1 - 0.999^1) = 0.00001 / 0.001 = 0.01
    // update = 0.1 / (sqrt(0.01) + 1e-8) = 0.1 / (0.1 + 1e-8) approx 1.0
    // value = 1.0 - 0.001 * 1.0 = 0.999
    
    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::Adam, 0);
    
    EXPECT_NEAR(values[0], 0.999, 1e-6);
    EXPECT_NEAR(m1[0], 0.1, 1e-9);
    EXPECT_NEAR(m2[0], 0.00001, 1e-9);
}

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightAdamW) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 1.0 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0 };
    std::vector<double> m2 = { 0.0 };
    std::vector<long long> timesteps = { 0 };
    std::vector<double> decays = { 0.01 }; // Weight decay
    
    double input_grad = 0.1;
    double lr = 0.001;
    double clipping = 1.0;
    
    // AdamW Update:
    // weight_decay applied directly: current_weight *= (1 - lr * decay)
    // then subtract lr * update
    // Step 1: current_weight = 1.0 * (1 - 0.001 * 0.01) = 1.0 * (1 - 0.00001) = 0.99999
    // update approx 1.0 (same as Adam above)
    // value = 0.99999 - 0.001 * 1.0 = 0.99899
    
    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::AdamW, 0);
    
    EXPECT_NEAR(values[0], 0.99899, 1e-6);
}

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightNadam) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 1.0 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0 };
    std::vector<double> m2 = { 0.0 };
    std::vector<long long> timesteps = { 0 };
    std::vector<double> decays = { 0.0 };
    
    double input_grad = 0.1;
    double lr = 0.001;
    double clipping = 1.0;
    
    // Nadam Update (first step, t=1, beta1=0.0):
    // m1 = 0.1, m2 = 0.00001, m_hat = 0.1, v_hat = 0.01 (same as Adam)
    // m_nadam = beta1 * m_hat + ((1-beta1)*grad)/(1-beta1^t)
    // If beta1=0.0: m_nadam = 0.0 + (1.0 * 0.1) / (1 - 0) = 0.1
    // update = 0.1 / (sqrt(0.01) + 1e-8) approx 1.0
    // value = 1.0 - 0.001 * 1.0 = 0.999
    
    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::Nadam, 0);
    
    EXPECT_NEAR(values[0], 0.999, 1e-6);
}

TEST_F(LayerOptimizerTest, ClippingRobustness) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 1.0 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities = { 0.0 };
    std::vector<double> m1, m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.0 };

    // Extremely large gradient should panic
    EXPECT_ANY_THROW(layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, 1e7, 0.01, 1.0, OptimiserType::SGD, 0));
    
    // Non-finite gradient should panic
    EXPECT_ANY_THROW(layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, std::nan(""), 0.01, 1.0, OptimiserType::SGD, 0));
}

TEST_F(LayerOptimizerTest, ApplyUpdateToVectorSlice) {
    MockOptimizerLayer layer(10, 1);
    std::vector<double> values = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    std::vector<double> grads = { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 };
    std::vector<double> velocities = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    std::vector<double> m1, m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    
    double lr = 0.01;
    double clipping = 1.0;
    
    layer.apply_update_to_vector(values, grads, velocities, m1, m2, timesteps, decays, lr, clipping, false, OptimiserType::SGD, 3, 4);
    
    for (size_t i = 0; i < 3; ++i)
    {
      EXPECT_NEAR(values[i], 1.0, 1e-9);
      EXPECT_NEAR(grads[i], 0.2, 1e-9);
      EXPECT_NEAR(velocities[i], 0.5, 1e-9);
    }
    for (size_t i = 7; i < 10; ++i)
    {
      EXPECT_NEAR(values[i], 1.0, 1e-9);
      EXPECT_NEAR(grads[i], 0.2, 1e-9);
      EXPECT_NEAR(velocities[i], 0.5, 1e-9);
    }
    
    for (size_t i = 3; i < 7; ++i)
    {
      EXPECT_NEAR(values[i], 0.997, 1e-9);
      EXPECT_NEAR(grads[i], 0.3, 1e-9);
      EXPECT_NEAR(velocities[i], 0.3, 1e-9);
    }
}

