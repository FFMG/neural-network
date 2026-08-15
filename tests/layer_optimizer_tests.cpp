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

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightLion) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 1.0 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0 };
    std::vector<double> m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.0 };
    
    double input_grad = 0.2;
    double lr = 0.01;
    double clipping = 1.0;
    
    // Lion Update (first step, beta1=0.0 since MockOptimizerLayer has momentum=0.0):
    // beta1 = get_momentum(0) = 0.0, beta2 = 0.99
    // update = beta1 * m1[0] + (1 - beta1) * final_grad = 0.0 * 0.0 + 1.0 * 0.2 = 0.2
    // sign_update = sign(0.2) = 1.0
    // values[0] = 1.0 - lr * sign_update = 1.0 - 0.01 * 1.0 = 0.99
    // m1[0] = beta2 * m1[0] + (1 - beta2) * final_grad = 0.99 * 0.0 + 0.01 * 0.2 = 0.002
    // grads[0] = 0.2
    
    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::Lion, 0);
    
    EXPECT_NEAR(values[0], 0.99, 1e-9);
    EXPECT_NEAR(m1[0], 0.002, 1e-9);
    EXPECT_NEAR(grads[0], 0.2, 1e-9);

    // Second step with negative gradient:
    input_grad = -0.5;
    // update = 0.0 * 0.002 + 1.0 * (-0.5) = -0.5
    // sign_update = sign(-0.5) = -1.0
    // values[0] = 0.99 - 0.01 * (-1.0) = 1.00
    // m1[0] = 0.99 * 0.002 + 0.01 * (-0.5) = 0.00198 - 0.005 = -0.00302
    // grads[0] = -0.5
    
    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::Lion, 0);
    
    EXPECT_NEAR(values[0], 1.00, 1e-9);
    EXPECT_NEAR(m1[0], -0.00302, 1e-9);
    EXPECT_NEAR(grads[0], -0.5, 1e-9);
}

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightLionWithDecay) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 2.0 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0 };
    std::vector<double> m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.05 }; // Weight decay
    
    double input_grad = 0.1;
    double lr = 0.01;
    double clipping = 1.0;
    
    // Lion with decoupled weight decay:
    // current_weight = values[0] * (1.0 - lr * decays[0]) = 2.0 * (1.0 - 0.01 * 0.05) = 2.0 * 0.9995 = 1.999
    // update = 0.0 * 0.0 + 1.0 * 0.1 = 0.1 -> sign_update = 1.0
    // values[0] = 1.999 - lr * 1.0 = 1.989
    // m1[0] = 0.99 * 0.0 + 0.01 * 0.1 = 0.001
    
    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::Lion, 0);
    
    EXPECT_NEAR(values[0], 1.989, 1e-9);
    EXPECT_NEAR(m1[0], 0.001, 1e-9);
}

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightLionZeroGradient) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 1.5 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0 };
    std::vector<double> m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.0 };
    
    double input_grad = 0.0;
    double lr = 0.01;
    double clipping = 1.0;
    
    // Gradient is zero and momentum is zero:
    // update = 0.0 -> sign_update = 0.0
    // values[0] remains 1.5
    // m1[0] remains 0.0
    
    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::Lion, 0);
    
    EXPECT_NEAR(values[0], 1.5, 1e-9);
    EXPECT_NEAR(m1[0], 0.0, 1e-9);
    EXPECT_NEAR(grads[0], 0.0, 1e-9);
}

TEST_F(LayerOptimizerTest, ApplyUpdateToVectorLion) {
    MockOptimizerLayer layer(4, 1);
    std::vector<double> values = { 1.0, 2.0, 3.0, 4.0 };
    std::vector<double> grads = { 0.5, -0.5, 0.0, 1.2 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.0, 0.0, 0.0, 0.0 };
    
    double lr = 0.01;
    double clipping = 1.0;
    
    layer.apply_update_to_vector(values, grads, velocities, m1, m2, timesteps, decays, lr, clipping, false, OptimiserType::Lion, 0, 4);
    
    // beta1 = 0.0, beta2 = 0.99
    // elem 0: c = 0.5 > 0 -> sign = 1.0 -> val = 1.0 - 0.01 * 1.0 = 0.99, m1 = 0.01 * 0.5 = 0.005
    // elem 1: c = -0.5 < 0 -> sign = -1.0 -> val = 2.0 - 0.01 * (-1.0) = 2.01, m1 = 0.01 * (-0.5) = -0.005
    // elem 2: c = 0.0 -> sign = 0.0 -> val = 3.0, m1 = 0.0
    // elem 3: c = 1.2 > 0 -> sign = 1.0 -> val = 4.0 - 0.01 * 1.0 = 3.99, m1 = 0.01 * 1.2 = 0.012
    
    EXPECT_NEAR(values[0], 0.99, 1e-9);
    EXPECT_NEAR(values[1], 2.01, 1e-9);
    EXPECT_NEAR(values[2], 3.0, 1e-9);
    EXPECT_NEAR(values[3], 3.99, 1e-9);

    EXPECT_NEAR(m1[0], 0.005, 1e-9);
    EXPECT_NEAR(m1[1], -0.005, 1e-9);
    EXPECT_NEAR(m1[2], 0.0, 1e-9);
    EXPECT_NEAR(m1[3], 0.012, 1e-9);
}

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightLionClampsExtremeValues) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> velocities;
    std::vector<double> m2;
    std::vector<long long> timesteps;
    double lr = 0.01;
    double clipping = 1.0;

    // Positive side: weight is just below the clamp bound and the update pushes it over.
    // beta1 = 0.0 -> update = final_gradient = -1.0 -> sign_update = -1.0
    // unclamped: 99999.995 - lr * (-1.0) = 100000.005 -> clamps to 100000.0
    {
        std::vector<double> values = { 99999.995 };
        std::vector<double> grads = { 0.0 };
        std::vector<double> m1 = { 0.0 };
        std::vector<double> decays = { 0.0 };

        layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, -1.0, lr, clipping, OptimiserType::Lion, 0);

        EXPECT_NEAR(values[0], 100000.0, 1e-9);
    }

    // Negative side: weight is just above the negative clamp bound and the update pushes it under.
    // beta1 = 0.0 -> update = final_gradient = 1.0 -> sign_update = 1.0
    // unclamped: -99999.995 - lr * 1.0 = -100000.005 -> clamps to -100000.0
    {
        std::vector<double> values = { -99999.995 };
        std::vector<double> grads = { 0.0 };
        std::vector<double> m1 = { 0.0 };
        std::vector<double> decays = { 0.0 };

        layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, 1.0, lr, clipping, OptimiserType::Lion, 0);

        EXPECT_NEAR(values[0], -100000.0, 1e-9);
    }
}

TEST_F(LayerOptimizerTest, ApplyUpdateToWeightLionWithClipping) {
    MockOptimizerLayer layer(1, 1);
    std::vector<double> values = { 1.0 };
    std::vector<double> grads = { 0.0 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0 };
    std::vector<double> m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.0 };

    double input_grad = 2.0;
    double lr = 0.01;
    double clipping = 0.25;

    // final_gradient = input_grad * clipping = 2.0 * 0.25 = 0.5
    // beta1 = 0.0 -> update = 0.5 -> sign_update = 1.0
    // values[0] = 1.0 - 0.01 * 1.0 = 0.99
    // m1[0] = 0.99 * 0.0 + 0.01 * 0.5 = 0.005
    // grads[0] = 0.5 (the clipped gradient, not the raw 2.0)

    layer.apply_update_to_weight(values, grads, velocities, m1, m2, timesteps, decays, 0, input_grad, lr, clipping, OptimiserType::Lion, 0);

    EXPECT_NEAR(values[0], 0.99, 1e-9);
    EXPECT_NEAR(m1[0], 0.005, 1e-9);
    EXPECT_NEAR(grads[0], 0.5, 1e-9);
}

TEST_F(LayerOptimizerTest, ApplyUpdateToVectorLionSkipsDecayForBias) {
    MockOptimizerLayer layer(4, 1);
    std::vector<double> values = { 1.0, 2.0, 3.0, 4.0 };
    std::vector<double> grads = { 0.5, -0.5, 0.0, 1.2 };
    std::vector<double> velocities;
    std::vector<double> m1 = { 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> m2;
    std::vector<long long> timesteps;
    std::vector<double> decays = { 0.5, 0.5, 0.5, 0.5 }; // Large decay that would be very visible if wrongly applied.

    double lr = 0.01;
    double clipping = 1.0;

    // is_bias = true -> decay must be skipped even though decays[] is populated.
    // Result should be identical to ApplyUpdateToVectorLion (decays = 0) above.
    layer.apply_update_to_vector(values, grads, velocities, m1, m2, timesteps, decays, lr, clipping, true, OptimiserType::Lion, 0, 4);

    EXPECT_NEAR(values[0], 0.99, 1e-9);
    EXPECT_NEAR(values[1], 2.01, 1e-9);
    EXPECT_NEAR(values[2], 3.0, 1e-9);
    EXPECT_NEAR(values[3], 3.99, 1e-9);

    EXPECT_NEAR(m1[0], 0.005, 1e-9);
    EXPECT_NEAR(m1[1], -0.005, 1e-9);
    EXPECT_NEAR(m1[2], 0.0, 1e-9);
    EXPECT_NEAR(m1[3], 0.012, 1e-9);
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

TEST_F(LayerOptimizerTest, LazyTimestepSynchronization) {
    MockOptimizerLayer layer(5, 1);
    std::vector<double> values = { 1.0, 1.0, 1.0, 1.0, 1.0 };
    std::vector<double> grads = { 0.2, 0.2, 0.2, 0.2, 0.2 };
    std::vector<double> velocities = { 0.5, 0.5, 0.5, 0.5, 0.5 };
    std::vector<double> m1 = { 0.1, 0.1, 0.1, 0.1, 0.1 };
    std::vector<double> m2 = { 0.1, 0.1, 0.1, 0.1, 0.1 };
    std::vector<long long> timesteps = { 0, 0, 0, 0, 0 };
    std::vector<double> decays = { 0.1, 0.1, 0.1, 0.1, 0.1 };
    
    double lr = 0.01;
    double clipping = 1.0;
    
    layer.apply_update_to_vector(values, grads, velocities, m1, m2, timesteps, decays, lr, clipping, false, OptimiserType::Adam, 0, 5);
    
    EXPECT_EQ(timesteps[0], 1);
    for (size_t i = 1; i < 5; ++i)
    {
      EXPECT_EQ(timesteps[i], 0);
    }
    
    layer.set_w_timesteps(timesteps);
    
    const auto& synchronized = layer.get_w_timesteps();
    for (size_t i = 0; i < 5; ++i)
    {
      EXPECT_EQ(synchronized[i], 1);
    }
}

