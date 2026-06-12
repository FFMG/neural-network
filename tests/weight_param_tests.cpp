#include <gtest/gtest.h>
#include "common/weightparam.h"
#include <cmath>
#include <limits>


using namespace myoddweb::nn;
TEST(WeightParamTest, ConstructorInitialization) {
    // Test the full constructor
    WeightParam wp(1.5, 0.1, 0.01, 0.001, 0.0001, 42, 0.05);
    
    EXPECT_DOUBLE_EQ(wp.get_value(), 1.5);
    EXPECT_DOUBLE_EQ(wp.get_raw_gradient(), 0.1);
    EXPECT_DOUBLE_EQ(wp.get_velocity(), 0.01);
    EXPECT_DOUBLE_EQ(wp.get_first_moment_estimate(), 0.001);
    EXPECT_DOUBLE_EQ(wp.get_second_moment_estimate(), 0.0001);
    EXPECT_EQ(wp.get_timestep(), 42);
    EXPECT_DOUBLE_EQ(wp.get_weight_decay(), 0.05);
}

TEST(WeightParamTest, ShortConstructor) {
    // Test the constructor for simpler optimisers
    WeightParam wp(1.5, 0.1, 0.01, 0.05);
    
    EXPECT_DOUBLE_EQ(wp.get_value(), 1.5);
    EXPECT_DOUBLE_EQ(wp.get_raw_gradient(), 0.1);
    EXPECT_DOUBLE_EQ(wp.get_velocity(), 0.01);
    EXPECT_DOUBLE_EQ(wp.get_first_moment_estimate(), 0.0);
    EXPECT_DOUBLE_EQ(wp.get_second_moment_estimate(), 0.0);
    EXPECT_EQ(wp.get_timestep(), 0);
    EXPECT_DOUBLE_EQ(wp.get_weight_decay(), 0.05);
}

TEST(WeightParamTest, CopySemantics) {
    WeightParam original(1.0, 2.0, 3.0, 4.0, 5.0, 6, 7.0);
    WeightParam copy = original;
    
    EXPECT_DOUBLE_EQ(copy.get_value(), 1.0);
    EXPECT_DOUBLE_EQ(copy.get_raw_gradient(), 2.0);
    EXPECT_DOUBLE_EQ(copy.get_velocity(), 3.0);
    EXPECT_DOUBLE_EQ(copy.get_first_moment_estimate(), 4.0);
    EXPECT_DOUBLE_EQ(copy.get_second_moment_estimate(), 5.0);
    EXPECT_EQ(copy.get_timestep(), 6);
    EXPECT_DOUBLE_EQ(copy.get_weight_decay(), 7.0);
    
    // Modify original, copy should remain unchanged
    original.set_value(10.0);
    EXPECT_DOUBLE_EQ(copy.get_value(), 1.0);
}

TEST(WeightParamTest, MoveSemantics) {
    WeightParam original(1.0, 2.0, 3.0, 4.0, 5.0, 6, 7.0);
    WeightParam moved = std::move(original);
    
    EXPECT_DOUBLE_EQ(moved.get_value(), 1.0);
    EXPECT_DOUBLE_EQ(moved.get_raw_gradient(), 2.0);
    EXPECT_DOUBLE_EQ(moved.get_velocity(), 3.0);
    EXPECT_DOUBLE_EQ(moved.get_first_moment_estimate(), 4.0);
    EXPECT_DOUBLE_EQ(moved.get_second_moment_estimate(), 5.0);
    EXPECT_EQ(moved.get_timestep(), 6);
    EXPECT_DOUBLE_EQ(moved.get_weight_decay(), 7.0);
    
    // Original should be reset to 0/defaults after move
    EXPECT_DOUBLE_EQ(original.get_value(), 0.0);
    EXPECT_DOUBLE_EQ(original.get_raw_gradient(), 0.0);
    EXPECT_DOUBLE_EQ(original.get_velocity(), 0.0);
}

TEST(WeightParamTest, SettersAndValidation) {
    WeightParam wp(0.0, 0.0, 0.0, 0.0);
    
    wp.set_value(1.23);
    EXPECT_DOUBLE_EQ(wp.get_value(), 1.23);
    
    wp.set_raw_gradient(4.56);
    EXPECT_DOUBLE_EQ(wp.get_raw_gradient(), 4.56);
    
    wp.set_velocity(7.89);
    EXPECT_DOUBLE_EQ(wp.get_velocity(), 7.89);
    
    wp.set_first_moment_estimate(0.12);
    EXPECT_DOUBLE_EQ(wp.get_first_moment_estimate(), 0.12);
    
    wp.set_second_moment_estimate(0.34);
    EXPECT_DOUBLE_EQ(wp.get_second_moment_estimate(), 0.34);
    
    wp.increment_timestep();
    EXPECT_EQ(wp.get_timestep(), 1);
    wp.increment_timestep();
    EXPECT_EQ(wp.get_timestep(), 2);
}

#if VALIDATE_DATA == 1
TEST(WeightParamTest, PanicOnInvalidValues) {
    WeightParam wp(0.0, 0.0, 0.0, 0.0);
    double nan = std::numeric_limits<double>::quiet_NaN();
    double inf = std::numeric_limits<double>::infinity();
    
    EXPECT_THROW(wp.set_value(nan), std::runtime_error);
    EXPECT_THROW(wp.set_value(inf), std::runtime_error);
    
    EXPECT_THROW(wp.set_raw_gradient(nan), std::runtime_error);
    EXPECT_THROW(wp.set_velocity(nan), std::runtime_error);
    EXPECT_THROW(wp.set_first_moment_estimate(nan), std::runtime_error);
    EXPECT_THROW(wp.set_second_moment_estimate(nan), std::runtime_error);
}
#endif
