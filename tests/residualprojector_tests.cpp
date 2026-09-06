#include <gtest/gtest.h>
#include "layers/residualprojector.h"
#include <cmath>
#include <numeric>
#include <vector>

using namespace myoddweb::nn;

TEST(ResidualProjectorTest, ConstructorWithActivationAndSeed)
{
  const unsigned input_size = 5;
  const unsigned output_size = 8;
  const double weight_decay = 0.02;
  const activation act(activation::method::relu, 0.0);

  ResidualProjector projector(input_size, output_size, act, weight_decay, 12345);

  EXPECT_EQ(projector.get_input_size(), input_size);
  EXPECT_EQ(projector.get_output_size(), output_size);

  const auto& w_values = projector.get_w_values();
  EXPECT_EQ(w_values.size(), static_cast<size_t>(input_size) * output_size);

  const auto& w_grads = projector.get_w_grads();
  EXPECT_EQ(w_grads.size(), static_cast<size_t>(input_size) * output_size);
  for (double g : w_grads)
  {
    EXPECT_DOUBLE_EQ(g, 0.0);
  }

  const auto& w_velocities = projector.get_w_velocities();
  EXPECT_EQ(w_velocities.size(), static_cast<size_t>(input_size) * output_size);
  for (double v : w_velocities)
  {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }

  const auto& w_m1 = projector.get_w_m1();
  EXPECT_EQ(w_m1.size(), static_cast<size_t>(input_size) * output_size);
  for (double m : w_m1)
  {
    EXPECT_DOUBLE_EQ(m, 0.0);
  }

  const auto& w_m2 = projector.get_w_m2();
  EXPECT_EQ(w_m2.size(), static_cast<size_t>(input_size) * output_size);
  for (double m : w_m2)
  {
    EXPECT_DOUBLE_EQ(m, 0.0);
  }

  const auto& w_timesteps = projector.get_w_timesteps();
  EXPECT_EQ(w_timesteps.size(), static_cast<size_t>(input_size) * output_size);
  for (long long t : w_timesteps)
  {
    EXPECT_EQ(t, 0);
  }

  const auto& w_decays = projector.get_w_decays();
  EXPECT_EQ(w_decays.size(), static_cast<size_t>(input_size) * output_size);
  for (double d : w_decays)
  {
    EXPECT_DOUBLE_EQ(d, weight_decay);
  }

  // Verify seed determinism: identical seed produces identical initial weights
  ResidualProjector identical_projector(input_size, output_size, act, weight_decay, 12345);
  EXPECT_EQ(projector.get_w_values(), identical_projector.get_w_values());

  // Different seed produces different initial weights
  ResidualProjector different_projector(input_size, output_size, act, weight_decay, 99999);
  EXPECT_NE(projector.get_w_values(), different_projector.get_w_values());

  // Nullopt seed produces non-empty weights
  ResidualProjector unseeded_projector(input_size, output_size, act, weight_decay, std::nullopt);
  EXPECT_EQ(unseeded_projector.get_w_values().size(), static_cast<size_t>(input_size) * output_size);
}

TEST(ResidualProjectorTest, ConstructorFromWeightParams)
{
  const unsigned output_size = 3;
  const unsigned input_size = 2;

  // 2D matrix: [output_size][input_size]
  std::vector<std::vector<WeightParam>> params(output_size);
  for (unsigned j = 0; j < output_size; ++j)
  {
    params[j].reserve(input_size);
    for (unsigned i = 0; i < input_size; ++i)
    {
      const double val = 1.0 + i * 0.1 + j * 0.01;
      const double grad = 0.05 * (i + j + 1);
      const double vel = 0.001 * (i + 1);
      const double m1 = 0.002 * (j + 1);
      const double m2 = 0.0003 * (i + j + 1);
      const long long t = 10 + i + j;
      const double decay = 0.01;
      params[j].emplace_back(val, grad, vel, m1, m2, t, decay);
    }
  }

  ResidualProjector projector(params);

  EXPECT_EQ(projector.get_input_size(), input_size);
  EXPECT_EQ(projector.get_output_size(), output_size);

  for (unsigned j = 0; j < output_size; ++j)
  {
    for (unsigned i = 0; i < input_size; ++i)
    {
      const size_t idx = i * output_size + j;
      const auto& expected = params[j][i];
      EXPECT_DOUBLE_EQ(projector.get_w_values()[idx], expected.get_value());
      EXPECT_DOUBLE_EQ(projector.get_w_grads()[idx], expected.get_raw_gradient());
      EXPECT_DOUBLE_EQ(projector.get_w_velocities()[idx], expected.get_velocity());
      EXPECT_DOUBLE_EQ(projector.get_w_m1()[idx], expected.get_first_moment_estimate());
      EXPECT_DOUBLE_EQ(projector.get_w_m2()[idx], expected.get_second_moment_estimate());
      EXPECT_EQ(projector.get_w_timesteps()[idx], expected.get_timestep());
      EXPECT_DOUBLE_EQ(projector.get_w_decays()[idx], expected.get_weight_decay());
    }
  }

  // Verify get_weight_params() reconstructs the identical 2D structure
  const auto& retrieved_params = projector.get_weight_params();
  ASSERT_EQ(retrieved_params.size(), output_size);
  for (unsigned j = 0; j < output_size; ++j)
  {
    ASSERT_EQ(retrieved_params[j].size(), input_size);
    for (unsigned i = 0; i < input_size; ++i)
    {
      EXPECT_DOUBLE_EQ(retrieved_params[j][i].get_value(), params[j][i].get_value());
      EXPECT_DOUBLE_EQ(retrieved_params[j][i].get_raw_gradient(), params[j][i].get_raw_gradient());
      EXPECT_DOUBLE_EQ(retrieved_params[j][i].get_velocity(), params[j][i].get_velocity());
      EXPECT_DOUBLE_EQ(retrieved_params[j][i].get_first_moment_estimate(), params[j][i].get_first_moment_estimate());
      EXPECT_DOUBLE_EQ(retrieved_params[j][i].get_second_moment_estimate(), params[j][i].get_second_moment_estimate());
      EXPECT_EQ(retrieved_params[j][i].get_timestep(), params[j][i].get_timestep());
      EXPECT_DOUBLE_EQ(retrieved_params[j][i].get_weight_decay(), params[j][i].get_weight_decay());
    }
  }
}

TEST(ResidualProjectorTest, ConstructorFromEmptyWeightParams)
{
  const std::vector<std::vector<WeightParam>> empty_params;
  ResidualProjector projector(empty_params);

  EXPECT_EQ(projector.get_input_size(), 0u);
  EXPECT_EQ(projector.get_output_size(), 0u);
  EXPECT_TRUE(projector.get_w_values().empty());
  EXPECT_TRUE(projector.get_w_grads().empty());
}

TEST(ResidualProjectorTest, CopyAndMoveSemantics)
{
  const unsigned input_size = 4;
  const unsigned output_size = 3;
  const activation act(activation::method::relu, 0.0);

  ResidualProjector original(input_size, output_size, act, 0.01, 777);

  // Copy construction
  ResidualProjector copy_proj(original);
  EXPECT_EQ(copy_proj.get_input_size(), original.get_input_size());
  EXPECT_EQ(copy_proj.get_output_size(), original.get_output_size());
  EXPECT_EQ(copy_proj.get_w_values(), original.get_w_values());

  // Mutating original does not mutate copy (deep copy)
  original.update_weight(0, 0, 5.0);
  EXPECT_NE(original.get_w_values(), copy_proj.get_w_values());

  // Move construction
  const auto original_values_before_move = original.get_w_values();
  ResidualProjector moved_proj(std::move(original));
  EXPECT_EQ(moved_proj.get_input_size(), input_size);
  EXPECT_EQ(moved_proj.get_output_size(), output_size);
  EXPECT_EQ(moved_proj.get_w_values(), original_values_before_move);
}

TEST(ResidualProjectorTest, ExplicitVectorsConstructor)
{
  const unsigned input_size = 2;
  const unsigned output_size = 2;
  const std::vector<double> w_values = { 0.5, -0.5, 1.0, -1.0 };
  const std::vector<double> w_grads = { 0.01, 0.02, 0.03, 0.04 };
  const std::vector<double> w_velocities = { 0.001, 0.002, 0.003, 0.004 };
  const std::vector<double> w_m1 = { 0.1, 0.2, 0.3, 0.4 };
  const std::vector<double> w_m2 = { 0.01, 0.04, 0.09, 0.16 };
  const std::vector<long long> w_timesteps = { 1, 2, 3, 4 };
  const std::vector<double> w_decays = { 0.05, 0.05, 0.05, 0.05 };

  ResidualProjector proj(
    input_size,
    output_size,
    w_values,
    w_grads,
    w_velocities,
    w_m1,
    w_m2,
    w_timesteps,
    w_decays
  );

  EXPECT_EQ(proj.get_input_size(), input_size);
  EXPECT_EQ(proj.get_output_size(), output_size);
  EXPECT_EQ(proj.get_w_values(), w_values);
  EXPECT_EQ(proj.get_w_grads(), w_grads);
  EXPECT_EQ(proj.get_w_velocities(), w_velocities);
  EXPECT_EQ(proj.get_w_m1(), w_m1);
  EXPECT_EQ(proj.get_w_m2(), w_m2);
  EXPECT_EQ(proj.get_w_timesteps(), w_timesteps);
  EXPECT_EQ(proj.get_w_decays(), w_decays);
}

TEST(ResidualProjectorTest, CreateFactoryMethods)
{
  // 1. Factory from WeightParam vector
  std::vector<std::vector<WeightParam>> empty_params;
  ResidualProjector* null_proj = ResidualProjector::create(empty_params);
  EXPECT_EQ(null_proj, nullptr);

  std::vector<std::vector<WeightParam>> valid_params(2, std::vector<WeightParam>(3, WeightParam(0.5, 0.0, 0.0, 0.0)));
  ResidualProjector* valid_proj = ResidualProjector::create(valid_params);
  ASSERT_NE(valid_proj, nullptr);
  EXPECT_EQ(valid_proj->get_input_size(), 3u);
  EXPECT_EQ(valid_proj->get_output_size(), 2u);
  delete valid_proj;

  // 2. Factory from layer parameters
  const activation act(activation::method::relu, 0.0);
  ResidualProjector* layer_zero_proj = ResidualProjector::create(0, act, 4, 4, 0.0, std::nullopt);
  EXPECT_EQ(layer_zero_proj, nullptr);

  ResidualProjector* layer_non_zero_proj = ResidualProjector::create(2, act, 6, 8, 0.01, 42);
  ASSERT_NE(layer_non_zero_proj, nullptr);
  EXPECT_EQ(layer_non_zero_proj->get_input_size(), 6u);
  EXPECT_EQ(layer_non_zero_proj->get_output_size(), 8u);
  delete layer_non_zero_proj;
}

TEST(ResidualProjectorTest, ProjectSingleVectorNumericalCorrectness)
{
  // Known matrix: input_size = 2, output_size = 3
  // W matrix:
  // in 0 -> [0.1, 0.2, 0.3]
  // in 1 -> [0.4, 0.5, 0.6]
  const unsigned input_size = 2;
  const unsigned output_size = 3;
  const std::vector<double> w_values = {
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
  };
  const std::vector<double> zeros_d(6, 0.0);
  const std::vector<long long> zeros_ll(6, 0);

  ResidualProjector proj(
    input_size,
    output_size,
    w_values,
    zeros_d,
    zeros_d,
    zeros_d,
    zeros_d,
    zeros_ll,
    zeros_d
  );

  const std::vector<double> inputs = { 2.0, 3.0 };
  // Expected:
  // out[0] = 2.0 * 0.1 + 3.0 * 0.4 = 0.2 + 1.2 = 1.4
  // out[1] = 2.0 * 0.2 + 3.0 * 0.5 = 0.4 + 1.5 = 1.9
  // out[2] = 2.0 * 0.3 + 3.0 * 0.6 = 0.6 + 1.8 = 2.4
  const auto actual = proj.project(inputs);

  ASSERT_EQ(actual.size(), output_size);
  EXPECT_NEAR(actual[0], 1.4, 1e-12);
  EXPECT_NEAR(actual[1], 1.9, 1e-12);
  EXPECT_NEAR(actual[2], 2.4, 1e-12);
}

TEST(ResidualProjectorTest, ProjectBatchVectorEquivalence)
{
  const unsigned input_size = 4;
  const unsigned output_size = 6;
  const activation act(activation::method::relu, 0.0);
  ResidualProjector projector(input_size, output_size, act, 0.0, 42);

  const std::vector<std::vector<double>> batch_inputs = {
    { 0.1, 0.2, 0.3, 0.4 },
    { -0.5, 0.0, 1.5, -2.0 },
    { 1.0, 1.0, 1.0, 1.0 },
    { 0.0, 0.0, 0.0, 0.0 }
  };

  const auto batch_output = projector.project_batch(batch_inputs);
  ASSERT_EQ(batch_output.size(), batch_inputs.size());

  for (size_t b = 0; b < batch_inputs.size(); ++b)
  {
    const auto single_output = projector.project(batch_inputs[b]);
    ASSERT_EQ(batch_output[b].size(), single_output.size());
    for (size_t j = 0; j < output_size; ++j)
    {
      EXPECT_NEAR(batch_output[b][j], single_output[j], 1e-12);
    }
  }

  // Empty batch returns empty result
  const std::vector<std::vector<double>> empty_batch;
  EXPECT_TRUE(projector.project_batch(empty_batch).empty());
}

TEST(ResidualProjectorTest, ProjectBatchRawPointersEquivalence)
{
  const unsigned input_size = 3;
  const unsigned output_size = 4;
  const activation act(activation::method::relu, 0.0);
  ResidualProjector projector(input_size, output_size, act, 0.0, 888);

  const std::vector<std::vector<double>> batch_inputs = {
    { 1.2, -0.4, 0.8 },
    { 0.0, 2.5, -1.1 },
    { -3.0, 0.1, 0.5 }
  };

  std::vector<const double*> raw_ptrs;
  raw_ptrs.reserve(batch_inputs.size());
  for (const auto& sample : batch_inputs)
  {
    raw_ptrs.push_back(sample.data());
  }

  const auto expected_from_vectors = projector.project_batch(batch_inputs);
  const auto actual_from_ptrs = projector.project_batch(raw_ptrs);

  ASSERT_EQ(actual_from_ptrs.size(), expected_from_vectors.size());
  for (size_t b = 0; b < expected_from_vectors.size(); ++b)
  {
    ASSERT_EQ(actual_from_ptrs[b].size(), expected_from_vectors[b].size());
    for (size_t j = 0; j < output_size; ++j)
    {
      EXPECT_DOUBLE_EQ(actual_from_ptrs[b][j], expected_from_vectors[b][j]);
    }
  }
}

TEST(ResidualProjectorTest, ProjectBatchIntoVariousBufferStates)
{
  const unsigned input_size = 4;
  const unsigned output_size = 5;
  const activation act(activation::method::relu, 0.0);
  ResidualProjector projector(input_size, output_size, act, 0.0, 314);

  const std::vector<std::vector<double>> batch_inputs = {
    { 0.2, 0.4, 0.6, 0.8 },
    { -0.1, -0.3, -0.5, -0.7 }
  };

  std::vector<const double*> raw_ptrs = {
    batch_inputs[0].data(),
    batch_inputs[1].data()
  };

  const auto expected = projector.project_batch(raw_ptrs);

  // State 1: out is completely empty
  std::vector<std::vector<double>> actual_empty;
  projector.project_batch_into(raw_ptrs, actual_empty);
  ASSERT_EQ(actual_empty.size(), expected.size());
  for (size_t b = 0; b < expected.size(); ++b)
  {
    ASSERT_EQ(actual_empty[b].size(), expected[b].size());
    for (size_t j = 0; j < output_size; ++j)
    {
      EXPECT_DOUBLE_EQ(actual_empty[b][j], expected[b][j]);
    }
  }

  // State 2: out is already correctly sized but filled with non-zero stale data
  std::vector<std::vector<double>> actual_stale(2, std::vector<double>(output_size, 999.0));
  projector.project_batch_into(raw_ptrs, actual_stale);
  ASSERT_EQ(actual_stale.size(), expected.size());
  for (size_t b = 0; b < expected.size(); ++b)
  {
    ASSERT_EQ(actual_stale[b].size(), expected[b].size());
    for (size_t j = 0; j < output_size; ++j)
    {
      EXPECT_DOUBLE_EQ(actual_stale[b][j], expected[b][j]);
    }
  }

  // State 3: out has incorrect dimensions (e.g. 5 batches with size 2)
  std::vector<std::vector<double>> actual_mismatched(5, std::vector<double>(2, -42.0));
  projector.project_batch_into(raw_ptrs, actual_mismatched);
  ASSERT_EQ(actual_mismatched.size(), expected.size());
  for (size_t b = 0; b < expected.size(); ++b)
  {
    ASSERT_EQ(actual_mismatched[b].size(), expected[b].size());
    for (size_t j = 0; j < output_size; ++j)
    {
      EXPECT_DOUBLE_EQ(actual_mismatched[b][j], expected[b][j]);
    }
  }
}

TEST(ResidualProjectorTest, ApplyWeightGradientWithoutDecay)
{
  const unsigned input_size = 2;
  const unsigned output_size = 2;
  const activation act(activation::method::relu, 0.0);
  ResidualProjector projector(input_size, output_size, act, 0.0, 100);

  const unsigned in = 1;
  const unsigned out = 0;
  const size_t idx = in * output_size + out;

  const double old_weight = projector.get_w_values()[idx];
  const double gradient = 0.5;
  const double learning_rate = 0.1;
  const double clipping_scale = 1.0;

  projector.apply_weight_gradient(gradient, learning_rate, in, out, clipping_scale);

  // w_new = w_old - lr * grad
  const double expected_new_weight = old_weight - learning_rate * gradient;
  EXPECT_DOUBLE_EQ(projector.get_w_values()[idx], expected_new_weight);
  EXPECT_DOUBLE_EQ(projector.get_w_grads()[idx], gradient);

  // Verify get_weight_params() reflects updated weight and gradient
  const auto& params = projector.get_weight_params();
  EXPECT_DOUBLE_EQ(params[out][in].get_value(), expected_new_weight);
  EXPECT_DOUBLE_EQ(params[out][in].get_raw_gradient(), gradient);
}

TEST(ResidualProjectorTest, ApplyWeightGradientWithDecayAndClipping)
{
  const unsigned input_size = 2;
  const unsigned output_size = 2;
  const double weight_decay = 0.05;
  const activation act(activation::method::relu, 0.0);
  ResidualProjector projector(input_size, output_size, act, weight_decay, 200);

  const unsigned in = 0;
  const unsigned out = 1;
  const size_t idx = in * output_size + out;

  const double old_weight = projector.get_w_values()[idx];
  const double gradient = 0.8;
  const double learning_rate = 0.02;
  const double clipping_scale = 0.5;

  projector.apply_weight_gradient(gradient, learning_rate, in, out, clipping_scale);

  // final_gradient = (gradient * clipping_scale) + (weight_decay * old_weight)
  const double expected_final_gradient = (gradient * clipping_scale) + (weight_decay * old_weight);
  const double expected_new_weight = old_weight - learning_rate * expected_final_gradient;

  EXPECT_NEAR(projector.get_w_values()[idx], expected_new_weight, 1e-12);
  EXPECT_NEAR(projector.get_w_grads()[idx], expected_final_gradient, 1e-12);

  const auto& params = projector.get_weight_params();
  EXPECT_NEAR(params[out][in].get_value(), expected_new_weight, 1e-12);
  EXPECT_NEAR(params[out][in].get_raw_gradient(), expected_final_gradient, 1e-12);
}

TEST(ResidualProjectorTest, UpdateWeightDirectDelta)
{
  const unsigned input_size = 3;
  const unsigned output_size = 2;
  const activation act(activation::method::relu, 0.0);
  ResidualProjector projector(input_size, output_size, act, 0.0, 300);

  const size_t out = 1;
  const size_t in = 2;
  const size_t idx = in * output_size + out;

  const double initial_val = projector.get_w_values()[idx];
  const double delta = 0.75;

  projector.update_weight(out, in, delta);

  EXPECT_DOUBLE_EQ(projector.get_w_values()[idx], initial_val + delta);
  EXPECT_DOUBLE_EQ(projector.get_weight_params()[out][in].get_value(), initial_val + delta);
}

TEST(ResidualProjectorTest, WeightParamsCachingAndInvalidation)
{
  const unsigned input_size = 2;
  const unsigned output_size = 2;
  const activation act(activation::method::relu, 0.0);
  ResidualProjector projector(input_size, output_size, act, 0.0, 400);

  // Consecutive calls to get_weight_params() should return the same cached object
  const auto& first_call = projector.get_weight_params();
  const auto& second_call = projector.get_weight_params();
  EXPECT_EQ(&first_call, &second_call);

  // Mutating via update_weight invalidates cache
  projector.update_weight(0, 0, 1.23);
  const auto& third_call = projector.get_weight_params();
  EXPECT_DOUBLE_EQ(third_call[0][0].get_value(), projector.get_w_values()[0]);

  // Mutating via apply_weight_gradient invalidates cache
  projector.apply_weight_gradient(0.1, 0.05, 1, 1, 1.0);
  const auto& fourth_call = projector.get_weight_params();
  EXPECT_DOUBLE_EQ(fourth_call[1][1].get_value(), projector.get_w_values()[1 * output_size + 1]);
}

TEST(ResidualProjectorTest, AccumulateSwaAverageRunningMean)
{
  const unsigned input_size = 2;
  const unsigned output_size = 2;
  const activation act(activation::method::relu, 0.0);

  ResidualProjector base_proj(input_size, output_size, act, 0.01, 1);
  ResidualProjector snap1(input_size, output_size, act, 0.01, 10);
  ResidualProjector snap2(input_size, output_size, act, 0.01, 20);
  ResidualProjector snap3(input_size, output_size, act, 0.01, 30);

  // Set specific values for snapshots
  for (size_t i = 0; i < 4; ++i)
  {
    snap1.update_weight(i % output_size, i / output_size, 1.0);
    snap2.update_weight(i % output_size, i / output_size, 3.0);
    snap3.update_weight(i % output_size, i / output_size, 5.0);
  }

  const auto snap1_vals = snap1.get_w_values();
  const auto snap2_vals = snap2.get_w_values();
  const auto snap3_vals = snap3.get_w_values();

  // Accumulate checkpoint 1: existing_swa_count = 0 -> base becomes exactly snap1
  base_proj.accumulate_swa_average(snap1, 0);
  for (size_t i = 0; i < 4; ++i)
  {
    EXPECT_NEAR(base_proj.get_w_values()[i], snap1_vals[i], 1e-12);
  }

  // Accumulate checkpoint 2: existing_swa_count = 1 -> base becomes (snap1 + snap2) / 2
  base_proj.accumulate_swa_average(snap2, 1);
  for (size_t i = 0; i < 4; ++i)
  {
    const double expected_mean = (snap1_vals[i] + snap2_vals[i]) / 2.0;
    EXPECT_NEAR(base_proj.get_w_values()[i], expected_mean, 1e-12);
  }

  // Accumulate checkpoint 3: existing_swa_count = 2 -> base becomes (snap1 + snap2 + snap3) / 3
  base_proj.accumulate_swa_average(snap3, 2);
  for (size_t i = 0; i < 4; ++i)
  {
    const double expected_mean = (snap1_vals[i] + snap2_vals[i] + snap3_vals[i]) / 3.0;
    EXPECT_NEAR(base_proj.get_w_values()[i], expected_mean, 1e-12);
  }

  // Verify non-weight properties (velocities, moments, timesteps, decays) were not modified
  for (double v : base_proj.get_w_velocities())
  {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
  for (double m : base_proj.get_w_m1())
  {
    EXPECT_DOUBLE_EQ(m, 0.0);
  }
  for (long long t : base_proj.get_w_timesteps())
  {
    EXPECT_EQ(t, 0);
  }
}

TEST(ResidualProjectorTest, UpdateLookaheadSlowWeights)
{
  const unsigned input_size = 3;
  const unsigned output_size = 2;
  const activation act(activation::method::relu, 0.0);

  ResidualProjector slow_proj(input_size, output_size, act, 0.0, 50);
  ResidualProjector fast_proj(input_size, output_size, act, 0.0, 51);

  const auto slow_before = slow_proj.get_w_values();
  const auto fast_before = fast_proj.get_w_values();

  const double alpha = 0.5;
  slow_proj.update_lookahead_slow_weights(fast_proj, alpha);

  const auto slow_after = slow_proj.get_w_values();
  const auto fast_after = fast_proj.get_w_values();

  for (size_t i = 0; i < slow_after.size(); ++i)
  {
    const double expected = slow_before[i] + alpha * (fast_before[i] - slow_before[i]);
    EXPECT_NEAR(slow_after[i], expected, 1e-12);
    // Fast projector weights are synchronised to new slow weights
    EXPECT_NEAR(fast_after[i], expected, 1e-12);
  }
}

TEST(ResidualProjectorTest, DimensionHandlingAndEdgeCases)
{
  const activation act(activation::method::relu, 0.0);

  // 1. Single scalar projection (1 x 1)
  ResidualProjector scalar_proj(1, 1, act, 0.0, 10);
  std::vector<double> scalar_in = { 3.0 };
  auto scalar_out = scalar_proj.project(scalar_in);
  ASSERT_EQ(scalar_out.size(), 1u);
  EXPECT_DOUBLE_EQ(scalar_out[0], 3.0 * scalar_proj.get_w_values()[0]);

  // 2. High reduction: 16 inputs -> 2 outputs
  ResidualProjector reduction_proj(16, 2, act, 0.0, 20);
  std::vector<double> reduction_in(16, 1.0);
  auto reduction_out = reduction_proj.project(reduction_in);
  ASSERT_EQ(reduction_out.size(), 2u);

  // 3. High expansion: 2 inputs -> 16 outputs
  ResidualProjector expansion_proj(2, 16, act, 0.0, 30);
  std::vector<double> expansion_in = { 1.0, -1.0 };
  auto expansion_out = expansion_proj.project(expansion_in);
  ASSERT_EQ(expansion_out.size(), 16u);
}

TEST(ResidualProjectorTest, CopyAndMoveAssignmentOperators)
{
  const unsigned in_size1 = 4;
  const unsigned out_size1 = 3;
  const activation act(activation::method::relu, 0.0);

  ResidualProjector proj1(in_size1, out_size1, act, 0.02, 101);
  proj1.apply_weight_gradient(0.5, 0.01, 1, 2, 1.0);
  proj1.update_weight(0, 3, 2.5);

  ResidualProjector proj2(2, 2, act, 0.0, 202);

  // 1. Copy assignment
  proj2 = proj1;
  EXPECT_EQ(proj2.get_input_size(), in_size1);
  EXPECT_EQ(proj2.get_output_size(), out_size1);
  EXPECT_EQ(proj2.get_w_values(), proj1.get_w_values());
  EXPECT_EQ(proj2.get_w_grads(), proj1.get_w_grads());
  EXPECT_EQ(proj2.get_weight_params()[0][3].get_value(), proj1.get_weight_params()[0][3].get_value());

  // Deep copy verification
  proj1.update_weight(0, 0, 9.99);
  EXPECT_NE(proj1.get_w_values()[0], proj2.get_w_values()[0]);

  // Self-assignment
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
  ResidualProjector& proj2_ref = proj2;
  proj2 = proj2_ref;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  EXPECT_EQ(proj2.get_input_size(), in_size1);
  EXPECT_EQ(proj2.get_output_size(), out_size1);

  // 2. Move assignment
  const auto values_before_move = proj2.get_w_values();
  ResidualProjector proj3(1, 1, act, 0.0, 303);
  proj3 = std::move(proj2);
  EXPECT_EQ(proj3.get_input_size(), in_size1);
  EXPECT_EQ(proj3.get_output_size(), out_size1);
  EXPECT_EQ(proj3.get_w_values(), values_before_move);
}

TEST(ResidualProjectorTest, ProjectBatchMultiBatchSizesEquivalence)
{
  const unsigned in_size = 5;
  const unsigned out_size = 7;
  const activation act(activation::method::linear, 0.0);
  ResidualProjector projector(in_size, out_size, act, 0.0, 777);

  const std::vector<size_t> test_batch_sizes = { 1, 2, 3, 4, 7, 8, 15, 16, 33, 64 };

  for (size_t batch_size : test_batch_sizes)
  {
    std::vector<std::vector<double>> batch_inputs(batch_size, std::vector<double>(in_size));
    std::vector<const double*> raw_ptrs(batch_size);
    std::vector<double> flat_inputs(batch_size * in_size);
    std::vector<double> flat_outputs(batch_size * out_size, 0.0);

    for (size_t b = 0; b < batch_size; ++b)
    {
      for (size_t i = 0; i < in_size; ++i)
      {
        const double val = std::sin(static_cast<double>(b * in_size + i + 1));
        batch_inputs[b][i] = val;
        flat_inputs[b * in_size + i] = val;
      }
      raw_ptrs[b] = batch_inputs[b].data();
    }

    // Method 1: project single
    std::vector<std::vector<double>> expected(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
      expected[b] = projector.project(batch_inputs[b]);
    }

    // Method 2: project_batch(vector of vectors)
    const auto out_vec = projector.project_batch(batch_inputs);

    // Method 3: project_batch_into(vector of vectors)
    std::vector<std::vector<double>> out_vec_into;
    projector.project_batch_into(batch_inputs, out_vec_into);

    // Method 4: project_batch(vector of raw pointers)
    const auto out_ptrs = projector.project_batch(raw_ptrs);

    // Method 5: project_batch_into(vector of raw pointers)
    std::vector<std::vector<double>> out_ptrs_into;
    projector.project_batch_into(raw_ptrs, out_ptrs_into);

    // Method 6: project_batch flat buffer
    projector.project_batch(flat_inputs.data(), flat_outputs.data(), batch_size);

    ASSERT_EQ(out_vec.size(), batch_size);
    ASSERT_EQ(out_vec_into.size(), batch_size);
    ASSERT_EQ(out_ptrs.size(), batch_size);
    ASSERT_EQ(out_ptrs_into.size(), batch_size);

    for (size_t b = 0; b < batch_size; ++b)
    {
      for (size_t j = 0; j < out_size; ++j)
      {
        const double exp_val = expected[b][j];
        EXPECT_NEAR(out_vec[b][j], exp_val, 1e-12);
        EXPECT_NEAR(out_vec_into[b][j], exp_val, 1e-12);
        EXPECT_NEAR(out_ptrs[b][j], exp_val, 1e-12);
        EXPECT_NEAR(out_ptrs_into[b][j], exp_val, 1e-12);
        EXPECT_NEAR(flat_outputs[b * out_size + j], exp_val, 1e-12);
      }
    }
  }
}

TEST(ResidualProjectorTest, ProjectRawPointersBufferInto)
{
  const unsigned in_size = 4;
  const unsigned out_size = 4;
  const activation act(activation::method::relu, 0.0);
  ResidualProjector projector(in_size, out_size, act, 0.0, 123);

  const std::vector<double> single_in = { 1.0, 2.0, 3.0, 4.0 };
  std::vector<double> single_out(out_size, 0.0);

  projector.project(single_in.data(), single_out.data());

  const auto expected = projector.project(single_in);
  for (size_t j = 0; j < out_size; ++j)
  {
    EXPECT_NEAR(single_out[j], expected[j], 1e-12);
  }

  // Nullptr safety
  projector.project(nullptr, single_out.data());
  projector.project(single_in.data(), nullptr);
  projector.project_batch(nullptr, single_out.data(), 1);
  projector.project_batch(single_in.data(), nullptr, 1);
}

TEST(ResidualProjectorTest, AccumulateSwaAverageVectorisedEquivalence)
{
  const unsigned in_size = 8;
  const unsigned out_size = 6;
  const activation act(activation::method::relu, 0.0);

  ResidualProjector swa_proj(in_size, out_size, act, 0.0, 1);
  const size_t num_snapshots = 5;
  std::vector<ResidualProjector> snapshots;
  snapshots.reserve(num_snapshots);

  for (size_t s = 0; s < num_snapshots; ++s)
  {
    snapshots.emplace_back(in_size, out_size, act, 0.0, static_cast<uint32_t>(100 + s * 17));
  }

  // Accumulate snapshots into swa_proj
  for (size_t s = 0; s < num_snapshots; ++s)
  {
    swa_proj.accumulate_swa_average(snapshots[s], s);
  }

  // Compute exact arithmetic mean across all snapshots
  const size_t total_weights = static_cast<size_t>(in_size) * out_size;
  std::vector<double> expected_mean(total_weights, 0.0);
  for (size_t s = 0; s < num_snapshots; ++s)
  {
    const auto& snap_w = snapshots[s].get_w_values();
    for (size_t i = 0; i < total_weights; ++i)
    {
      expected_mean[i] += snap_w[i];
    }
  }
  for (size_t i = 0; i < total_weights; ++i)
  {
    expected_mean[i] /= static_cast<double>(num_snapshots);
  }

  const auto& actual_swa_w = swa_proj.get_w_values();
  for (size_t i = 0; i < total_weights; ++i)
  {
    EXPECT_NEAR(actual_swa_w[i], expected_mean[i], 1e-11);
  }
}

TEST(ResidualProjectorTest, ZeroDimensionEdgeCases)
{
  const activation act(activation::method::relu, 0.0);

  // 1. Zero output size
  ResidualProjector zero_out(4, 0, act, 0.0, 10);
  EXPECT_EQ(zero_out.get_input_size(), 4u);
  EXPECT_EQ(zero_out.get_output_size(), 0u);
  EXPECT_TRUE(zero_out.project({ 1.0, 2.0, 3.0, 4.0 }).empty());

  std::vector<std::vector<double>> batch_in = { { 1.0, 2.0, 3.0, 4.0 } };
  auto batch_out = zero_out.project_batch(batch_in);
  ASSERT_EQ(batch_out.size(), 1u);
  EXPECT_TRUE(batch_out[0].empty());

  // 2. Zero input size
  ResidualProjector zero_in(0, 4, act, 0.0, 10);
  EXPECT_EQ(zero_in.get_input_size(), 0u);
  EXPECT_EQ(zero_in.get_output_size(), 4u);
  auto proj_zero_in = zero_in.project({});
  ASSERT_EQ(proj_zero_in.size(), 4u);
  for (double val : proj_zero_in)
  {
    EXPECT_DOUBLE_EQ(val, 0.0);
  }
}

