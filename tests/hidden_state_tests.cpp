#include <gtest/gtest.h>
#include "../src/neuralnetwork/hiddenstate.h"
#include <vector>

TEST(HiddenStateTest, ConstructorInitializesWithZeros)
{
  unsigned num_neurons = 5;
  HiddenState state(num_neurons);

  const auto& sums = state.get_pre_activation_sums();
  const auto& values = state.get_hidden_state_values();

  ASSERT_EQ(sums.size(), num_neurons);
  ASSERT_EQ(values.size(), num_neurons);

  for (unsigned i = 0; i < num_neurons; ++i)
  {
    EXPECT_DOUBLE_EQ(sums[i], 0.0);
    EXPECT_DOUBLE_EQ(values[i], 0.0);
  }
}

TEST(HiddenStateTest, SetAndGetPreActivationSums)
{
  unsigned num_neurons = 3;
  HiddenState state(num_neurons);
  std::vector<double> new_sums = { 1.1, 2.2, 3.3 };

  state.set_pre_activation_sums(new_sums);

  EXPECT_EQ(state.get_pre_activation_sums(), new_sums);
  EXPECT_DOUBLE_EQ(state.get_pre_activation_sum_at_neuron(0), 1.1);
  EXPECT_DOUBLE_EQ(state.get_pre_activation_sum_at_neuron(1), 2.2);
  EXPECT_DOUBLE_EQ(state.get_pre_activation_sum_at_neuron(2), 3.3);
}

TEST(HiddenStateTest, SetAndGetHiddenStateValues)
{
  unsigned num_neurons = 3;
  HiddenState state(num_neurons);
  std::vector<double> new_values = { -0.5, 0.0, 0.5 };

  state.set_hidden_state_values(new_values);

  EXPECT_EQ(state.get_hidden_state_values(), new_values);
  EXPECT_DOUBLE_EQ(state.get_hidden_state_value_at_neuron(0), -0.5);
  EXPECT_DOUBLE_EQ(state.get_hidden_state_value_at_neuron(1), 0.0);
  EXPECT_DOUBLE_EQ(state.get_hidden_state_value_at_neuron(2), 0.5);
}

TEST(HiddenStateTest, CopyConstructorAndAssignment)
{
  unsigned num_neurons = 2;
  HiddenState state1(num_neurons);
  state1.set_pre_activation_sums({ 1.0, 2.0 });
  state1.set_hidden_state_values({ 3.0, 4.0 });

  HiddenState state2(state1);
  EXPECT_EQ(state2.get_pre_activation_sums(), state1.get_pre_activation_sums());
  EXPECT_EQ(state2.get_hidden_state_values(), state1.get_hidden_state_values());

  HiddenState state3(1);
  state3 = state1;
  EXPECT_EQ(state3.get_pre_activation_sums(), state1.get_pre_activation_sums());
  EXPECT_EQ(state3.get_hidden_state_values(), state1.get_hidden_state_values());
}

TEST(HiddenStateTest, MoveConstructorAndAssignment)
{
  unsigned num_neurons = 2;
  HiddenState state1(num_neurons);
  std::vector<double> sums = { 1.0, 2.0 };
  std::vector<double> values = { 3.0, 4.0 };
  state1.set_pre_activation_sums(sums);
  state1.set_hidden_state_values(values);

  HiddenState state2(std::move(state1));
  EXPECT_EQ(state2.get_pre_activation_sums(), sums);
  EXPECT_EQ(state2.get_hidden_state_values(), values);

  HiddenState state3(1);
  state3 = std::move(state2);
  EXPECT_EQ(state3.get_pre_activation_sums(), sums);
  EXPECT_EQ(state3.get_hidden_state_values(), values);
}

TEST(HiddenStateTest, ValidationLogic)
{
#if VALIDATE_DATA == 1
  unsigned num_neurons = 2;
  HiddenState state(num_neurons);

  EXPECT_THROW(state.get_pre_activation_sum_at_neuron(2), std::runtime_error);
  EXPECT_THROW(state.get_hidden_state_value_at_neuron(2), std::runtime_error);
#endif
}
