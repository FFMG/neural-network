#include <gtest/gtest.h>
#include "common/hiddenstate.h"
#include <vector>
#include <algorithm>


using namespace myoddweb::nn;
TEST(HiddenStateTest, ConstructorInitializesCorrectly)
{
  unsigned num_neurons = 5;
  unsigned num_pre_activations = 10;
  std::vector<double> pre_sums(num_pre_activations, 0.0);
  std::vector<double> hidden_values(num_neurons, 0.0);
  std::vector<double> cell_values(num_neurons, 0.0);

  HiddenState state(pre_sums.data(), hidden_values.data(), cell_values.data(), num_neurons, num_pre_activations);

  const auto sums = state.get_pre_activation_sums();
  const auto values = state.get_hidden_state_values();
  const auto cells = state.get_cell_state_values();

  ASSERT_EQ(sums.size(), num_pre_activations);
  ASSERT_EQ(values.size(), num_neurons);
  ASSERT_EQ(cells.size(), num_neurons);

  for (unsigned i = 0; i < num_pre_activations; ++i)
  {
    EXPECT_DOUBLE_EQ(sums[i], 0.0);
  }
  for (unsigned i = 0; i < num_neurons; ++i)
  {
    EXPECT_DOUBLE_EQ(values[i], 0.0);
    EXPECT_DOUBLE_EQ(cells[i], 0.0);
  }
}

TEST(HiddenStateTest, SetAndGetPreActivationSums)
{
  unsigned num_neurons = 3;
  std::vector<double> pre_sums(num_neurons, 0.0);
  std::vector<double> hidden_values(num_neurons, 0.0);
  std::vector<double> cell_values(num_neurons, 0.0);
  HiddenState state(pre_sums.data(), hidden_values.data(), cell_values.data(), num_neurons, num_neurons);

  std::vector<double> new_sums = { 1.1, 2.2, 3.3 };
  state.set_pre_activation_sums(new_sums);

  auto sums = state.get_pre_activation_sums();
  ASSERT_EQ(sums.size(), new_sums.size());
  for (size_t i = 0; i < sums.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(sums[i], new_sums[i]);
  }
  EXPECT_DOUBLE_EQ(state.get_pre_activation_sum_at_neuron(0), 1.1);
  EXPECT_DOUBLE_EQ(state.get_pre_activation_sum_at_neuron(1), 2.2);
  EXPECT_DOUBLE_EQ(state.get_pre_activation_sum_at_neuron(2), 3.3);
}

TEST(HiddenStateTest, SetAndGetHiddenStateValues)
{
  unsigned num_neurons = 3;
  std::vector<double> pre_sums(num_neurons, 0.0);
  std::vector<double> hidden_values(num_neurons, 0.0);
  std::vector<double> cell_values(num_neurons, 0.0);
  HiddenState state(pre_sums.data(), hidden_values.data(), cell_values.data(), num_neurons, num_neurons);

  std::vector<double> new_values = { -0.5, 0.0, 0.5 };
  state.set_hidden_state_values(new_values);

  auto values = state.get_hidden_state_values();
  ASSERT_EQ(values.size(), new_values.size());
  for (size_t i = 0; i < values.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(values[i], new_values[i]);
  }
  EXPECT_DOUBLE_EQ(state.get_hidden_state_value_at_neuron(0), -0.5);
  EXPECT_DOUBLE_EQ(state.get_hidden_state_value_at_neuron(1), 0.0);
  EXPECT_DOUBLE_EQ(state.get_hidden_state_value_at_neuron(2), 0.5);
}

TEST(HiddenStateTest, SetAndGetCellStateValues)
{
  unsigned num_neurons = 3;
  std::vector<double> pre_sums(num_neurons, 0.0);
  std::vector<double> hidden_values(num_neurons, 0.0);
  std::vector<double> cell_values(num_neurons, 0.0);
  HiddenState state(pre_sums.data(), hidden_values.data(), cell_values.data(), num_neurons, num_neurons);

  std::vector<double> new_values = { 0.1, 0.2, 0.3 };
  state.set_cell_state_values(new_values);

  auto values = state.get_cell_state_values();
  ASSERT_EQ(values.size(), new_values.size());
  for (size_t i = 0; i < values.size(); ++i)
  {
    EXPECT_DOUBLE_EQ(values[i], new_values[i]);
  }
  EXPECT_DOUBLE_EQ(state.get_cell_state_value_at_neuron(0), 0.1);
  EXPECT_DOUBLE_EQ(state.get_cell_state_value_at_neuron(1), 0.2);
  EXPECT_DOUBLE_EQ(state.get_cell_state_value_at_neuron(2), 0.3);
}

TEST(HiddenStateTest, SetIndividualValues)
{
  unsigned num_neurons = 3;
  std::vector<double> pre_sums(num_neurons, 0.0);
  std::vector<double> hidden_values(num_neurons, 0.0);
  std::vector<double> cell_values(num_neurons, 0.0);
  HiddenState state(pre_sums.data(), hidden_values.data(), cell_values.data(), num_neurons, num_neurons);

  state.set_pre_activation_sum(1, 4.4);
  EXPECT_DOUBLE_EQ(state.get_pre_activation_sum_at_neuron(1), 4.4);
  EXPECT_DOUBLE_EQ(pre_sums[1], 4.4);

  state.set_hidden_state_value(0, -1.1);
  EXPECT_DOUBLE_EQ(state.get_hidden_state_value_at_neuron(0), -1.1);
  EXPECT_DOUBLE_EQ(hidden_values[0], -1.1);

  state.set_cell_state_value(2, 9.9);
  EXPECT_DOUBLE_EQ(state.get_cell_state_value_at_neuron(2), 9.9);
  EXPECT_DOUBLE_EQ(cell_values[2], 9.9);

  state.get_pre_activation_sum(0) = 7.7;
  EXPECT_DOUBLE_EQ(state.get_pre_activation_sum_at_neuron(0), 7.7);
  EXPECT_DOUBLE_EQ(pre_sums[0], 7.7);
}

TEST(HiddenStateTest, ShallowCopyConstructorAndAssignment)
{
  unsigned num_neurons = 2;
  std::vector<double> pre_sums(num_neurons, 0.0);
  std::vector<double> hidden_values(num_neurons, 0.0);
  std::vector<double> cell_values(num_neurons, 0.0);
  HiddenState state1(pre_sums.data(), hidden_values.data(), cell_values.data(), num_neurons, num_neurons);

  state1.set_pre_activation_sums({ 1.0, 2.0 });
  state1.set_hidden_state_values({ 3.0, 4.0 });

  HiddenState state2(state1);
  EXPECT_EQ(state2.get_pre_activation_sums().data(), state1.get_pre_activation_sums().data());
  EXPECT_EQ(state2.get_hidden_state_values().data(), state1.get_hidden_state_values().data());

  HiddenState state3;
  state3 = state1;
  EXPECT_EQ(state3.get_pre_activation_sums().data(), state1.get_pre_activation_sums().data());
  EXPECT_EQ(state3.get_hidden_state_values().data(), state1.get_hidden_state_values().data());
}

TEST(HiddenStateTest, ShallowMoveConstructorAndAssignment)
{
  unsigned num_neurons = 2;
  std::vector<double> pre_sums(num_neurons, 0.0);
  std::vector<double> hidden_values(num_neurons, 0.0);
  std::vector<double> cell_values(num_neurons, 0.0);
  HiddenState state1(pre_sums.data(), hidden_values.data(), cell_values.data(), num_neurons, num_neurons);

  double* original_pre_ptr = pre_sums.data();
  double* original_hid_ptr = hidden_values.data();

  HiddenState state2(std::move(state1));
  EXPECT_EQ(state2.get_pre_activation_sums().data(), original_pre_ptr);
  EXPECT_EQ(state2.get_hidden_state_values().data(), original_hid_ptr);

  HiddenState state3;
  state3 = std::move(state2);
  EXPECT_EQ(state3.get_pre_activation_sums().data(), original_pre_ptr);
  EXPECT_EQ(state3.get_hidden_state_values().data(), original_hid_ptr);
}
