#include <gtest/gtest.h>
#include "../src/neuralnetwork/layersandneuronscontainer.h"
#include <vector>
#include <numeric>


using namespace myoddweb::nn;
TEST(LayersAndNeuronsContainerTest, ConstructorInitializesTopologyAndOffsets)
{
  std::vector<unsigned> topology = { 3, 5, 2 };
  LayersAndNeuronsContainer container(topology);

  EXPECT_EQ(container.number_layers(), 3);
  EXPECT_EQ(container.number_neurons(0), 3);
  EXPECT_EQ(container.number_neurons(1), 5);
  EXPECT_EQ(container.number_neurons(2), 2);
}

TEST(LayersAndNeuronsContainerTest, ZeroInitializesDataToZero)
{
  std::vector<unsigned> topology = { 2, 2 };
  LayersAndNeuronsContainer container(topology);

  container.set(0, 0, 1.0);
  container.set(1, 1, 2.0);

  container.zero();

  EXPECT_DOUBLE_EQ(container.get(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(container.get(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(container.get(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(container.get(1, 1), 0.0);
}

TEST(LayersAndNeuronsContainerTest, SetAndGetSpecificNeuron)
{
  std::vector<unsigned> topology = { 3, 4 };
  LayersAndNeuronsContainer container(topology);

  container.set(0, 2, 42.0);
  container.set(1, 0, -1.5);

  EXPECT_DOUBLE_EQ(container.get(0, 2), 42.0);
  EXPECT_DOUBLE_EQ(container.get(1, 0), -1.5);
  EXPECT_DOUBLE_EQ(container.get(0, 0), 0.0);
}

TEST(LayersAndNeuronsContainerTest, SetLayerWithVector)
{
  std::vector<unsigned> topology = { 3, 3 };
  LayersAndNeuronsContainer container(topology);

  std::vector<double> layer_data = { 1.1, 2.2, 3.3 };
  container.set(1, layer_data);

  EXPECT_DOUBLE_EQ(container.get(1, 0), 1.1);
  EXPECT_DOUBLE_EQ(container.get(1, 1), 2.2);
  EXPECT_DOUBLE_EQ(container.get(1, 2), 3.3);
  EXPECT_DOUBLE_EQ(container.get(0, 0), 0.0);
}

TEST(LayersAndNeuronsContainerTest, GetRawPtr)
{
  std::vector<unsigned> topology = { 2, 3 };
  LayersAndNeuronsContainer container(topology);

  double* ptr0 = container.get_raw_ptr(0);
  double* ptr1 = container.get_raw_ptr(1);

  ptr0[0] = 10.0;
  ptr0[1] = 20.0;
  ptr1[0] = 30.0;

  EXPECT_DOUBLE_EQ(container.get(0, 0), 10.0);
  EXPECT_DOUBLE_EQ(container.get(0, 1), 20.0);
  EXPECT_DOUBLE_EQ(container.get(1, 0), 30.0);

  const LayersAndNeuronsContainer& const_container = container;
  const double* cptr1 = const_container.get_raw_ptr(1);
  EXPECT_DOUBLE_EQ(cptr1[0], 30.0);
}

TEST(LayersAndNeuronsContainerTest, GetSpanAndNeurons)
{
  std::vector<unsigned> topology = { 2, 2 };
  LayersAndNeuronsContainer container(topology);
  container.set(1, std::vector<double>{ 5.0, 6.0 });

  auto span = container.get_span(1);
  ASSERT_EQ(span.size(), 2);
  EXPECT_DOUBLE_EQ(span[0], 5.0);
  EXPECT_DOUBLE_EQ(span[1], 6.0);

  auto vec = container.get_neurons(1);
  ASSERT_EQ(vec.size(), 2);
  EXPECT_DOUBLE_EQ(vec[0], 5.0);
  EXPECT_DOUBLE_EQ(vec[1], 6.0);
}

TEST(LayersAndNeuronsContainerTest, CopyConstructorAndAssignment)
{
  std::vector<unsigned> topology = { 2, 2 };
  LayersAndNeuronsContainer container1(topology);
  container1.set(0, 0, 7.7);

  LayersAndNeuronsContainer container2(container1);
  EXPECT_DOUBLE_EQ(container2.get(0, 0), 7.7);

  container2.set(0, 0, 8.8);
  EXPECT_DOUBLE_EQ(container1.get(0, 0), 7.7);

  LayersAndNeuronsContainer container3;
  container3 = container2;
  EXPECT_DOUBLE_EQ(container3.get(0, 0), 8.8);
}

TEST(LayersAndNeuronsContainerTest, MoveConstructorAndAssignment)
{
  std::vector<unsigned> topology = { 2, 2 };
  LayersAndNeuronsContainer container1(topology);
  container1.set(0, 0, 9.9);

  LayersAndNeuronsContainer container2(std::move(container1));
  EXPECT_DOUBLE_EQ(container2.get(0, 0), 9.9);
  EXPECT_EQ(container1.number_layers(), 0);

  LayersAndNeuronsContainer container3;
  container3 = std::move(container2);
  EXPECT_DOUBLE_EQ(container3.get(0, 0), 9.9);
  EXPECT_EQ(container2.number_layers(), 0);
}

TEST(LayersAndNeuronsContainerTest, ValidationLogic)
{
#if VALIDATE_DATA == 1
  std::vector<unsigned> topology = { 2, 2 };
  LayersAndNeuronsContainer container(topology);

  // Layer out of bounds
  EXPECT_THROW(container.set(2, 0, 1.0), std::runtime_error);
  EXPECT_THROW((void)container.get_raw_ptr(2), std::runtime_error);
  EXPECT_THROW((void)container.get_span(2), std::runtime_error);

  // Neuron out of bounds
  EXPECT_THROW(container.set(0, 2, 1.0), std::runtime_error);
  EXPECT_THROW((void)container.get(0, 2), std::runtime_error);
  
  // Data size mismatch
  std::vector<double> too_much_data = { 1.0, 2.0, 3.0 };
  std::vector<double> too_little_data = { 1.0 };
  
  // It should fail if we try to set more data than the layer can hold (overflow)
  EXPECT_THROW(container.set(0, too_much_data), std::runtime_error);
  // It should fail if we try to set less data than the layer has (incomplete)
  EXPECT_THROW(container.set(0, too_little_data), std::runtime_error);
#endif
}
