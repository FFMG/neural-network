#include <gtest/gtest.h>
#include "neuron.h"
#include <vector>


using namespace myoddweb::nn;
TEST(NeuronTest, NormalNeuronInitialization)
{
  Neuron n(42, Neuron::Type::Normal, 0.0);
  EXPECT_EQ(n.get_index(), 42);
  EXPECT_EQ(n.get_type(), Neuron::Type::Normal);
  EXPECT_FALSE(n.is_dropout());
}

TEST(NeuronTest, DropoutNeuronInitialization)
{
  Neuron n(7, Neuron::Type::Dropout, 0.25);
  EXPECT_EQ(n.get_index(), 7);
  EXPECT_EQ(n.get_type(), Neuron::Type::Dropout);
  EXPECT_TRUE(n.is_dropout());
  EXPECT_DOUBLE_EQ(n.get_dropout_rate(), 0.25);
}

TEST(NeuronTest, CopySemantics)
{
  Neuron n1(10, Neuron::Type::Dropout, 0.3);
  Neuron n2 = n1;

  EXPECT_EQ(n2.get_index(), 10);
  EXPECT_EQ(n2.get_type(), Neuron::Type::Dropout);
  EXPECT_DOUBLE_EQ(n2.get_dropout_rate(), 0.3);
}

TEST(NeuronTest, MoveSemantics)
{
  Neuron n1(10, Neuron::Type::Dropout, 0.3);
  Neuron n2 = std::move(n1);

  EXPECT_EQ(n2.get_index(), 10);
  EXPECT_EQ(n2.get_type(), Neuron::Type::Dropout);
  EXPECT_DOUBLE_EQ(n2.get_dropout_rate(), 0.3);

  // n1 is reset
  EXPECT_EQ(n1.get_index(), 0);
  EXPECT_EQ(n1.get_type(), Neuron::Type::Normal);
}

TEST(NeuronTest, DropoutBehaviorBoundaries)
{
  // 0% dropout should never drop
  Neuron n_never(1, Neuron::Type::Dropout, 0.0);
  for(int i=0; i<100; ++i) {
    EXPECT_FALSE(n_never.must_randomly_drop());
  }

  // 100% dropout should always drop
  Neuron n_always(2, Neuron::Type::Dropout, 1.0);
  for(int i=0; i<100; ++i) {
    EXPECT_TRUE(n_always.must_randomly_drop());
  }
}

#if VALIDATE_DATA == 1
TEST(NeuronTest, ValidationLogic)
{
  Neuron n(1, Neuron::Type::Normal, 0.0);
  
  // These should panic for Normal neurons
  EXPECT_THROW((void)n.get_dropout_rate(), std::runtime_error);
  EXPECT_THROW((void)n.must_randomly_drop(), std::runtime_error);
}
#endif

TEST(NeuronTest, DropoutStatisticalDistribution)
{
  Neuron n(3, Neuron::Type::Dropout, 0.5);
  int drop_count = 0;
  const int total = 10000;
  for (int i = 0; i < total; ++i)
  {
    if (n.must_randomly_drop())
    {
      ++drop_count;
    }
  }
  // Expected value is 5000, standard deviation is sqrt(10000 * 0.5 * 0.5) = 50.
  // Within 4 standard deviations (99.993% confidence): [4800, 5200]
  EXPECT_GE(drop_count, 4700);
  EXPECT_LE(drop_count, 5300);
}

TEST(NeuronTest, Xorshift64DistributionAndBounds)
{
  Neuron n(0, Neuron::Type::Dropout, 0.5);
  
  const int num_iterations = 20000;
  int dropped = 0;
  for (int i = 0; i < num_iterations; ++i)
  {
    if (n.must_randomly_drop())
    {
      dropped++;
    }
  }
  
  double actual_rate = static_cast<double>(dropped) / num_iterations;
  // 5% tolerance is extremely safe for 20000 iterations (approx 7 standard deviations)
  EXPECT_NEAR(actual_rate, 0.5, 0.05);
}

