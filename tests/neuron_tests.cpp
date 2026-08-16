#include <gtest/gtest.h>
#include "neuron.h"
#include <optional>
#include <vector>


using namespace myoddweb::nn;
TEST(NeuronTest, NormalNeuronInitialization)
{
  Neuron n(42, Neuron::Type::Normal, 0.0, std::nullopt);
  EXPECT_EQ(n.get_index(), 42);
  EXPECT_EQ(n.get_type(), Neuron::Type::Normal);
  EXPECT_FALSE(n.is_dropout());
}

TEST(NeuronTest, DropoutNeuronInitialization)
{
  Neuron n(7, Neuron::Type::Dropout, 0.25, std::nullopt);
  EXPECT_EQ(n.get_index(), 7);
  EXPECT_EQ(n.get_type(), Neuron::Type::Dropout);
  EXPECT_TRUE(n.is_dropout());
  EXPECT_DOUBLE_EQ(n.get_dropout_rate(), 0.25);
}

TEST(NeuronTest, CopySemantics)
{
  Neuron n1(10, Neuron::Type::Dropout, 0.3, std::nullopt);
  Neuron n2 = n1;

  EXPECT_EQ(n2.get_index(), 10);
  EXPECT_EQ(n2.get_type(), Neuron::Type::Dropout);
  EXPECT_DOUBLE_EQ(n2.get_dropout_rate(), 0.3);
}

TEST(NeuronTest, MoveSemantics)
{
  Neuron n1(10, Neuron::Type::Dropout, 0.3, std::nullopt);
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
  Neuron n_never(1, Neuron::Type::Dropout, 0.0, std::nullopt);
  for (int i = 0; i < 100; ++i)
  {
    EXPECT_FALSE(n_never.must_randomly_drop(static_cast<uint64_t>(i)));
  }

  // 100% dropout should always drop
  Neuron n_always(2, Neuron::Type::Dropout, 1.0, std::nullopt);
  for (int i = 0; i < 100; ++i)
  {
    EXPECT_TRUE(n_always.must_randomly_drop(static_cast<uint64_t>(i)));
  }
}

#if VALIDATE_DATA == 1
TEST(NeuronTest, ValidationLogic)
{
  Neuron n(1, Neuron::Type::Normal, 0.0, std::nullopt);

  // These should panic for Normal neurons
  EXPECT_THROW((void)n.get_dropout_rate(), std::runtime_error);
  EXPECT_THROW((void)n.must_randomly_drop(0), std::runtime_error);
}
#endif

TEST(NeuronTest, DropoutStatisticalDistribution)
{
  Neuron n(3, Neuron::Type::Dropout, 0.5, std::nullopt);
  int drop_count = 0;
  const int total = 10000;
  for (int i = 0; i < total; ++i)
  {
    if (n.must_randomly_drop(static_cast<uint64_t>(i)))
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
  Neuron n(0, Neuron::Type::Dropout, 0.5, std::nullopt);

  const int num_iterations = 20000;
  int dropped = 0;
  for (int i = 0; i < num_iterations; ++i)
  {
    if (n.must_randomly_drop(static_cast<uint64_t>(i)))
    {
      dropped++;
    }
  }

  double actual_rate = static_cast<double>(dropped) / num_iterations;
  // 5% tolerance is extremely safe for 20000 iterations (approx 7 standard deviations)
  EXPECT_NEAR(actual_rate, 0.5, 0.05);
}

TEST(NeuronTest, SeededDropoutIsDeterministic)
{
  // Same seed_base and call_index must always produce the same decision -
  // this is the core determinism guarantee the seed feature exists for.
  Neuron n1(5, Neuron::Type::Dropout, 0.5, std::optional<uint64_t>(123456789ULL));
  Neuron n2(5, Neuron::Type::Dropout, 0.5, std::optional<uint64_t>(123456789ULL));

  for (uint64_t call_index = 0; call_index < 500; ++call_index)
  {
    EXPECT_EQ(n1.must_randomly_drop(call_index), n2.must_randomly_drop(call_index));
  }
}

TEST(NeuronTest, SeededDropoutDiffersByCallIndexAndSeed)
{
  Neuron n(5, Neuron::Type::Dropout, 0.5, std::optional<uint64_t>(42ULL));

  // Across many call_index values, the decision must not be constant -
  // otherwise seeding would have silently disabled dropout entirely.
  bool saw_drop = false;
  bool saw_keep = false;
  for (uint64_t call_index = 0; call_index < 200; ++call_index)
  {
    if (n.must_randomly_drop(call_index))
    {
      saw_drop = true;
    }
    else
    {
      saw_keep = true;
    }
  }
  EXPECT_TRUE(saw_drop);
  EXPECT_TRUE(saw_keep);

  // A different seed_base at the same call_index should, overwhelmingly
  // likely, diverge from the original at least once across many samples.
  Neuron n_other_seed(5, Neuron::Type::Dropout, 0.5, std::optional<uint64_t>(43ULL));
  bool diverged = false;
  for (uint64_t call_index = 0; call_index < 200; ++call_index)
  {
    if (n.must_randomly_drop(call_index) != n_other_seed.must_randomly_drop(call_index))
    {
      diverged = true;
      break;
    }
  }
  EXPECT_TRUE(diverged);
}

TEST(NeuronTest, SeededDropoutStatisticalDistribution)
{
  Neuron n(3, Neuron::Type::Dropout, 0.5, std::optional<uint64_t>(987654321ULL));
  int drop_count = 0;
  const int total = 10000;
  for (int i = 0; i < total; ++i)
  {
    if (n.must_randomly_drop(static_cast<uint64_t>(i)))
    {
      ++drop_count;
    }
  }
  EXPECT_GE(drop_count, 4700);
  EXPECT_LE(drop_count, 5300);
}
