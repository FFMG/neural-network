#include <gtest/gtest.h>
#include "layers/fflayer.h"
#include "layers/layers.h"
#include "neuralnetworkoptions.h"
#include "test_helper.h"
#include <vector>

using namespace myoddweb::nn;
using namespace test_helper;

TEST(SwaTests, LayerRunningMeanMatchesDirectArithmeticMean)
{
  // Three "snapshots" of the same 2-input/2-neuron FF layer. Folding them
  // one at a time via the incremental running-mean update should produce
  // exactly the same result as a direct arithmetic mean of the three.
  FFLayer running(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
  FFLayer snap2(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);
  FFLayer snap3(1, 2, 2, 0.0, Layer::Role::Hidden, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0.0, nullptr, 1, true, 0.0, std::nullopt);

  running.set_w_values({ 1.0, 2.0, 3.0, 4.0 });
  running.set_b_values({ 10.0, 20.0 });
  snap2.set_w_values({ 5.0, 6.0, 7.0, 8.0 });
  snap2.set_b_values({ 30.0, 40.0 });
  snap3.set_w_values({ 9.0, 10.0, 11.0, 12.0 });
  snap3.set_b_values({ 50.0, 60.0 });

  // `running` already represents 1 prior snapshot (itself); fold in snap2,
  // then snap3.
  running.accumulate_swa_average(snap2, 1);
  running.accumulate_swa_average(snap3, 2);

  const auto& w = running.get_w_values();
  const auto& b = running.get_b_values();

  const std::vector<double> expected_w = { (1.0 + 5.0 + 9.0) / 3.0, (2.0 + 6.0 + 10.0) / 3.0, (3.0 + 7.0 + 11.0) / 3.0, (4.0 + 8.0 + 12.0) / 3.0 };
  const std::vector<double> expected_b = { (10.0 + 30.0 + 50.0) / 3.0, (20.0 + 40.0 + 60.0) / 3.0 };

  ASSERT_EQ(w.size(), expected_w.size());
  for (size_t i = 0; i < w.size(); ++i)
  {
    EXPECT_NEAR(w[i], expected_w[i], 1e-9);
  }
  ASSERT_EQ(b.size(), expected_b.size());
  for (size_t i = 0; i < b.size(); ++i)
  {
    EXPECT_NEAR(b[i], expected_b[i], 1e-9);
  }
}

TEST(SwaTests, MockLayerAccumulateSwaAverageIsNoOp)
{
  // MockLayer (an Input-role layer with no trainable weights) must accept
  // the call without throwing, exercising the pure-virtual override that
  // every concrete Layer subclass -- including test doubles -- now needs.
  MockLayer running(0, 2, 0);
  MockLayer snapshot(0, 2, 0);
  EXPECT_NO_THROW(running.accumulate_swa_average(snapshot, 1));
}

TEST(SwaTests, LayersAccumulateSwaAveragePropagatesAcrossLayers)
{
  // Builds a real 2-layer (hidden + output) Layers instance and verifies
  // Layers::accumulate_swa_average folds a snapshot into every layer, not
  // just the first one.
  std::vector<LayerDetails> hidden_layers = {
    LayerDetails(Layer::Architecture::FF, 3, activation(activation::method::linear, 0.0), 0.0, 0.0, OptimiserType::None, 0.0, false, 0)
  };
  auto options = NeuralNetworkOptions::create({ 2, 3, 1 })
    .with_hidden_layers(hidden_layers)
    .with_output_layer_details(OutputLayerDetails(1, activation(activation::method::linear, 0.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::None, 0.0))
    .build();

  Layers running(options);
  Layers snapshot(options);

  running[1].set_w_values({ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });
  running[1].set_b_values({ 1.0, 1.0, 1.0 });
  snapshot[1].set_w_values({ 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 });
  snapshot[1].set_b_values({ 3.0, 3.0, 3.0 });

  running[2].set_w_values({ 2.0, 2.0, 2.0 });
  running[2].set_b_values({ 2.0 });
  snapshot[2].set_w_values({ 4.0, 4.0, 4.0 });
  snapshot[2].set_b_values({ 4.0 });

  running.accumulate_swa_average(snapshot, 1);

  for (double v : running[1].get_w_values())
  {
    EXPECT_NEAR(v, 2.0, 1e-9); // (1+3)/2
  }
  for (double v : running[1].get_b_values())
  {
    EXPECT_NEAR(v, 2.0, 1e-9);
  }
  for (double v : running[2].get_w_values())
  {
    EXPECT_NEAR(v, 3.0, 1e-9); // (2+4)/2
  }
  for (double v : running[2].get_b_values())
  {
    EXPECT_NEAR(v, 3.0, 1e-9);
  }
}
