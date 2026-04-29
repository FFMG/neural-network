#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include "../src/neuralnetwork/fflayer.h"
#include "../src/neuralnetwork/gradientsandoutputs.h"
#include "../src/neuralnetwork/hiddenstates.h"
#include "../src/neuralnetwork/layer.h"

namespace test_helper {
  inline std::vector<GradientsAndOutputs> create_batch_gradients_and_outputs(const std::vector<unsigned>& topology, size_t batch_size) {
    std::vector<GradientsAndOutputs> batch;
    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      batch.emplace_back(topology);
    }
    return batch;
  }

  inline std::vector<HiddenStates> create_batch_hidden_states(const std::vector<unsigned>& topology, size_t batch_size, size_t num_ticks) {
    std::vector<HiddenStates> batch;
    batch.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      HiddenStates hs(topology);
      for (size_t l = 0; l < topology.size(); ++l) {
        hs.assign(l, num_ticks, HiddenState(nullptr, nullptr, nullptr, 0, 0));
      }
      batch.push_back(std::move(hs));
    }
    return batch;
  }

  // A simple concrete layer to use as 'previous_layer'
  class MockLayer : public Layer {
  public:
    MockLayer(unsigned layer_index, unsigned num_neurons) :
      Layer(layer_index, Layer::Role::Input, activation(activation::method::linear, 0.0), OptimiserType::None, -1, 0, num_neurons, {}, false, 0.0, nullptr, 1, 0.0) {}

    Architecture get_layer_architecture() const override { return Architecture::None; }
    void calculate_forward_feed(std::vector<GradientsAndOutputs>&, const Layer&, const std::vector<std::vector<double>>&, std::vector<HiddenStates>&, size_t, bool) const override {}
    void calculate_output_gradients(std::vector<GradientsAndOutputs>&, std::vector<std::vector<double>>::const_iterator, const std::vector<HiddenStates>&, size_t) const override {}
    void calculate_hidden_gradients(std::vector<GradientsAndOutputs>&, const Layer&, const std::vector<std::vector<double>>&, const std::vector<HiddenStates>&, size_t, int) const override {}
    void calculate_hidden_gradients_from_output_gradients(std::vector<GradientsAndOutputs>&, const std::vector<std::vector<double>>&, const std::vector<HiddenStates>&, size_t, int) const override {}
    Layer* clone() const override { return new MockLayer(*this); }
    void calculate_and_store_gradients(const std::vector<GradientsAndOutputs>&, const std::vector<HiddenStates>&, const Layer&, size_t, int) override {}
    double get_gradient_norm_sq() const override { return 0.0; }
    void apply_stored_gradients(double, double) override {}
  };

  inline bool approx_equal(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
  }
}
