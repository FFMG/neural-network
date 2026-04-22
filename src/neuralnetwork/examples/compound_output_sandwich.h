#pragma once
#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetworkserializer.h"
#include "helper.h"
#include <iomanip>
#include <numeric>
#include <filesystem>

/**
 * ExampleCompoundOutputSandwich
 * 
 * THE PARANOID TEST: A "torture test" for the Neural Network library.
 * 
 * Complexity Level:
 * - Architecture: GRU -> FF (Multi-Head!) -> FF (with Residual Jump)
 * - Multi-Head Hidden: Layer 2 has 16 Mish neurons and 16 ReLU neurons.
 * - Output Heads: Softmax(3) -> Tanh(2) -> Softmax(4) -> Sigmoid(1) -> Linear(2)
 * - Validation: Overfitting -> Serialization (Save/Load) -> Re-Validation
 * 
 * This confirms that Softmax, compound slicing, multi-head hidden layers,
 * recurrent states, residual jumps, and persistence all work in perfect harmony.
 */
class ExampleCompoundOutputSandwich
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    // Topology: Input(3) -> GRU(32) -> FF(32) -> FF(16) -> Output(12)
    std::vector<unsigned> topology = { 3, 32, 32, 16, 12 };
    
    std::vector<LayerDetails> hidden_layers = {
      // Layer 1: GRU
      LayerDetails(LayerDetails::LayerType::Gru, 32, activation(activation::method::tanh, 0.01), 0.0, 0.0001, OptimiserType::NadamW, 0.9),
      // Layer 2: FF (We will manually make this multi-head below)
      LayerDetails(LayerDetails::LayerType::FF, 32, activation(activation::method::mish, 0.01), 0.0, 0.0001, OptimiserType::NadamW, 0.95),
      // Layer 3: FF
      LayerDetails(LayerDetails::LayerType::FF, 16, activation(activation::method::relu, 0.01), 0.0, 0.0001, OptimiserType::NadamW, 0.9)
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(4)
      .with_output_layer_details(
        {
          // Head 0: Softmax(3)
          OutputLayerDetails(3, activation(activation::method::softmax, 0.01), ErrorCalculation::type::cross_entropy, { 0.0, 0.5, 1.0, 1.0, true, 1.0 }, 0.0, OptimiserType::NadamW, 0.99),
          // Head 1: Tanh(2)
          OutputLayerDetails(2, activation(activation::method::tanh, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 1.0, true, 1.0 }, 0.0, OptimiserType::NadamW, 0.9),
          // Head 2: Softmax(4)
          OutputLayerDetails(4, activation(activation::method::softmax, 0.01), ErrorCalculation::type::cross_entropy, { 0.0, 0.5, 1.0, 1.0, true, 1.0 }, 0.0, OptimiserType::NadamW, 0.99),
          // Head 3: Sigmoid(1)
          OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 1.0, true, 1.0 }, 0.0, OptimiserType::NadamW, 0.99),
          // Head 4: Linear(2)
          OutputLayerDetails(2, activation(activation::method::linear, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 1.0, true, 1.0 }, 0.0, OptimiserType::NadamW, 0.9)
        }
      )
      .with_log_level(log_level)
      .with_learning_rate(0.005)
      .with_number_of_epoch(5000)
      .with_hidden_layers(hidden_layers)
      .with_residual_layer_jump(1) 
      .with_data_is_unique(true)
      .build();

    auto nn = new NeuralNetwork(options);

    // --- MANUALLY SETUP MULTI-HEAD HIDDEN LAYER ---
    // Layer 2 (index 2): neurons 0-15 = Mish, 16-31 = ReLU
    auto& layer2 = const_cast<Layer&>(nn->get_layer(2));
    layer2.get_activation_helper().set_bounds(activation(activation::method::relu, 0.01), 16, 32);

    return nn;
  }

  static void verify_network(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, const std::string& label)
  {
    auto results = nn.think(inputs);
    
    bool s0_sum_ok = true;
    bool s2_sum_ok = true;
    for (size_t i = 0; i < results.size(); ++i)
    {
      // Head 0: 0-2
      double sum0 = results[i][0] + results[i][1] + results[i][2];
      // Head 2: 5-8
      double sum2 = results[i][5] + results[i][6] + results[i][7] + results[i][8];
      
      if (std::abs(sum0 - 1.0) > 1e-4) s0_sum_ok = false;
      if (std::abs(sum2 - 1.0) > 1e-4) s2_sum_ok = false;
    }

    // Verify Multi-Head Hidden Layer Persistence (Layer 2)
    const auto& lah2 = nn.get_layer(2).get_activation_helper();
    bool hidden_multi_head_ok = (lah2.ranges().size() == 2) &&
                                (lah2.ranges()[0].activation_method.get_method() == activation::method::mish) &&
                                (lah2.ranges()[1].activation_method.get_method() == activation::method::relu);

    auto metrics = nn.calculate_forecast_metrics_all_layers({ 
      ErrorCalculation::type::directional_accuracy 
    }, true);

    Logger::info("--- Verification: ", label, " ---");
    Logger::info("Softmax Head 0 Sum-to-one: ", (s0_sum_ok ? "PASS" : "FAIL"));
    Logger::info("Softmax Head 2 Sum-to-one: ", (s2_sum_ok ? "PASS" : "FAIL"));
    Logger::info("Hidden Multi-Head Config : ", (hidden_multi_head_ok ? "PASS" : "FAIL"));
    
    if (metrics.size() >= 3) {
      double acc0 = metrics[0][0].error();
      double acc2 = metrics[2][0].error();
      
      Logger::info("Head 0 Acc: ", acc0 * 100.0, "%");
      Logger::info("Head 2 Acc: ", acc2 * 100.0, "%");
      
      if (s0_sum_ok && s2_sum_ok && hidden_multi_head_ok && acc0 > 0.9 && acc2 > 0.9) {
        Logger::info("RESULT: PASS");
      } else {
        Logger::error("RESULT: FAIL");
      }
    }
  }

public:
  static void Run(Logger::LogLevel log_level)
  {
    TEST_START("Paranoid Compound Output Test")

    const char* file_name = "./paranoid_sandwich.nn";
    NeuralNetwork* nn = create_neural_network(log_level);

    // 4 complex samples
    std::vector<std::vector<double>> inputs = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}
    };
    std::vector<std::vector<double>> outputs = {
      // S(3)    T(2)     S(4)       Si(1) L(2)
      {1,0,0,  -1,-1,   1,0,0,0,   0,    1, 2},
      {0,1,0,  -0.5,0.5, 0,1,0,0,   1,    -5, -5},
      {0,0,1,   0.5,-0.5, 0,0,1,0,   0,    10, 0},
      {1,0,0,   1,1,     0,0,0,1,   1,    0, 0}
    };

    Logger::info("Step 1: Training original network...");
    nn->train(inputs, outputs);
    verify_network(*nn, inputs, outputs, "Original Network");

    Logger::info("Step 2: Saving to disk and destroying...");
    NeuralNetworkSerializer::save(*nn, file_name);
    delete nn;

    Logger::info("Step 3: Loading from disk...");
    nn = NeuralNetworkSerializer::load(file_name);
    if (nn == nullptr) {
        Logger::error("CRITICAL: Failed to load network from disk!");
        TEST_END
        return;
    }

    verify_network(*nn, inputs, outputs, "Reloaded Network");

    delete nn;
    if (std::filesystem::exists(file_name)) std::filesystem::remove(file_name);
    
    TEST_END
  }
};
