#pragma once
#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetworkserializer.h"
#include "helper.h"
#include <iomanip>
#include <numeric>

/**
 * ExampleBranchedOutput
 * 
 * Tests the BranchedOutputLayer architecture.
 * Topology: Input(3) -> GRU(16) -> BRANCH({
 *    Branch 1: Dense(8) -> OutputRegression(2) [MSE]
 *    Branch 2: Dense(8) -> OutputSoftmax(3) [CrossEntropy]
 * })
 */
class ExampleBranchedOutput
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    std::vector<unsigned> topology = { 3, 16, 5 }; // 5 = 2 (Reg) + 3 (Softmax)
    
    std::vector<LayerDetails> trunk_hidden = {
      LayerDetails(LayerDetails::LayerType::Gru, 16, activation(activation::method::tanh, 1.0), 0.0, 0.0, OptimiserType::AdamW, 0.9)
    };

    // Branch 1: Regression
    LayerDetails::BranchDetails b1;
    b1.hidden_layers = {
      LayerDetails(LayerDetails::LayerType::FF, 8, activation(activation::method::mish, 1.0), 0.0, 0.0, OptimiserType::AdamW, 0.9)
    };
    b1.output_details = OutputLayerDetails(2, activation(activation::method::linear, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::AdamW, 0.9);

    // Branch 2: Classification
    LayerDetails::BranchDetails b2;
    b2.hidden_layers = {
      LayerDetails(LayerDetails::LayerType::FF, 8, activation(activation::method::relu, 1.0), 0.0, 0.0, OptimiserType::AdamW, 0.9)
    };
    b2.output_details = OutputLayerDetails(3, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, { 0.0, 0.5, 1.0, 1.0, true, 1.0 }, 0.0, OptimiserType::AdamW, 0.9);

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(4)
      .with_branched_outputs({ b1, b2 })
      .with_log_level(log_level)
      .with_learning_rate(0.01)
      .with_number_of_epoch(1000)
      .with_hidden_layers(trunk_hidden)
      .build();

    return new NeuralNetwork(options);
  }

public:
  static void Run(Logger::LogLevel log_level)
  {
    TEST_START("Branched Output Architecture Test")

    NeuralNetwork* nn = create_neural_network(log_level);

    // 4 samples
    std::vector<std::vector<double>> inputs = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}
    };
    std::vector<std::vector<double>> outputs = {
      // Reg(2)  Softmax(3)
      {0.1, 0.9,  1, 0, 0},
      {0.5, 0.5,  0, 1, 0},
      {0.9, 0.1,  0, 0, 1},
      {0.0, 0.0,  1, 0, 0}
    };

    Logger::info("Training branched network...");
    nn->train(inputs, outputs);

    Logger::info("Verifying outputs...");
    auto results = nn->think(inputs);

    bool all_ok = true;
    for (size_t i = 0; i < results.size(); ++i)
    {
      // Branch 2 is at index 2,3,4. Should sum to 1.0
      double softmax_sum = results[i][2] + results[i][3] + results[i][4];
      if (std::abs(softmax_sum - 1.0) > 1e-5)
      {
        Logger::error("Sample ", i, " Softmax sum FAIL: ", softmax_sum);
        all_ok = false;
      }
      
      Logger::info("Sample ", i, " Output: ", 
        results[i][0], ", ", results[i][1], " | ", 
        results[i][2], ", ", results[i][3], ", ", results[i][4]);
    }

    if (all_ok)
    {
      Logger::info("RESULT: PASS");
    }
    else
    {
      Logger::error("RESULT: FAIL");
    }

    delete nn;
    TEST_END
  }
};
