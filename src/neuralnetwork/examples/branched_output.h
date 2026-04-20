#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "../neuralnetwork.h"
#include "../neuralnetworkserializer.h"
#include "../logger.h"

class ExampleBranchedOutput
{
public:
  static void Run(Logger::LogLevel log_level)
  {
    Logger::info("Starting Branched Output Complexity & Serialization Test...");
    auto start = std::chrono::high_resolution_clock::now();

    // Topology: 3 inputs -> GRU(4) -> BranchedOutput
    // Branch 1: FF(2) -> Output(2) [Regression, MSE]
    // Branch 2: FF(4) -> FF(3) -> Output(3) [Classification, Softmax] - deeper!
    
    std::vector<unsigned> topology = { 3 };
    
    std::vector<LayerDetails::BranchDetails> branches;
    
    // Branch 1: Shallow Regression
    LayerDetails::BranchDetails b1;
    b1.hidden_layers.emplace_back(LayerDetails(LayerDetails::LayerType::FF, 2, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.9));
    b1.output_details = OutputLayerDetails(2, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::SGD, 0.9);
    branches.push_back(b1);

    // Branch 2: Deeper Classification
    LayerDetails::BranchDetails b2;
    b2.hidden_layers.emplace_back(LayerDetails(LayerDetails::LayerType::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.9));
    b2.hidden_layers.emplace_back(LayerDetails(LayerDetails::LayerType::FF, 3, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.9));
    b2.output_details = OutputLayerDetails(3, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::SGD, 0.9);
    branches.push_back(b2);

    auto options = NeuralNetworkOptions::create(topology)
      .with_hidden_layers({ LayerDetails(LayerDetails::LayerType::Gru, 4, activation(activation::method::tanh, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.9) })
      .with_branched_outputs(branches)
      .with_learning_rate(0.1)
      .with_number_of_epoch(500)
      .with_batch_size(1)
      .with_log_level(log_level)
      .build();

    NeuralNetwork nn(options);

    // Data: 3-bit input. 
    // Out1: {sum/3, avg} (Regression)
    // Out2: {all-zeros, all-ones, mixed} (Classification)
    std::vector<std::vector<double>> inputs = {
      {1.0, 1.0, 1.0}, // Sum 3, All ones
      {0.0, 0.0, 0.0}, // Sum 0, All zeros
      {1.0, 0.0, 1.0}, // Sum 2, Mixed
      {0.0, 0.0, 0.0}  // Sum 0, All zeros
    };

    std::vector<std::vector<double>> targets = {
      {1.0, 0.9,  0.0, 1.0, 0.0}, // Sum high, All ones
      {0.1, 0.1,  1.0, 0.0, 0.0}, // Sum low, All zeros
      {0.7, 0.5,  0.0, 0.0, 1.0}, // Sum mid, Mixed
      {0.0, 0.0,  1.0, 0.0, 0.0}  // Sum low, All zeros
    };

    nn.train(inputs, targets);

    // Save the network
    const std::string model_path = "branched_complex.nn";
    NeuralNetworkSerializer::save(nn, model_path);
    Logger::info("Model saved to ", model_path);

    // Load the network
    auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(model_path));
    if (!loaded_nn)
    {
      Logger::panic("Failed to load complex branched network!");
    }
    Logger::info("Model loaded successfully.");

    Logger::info("Verifying outputs from loaded network...");
    bool pass = true;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
      auto output = loaded_nn->think(inputs[i]);
      
      // Log outputs
      std::string out_str = "";
      for(auto d : output) out_str += std::to_string(d) + ", ";
      Logger::info("Sample ", i, " Output: ", out_str);

      // Simple checks
      if (i == 0 && output[2] < 0.8) pass = false; // Should be 'all ones'
      if (i == 1 && output[2] > 0.2) pass = false; // Should be 'all zeros'
    }

    if (pass)
      Logger::info("RESULT: PASS");
    else
      Logger::error("RESULT: FAIL (Convergence or Serialization issue)");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    Logger::info("Branched Complexity Test took ", (int)elapsed.count() / 60, " min ", (int)elapsed.count() % 60, " sec");
  }
};
