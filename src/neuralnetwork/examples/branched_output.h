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
    MYODDWEB_PROFILE_FUNCTION("ExampleBranchedOutput");
    Logger::info("Starting Branched Output Complexity & Serialization Test...");
    auto start = std::chrono::high_resolution_clock::now();

    // Topology: 3 inputs -> GRU(4) -> BranchedOutput
    // Branch 1: FF(2) -> Output(2) [Regression, MSE]
    // Branch 2: FF(4) -> FF(3) -> Output(3) [Classification, Softmax] - deeper!

    std::vector<unsigned> topology = { 3, 4, 5 }; // 5 = 2 (Reg) + 3 (Softmax)

    std::vector<MultiOutputLayerDetails> multi_output_layer_details;

    // Branch 1: Shallow Regression
    MultiOutputLayerDetails b1
    (
      { LayerDetails(LayerDetails::LayerType::FF, 2, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.9) },
      OutputLayerDetails(2, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::SGD, 0.9)
    );
    multi_output_layer_details.push_back(b1);

    // Branch 2: Deeper Classification
    MultiOutputLayerDetails b2
    (
      {
        LayerDetails(LayerDetails::LayerType::FF, 4, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.9),
        LayerDetails(LayerDetails::LayerType::FF, 3, activation(activation::method::sigmoid, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.9)
      },
      OutputLayerDetails(3, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0, OptimiserType::SGD, 0.9)
    );
    multi_output_layer_details.push_back(b2);

    auto options = NeuralNetworkOptions::create(topology)
      .with_hidden_layers({ LayerDetails(LayerDetails::LayerType::Gru, 4, activation(activation::method::tanh, 1.0), 0.0, 0.0, OptimiserType::SGD, 0.9) })
      .with_multi_output_layer_details(multi_output_layer_details)
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

    Logger::info("Verifying outputs from loaded network...");

    // from original
    {
      bool pass = true;
      for (size_t i = 0; i < inputs.size(); ++i)
      {
        auto output = nn.think(inputs[i]);

        // Log outputs
        std::string out_str = "";
        for (auto d : output)
        {
          out_str += std::to_string(d) + ", ";
        }
        Logger::info("Sample ", i, " Output: ", out_str);

        // Simple checks
        // Out2: {all-zeros, all-ones, mixed} (Classification)
        // Sample 0: all-ones -> expects output[3] to be high
        if (i == 0 && output[3] < 0.8)
        {
          pass = false;
        }
        // Sample 1: all-zeros -> expects output[2] to be high
        if (i == 1 && output[2] < 0.8)
        {
          pass = false;
        }
      }

      if (pass)
      {
        Logger::info("RESULT: PASS");
      }
      else
      {
        Logger::error("RESULT: FAIL (Convergence or Serialization issue)");
      }
    }

    // Load the network
    auto loaded_nn = std::unique_ptr<NeuralNetwork>(NeuralNetworkSerializer::load(model_path));
    if (!loaded_nn)
    {
      Logger::panic("Failed to load complex branched network!");
    }
    Logger::info("Model loaded successfully.");

    // from saved
    {
      bool pass = true;
      for (size_t i = 0; i < inputs.size(); ++i)
      {
        auto output = loaded_nn->think(inputs[i]);

        // Log outputs
        std::string out_str = "";
        for (auto d : output)
        {
          out_str += std::to_string(d) + ", ";
        }
        Logger::info("Sample ", i, " Output: ", out_str);

        // Simple checks
        // Sample 0: all-ones -> expects output[3] to be high
        if (i == 0 && output[3] < 0.8)
        {
          pass = false;
        }
        // Sample 1: all-zeros -> expects output[2] to be high
        if (i == 1 && output[2] < 0.8)
        {
          pass = false;
        }
      }

      if (pass)
      {
        Logger::info("RESULT: PASS");
      }
      else
      {
        Logger::error("RESULT: FAIL (Convergence or Serialization issue)");
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    Logger::info("Branched Complexity Test took ", (int)elapsed.count() / 60, " min ", (int)elapsed.count() % 60, " sec");
  }
};
