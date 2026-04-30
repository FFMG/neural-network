#pragma once
#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetwork.h"
#include "helper.h"
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

/**
 * ExampleReproIssue
 * 
 * Repro case for TANH x2 + SOFTMAX x5 issue.
 * Architecture:
 * - Input: 1 neuron
 * - Hidden: 2 neurons, Tanh
 * - Output 0: 1 neuron, MSE (Regression)
 * - Output 1: 5 neurons, Softmax + Cross Entropy (Classification)
 */
class ExampleReproIssue
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleReproIssue");

    std::vector<unsigned> topology = { 1, 16, 7 }; // 1 input, 16 hidden, 2 reg + 5 softmax = 7 outputs

    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::FF, 16, activation(activation::method::tanh, 0.01), 0.0, 0.5, OptimiserType::AdamW, 0.9)
    };

    // Define compound output layers
    auto output_layers = {
      // First output: Regression (2 Tanh neurons, MSE)
      OutputLayerDetails(2, activation(activation::method::tanh, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.0001, OptimiserType::AdamW, 0.9),
      // Second output: Softmax (5 classes, Cross-Entropy)
      OutputLayerDetails(5, activation(activation::method::softmax, 0.01), ErrorCalculation::type::cross_entropy, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.5, OptimiserType::AdamW, 0.99)
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(16)
      .with_output_layer_details(output_layers)
      .with_log_level(log_level)
      .with_learning_rate(0.0005)     // Further reduced LR
      .with_clip_threshold(1.0)      // Normal clipping
      .with_number_of_epoch(500)
      .with_hidden_layers(hidden_layers)
      .build();

    return new NeuralNetwork(options);
  }

  static int get_bucket(double x)
  {
    if (x < -0.6) return 0;
    if (x < -0.2) return 1;
    if (x < 0.2)  return 2;
    if (x < 0.6)  return 3;
    return 4;
  }

  static void generate_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs, size_t count)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleReproIssue");
    inputs.reserve(count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
      // Random value between -1.0 and 1.0
      double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
      
      inputs.push_back({ x });

      std::vector<double> y(7, 0.0); // Changed from 6 to 7
      
      // Target 0: Regression 1
      y[0] = x * x;
      // Target 1: Regression 2
      y[1] = x * x * x; // Added second regression target
      
      // Target 2-6: 5-class One-Hot encoding for Softmax
      int bucket = get_bucket(x);
      y[2 + bucket] = 1.0; // Changed index from 1 + bucket to 2 + bucket

      outputs.push_back(y);
    }
  }

public:
  static void ReproIssue(Logger::LogLevel log_level)
  {
    TEST_START("Repro Issue (TANH x2 + SOFTMAX x5)") // Changed description

    srand(42); // Seed for reproducible results

    NeuralNetwork* nn = create_neural_network(log_level);

    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;
    generate_data(training_inputs, training_outputs, 1000);

    Logger::info("Training network...");
    nn->train(training_inputs, training_outputs);

    Logger::info("Validation:");
    bool success = true;
    for (double x : { -0.8, -0.4, 0.0, 0.4, 0.8 })
    {
      auto res = nn->think({ x });
      
      // 1. Check Regression 1
      double exp_reg1 = x * x;
      double got_reg1 = res[0];

      // 2. Check Regression 2
      double exp_reg2 = x * x * x;
      double got_reg2 = res[1];

      // 3. Check Softmax
      int exp_bucket = get_bucket(x);
      auto softmax_slice_begin = res.begin() + 2; // Changed from 1 to 2
      auto softmax_slice_end = res.begin() + 7;   // Changed from 6 to 7
      auto max_it = std::max_element(softmax_slice_begin, softmax_slice_end);
      int got_bucket = (int)std::distance(softmax_slice_begin, max_it);
      double confidence = *max_it;

      Logger::info("x: ", std::fixed, std::setprecision(2), x, 
                   " | reg1: ", std::fixed, std::setprecision(4), got_reg1, " (exp ", exp_reg1, ")",
                   " | reg2: ", std::fixed, std::setprecision(4), got_reg2, " (exp ", exp_reg2, ")", // Added reg2 output
                   " | bucket: ", got_bucket, " (exp ", exp_bucket, ") conf: ", confidence);

      if (exp_bucket != got_bucket)
      {
        success = false;
      }
    }

    if (success)
    {
      Logger::info("Result: Model is learning correctly.");
    }
    else
    {
      Logger::error("Result: Model failed to learn classification correctly.");
    }

    delete nn;
    TEST_END
  }
};
