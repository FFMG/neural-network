#pragma once
#include "helpers/errorcalculation.h"
#include "common/logger.h"
#include "neuralnetwork.h"
#include "helper.h"
#include <vector>
#include <cmath>
#include <iomanip>
#include <numeric>

/**
 * ExampleMultiOutputGru
 * 
 * This example demonstrates a neural network with a GRU recurrent layer
 * and multiple logical output layers using different activation functions.
 * 
 * It uses sequential data (5 time steps) to predict:
 * 1. Sigmoid Output: Is the LAST value in the sequence positive?
 * 2. Tanh Output: What is the average value of the sequence?
 */

using namespace myoddweb::nn;
class ExampleMultiOutputGru
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleMultiOutputGru");

    // Input: 1 value per tick
    // Hidden: 1 GRU layer with 16 neurons
    // Output: 2 (1 for Sigmoid, 1 for Tanh)
    std::vector<unsigned> topology = { 1, 16, 2 };

    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Gru, 16, activation(activation::method::tanh, 0.01), 0.0, 0.5, OptimiserType::NadamW, 0.9)
    };

    // Define multiple output layers
    auto output_layers = {
      // First output: Sigmoid (Classification: Is last value positive?)
      OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.5, OptimiserType::NadamW, 0.99),
      // Second output: Tanh (Regression: Average value)
      OutputLayerDetails(1, activation(activation::method::tanh, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.5, OptimiserType::NadamW, 0.9)
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(16)
      .with_output_layer_details(output_layers)
      .with_log_level(log_level)
      .with_learning_rate(0.01)
      .with_number_of_epoch(500)
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(5) // We will use sequences of 5
      .build();

    return new NeuralNetwork(options);
  }

  static void generate_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs, size_t count, size_t sequence_length)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleMultiOutputGru");
    inputs.reserve(count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
      std::vector<double> seq;
      double sum = 0;
      for (size_t t = 0; t < sequence_length; ++t)
      {
        // Random value between -1.0 and 1.0
        double val = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        seq.push_back(val);
        sum += val;
      }
      
      inputs.push_back(seq);

      // Target 1: Is last value positive? (Sigmoid: 0 or 1)
      double last_val = seq.back();
      double y1 = (last_val > 0) ? 1.0 : 0.0;
      
      // Target 2: Average of sequence (Tanh: -1 to 1)
      double avg = sum / sequence_length;
      double y2 = avg; // Since avg is between -1 and 1, tanh is not strictly needed but matches multi_output style

      outputs.push_back({ y1, y2 });
    }
  }

  static bool run_tests(NeuralNetwork& nn, size_t sequence_length)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleMultiOutputGru");
    (void)sequence_length;
    Logger::info("Running validation tests...");
    
    struct TestCase {
      std::vector<double> input;
      double exp_class;
      double exp_reg;
    };

    std::vector<TestCase> tests = {
      { { -0.5, -0.2, 0.1, 0.4, 0.8 },  1.0,  ( -0.5 - 0.2 + 0.1 + 0.4 + 0.8 ) / 5.0 },
      { { 0.5, 0.2, -0.1, -0.4, -0.8 }, 0.0,  ( 0.5 + 0.2 - 0.1 - 0.4 - 0.8 ) / 5.0 },
      { { 0.1, 0.1, 0.1, 0.1, 0.1 },    1.0,  0.1 },
      { { -0.1, -0.1, -0.1, -0.1, -0.1 }, 0.0, -0.1 }
    };

    bool all_passed = true;
    const double tolerance = 0.2;

    for (const auto& test : tests)
    {
      auto result = nn.think(test.input);
      double got_class = result[0];
      double got_reg = result[1];

      bool class_ok = std::abs(got_class - test.exp_class) < tolerance;
      bool reg_ok = std::abs(got_reg - test.exp_reg) < tolerance;

      Logger::info("Input Last: ", std::fixed, std::setprecision(2), test.input.back(), 
                   " | Class: ", std::fixed, std::setprecision(4), got_class, " (exp ", test.exp_class, ") ", (class_ok ? "[OK]" : "[FAIL]"),
                   " | Reg: ", std::fixed, std::setprecision(4), got_reg, " (exp ", test.exp_reg, ") ", (reg_ok ? "[OK]" : "[FAIL]"));

      if (!class_ok || !reg_ok)
      {
        all_passed = false;
      }
    }

    return all_passed;
  }

public:
  static void MultiOutputGru(Logger::LogLevel log_level)
  {
    TEST_START("Multi-Output Layer (Sigmoid + Tanh) with GRU Example")

    srand(42); // Seed for reproducible results

    NeuralNetwork* nn = create_neural_network(log_level);

    const size_t sequence_length = 5;
    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;
    generate_data(training_inputs, training_outputs, 1000, sequence_length);

    Logger::info("Training network with GRU and multiple output layers...");
    nn->train(training_inputs, training_outputs);

    bool success = run_tests(*nn, sequence_length);

    if (success)
    {
      Logger::info("*********************************");
      Logger::info("*      OVERALL STATUS: SUCCESS  *");
      Logger::info("*********************************");
    }
    else
    {
      Logger::error("*********************************");
      Logger::error("*      OVERALL STATUS: FAILURE  *");
      Logger::error("*********************************");
    }

    delete nn;
    TEST_END
  }
};
