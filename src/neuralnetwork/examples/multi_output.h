#pragma once
#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetwork.h"
#include "helper.h"
#include <vector>
#include <cmath>
#include <iomanip>

/**
 * ExampleMultiOutput
 * 
 * This example demonstrates a neural network with multiple logical output layers
 * using different activation functions:
 * 1. Sigmoid Output: Classification (Is the input value positive?)
 * 2. Tanh Output: Regression (Mapping input to a symmetric range [-1, 1])
 * 
 * The network will learn to:
 * - Output ~1.0 if x > 0, ~0.0 if x < 0 (Sigmoid)
 * - Output tanh(x) as a regression value (Tanh)
 */
class ExampleMultiOutput
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleMultiOutput");

    // Input: 1 value
    // Hidden: 8, 8
    // Output: 2 (1 for Sigmoid, 1 for Tanh)
    std::vector<unsigned> topology = { 1, 8, 8, 2 };

    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::tanh, 0.01), 0.0, 0.5, OptimiserType::NadamW, 0.9),
      LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::tanh, 0.01), 0.0, 0.5, OptimiserType::NadamW, 0.9)
    };

    // Define multiple output layers
    auto output_layers = {
      // First output: Sigmoid (Classification: Is positive?)
      OutputLayerDetails(1, activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.5, OptimiserType::NadamW, 0.99),
      // Second output: Tanh (Regression: Scaled value)
      OutputLayerDetails(1, activation(activation::method::tanh, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.5, OptimiserType::NadamW, 0.9)
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(16)
      .with_output_layer_details(output_layers)
      .with_log_level(log_level)
      .with_learning_rate(0.01)
      .with_number_of_epoch(1000)
      .with_hidden_layers(hidden_layers)
      .build();

    return new NeuralNetwork(options);
  }

  static void generate_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs, size_t count)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleMultiOutput");
    inputs.reserve(count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
      // Random value between -2.0 and 2.0
      double x = ((double)rand() / RAND_MAX) * 4.0 - 2.0;
      
      inputs.push_back({ x });

      // Target 1: Is positive? (Sigmoid: 0 or 1)
      double y1 = (x > 0) ? 1.0 : 0.0;
      
      // Target 2: Tanh of x (Tanh: -1 to 1)
      double y2 = std::tanh(x);

      outputs.push_back({ y1, y2 });
    }
  }

  static bool run_tests(NeuralNetwork& nn)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleMultiOutput");
    Logger::info("Running validation tests...");
    
    struct TestCase {
      double input;
      double exp_class;
      double exp_reg;
    };

    std::vector<TestCase> tests = {
      { 1.5,  1.0,  std::tanh(1.5) },
      { -1.5, 0.0,  std::tanh(-1.5) },
      { 0.5,  1.0,  std::tanh(0.5) },
      { -0.5, 0.0,  std::tanh(-0.5) }
    };

    bool all_passed = true;
    const double tolerance = 0.15;

    for (const auto& test : tests)
    {
      auto result = nn.think({ test.input });
      double got_class = result[0];
      double got_reg = result[1];

      bool class_ok = std::abs(got_class - test.exp_class) < tolerance;
      bool reg_ok = std::abs(got_reg - test.exp_reg) < tolerance;

      Logger::info("Input: ", std::fixed, std::setprecision(2), test.input, 
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
  static void MultiOutput(Logger::LogLevel log_level)
  {
    TEST_START("Multi-Output Layer (Sigmoid + Tanh) Example")

    srand(42); // Seed for reproducible results

    NeuralNetwork* nn = create_neural_network(log_level);

    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;
    generate_data(training_inputs, training_outputs, 500);

    Logger::info("Training network with multiple output layers...");
    nn->train(training_inputs, training_outputs);

    bool success = run_tests(*nn);

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
