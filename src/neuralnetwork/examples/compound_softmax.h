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
 * ExampleCompoundSoftmax
 * 
 * This example demonstrates a neural network with a compound output layer:
 * 1. Sigmoid Output (1 neuron): Classification (Is the input value positive?)
 * 2. Softmax Output (5 neurons): Categorical classification into 5 buckets
 *    Buckets: 
 *    - Bucket 0: x < -1.2
 *    - Bucket 1: -1.2 <= x < -0.4
 *    - Bucket 2: -0.4 <= x < 0.4 (Neutral)
 *    - Bucket 3: 0.4 <= x < 1.2
 *    - Bucket 4: x >= 1.2
 */
class ExampleCompoundSoftmax
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleCompoundSoftmax");

    // Input: 1 value
    // Hidden: 16, 16
    // Output: 1 (Sigmoid) + 5 (Softmax) = 6 total neurons
    std::vector<unsigned> topology = { 1, 16, 16, 6 };

    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::FF, 16, activation(activation::method::mish, 1.0), 0.0, 0.01, OptimiserType::NadamW, 0.95),
      LayerDetails(Layer::Architecture::FF, 16, activation(activation::method::mish, 1.0), 0.0, 0.01, OptimiserType::NadamW, 0.95)
    };

    // Define compound output layers
    auto output_layers = {
      // First output: Sigmoid (Classification: Is positive?)
      OutputLayerDetails(1, activation(activation::method::sigmoid, 1.0), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.01, OptimiserType::NadamW, 0.95),
      // Second output: Softmax (5-bucket classification)
      OutputLayerDetails(5, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.01, OptimiserType::NadamW, 0.95)
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(32)
      .with_output_layer_details(output_layers)
      .with_log_level(log_level)
      .with_learning_rate(0.001)
      .with_number_of_epoch(2000)
      .with_hidden_layers(hidden_layers)
      .with_final_error_calculation_types({ 
          ErrorCalculation::type::rmse,
          ErrorCalculation::type::mape,
          ErrorCalculation::type::wape,
          ErrorCalculation::type::directional_accuracy,
          ErrorCalculation::type::directional_confidence_score
        })
      .build();

    return new NeuralNetwork(options);
  }

  static int get_bucket(double x)
  {
    if (x < -1.2) return 0;
    if (x < -0.4) return 1;
    if (x < 0.4)  return 2;
    if (x < 1.2)  return 3;
    return 4;
  }

  static void generate_data(std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs, size_t count)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleCompoundSoftmax");
    inputs.reserve(count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
      // Random value between -2.0 and 2.0
      double x = ((double)rand() / RAND_MAX) * 4.0 - 2.0;
      
      inputs.push_back({ x });

      std::vector<double> y(6, 0.0);
      
      // Target 0: Is positive? (Sigmoid: 0 or 1)
      y[0] = (x > 0) ? 1.0 : 0.0;
      
      // Target 1-5: 5-bucket One-Hot encoding for Softmax
      int bucket = get_bucket(x);
      y[1 + bucket] = 1.0;

      outputs.push_back(y);
    }
  }

  static bool run_tests(NeuralNetwork& nn)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleCompoundSoftmax");
    Logger::info("Running compound output validation tests...");
    
    std::vector<double> test_inputs = { -1.8, -0.8, -0.1, 0.8, 1.8 };

    bool all_passed = true;
    const double sigmoid_tolerance = 0.2;

    for (double x : test_inputs)
    {
      auto result = nn.think({ x });
      
      // 1. Check Sigmoid (Is positive?)
      double exp_sigmoid = (x > 0) ? 1.0 : 0.0;
      double got_sigmoid = result[0];
      bool sigmoid_ok = std::abs(got_sigmoid - exp_sigmoid) < sigmoid_tolerance;

      // 2. Check Softmax (Winning bucket)
      int exp_bucket = get_bucket(x);
      auto softmax_slice_begin = result.begin() + 1;
      auto softmax_slice_end = result.begin() + 6;
      auto max_it = std::max_element(softmax_slice_begin, softmax_slice_end);
      int got_bucket = (int)std::distance(softmax_slice_begin, max_it);
      bool softmax_ok = (exp_bucket == got_bucket);

      Logger::info("Input: ", std::fixed, std::setprecision(1), x, 
                   " | Sigmoid: ", std::fixed, std::setprecision(4), got_sigmoid, " (exp ", exp_sigmoid, ") ", (sigmoid_ok ? "[OK]" : "[FAIL]"),
                   " | Softmax Bucket: ", got_bucket, " (exp ", exp_bucket, ") ", (softmax_ok ? "[OK]" : "[FAIL]"));

      if (!sigmoid_ok || !softmax_ok)
      {
        all_passed = false;
      }
    }

    return all_passed;
  }

public:
  static void CompoundSoftmax(Logger::LogLevel log_level)
  {
    TEST_START("Compound Output Layer (Sigmoid + 5-neuron Softmax) Example")

    srand(123); // Seed for reproducible results

    NeuralNetwork* nn = create_neural_network(log_level);

    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;
    generate_data(training_inputs, training_outputs, 1000);

    Logger::info("Training network with compound output layers...");
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
