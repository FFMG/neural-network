#pragma once
#include "helpers/errorcalculation.h"
#include "common/logger.h"
#include "neuralnetwork.h"
#include "helper.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <vector>

#include <fstream>


/**
 * ExampleReproIssueSoftmax
 * 
 * Repro case for TANH x2 + SOFTMAX x5 issue.
 * Architecture:
 * - Input: 1 neuron
 * - Hidden: 2 neurons, Tanh
 * - Output 0: 1 neuron, MSE (Regression)
 * - Output 1: 5 neurons, Softmax + Cross Entropy (Classification)
 */

using namespace myoddweb::nn;
class ExampleReproIssueSoftmax
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleReproIssueSoftmax");

    // 17 inputs, 80 LSTM, 40 LSTM, 7 total outputs (2 in branch 0, 5 in branch 1)
    std::vector<unsigned> topology = { 17, 64, 32, 7 }; 

    std::vector<LayerDetails> trunk_hidden_layers = 
    {
      LayerDetails(Layer::Architecture::Gru, 64, activation(activation::method::tanh, 0.1), 0.1, 0.0001, OptimiserType::NadamW, 0.9),
      LayerDetails(Layer::Architecture::Gru, 32, activation(activation::method::tanh, 0.1), 0.1, 0.0001, OptimiserType::NadamW, 0.9)
    };

    std::vector<MultiOutputLayerDetails> multi_output_layer_details;

    // Branch 0: 2 outputs, Huber
    MultiOutputLayerDetails b0
    (
      { 
        LayerDetails(Layer::Architecture::FF, 32, activation(activation::method::tanh, 0.01), 0, 0.0001, OptimiserType::NadamW, 0.9),
        LayerDetails(Layer::Architecture::FF, 16, activation(activation::method::tanh, 0.01), 0, 0.0001, OptimiserType::NadamW, 0.9)
      },
      OutputLayerDetails(2, activation(activation::method::tanh, 0.1), ErrorCalculation::type::huber_direction_loss, EvaluationConfig(0.01, 0.15, 0.3, 0.3, false, 1.0), 0.0, OptimiserType::NadamW, 0.99)
    );

    // Branch 1: 5 outputs, Softmax
    MultiOutputLayerDetails b1
    (
      {
        LayerDetails(Layer::Architecture::FF, 32, activation(activation::method::tanh, 0.01), 0, 0.0001, OptimiserType::NadamW, 0.9),
        LayerDetails(Layer::Architecture::FF, 16, activation(activation::method::tanh, 0.01), 0, 0.0001, OptimiserType::NadamW, 0.9)
      },
      OutputLayerDetails(5, activation(activation::method::softmax, 0.01), ErrorCalculation::type::cross_entropy, EvaluationConfig(0.0, 0.2, 1.0, 0.3, false, 1.0), 0.0, OptimiserType::NadamW, 0.99)
    );

    multi_output_layer_details.push_back(b0);
    multi_output_layer_details.push_back(b1);

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(32)
      .with_output_layer_details(multi_output_layer_details)
      .with_log_level(log_level)
      .with_learning_rate(0.0005)
      .with_learning_rate_warmup(0.0002, 0.07)
      .with_learning_rate_decay_rate(0.985)
      .with_clip_threshold(1.5)
      .with_number_of_epoch(5000) 
      .with_hidden_layers(trunk_hidden_layers)
      .with_shuffle_training_data(false)
      .with_shuffle_bptt_batches(true)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(24)
      .with_final_error_calculation_types({
        ErrorCalculation::type::huber_loss,
        ErrorCalculation::type::huber_direction_loss,
        ErrorCalculation::type::directional_accuracy,
        ErrorCalculation::type::directional_confidence_score,
        ErrorCalculation::type::prediction_coverage
      })
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
    MYODDWEB_PROFILE_FUNCTION("ExampleReproIssueSoftmax");
    inputs.reserve(count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
      // Random value between -1.0 and 1.0
      double x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
      
      inputs.push_back({ x });

      std::vector<double> y(7, 0.0); 
      
      // Target 0: Regression 1
      y[0] = x * x;
      // Target 1: Regression 2
      y[1] = x * x * x; 
      
      // Target 2-6: 5-class One-Hot encoding for Softmax
      int bucket = get_bucket(x);
      y[2 + bucket] = 1.0; 

      outputs.push_back(y);
    }
  }

  static bool load_csv_data(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs, size_t max_rows = 0)
  {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    size_t rows_read = 0;
    while (std::getline(file, line) && (max_rows == 0 || rows_read < max_rows))
    {
      std::stringstream ss(line);
      std::string val;
      std::vector<double> row;
      while (std::getline(ss, val, ','))
      {
        row.push_back(std::stod(val));
      }
      if (row.size() >= 24) 
      {
        inputs.push_back(std::vector<double>(row.begin(), row.begin() + 17));
        outputs.push_back(std::vector<double>(row.begin() + 17, row.begin() + 24));
        rows_read++;
      }
    }
    return true;
  }

public:
  static void ReproIssueMarketData(const std::string& nn_file, const std::string& save_nn_file, const std::string& csv_file, Logger::LogLevel log_level, size_t max_rows = 0)
  {
    TEST_START("Repro Issue Market Data")
    
    NeuralNetwork* nn = nullptr;
    if (!nn_file.empty())
    {
      nn = NeuralNetworkSerializer::load(nn_file);
      if (nullptr == nn)
      {
        Logger::error("Failed to load NN : ", nn_file);
        return;
      }
    }
    else
    {
      nn = create_neural_network(log_level);
    }

    std::vector<std::vector<double>> inputs, outputs;
    if (!load_csv_data(csv_file, inputs, outputs, max_rows))
    {
      Logger::error("Failed to load CSV: ", csv_file);
      delete nn;
      return;
    }

    Logger::info("Loaded ", inputs.size(), " samples from ", csv_file);
    
    // Verify distribution
    std::vector<int> counts(5, 0);
    for (const auto& o : outputs)
    {
      for (int i = 0; i < 5; ++i)
      {
        if (o[2 + i] > 0.5) { counts[i]++; break; }
      }
    }
    for (int i = 0; i < 5; ++i)
      Logger::info("Class ", i, ": ", (double)counts[i] / outputs.size() * 100.0, "%");

    try {
      nn->train(inputs, outputs);

      // Metrics for both branches
      std::vector<ErrorCalculation::type> all_layer_metrics = { 
        ErrorCalculation::type::huber_direction_loss, 
        ErrorCalculation::type::cross_entropy, 
        ErrorCalculation::type::directional_confidence_score, 
        ErrorCalculation::type::prediction_coverage 
      };

      auto metrics_results = nn->calculate_forecast_metrics_all_layers(all_layer_metrics);

      Logger::info("Branch 0 Metrics (Regression/Huber):");
      if (metrics_results.size() > 0)
      {
        for (const auto& m : metrics_results[0])
        {
          Logger::info("  ", ErrorCalculation::type_to_string(m.error_type()), ": ", m.error());
        }
      }

      Logger::info("Branch 1 Metrics (Softmax/Cross-Entropy):");
      if (metrics_results.size() > 1)
      {
        for (const auto& m : metrics_results[1])
        {
          Logger::info("  ", ErrorCalculation::type_to_string(m.error_type()), ": ", m.error());
        }
      }

      // Detailed sample check
      Logger::info("Checking sample 0 results:");
      auto res0 = nn->think(inputs[0]);
      Logger::info("Think: ", res0);
      Logger::info("Given: ", outputs[0]);

      if (!save_nn_file.empty())
      {
        NeuralNetworkSerializer::save(*nn, save_nn_file);
      }
    } catch (const std::exception& e) {
      Logger::set_level(Logger::LogLevel::Error);
      Logger::error("ReproIssueMarketData failed with error: ", e.what());
    }
    
    delete nn;
    TEST_END
  }

  static void ReproIssueMarketData(const std::string& save_nn_file, const std::string& csv_file, Logger::LogLevel log_level, size_t max_rows = 0)
  {
    TEST_START("Repro Issue Market Data")
    NeuralNetwork* nn = create_neural_network(log_level);
    std::vector<std::vector<double>> inputs, outputs;

    if (!load_csv_data(csv_file, inputs, outputs, max_rows))
    {
      Logger::error("Failed to load CSV: ", csv_file);
      delete nn;
      return;
    }

    Logger::info("Loaded ", inputs.size(), " samples from ", csv_file);
    // Verify distribution
    std::vector<int> counts(5, 0);
    for (const auto& o : outputs)
    {
      for (int i = 0; i < 5; ++i)
      {
        if (o[2 + i] > 0.5) { counts[i]++; break; }
      }
    }
    for (int i = 0; i < 5; ++i)
    {
      Logger::info("Class ", i, ": ", (double)counts[i] / outputs.size() * 100.0, "%");
    }

    try {
      nn->train(inputs, outputs);

      // Monitoring
      // Logger::info("Final Softmax Weights Sum: ", nn->get_output_layer_weights_sum(1));
      // Logger::info("Final Logits (sample 0): ", Logger::vec_to_string(nn->get_output_layer_logits(1, inputs[0])));

      // Calculate metrics for both layers - request only appropriate metrics for each
      std::vector<ErrorCalculation::type> all_layer_metrics = { ErrorCalculation::type::huber_direction_loss, ErrorCalculation::type::cross_entropy, ErrorCalculation::type::directional_confidence_score, ErrorCalculation::type::prediction_coverage };

      auto all_layer_metrics_results = nn->calculate_forecast_metrics_all_layers(all_layer_metrics);

      Logger::info("Layer 0 Metrics (Regression):");
      for (const auto& m : all_layer_metrics_results[0])
      {
        Logger::info("  ", ErrorCalculation::type_to_string(m.error_type()), ": ", m.error());
      }

      Logger::info("Layer 1 Metrics (Softmax):");
      for (const auto& m : all_layer_metrics_results[1])
      {
        Logger::info("  ", ErrorCalculation::type_to_string(m.error_type()), ": ", m.error());
      }

      auto size = static_cast<size_t>(inputs.size() / 2.0);

      auto res0 = nn->think(inputs[0]);
      Logger::debug("Think:", res0);
      Logger::debug("Given:", outputs[0]);

      auto res = nn->think(inputs[size]);
      Logger::debug("Think:", res);
      Logger::debug("Given:", outputs[size]);

      auto res1 = nn->think(inputs[1]);
      Logger::debug("Think:", res1);
      Logger::debug("Given:", outputs[1]);

      NeuralNetworkSerializer::save(*nn, save_nn_file);
    } 
    catch (const std::exception& e) 
    {
      Logger::set_level(Logger::LogLevel::Error);
      Logger::error("ReproIssueMarketData failed with error: ", e.what());
    }
    delete nn;
    TEST_END
  }

  static void ReproIssueSoftmax(Logger::LogLevel log_level)
  {
    TEST_START("Repro Issue Market Data")

    srand(42); // Seed for reproducible results

    NeuralNetwork* nn = create_neural_network(log_level);

    // --- DEBUGGING: Softmax Weight Monitoring (Before Training) ---
    //double initial_softmax_weights_sum = nn->get_output_layer_weights_sum(1);
    //Logger::info("DEBUG: Initial Softmax Layer (Output Head 1) Weights Sum: ", std::fixed, std::setprecision(10), initial_softmax_weights_sum);

    // --- DEBUGGING: Softmax Logit Monitoring (Before Training) ---
    // Use a fixed input to observe logit changes
    std::vector<double> fixed_input = { 0.5 };
    //std::vector<double> initial_softmax_logits = nn->get_output_layer_logits(1, fixed_input);
    //Logger::info("DEBUG: Initial Softmax Layer (Output Head 1) Logits for input {0.5}: ", Logger::vec_to_string(initial_softmax_logits));

    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;
    generate_data(training_inputs, training_outputs, 1000);

    Logger::info("Training network...");
    nn->train(training_inputs, training_outputs);

    // --- DEBUGGING: Softmax Weight Monitoring (After Training) ---
    //double final_softmax_weights_sum = nn->get_output_layer_weights_sum(1);
    //Logger::info("DEBUG: Final Softmax Layer (Output Head 1) Weights Sum: ", std::fixed, std::setprecision(10), final_softmax_weights_sum);

    // --- DEBUGGING: Softmax Logit Monitoring (After Training) ---
    //std::vector<double> final_softmax_logits = nn->get_output_layer_logits(1, fixed_input);
    //Logger::info("DEBUG: Final Softmax Layer (Output Head 1) Logits for input {0.5}: ", Logger::vec_to_string(final_softmax_logits));

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
      auto softmax_slice_begin = res.begin() + 2;
      auto softmax_slice_end = res.begin() + 7;
      auto max_it = std::max_element(softmax_slice_begin, softmax_slice_end);
      int got_bucket = (int)std::distance(softmax_slice_begin, max_it);
      double confidence = *max_it;

      Logger::info("x: ", std::fixed, std::setprecision(2), x, 
                   " | reg1: ", std::fixed, std::setprecision(4), got_reg1, " (exp ", exp_reg1, ")",
                   " | reg2: ", std::fixed, std::setprecision(4), got_reg2, " (exp ", exp_reg2, ")",
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

    // --- DEBUGGING: Force a Gradient (Manual Test, if needed) ---
    // If the above debugging indicates no weight or logit changes,
    // you might need to manually force a gradient in FFOutputLayer::run_output_gradients.
    //
    // Navigate to src/neuralnetwork/ffoutputlayer.cpp, locate `run_output_gradients`.
    // Inside the loop for `b` (batch) and `neuron_index`, find where `deltas[neuron_index]` is set.
    // For the softmax head (Output Head 1), you could temporarily set:
    //
    // if (output_head_index == 1) // Assuming you can determine the current output head
    // {
    //   // For a specific neuron in the softmax output (e.g., neuron_index = 2, target class)
    //   // deltas[neuron_index] = (neuron_index == target_class_index ? -1.0 : 1.0);
    //   // This is a simplistic way to force a large gradient.
    // }
    //
    // Remember to revert this change after testing.

    // --- DEBUGGING: Verify Optimizer Loop / Parameter Registration ---
    // If weights are not changing, inspect the `Layers::update_weights` method (src/neuralnetwork/layers.cpp).
    // Ensure that `_layers[i]->calculate_and_store_gradients(...)` and `_layers[i]->apply_stored_gradients(...)`
    // are being called for ALL layers, including the output layers.
    // Also, verify that the weights of the softmax layer are correctly registered with the optimizer.
    // This typically happens during layer construction in FFLayer (which FFOutputLayer uses).

    delete nn;
    TEST_END
  }
};
