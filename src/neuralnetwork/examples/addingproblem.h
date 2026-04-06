#pragma once

#include "helper.h"
#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetworkserializer.h"
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

class ExampleAddingProblem
{
public:
  static void generate_data(
      size_t num_samples,
      size_t sequence_length,
      std::vector<std::vector<double>>& inputs,
      std::vector<std::vector<double>>& outputs)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> val_dist(0.0, 1.0);
    std::uniform_int_distribution<> idx_dist(0, (int)sequence_length - 1);

    inputs.clear();
    outputs.clear();
    inputs.reserve(num_samples);
    outputs.reserve(num_samples);

    for (size_t i = 0; i < num_samples; ++i)
    {
      std::vector<double> seq;
      seq.reserve(sequence_length * 2);
      double sum_val = 0.0;
            
      // Pick two distinct indices
      int t1 = idx_dist(gen);
      int t2 = idx_dist(gen);
      while (t1 == t2)
      {
        t2 = idx_dist(gen);
      }

      for (size_t t = 0; t < sequence_length; ++t)
      {
        double val = val_dist(gen);
        double mask = 0.0;
        if (t == t1 || t == t2)
        {
            mask = 1.0;
            sum_val += val;
        }
        seq.push_back(val);
        seq.push_back(mask);
      }
      inputs.push_back(seq);
      outputs.push_back({ sum_val });
    }
  }

  static void AddingProblem(Logger::LogLevel log_level)
  {
    TEST_START("Adding Problem Test");

    const size_t sequence_length = 50; // Standard length for this problem
    const size_t train_samples = 2000;
    const size_t test_samples = 100;
    const unsigned epochs = 20;
    const unsigned batch_size = 32;
    const double learning_rate = 0.001; 

    std::vector<std::vector<double>> train_inputs, train_outputs;
    std::vector<std::vector<double>> test_inputs, test_outputs;

    Logger::info("Generating Training Data (", train_samples, " samples, length ", sequence_length, ")...");
    generate_data(train_samples, sequence_length, train_inputs, train_outputs);
    Logger::info("Generating Test Data (", test_samples, " samples)...");
    generate_data(test_samples, sequence_length, test_inputs, test_outputs);

    // Topology:
    // Input: 2 neurons (but data is flattened 2*T, handled by BPTT logic)
    // Hidden: 1 layer of GRU with 100 neurons
    // Output: 1 neuron (Linear)
    std::vector<unsigned> topology = { 2, 100, 1 };
    std::vector<LayerDetails> hidden_layers = {
        LayerDetails(LayerDetails::LayerType::Gru, 100, activation(activation::method::tanh, 0.01), 0.0)
    };
    auto output_layer = OutputLayerDetails(topology.back(), activation(activation::method::linear, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 });
    
    auto options = NeuralNetworkOptions::create(topology)
        .with_batch_size(batch_size)
        .with_output_layer_details(output_layer) // Regression output
        .with_log_level(log_level)
        .with_learning_rate(learning_rate)
        .with_optimiser_type(OptimiserType::Adam)
        .with_hidden_layers(hidden_layers)
        .with_number_of_epoch(epochs)
        .with_enable_bptt(true)
        .with_bptt_max_ticks((int)sequence_length)
        .with_data_is_unique(true)
        .with_clip_threshold(1.0) // Gradient clipping is important for RNNs
        .build();    
    NeuralNetwork nn(options);

    // Train
    Logger::info("Training...");
    nn.train(train_inputs, train_outputs);

    // Test
    Logger::info("Testing...");
    auto results = nn.think(test_inputs);
        
    double total_mse = 0.0;
    for(size_t i=0; i<test_samples; ++i)
    {
      double diff = results[i][0] - test_outputs[i][0];
      total_mse += diff * diff;
      if (i < 5) 
      {
        Logger::info(" Sample ", i, ": Pred=", std::fixed, std::setprecision(4), results[i][0], 
                    " Actual=", test_outputs[i][0], " Diff=", diff);
      }
    }
    Logger::info("Test MSE: ", std::fixed, std::setprecision(6), total_mse / test_samples);
        
    // Benchmark trivial solution (predicting 1.0 always)
    // Expected value of sum of 2 U[0,1] is 1.0.
    // Variance of U[0,1] is 1/12. Variance of sum is 2/12 = 1/6 ~= 0.1667.
    // So a baseline MSE should be around 0.1667.
    Logger::info("Baseline MSE (predicting 1.0): ~0.1667");
    if ((total_mse / test_samples) < 0.16)
    {
      Logger::info("SUCCESS: Model beat the baseline!");
    }
    else
    {
      Logger::warning("FAILURE: Model did not beat the baseline.");
    }

    TEST_END
    std::cout << std::endl;
  }
};
