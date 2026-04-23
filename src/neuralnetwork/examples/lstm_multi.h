#pragma once
#include "../neuralnetwork.h"
#include "../neuralnetworkoptions.h"
#include "../logger.h"
#include "helper.h"
#include <vector>

/**
 * ExampleLstmMulti
 * 
 * This example demonstrates a neural network with an LSTM recurrent layer
 * and multiple logical output layers.
 * 
 * It is used to verify the integration of LSTM with Multi-Output functionality.
 */
class ExampleLstmMulti
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleLstmMulti");
    
    // 1. Topology: 1 input, 10 hidden neurons, 2 total output neurons
    std::vector<unsigned> topology = { 1, 10, 20, 2 };

    // 2. Define multiple output layers
    // We want 2 separate output layers, each with 1 neuron.
    auto output_layers = {
      // First output: Tanh, Temperature 1.0
      OutputLayerDetails(1, activation(activation::method::tanh, 0.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::Adam, 0.9),
      // Second output: Tanh, Temperature 1.0
      OutputLayerDetails(1, activation(activation::method::tanh, 0.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::Adam, 0.9)
    };

    // 3. Configure options
    auto options = NeuralNetworkOptions::create(topology)
      .with_log_level(log_level)
      .with_learning_rate(0.01)
      .with_number_of_epoch(500)
      .with_batch_size(1)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(4) // sequence length
      .with_output_layer_details(output_layers);

    // Set the hidden layer to LSTM
    std::vector<LayerDetails> hidden_layers;
    hidden_layers.emplace_back(
      LayerDetails::LayerType::Lstm,
      10,
      activation(activation::method::tanh, 0.0),
      0.0, // dropout
      0.01, // weight decay
      OptimiserType::Adam,
      0.9
    );
    hidden_layers.emplace_back(
      LayerDetails::LayerType::Lstm,
      20,
      activation(activation::method::tanh, 0.0),
      0.0, // dropout
      0.01, // weight decay
      OptimiserType::Adam,
      0.9
    );
    options.with_hidden_layers(hidden_layers);
    options.build();

    return new NeuralNetwork(options);
  }

public:
  static void Run(Logger::LogLevel log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleLstmMulti");
    TEST_START("LSTM Multi-Output sequence test.")
    NeuralNetwork* nn = create_neural_network(log_level);

    // 4. Prepare training data for a uni variate sequence:
    // Sequence: [0.1, 0.2, 0.3, 0.4] -> Targets: [0.5, 0.25]
    std::vector<double> sequence = { 0.1, 0.2, 0.3, 0.4 };
    // wrap as batch of one sequence (sequence length must be divisible by input_size (1))
    std::vector<std::vector<double>> inputs = { sequence };
    std::vector<std::vector<double>> targets = { { 0.5, 0.25 } };

    // 5. Train
    Logger::info("Training LSTM Multi-Output on simple sequence...");
    nn->train(inputs, targets);

    // 6. Test prediction
    auto prediction_batch = nn->think(inputs);
    if (!prediction_batch.empty())
    {
      Logger::info("Prediction for [0.1, 0.2, 0.3, 0.4]: ");
      Logger::info("  Output 1: ", prediction_batch[0][0], " (Expected ~0.5)");
      Logger::info("  Output 2: ", prediction_batch[0][1], " (Expected ~0.25)");
    }

    // 7. Test Serialization
    Logger::info("Testing serialization...");
    std::string model_path = "lstm_multi_model.nn";
    NeuralNetworkSerializer::save(*nn, model_path);

    auto loaded_nn = NeuralNetworkSerializer::load(model_path);
    if (loaded_nn != nullptr)
    {
      auto loaded_prediction_batch = loaded_nn->think(inputs);
      if (!loaded_prediction_batch.empty())
      {
        Logger::info("Loaded model prediction: ", loaded_prediction_batch[0][0], ", ", loaded_prediction_batch[0][1]);
        
        bool match = true;
        for(size_t i = 0; i < prediction_batch[0].size(); ++i)
        {
          if (std::abs(prediction_batch[0][i] - loaded_prediction_batch[0][i]) > 1e-9)
          {
            match = false;
            break;
          }
        }

        if (match)
        {
          Logger::info("SUCCESS: Serialization verified. Predictions match.");
        }
        else
        {
          Logger::error("FAILURE: Serialization failed. Predictions differ!");
        }
      }
      delete loaded_nn;
    }
    else
    {
      Logger::error("FAILURE: Could not load the model!");
    }

    delete nn;
    TEST_END
  }
};
