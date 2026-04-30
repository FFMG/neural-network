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
    
    // 1. Topology: 17 inputs, 80 hidden neurons, 7 total output neurons
    std::vector<unsigned> topology = { 17, 80, 40, 7 };

    // 2. Define multiple output layers
    // We want 2 separate output layers, one with 5 neurons and one with 2 neurons.
    auto output_layers = {
      MultiOutputLayerDetails(
        {
          LayerDetails(Layer::Architecture::FF,
                16,
                activation(activation::method::tanh, 0.0, 1.0),
                0.0, // dropout
                0.01, // weight decay
                OptimiserType::Adam,
                0.9),
          LayerDetails(Layer::Architecture::FF,
                32,
                activation(activation::method::tanh, 0.0, 1.0),
                0.0, // dropout
                0.01, // weight decay
                OptimiserType::Adam,
                0.9)
        },
        OutputLayerDetails(5, activation(activation::method::tanh, 0.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::Adam, 0.9)
      ),
      MultiOutputLayerDetails(
        {
          LayerDetails(Layer::Architecture::FF,
                16,
                activation(activation::method::tanh, 0.0, 1.0),
                0.0, // dropout
                0.01, // weight decay
                OptimiserType::Adam,
                0.9),
          LayerDetails(Layer::Architecture::FF,
                32,
                activation(activation::method::tanh, 0.0, 1.0),
                0.0, // dropout
                0.01, // weight decay
                OptimiserType::Adam,
                0.9)
        },
        // Second output: Tanh, Temperature 1.0
        OutputLayerDetails(2, activation(activation::method::tanh, 0.0, 1.0), ErrorCalculation::type::mse, EvaluationConfig(), 0.01, OptimiserType::Adam, 0.9)
        )

    };

    // 3. Configure options
    auto options = NeuralNetworkOptions::create(topology)
      .with_log_level(log_level)
      .with_learning_rate(0.01)
      .with_number_of_epoch(5000)
      .with_batch_size(1)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(4) // sequence length
      .with_output_layer_details(output_layers);

    // Set the hidden layer to LSTM
    std::vector<LayerDetails> hidden_layers;
    hidden_layers.emplace_back(
      Layer::Architecture::Lstm,
      80,
      activation(activation::method::tanh, 0.0, 1.0),
      0.0, // dropout
      0.01, // weight decay
      OptimiserType::Adam,
      0.9
    );
    hidden_layers.emplace_back(
      Layer::Architecture::Lstm,
      40,
      activation(activation::method::tanh, 0.0, 1.0),
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

    // 4. Prepare training data for a multivariate sequence:
    // Sequence: [0.1, ..., 0.1] (17 values)
    std::vector<double> sequence(17, 0.1);
    // Provide 4 samples so BPTT can form a batch of length 4
    std::vector<std::vector<double>> inputs = { sequence, sequence, sequence, sequence };
    std::vector<double> target = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25 };
    std::vector<std::vector<double>> targets = { target, target, target, target };

    // 5. Train
    Logger::info("Training LSTM Multi-Output on simple sequence...");
    nn->train(inputs, targets);

    // 6. Test prediction
    auto prediction = nn->think(sequence);
    if (!prediction.empty())
    {
      Logger::info("Prediction for [0.1 (x17)]: ");
      Logger::info("  Output 1: ", prediction[0], " (Expected ~0.5)");
      Logger::info("  Output 2: ", prediction[5], " (Expected ~0.25)");
    }

    // 7. Test Serialization
    Logger::info("Testing serialization...");
    std::string model_path = "lstm_multi_model.nn";
    NeuralNetworkSerializer::save(*nn, model_path);

    auto loaded_nn = NeuralNetworkSerializer::load(model_path);
    if (loaded_nn != nullptr)
    {
      auto loaded_prediction = loaded_nn->think(sequence);
      if (!loaded_prediction.empty())
      {
        Logger::info("Loaded model prediction: ", loaded_prediction[0], ", ", loaded_prediction[5]);
        
        bool match = true;
        for(size_t i = 0; i < prediction.size(); ++i)
        {
          if (std::abs(prediction[i] - loaded_prediction[i]) > 1e-9)
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
