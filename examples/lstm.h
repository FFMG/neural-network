#pragma once
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"
#include "common/logger.h"
#include "helper.h"
#include <vector>


using namespace myoddweb::nn;
class ExampleLstm
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level)
  {
    MYODDWEB_PROFILE_FUNCTION("ExampleLstm");
    // 1. For un ivariate sequence forecasting use input size = 1
    std::vector<unsigned> topology = { 1, 10, 1 };

    // 2. Configure options
    auto options = NeuralNetworkOptions::create(topology)
      .with_log_level(log_level)
      .with_learning_rate(0.01)
      .with_number_of_epoch(500)
      .with_batch_size(1)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(4); // sequence length

    // Set the hidden layer to LSTM
    std::vector<LayerDetails> hidden_layers;
    hidden_layers.emplace_back(
      Layer::Architecture::Lstm,
      10,
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
    MYODDWEB_PROFILE_FUNCTION("ExampleLstm");
    TEST_START("LSTM sequence test.")
    NeuralNetwork* nn = create_neural_network(log_level);

    // 3. Prepare training data for a uni variate sequence:
    // Sequence: [0.1, 0.2, 0.3, 0.4] -> Target: [0.5]
    std::vector<double> sequence = { 0.1, 0.2, 0.3, 0.4 };
    // wrap as batch of one sequence (sequence length must be divisible by input_size (1))
    std::vector<std::vector<double>> inputs = { sequence };
    std::vector<std::vector<double>> targets = { { 0.5 } };

    // 4. Train
    Logger::info("Training LSTM on simple sequence...");
    nn->train(inputs, targets);

    // 5. Test — call the batch think overload (pass the sequence in the same form)
    auto prediction_batch = nn->think(inputs);
    if (!prediction_batch.empty())
    {
      Logger::info("Prediction for next value after [0.1, 0.2, 0.3, 0.4]: ", prediction_batch[0][0], " (Expected ~0.5)");
    }

    // 6. Test Serialization
    Logger::info("Testing serialization...");
    std::string model_path = "lstm_model.nn";
    NeuralNetworkSerializer::save(*nn, model_path);

    auto loaded_nn = NeuralNetworkSerializer::load(model_path);
    if (loaded_nn != nullptr)
    {
      auto loaded_prediction_batch = loaded_nn->think(inputs);
      if (!loaded_prediction_batch.empty())
      {
        Logger::info("Loaded model prediction: ", loaded_prediction_batch[0][0]);
        if (std::abs(prediction_batch[0][0] - loaded_prediction_batch[0][0]) < 1e-9)
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
