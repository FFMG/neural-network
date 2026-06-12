#pragma once
#include "helpers/errorcalculation.h"
#include "common/logger.h"
#include "helpers/neuralnetworkserializer.h"
#include "helper.h"
#include <iomanip>


using namespace myoddweb::nn;
class ExampleTrivialSoftmax
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level, unsigned epoch, unsigned batch_size)
  {
    std::vector<unsigned> topology = { 1, 32, 64, 5 };
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Gru, 32, activation(activation::method::tanh, 0.01), 0.0, 0.5, OptimiserType::NadamW, 0.9),
      LayerDetails(Layer::Architecture::Gru, 64, activation(activation::method::tanh, 0.01), 0.0, 0.5, OptimiserType::NadamW, 0.9)
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(batch_size)
      .with_output_layer_details(
        OutputLayerDetails(5, activation(activation::method::softmax, 0.0), ErrorCalculation::type::cross_entropy, { 0.001, 0.001, 1.0, 0.1, true, 1.0 }, 0.001, OptimiserType::NadamW, 0.99))
      .with_log_level(log_level)
      .with_learning_rate(0.001)
      .with_number_of_epoch(epoch)
      .with_hidden_layers(hidden_layers)
      .build();

    return new NeuralNetwork(options);
  }

  static void train_neural_network(NeuralNetwork& nn)
  {
    // Normalized inputs: (val - 30)/20 -> range [-1, 1]
    std::vector<std::vector<double>> training_inputs = {
        {-1.0}, {-0.5}, {0.0}, {0.5}, {1.0}
    };
    std::vector<std::vector<double>> training_outputs = {
        {1.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0}
    };

    nn.train(training_inputs, training_outputs);
  }

public:
  static void Run(Logger::LogLevel log_level)
  {
    TEST_START("Trivial Softmax test.")

    const unsigned epoch = 500;
    const unsigned batch_size = 5;

    NeuralNetwork* nn = create_neural_network(log_level, epoch, batch_size);
    train_neural_network(*nn);

    // Normalized inputs for inference
    std::vector<std::vector<double>> inputs = {
        {-1.0}, {-0.5}, {0.0}, {0.5}, {1.0}
    };

    auto outputs = nn->think(inputs);
    Logger::info("Output After Training:", std::fixed, std::setprecision(4));
    for (size_t i = 0; i < inputs.size(); ++i)
    {
      // Denormalize input for logging: val * 20 + 30
      double denorm_input = inputs[i][0] * 20.0 + 30.0;
      Logger::info("Input ", denorm_input, " -> ");
      for (double val : outputs[i])
      {
        Logger::info("  ", val);
      }
    }

    delete nn;
    TEST_END
  }
};
