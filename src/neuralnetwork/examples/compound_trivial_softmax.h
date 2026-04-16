#pragma once
#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetworkserializer.h"
#include "helper.h"
#include <iomanip>

// Compound 
class ExampleCompoundTrivialSoftmax
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level, unsigned epoch, unsigned batch_size)
  {
    std::vector<unsigned> topology = { 1, 32, 64, 6 };
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(LayerDetails::LayerType::Gru, 32, activation(activation::method::tanh, 0.01), 0.0, 0.5),
      LayerDetails(LayerDetails::LayerType::Gru, 64, activation(activation::method::tanh, 0.01), 0.0, 0.5)
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(batch_size)
      .with_output_layer_details(
        {
          OutputLayerDetails(1, activation(activation::method::tanh, 0.01), ErrorCalculation::type::huber_direction_loss, { 0.001, 0.001, 1.0, 0.1, true, 1.0 }, 0.001, OptimiserType::NadamW),
          OutputLayerDetails(5, activation(activation::method::softmax, 0.01), ErrorCalculation::type::cross_entropy, { 0.0, 0.30, 1.0, 5.0, false, 1.0 }, 0.001, OptimiserType::NadamW)
        }
      )
      .with_log_level(log_level)
      .with_learning_rate(0.001)
      .with_number_of_epoch(epoch)
      .with_optimiser_type(OptimiserType::NadamW)
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
        {-0.10, 1.0, 0.0, 0.0, 0.0, 0.0},
        {-0.05, 0.0, 1.0, 0.0, 0.0, 0.0},
        { 0.00, 0.0, 0.0, 1.0, 0.0, 0.0},
        { 0.10, 0.0, 0.0, 0.0, 1.0, 0.0},
        { 0.05, 0.0, 0.0, 0.0, 0.0, 1.0}
    };

    nn.train(training_inputs, training_outputs);
  }

public:
  static void Run(Logger::LogLevel log_level)
  {
    TEST_START("Trivial Softmax test.")

    const unsigned epoch = 15000;
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

    // Calculate metrics for both layers - request only appropriate metrics for each
    std::vector<ErrorCalculation::type> all_layer_metrics = { ErrorCalculation::type::huber_direction_loss, ErrorCalculation::type::cross_entropy, ErrorCalculation::type::directional_confidence_score, ErrorCalculation::type::prediction_coverage };

    auto all_layer_metrics_results = nn->calculate_forecast_metrics_all_layers(all_layer_metrics, true);

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

    delete nn;
    TEST_END
  }
};
