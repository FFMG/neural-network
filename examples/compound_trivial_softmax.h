#pragma once
#include "helpers/errorcalculation.h"
#include "common/logger.h"
#include "helpers/neuralnetworkserializer.h"
#include "helper.h"
#include <iomanip>

// Compound 

using namespace myoddweb::nn;
class ExampleCompoundTrivialSoftmax
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level, unsigned epoch, unsigned batch_size)
  {
    std::vector<unsigned> topology = { 1, 32, 64, 6 };
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Gru, 32, activation(activation::method::tanh, 0.01), 0.5, 0.0001, OptimiserType::NadamW, 0.9),
      LayerDetails(Layer::Architecture::Gru, 64, activation(activation::method::tanh, 0.01), 0.5, 0.0001, OptimiserType::NadamW, 0.9)
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(batch_size)
      .with_output_layer_details(
        {
          OutputLayerDetails(1, activation(activation::method::tanh, 0.01), ErrorCalculation::type::huber_direction_loss, { 0.01, 0.1, 0.1, 1.0, true, 1.0 }, 0.0, OptimiserType::NadamW, 0.9),
          OutputLayerDetails(5, activation(activation::method::softmax, 0.01), ErrorCalculation::type::cross_entropy, { 0.0, 0.5, 1.0, 1.0, true, 5.0 }, 0.0, OptimiserType::NadamW, 0.9)
        }
      )
      .with_log_level(log_level)
      .with_learning_rate(0.001)
      .with_number_of_epoch(epoch)
      .with_hidden_layers(hidden_layers)
      .with_data_is_unique(true)
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
        {-0.50, 1.0, 0.0, 0.0, 0.0, 0.0},
        {-0.25, 0.0, 1.0, 0.0, 0.0, 0.0},
        { 0.00, 0.0, 0.0, 1.0, 0.0, 0.0},
        { 0.25, 0.0, 0.0, 0.0, 1.0, 0.0},
        { 0.50, 0.0, 0.0, 0.0, 0.0, 1.0}
    };

    nn.train(training_inputs, training_outputs);
  }

public:
  static void Run(Logger::LogLevel log_level, bool use_file, bool continue_train)
  {
    TEST_START("Compound Trivial Softmax test.")

    const char* file_name = "./cts.nn";
    const unsigned epoch = 2000;
    const unsigned batch_size = 5;

    NeuralNetwork* nn = nullptr;
    if (use_file)
    {
      nn = NeuralNetworkSerializer::load(file_name);
      if (nullptr == nn)
      {
        // we need to create it
        nn = create_neural_network(log_level, epoch, batch_size);

        // train it
        train_neural_network(*nn);

        // save it
        NeuralNetworkSerializer::save(*nn, file_name);
      }

      if (continue_train)
      {
        // train it again
        train_neural_network(*nn);
      }
    }
    else
    {
      // we need to create it
      nn = create_neural_network(log_level, epoch, batch_size);

      // train it
      train_neural_network(*nn);
    }

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
      std::string s1 = Logger::factory("Input ", denorm_input, " -> ");
      for (double val : outputs[i])
      {
        s1 += Logger::factory(val, " | ");
      }
      Logger::info(s1);
    }

    // Calculate metrics for both layers
    auto all_layer_metrics_results = nn->calculate_forecast_metrics_all_layers({ ErrorCalculation::type::huber_direction_loss, ErrorCalculation::type::mse, ErrorCalculation::type::directional_confidence_score, ErrorCalculation::type::prediction_coverage });

    Logger::info("Layer 0 Metrics (Regression):");
    for (const auto& m : all_layer_metrics_results[0])
    {
      Logger::info("  ", ErrorCalculation::type_to_string(m.error_type()), ": ", m.error());
    }

    auto softmax_metrics = nn->calculate_forecast_metrics_all_layers({ ErrorCalculation::type::cross_entropy, ErrorCalculation::type::directional_accuracy, ErrorCalculation::type::directional_confidence_score, ErrorCalculation::type::prediction_coverage });

    Logger::info("Layer 1 Metrics (Softmax):");
    for (const auto& m : softmax_metrics[1])
    {
      Logger::info("  ", ErrorCalculation::type_to_string(m.error_type()), ": ", m.error());
    }

    delete nn;
    TEST_END
  }
};
