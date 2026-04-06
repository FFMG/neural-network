#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetworkserializer.h"
#include "helper.h"

class ExampleXor
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level, unsigned epoch, unsigned batch_size)
  {
    std::vector<unsigned> topology = {3,2,1};
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(LayerDetails::LayerType::FF, 2, activation(activation::method::sigmoid, 0.01), 0.2)
    };

    auto output_layer = OutputLayerDetails(topology.back(), activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 });

    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(batch_size)
      .with_output_layer_details(output_layer)
      .with_log_level(log_level)
      .with_learning_rate(0.1)
      .with_learning_rate_warmup(0.01, 0.075)
      .with_learning_rate_decay_rate(0.0)
      .with_learning_rate_boost_rate(0.25, 0.05) // 5% total, boost 5% of the training
      .with_number_of_epoch(epoch)
      .with_optimiser_type(OptimiserType::SGD)
      .with_hidden_layers(hidden_layers)
      .build();

    return new NeuralNetwork(options);
  }

  static void train_neural_network(NeuralNetwork& nn)
  {
    // XOR training input, 3 values in at a time.
    std::vector<std::vector<double>> training_inputs = {
      {0, 0, 1},
      {1, 1, 1},
      {1, 0, 1},
      {0, 1, 1}
    };

    // output, one single value for each input.
    std::vector<std::vector<double>> training_outputs = {
        {0}, {1}, {1}, {0}
    };

    // the topology is 3 input, 1 output and one hidden layer with 3 neuron
    nn.train(training_inputs, training_outputs);

    // pass an array of array to think about
    // the result should be close to the training output.
    auto outputs = nn.think(training_inputs);
    Logger::info("Output After Training:", std::fixed, std::setprecision(10));
    for (const auto& row : outputs) 
    {
      for (double val : row) 
      {
        Logger::info("  ", val);
      }
    }
  }

public:
  static void Xor(Logger::LogLevel log_level, bool use_file)
  {
    TEST_START("Xor test.")

    // the file we will be loading from
    const char* file_name = "./xor.nn";
    const unsigned epoch = 5000;
    const unsigned batch_size = 1;

    // assume that it does not exist
    NeuralNetwork* nn = nullptr;
    if(use_file)
    {
      nn = NeuralNetworkSerializer::load(file_name);
      if( nullptr == nn )
      {
        // we need to create it
        nn = create_neural_network(log_level, epoch, batch_size);

        // train it
        train_neural_network(*nn);

        // save it
        NeuralNetworkSerializer::save(*nn, file_name);

        auto nn_saved = NeuralNetworkSerializer::load(file_name);
        Logger::info("Output from saved file:", std::fixed, std::setprecision(10));
        auto t1 = nn_saved->think({ 0, 0, 1 });
        Logger::info("  ", t1.front(), " (should be close to 0)");
        auto t2 = nn_saved->think({ 1, 1, 1 });
        Logger::info("  ", t2.front(), " (should be close to 1)");

        delete nn_saved;
      }
    }
    else
    {
      // we need to create it
      nn = create_neural_network(log_level, epoch, batch_size);

      // train it
      train_neural_network(*nn);
    }

    auto metrics = nn->calculate_forecast_metric( ErrorCalculation::type::rmse);

    Logger::info("Error (rmse): ", metrics.error());

    Logger::info("Output After Training:", std::fixed, std::setprecision(10));

    // or we can train with a single input
    // we know that the output only has one value.
    auto t1 = nn->think({ 0, 0, 1 });
    Logger::info("  ", t1.front(), " (should be close to 0)");

    // or we can train with a single input
    // we know that the output only has one value.
    auto t2 = nn->think({ 1, 1, 1 });
    Logger::info("  ", t2.front(), " (should be close to 1)");

    delete nn;
    TEST_END
  }
};