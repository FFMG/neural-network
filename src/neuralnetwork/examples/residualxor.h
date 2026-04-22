#include "../neuralnetworkserializer.h"
#include "helper.h"
#include "../logger.h"

#include <iomanip>

class ExampleResidualXor
{
public:
  static void Xor(Logger::LogLevel log_level,  bool use_file)
  {
    TEST_START("Residual Xor test.")

    auto batch_size = 3;
    auto epoch = 10000;

    // XOR training input, 3 values in at a time.
    std::vector<std::vector<double>> training_inputs = {
      {0.0, 0.0},  // 0 XOR 0 = 0
      {0.0, 1.0},  // 0 XOR 1 = 1
      {1.0, 0.0},  // 1 XOR 0 = 1
      {1.0, 1.0}   // 1 XOR 1 = 0
    };

    // output, one single value for each input.
    std::vector<std::vector<double>> training_outputs = {
      {0.0},
      {1.0},
      {1.0},
      {0.0}
    };

    std::vector<unsigned> topology = { 2, 8, 8, 8, 8, 1 };

    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(LayerDetails::LayerType::Elman, 8, activation(activation::method::relu, 0.01), 0.0, 0.05, OptimiserType::NadamW, 0.95),
      LayerDetails(LayerDetails::LayerType::Elman, 8, activation(activation::method::relu, 0.01), 0.0, 0.05, OptimiserType::NadamW, 0.95),
      LayerDetails(LayerDetails::LayerType::Elman, 8, activation(activation::method::relu, 0.01), 0.2, 0.05, OptimiserType::NadamW, 0.95),
      LayerDetails(LayerDetails::LayerType::Elman, 8, activation(activation::method::relu, 0.01), 0.0, 0.05, OptimiserType::NadamW, 0.95),
    };

    auto output_layer = OutputLayerDetails(topology.back(), activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.05, OptimiserType::NadamW, 0.99);
    
    // std::vector<unsigned> topology = {2, 3, 1};
    // std::vector<double> dropout = { 0.0 };
    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(batch_size)
      .with_output_layer_details(output_layer)
      .with_log_level(log_level)
      .with_learning_rate(0.0003)
      .with_clip_threshold(2.0)
      .with_learning_rate_decay_rate(0.0)
      .with_learning_rate_boost_rate(0.00, 1.0)
      .with_number_of_epoch(epoch)
      .with_residual_layer_jump(3)
      .with_hidden_layers(hidden_layers)
      .build();

    const char* file_name = "./residualxor.nn";
    NeuralNetwork* nn = nullptr;
    if(use_file)
    {
      nn = NeuralNetworkSerializer::load(file_name);
      if( nullptr == nn )
      {
        nn = new NeuralNetwork(options);
        nn->train(training_inputs, training_outputs);
        Logger::debug("Output After Training: ", std::fixed, std::setprecision(10));

        // save it
        NeuralNetworkSerializer::save(*nn, file_name);
      }
    }
    else
    {
      nn = new NeuralNetwork(options);
      nn->train(training_inputs, training_outputs);
      Logger::debug("Output After Training: ", std::fixed, std::setprecision(10));
    }

    // or we can train with a single inut
    // we know that the output only has one value.
    auto t1 = nn->think({ 0, 0 });
    Logger::debug(t1.front(), " (should be close to 0)");//  should be close to 0

    // or we can train with a single inut
    // we know that the output only has one value.
    auto t2 = nn->think({ 1, 1 });
    Logger::debug(t2.front(), " (should be close to 0)"); //  should be close to 1

    // or we can train with a single inut
    // we know that the output only has one value.
    auto t3 = nn->think({ 1, 0 });
    Logger::debug(t3.front(), " (should be close to 1)"); //  should be close to 1

    delete nn;

    TEST_END
  }
};