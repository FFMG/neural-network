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
    auto epoch = 100000;

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

    std::vector<unsigned> topology = {2, 8, 8, 8, 8, 1};
    std::vector<double> dropout = { 0.0, 0.0, 0.2, 0.0 };
    // std::vector<unsigned> topology = {2, 3, 1};
    // std::vector<double> dropout = { 0.0 };
    auto options = NeuralNetworkOptions::create(topology)
      .with_dropout(dropout)
      .with_batch_size(batch_size)
      .with_hidden_activation_method(activation::method::relu)
      .with_output_activation_method(activation::method::sigmoid)
      .with_log_level(log_level)
      .with_learning_rate(0.0003)
      .with_clip_threshold(1.5)
      .with_learning_rate_decay_rate(0.0)
      .with_learning_rate_boost_rate(0.00, 1.0)
      .with_number_of_epoch(epoch)
      .with_optimiser_type(OptimiserType::NadamW)
      .with_residual_layer_jump(3)
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
    std::cout << std::endl;
  }
};