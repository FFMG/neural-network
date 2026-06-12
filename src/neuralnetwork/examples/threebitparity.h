#include <vector>

#include "../logger.h"
#include "../neuralnetwork.h"
#include "helper.h"


using namespace myoddweb::nn;
class ExampleThreebitParity
{
public:
  static void ThreebitParity(Logger::LogLevel log_level)
  {
    std::vector<unsigned> topology = { 3, 8, 1 };
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Elman, 8, activation(activation::method::tanh, 0.01), 0.0, 0.05, OptimiserType::NadamW, 0.9),
    };
    auto output_layer = OutputLayerDetails(topology.back(), activation(activation::method::sigmoid, 0.01), ErrorCalculation::type::mse, { 0.0, 0.0, 1.0, 0.0, false, 1.0 }, 0.05, OptimiserType::NadamW, 0.99);
    
    const int number_of_epoch = 5000;
    const double learning_rate = 0.01;

    std::vector<std::vector<double>> training_inputs = {
      {0, 0, 0},
      {0, 0, 1},
      {0, 1, 0},
      {0, 1, 1},
      {1, 0, 0},
      {1, 0, 1},
      {1, 1, 0},
      {1, 1, 1}
    };
    std::vector<std::vector<double>> training_outputs = {
        {0},  // 0 ones → even
        {1},  // 1 one  → odd
        {1},  // 1 one  → odd
        {0},  // 2 ones → even
        {1},  // 1 one  → odd
        {0},  // 2 ones → even
        {0},  // 2 ones → even
        {1}   // 3 ones → odd
    };

    {
      TEST_START("ThreebitParity test - No Batch.")
        Logger::info("No Batch:");
      auto options = NeuralNetworkOptions::create(topology)
        .with_batch_size(1)
        .with_output_layer_details(output_layer)
        .with_log_level(log_level)
        .with_learning_rate(learning_rate)
        .with_number_of_epoch(number_of_epoch)
        .with_adaptive_learning_rates(false)
        .with_clip_threshold(1.0)
        .with_data_is_unique(true)
        .with_hidden_layers(hidden_layers)
        .with_enable_bptt(true)
        .with_bptt_max_ticks(16)
        .build();

      auto* nn = new NeuralNetwork(options);
      nn->train(training_inputs, training_outputs);

      std::vector<double> test_input1 = { 1, 1, 1 };
      //std::vector<double> expected_output1 = {1};
      auto output1 = nn->think(test_input1);
      Logger::info(output1.front(), " (should be close to 1)"); //  should be close to 1

      std::vector<double> test_input2 = { 1, 0, 1 };       // 2 ones → even
      //std::vector<double> expected_output2 = {0};
      auto output2 = nn->think(test_input2);
      Logger::info(output2.front(), " (should be close to 0)"); //  should be close to 0

      delete nn;
      TEST_END
    }

    /*
     * For testting batch size > 1
    {
      for( int batch_size = 1; batch_size <= 4; ++batch_size)
      {
        TEST_START("ThreebitParity test - Batch.")
        Logger::info("Batch size=", batch_size, ":");

        auto options = NeuralNetworkOptions::create(topology)
          .with_batch_size(batch_size)
          .with_hidden_activation_method(activation::method::tanh)
          .with_output_activation_method(activation::method::sigmoid)
          .with_log_level(log_level)
          .with_learning_rate(learning_rate)
          .with_number_of_epoch(number_of_epoch)
          .with_adaptive_learning_rates(false)
          .with_optimiser_type(OptimiserType::SGD)
          .with_clip_threshold(1.0)
          .with_data_is_unique(true)
          .with_recurrent_layers(rlayers)
          .build();

        auto* nn = new NeuralNetwork(options);
        nn->train(training_inputs, training_outputs);

        std::vector<double> test_input1 = {1, 1, 1};
        //std::vector<double> expected_output1 = {1};
        auto output1 = nn->think(test_input1);
        Logger::info(output1.front(), " (should be close to 1)"); //  should be close to 1

        std::vector<double> test_input2 = {1, 0, 1};       // 2 ones → even
        //std::vector<double> expected_output2 = {0};
        auto output2 = nn->think(test_input2);
        Logger::info(output2.front(), " (should be close to 0)"); //  should be close to 0

        delete nn;
        TEST_END
      }
    }
    */
  }
};