#include <iostream>
#include <vector>

#include "../logger.h"
#include "../neuralnetwork.h"
#include "helper.h"

class ExampleThreebitParity
{
public:
  static void ThreebitParity(Logger::LogLevel log_level)
  {
    std::vector<unsigned> topology = {3, 4, 1};
    const int number_of_epoch = 100000;
    const double learning_rate = 0.001;

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
      std::cout << "No Batch:" << std::endl;
      auto options = NeuralNetworkOptions::create(topology)
        .with_batch_size(1)
        .with_hidden_activation_method(activation::method::relu)
        .with_output_activation_method(activation::method::sigmoid)
        .with_log_level(log_level)
        .with_learning_rate(learning_rate)
        .with_number_of_epoch(number_of_epoch)
        .with_adaptive_learning_rates(false)
        .with_optimiser_type(OptimiserType::SGD)
        .build();

      auto* nn = new NeuralNetwork(options);
      nn->train(training_inputs, training_outputs);

      std::vector<double> test_input1 = {1, 1, 1};
      //std::vector<double> expected_output1 = {1};
      auto output1 = nn->think(test_input1);
      std::cout << output1.front() << " (should be close to 1)" << std::endl; //  should be close to 1

      std::vector<double> test_input2 = {1, 0, 1};       // 2 ones → even
      //std::vector<double> expected_output2 = {0};
      auto output2 = nn->think(test_input2);
      std::cout << output2.front() << " (should be close to 0)" << std::endl; //  should be close to 0

      delete nn;
      TEST_END
      std::cout << std::endl;
    }

    {
      for( int batch_size = 1; batch_size <= 4; ++batch_size)
      {
        TEST_START("ThreebitParity test - Batch.")
        std::cout << "Batch size=" << batch_size <<":" << std::endl;

        auto options = NeuralNetworkOptions::create(topology)
          .with_batch_size(batch_size)
          .with_hidden_activation_method(activation::method::relu)
          .with_output_activation_method(activation::method::sigmoid)
          .with_log_level(log_level)
          .with_learning_rate(learning_rate)
          .with_number_of_epoch(number_of_epoch)
          .with_adaptive_learning_rates(false)
          .with_optimiser_type(OptimiserType::SGD)
          .build();

        auto* nn = new NeuralNetwork(options);
        nn->train(training_inputs, training_outputs);

        std::vector<double> test_input1 = {1, 1, 1};
        //std::vector<double> expected_output1 = {1};
        auto output1 = nn->think(test_input1);
        std::cout << output1.front() << " (should be close to 1)" << std::endl; //  should be close to 1

        std::vector<double> test_input2 = {1, 0, 1};       // 2 ones → even
        //std::vector<double> expected_output2 = {0};
        auto output2 = nn->think(test_input2);
        std::cout << output2.front() << " (should be close to 0)" << std::endl; //  should be close to 0

        delete nn;
        TEST_END
        std::cout << std::endl;
      }
    }  
  }
};