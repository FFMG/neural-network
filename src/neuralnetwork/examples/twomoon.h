#include "../neuralnetworkserializer.h"
#include "helper.h"
#include "../logger.h"

#include <cerrno>  // For errno
#include <cstring> // For strerror
#include <iomanip>

class TwoMoonLoader 
{
public:
  static void load_csv(
    const std::string& path,
    std::vector<std::vector<double>>& inputs,
    std::vector<std::vector<double>>& outputs
  ) 
  {
    std::ifstream file(path);
    if (!file.is_open()) 
    {
      int errnum = errno; 

      Logger::error("Error opening file: ", path);
      Logger::error("C++ stream error state: ");
      if (file.fail()) Logger::error("  failbit ");
      if (file.bad())  Logger::error("  badbit ");
      if (file.eof())  Logger::error("  eofbit ");
      Logger::error("Could not open file: ", path);
      Logger::error("System Error (", errnum, "): ", std::strerror(errnum));
      throw std::runtime_error("Could not open file: " + path);
    }

    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) 
    {
      std::stringstream ss(line);
      std::string item;

      double x1, x2;
      int label;

      std::getline(ss, item, ',');
      x1 = std::stod(item);

      std::getline(ss, item, ',');
      x2 = std::stod(item);

      std::getline(ss, item, ',');
      label = std::stoi(item);

      inputs.push_back({ x1, x2 });

      // One-hot encode if you want: {1,0} or {0,1}
      outputs.push_back({ static_cast<double>(label) });
    }
  }
};

class ExampleTwoMoon
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level, unsigned epoch, unsigned batch_size)
  {
    std::vector<unsigned> topology = {2, 8, 8, 1};
    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(batch_size)
      .with_hidden_activation_method(activation::method::tanh)
      .with_output_activation_method(activation::method::sigmoid)
      .with_log_level(log_level)
      .with_dropout({0.0, 0.0})
      .with_learning_rate(0.80)
      .with_clip_threshold(2)
      .with_learning_rate_warmup(0.50, 0.80) // from 0.01 to 0.05
      .with_learning_rate_decay_rate(0.0)
      .with_learning_rate_boost_rate(0.25, 0.05) // 5% total, boost 5% of the training
      .with_number_of_epoch(epoch)
      .with_optimiser_type(OptimiserType::SGD)
      .with_batch_size(16)
      .build();

    return new NeuralNetwork(options);
  }

  static void train_neural_network(NeuralNetwork& nn)
  {
    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;

    TwoMoonLoader::load_csv("./src/neuralnetwork/examples/two_moons.csv", training_inputs, training_outputs);

    Logger::debug("Loaded ", training_inputs.size(), " samples.");
    Logger::debug("First sample: x1=" ,training_inputs[0][0] 
                 ," x2=", training_inputs[0][1] 
                 ," label=", training_outputs[0][0]);    
    // the topology is 3 input, 1 output and one hidden layer with 3 neuron
    nn.train(training_inputs, training_outputs);

    // pass an array of array to think about
    // the result should be close to the training output.
    auto outputs = nn.think(training_inputs);
    std::cout << "Output After Training:" << std::endl;
    std::cout << std::fixed << std::setprecision(10);
    for (const auto& row : outputs) 
    {
      for (double val : row) 
      {
        std::cout << val << std::endl;
      }
    }
    std::cout << std::endl;  
  }

public:
  static void TwoMoon(Logger::LogLevel log_level, bool use_file)
  {
    TEST_START("TwoMoon test.")

    // the file we will be loading from
    const char* file_name = "./twomoon.nn";
    const unsigned epoch = 300;
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
        std::cout << "Output from saved file:" << std::endl;
        std::cout << std::fixed << std::setprecision(10);
        auto t1 = nn_saved->think({ 0, 0, 1 });
        std::cout << t1.front() << std::endl;//  should be close to 0
        auto t2 = nn_saved->think({ 1, 1, 1 });
        std::cout << t2.front() << std::endl; //  should be close to 1

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

    auto metrics = nn->calculate_forecast_metric( NeuralNetworkOptions::ErrorCalculation::rmse);

    std::cout << "Error: " << metrics.error() << std::endl;

    std::cout << "Output After Training:" << std::endl;
    std::cout << std::fixed << std::setprecision(10);

    // or we can train with a single inut
    // we know that the output only has one value.
    auto t1 = nn->think({ 0, 0, 1 });
    std::cout << t1.front() << " (should be close to 0)" << std::endl;//  should be close to 0

    // or we can train with a single inut
    // we know that the output only has one value.
    auto t2 = nn->think({ 1, 1, 1 });
    std::cout << t2.front() << " (should be close to 1)" << std::endl; //  should be close to 1

    delete nn;
    TEST_END
    std::cout << std::endl;
  }
};