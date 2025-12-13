#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetworkserializer.h"

#include "helper.h"

#include <cerrno>  // For errno
#include <cstring> // For strerror
#include <fstream>
#include <iomanip>

class SpiralLoader 
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
      double y1, y2;

      std::getline(ss, item, ',');
      x1 = std::stod(item);

      std::getline(ss, item, ',');
      x2 = std::stod(item);

      std::getline(ss, item, ',');
      y1 = std::stoi(item);

      std::getline(ss, item, ',');
      y2 = std::stoi(item);

      inputs.push_back({ x1, x2 });
      outputs.push_back({ y1, y2 });
    }
  }
};

class ExampleSpiral
{
private:
  static NeuralNetwork* create_neural_network(Logger::LogLevel log_level, unsigned epoch, unsigned batch_size, double learning_rate)
  {
    std::vector<unsigned> topology = {2, 32, 32, 2};
    auto options = NeuralNetworkOptions::create(topology)
      .with_batch_size(batch_size)
      .with_hidden_activation_method(activation::method::tanh)
      .with_output_activation_method(activation::method::sigmoid)
      .with_log_level(log_level)
      .with_dropout({0.0, 0.0})
      .with_learning_rate(learning_rate)
      .with_clip_threshold(2)
      .with_learning_rate_warmup(learning_rate * 0.5, 0.075) // from 1/2 LR to LR
      .with_learning_rate_decay_rate(0.0)
      .with_learning_rate_boost_rate(0.25, 0.05) // 5% total, boost 5% of the training
      .with_number_of_epoch(epoch)
      .with_optimiser_type(OptimiserType::NadamW)
      .build();

    return new NeuralNetwork(options);
  }

  static void train_neural_network(NeuralNetwork& nn)
  {
    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;
    SpiralLoader::load_csv("./examples/two_moons.csv", training_inputs, training_outputs);

    Logger::debug("Loaded ", training_inputs.size(), " samples.");
    Logger::debug("First sample: x1=" ,training_inputs[0][0] 
                 ," x2=", training_inputs[0][1] 
                 ," y1=", training_outputs[0][0]
                 ," y2=", training_outputs[0][1]);
    // the topology is 3 input, 1 output and one hidden layer with 3 neuron
    nn.train(training_inputs, training_outputs);

    // pass an array of array to think about
    // the result should be close to the training output.
    auto outputs = nn.think(training_inputs);
    Logger::info("Output After Training:");
    for (size_t i = 0; i < outputs.size(); ++i) 
    {
      Logger::info(" ", std::fixed, std::setprecision(10), 
        "y1=", outputs[i][0], ", y2=", outputs[i][1],
        " exp: ", 
        "y1=", training_outputs[i][0], ", y2=", training_outputs[i][1]);
    }
  }

public:
  static void Spiral(Logger::LogLevel log_level, bool use_file)
  {
    TEST_START("Spiral test.")

    // the file we will be loading from
    const char* file_name = "./spiral.nn";

    const unsigned epoch = 200;
    const unsigned batch_size = 32;
    const double one_batch_learning_rate = 0.001;
    const double learning_rate = (batch_size * one_batch_learning_rate);

    // assume that it does not exist
    NeuralNetwork* nn = nullptr;
    if(use_file)
    {
      // nn = NeuralNetworkSerializer::load(file_name);
      if( nullptr == nn )
      {
        // we need to create it
        nn = create_neural_network(log_level, epoch, batch_size, learning_rate);

        // train it
        train_neural_network(*nn);

        // save it
        NeuralNetworkSerializer::save(*nn, file_name);
      }
    }
    else
    {
      // we need to create it
      nn = create_neural_network(log_level, epoch, batch_size, learning_rate);

      // train it
      train_neural_network(*nn);
    }

    auto metrics = nn->calculate_forecast_metrics( 
      {
        ErrorCalculation::type::rmse,
        ErrorCalculation::type::bce_loss });
    Logger::debug("Error rmse: ", metrics[0].error());
    Logger::debug("Error bce:  ", metrics[1].error());

    delete nn;
    TEST_END
    std::cout << std::endl;
  }
};