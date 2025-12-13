#include <iostream>
#include <vector>
#include <random>

#include "../logger.h"
#include "../neuralnetwork.h"
#include "helper.h"

class ExampleCopyMemory
{
public:
  // Copy Memory task:
  // - sequence_length (S) bits presented first
  // - followed by delay (T) timesteps of zeros
  // - network must output the original S bits at the final timestep
  static void MemoryCopy(Logger::LogLevel log_level)
  {
    const unsigned sequence_length = 8;   // S
    const unsigned delay = 20;            // T
    const unsigned input_time_steps = sequence_length + delay; // total timesteps presented to input layer
    const unsigned output_size = sequence_length; // we want the network to reproduce the S bits at the end

    // topology: input_time_steps -> two recurrent hidden layers -> output_size
    std::vector<unsigned> topology = {
      input_time_steps,
      64,
      64,
      output_size
    };

    // mark which layers are recurrent (0 = input, last = output not recurrent)
    std::vector<unsigned> recurrent_layers = { 0, 1, 1, 0 };

    const int number_of_epoch = 2000;
    const double learning_rate = 0.01;

    // generate training set
    const size_t training_samples = 2000;
    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;
    training_inputs.reserve(training_samples);
    training_outputs.reserve(training_samples);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution bit_dist(0.5);

    for (size_t s = 0; s < training_samples; ++s)
    {
      std::vector<double> seq(sequence_length);
      for (unsigned i = 0; i < sequence_length; ++i)
      {
        seq[i] = bit_dist(gen) ? 1.0 : 0.0;
      }

      // input is flattened time series of length input_time_steps; each timestep has 1 feature
      std::vector<double> input(input_time_steps, 0.0);
      // first S timesteps are the sequence, remaining timesteps are zeros (the delay)
      for (unsigned t = 0; t < sequence_length; ++t)
      {
        input[t] = seq[t];
      }

      // target is the original sequence (we expect the network to output this vector at final timestep)
      std::vector<double> target = seq;

      training_inputs.emplace_back(std::move(input));
      training_outputs.emplace_back(std::move(target));
    }

    {
      TEST_START("CopyMemory test - No Batch.")
      Logger::info("CopyMemory: sequence_length=", sequence_length, ", delay=", delay, ", training_samples=", training_samples);

      auto options = NeuralNetworkOptions::create(topology)
        .with_batch_size(64)
        .with_hidden_activation_method(activation::method::tanh)
        .with_output_activation_method(activation::method::sigmoid)
        .with_log_level(log_level)
        .with_learning_rate(learning_rate)
        .with_number_of_epoch(number_of_epoch)
        .with_adaptive_learning_rates(false)
        .with_optimiser_type(OptimiserType::NadamW)
        .with_clip_threshold(5.0)
        .with_data_is_unique(false)
        .with_recurrent_layers(recurrent_layers)
        .build();

      auto* nn = new NeuralNetwork(options);
      nn->train(training_inputs, training_outputs);

      // test on a few random examples
      for (int test_i = 0; test_i < 5; ++test_i)
      {
        std::vector<double> seq(sequence_length);
        for (unsigned i = 0; i < sequence_length; ++i)
        {
          seq[i] = bit_dist(gen) ? 1.0 : 0.0;
        }

        std::vector<double> input(input_time_steps, 0.0);
        for (unsigned t = 0; t < sequence_length; ++t)
        {
          input[t] = seq[t];
        }

        auto output = nn->think(input);
        // output is a vector of size output_size (sequence_length)
        Logger::info("Test sample ", test_i, ": target=");
        std::string targ;
        for (auto v : seq) { targ += (v > 0.5 ? '1' : '0'); }
        Logger::info(targ);

        std::string outstr;
        outstr.reserve(output.size());
        for (auto v : output)
        {
          outstr += (v > 0.5 ? '1' : '0');
        }
        Logger::info(" prediction=", outstr);

        // simple accuracy metric: fraction of bits predicted correctly
        unsigned correct = 0;
        for (unsigned k = 0; k < output_size; ++k)
        {
          double pred_bit = output[k] > 0.5 ? 1.0 : 0.0;
          if (pred_bit == seq[k]) ++correct;
        }
        Logger::info(" accuracy=", static_cast<double>(correct) / static_cast<double>(output_size) * 100.0, "%");
      }

      delete nn;
      TEST_END
    }
  }
};