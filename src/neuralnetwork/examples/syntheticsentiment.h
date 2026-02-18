#pragma once

#include "helper.h"
#include "../errorcalculation.h"
#include "../logger.h"
#include "../neuralnetworkserializer.h"
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

class ExampleSyntheticSentiment
{
public:
  // Vocabulary Size = 16
  // 0: PAD
  // 1: "the"
  // 2: "this"
  // 3: "a"
  // 4: "movie"
  // 5: "film"
  // 6: "is"
  // 7: "was"
  // 8: "very"
  // 9: "not"
  // 10: "good" (+)
  // 11: "great" (+)
  // 12: "excellent" (+)
  // 13: "bad" (-)
  // 14: "awful" (-)
  // 15: "terrible" (-)
  static const int VOCAB_SIZE = 16;

  static void generate_data(
      size_t num_samples,
      size_t sequence_length,
      std::vector<std::vector<double>>& inputs,
      std::vector<std::vector<double>>& outputs)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Lists of word indices
    std::vector<int> starts = { 1, 2 }; // "the", "this"
    std::vector<int> nouns = { 4, 5 }; // "movie", "film"
    std::vector<int> verbs = { 6, 7 }; // "is", "was"
    std::vector<int> articles = { 3 }; // "a"
    std::vector<int> adverbs = { 8 }; // "very"
    std::vector<int> negations = { 9 }; // "not"
    std::vector<int> positives = { 10, 11, 12 };
    std::vector<int> negatives = { 13, 14, 15 };

    inputs.clear();
    outputs.clear();
    inputs.reserve(num_samples);
    outputs.reserve(num_samples);

    for (size_t i = 0; i < num_samples; ++i)
    {
      std::vector<int> seq_indices;
      double sentiment = 0.5; // undefined

      // Simple template choice
      // 0: "The movie is [not] [very] good/bad"
      // 1: "This is a [not] [very] good/bad movie"
      int template_id = std::uniform_int_distribution<>(0, 1)(gen);

      bool use_not = std::bernoulli_distribution(0.3)(gen); // 30% chance of 'not'
      bool use_very = std::bernoulli_distribution(0.3)(gen); // 30% chance of 'very'
      bool is_positive_word = std::bernoulli_distribution(0.5)(gen);

      int sentiment_word = is_positive_word 
          ? positives[std::uniform_int_distribution<size_t>(0, positives.size()-1)(gen)]
          : negatives[std::uniform_int_distribution<size_t>(0, negatives.size()-1)(gen)];
      
      // Determine label
      // Base: Pos=1, Neg=0
      double label = is_positive_word ? 1.0 : 0.0;
      if (use_not) label = 1.0 - label; // Flip
      // "Very" doesn't change polarity in this simple binary model

      if (template_id == 0)
      {
          seq_indices.push_back(starts[std::uniform_int_distribution<size_t>(0, starts.size()-1)(gen)]);
          seq_indices.push_back(nouns[std::uniform_int_distribution<size_t>(0, nouns.size()-1)(gen)]);
          seq_indices.push_back(verbs[std::uniform_int_distribution<size_t>(0, verbs.size()-1)(gen)]);
          if (use_not) seq_indices.push_back(9);
          if (use_very) seq_indices.push_back(8);
          seq_indices.push_back(sentiment_word);
      }
      else
      {
          seq_indices.push_back(2); // This
          seq_indices.push_back(verbs[std::uniform_int_distribution<size_t>(0, verbs.size()-1)(gen)]);
          seq_indices.push_back(3); // a
          if (use_not) seq_indices.push_back(9);
          if (use_very) seq_indices.push_back(8);
          seq_indices.push_back(sentiment_word);
          seq_indices.push_back(nouns[std::uniform_int_distribution<size_t>(0, nouns.size()-1)(gen)]);
      }

      // Pad with 0
      while (seq_indices.size() < sequence_length)
      {
          seq_indices.push_back(0);
      }
      // Truncate if too long (shouldn't happen with these templates and len=10)
      if (seq_indices.size() > sequence_length)
      {
          seq_indices.resize(sequence_length);
      }

      // One-hot encode
      std::vector<double> input_vec;
      input_vec.reserve(sequence_length * VOCAB_SIZE);
      for (int idx : seq_indices)
      {
          for (int v = 0; v < VOCAB_SIZE; ++v)
          {
              input_vec.push_back(v == idx ? 1.0 : 0.0);
          }
      }

      inputs.push_back(input_vec);
      outputs.push_back({ label });
    }
  }

  static void SyntheticSentiment(Logger::LogLevel log_level)
  {
    TEST_START("Synthetic Sentiment Analysis (GRU)");

    const size_t sequence_length = 32;
    const size_t train_samples = 2000;
    const size_t test_samples = 200;
    const unsigned epochs = 20; 
    const unsigned batch_size = 16;
    const double learning_rate = 0.005;

    std::vector<std::vector<double>> train_inputs, train_outputs;
    std::vector<std::vector<double>> test_inputs, test_outputs;

    Logger::info("Generating Data...");
    generate_data(train_samples, sequence_length, train_inputs, train_outputs);
    generate_data(test_samples, sequence_length, test_inputs, test_outputs);

    // Topology
    // Input: 16 (Vocab size)
    // Hidden: 24 (GRU)
    // Output: 1 (Sigmoid)
    std::vector<unsigned> topology = { VOCAB_SIZE, 24, 1 };
    std::vector<LayerDetails> hidden_layers = {
        LayerDetails(LayerDetails::LayerType::Gru, 24)
    };

    auto options = NeuralNetworkOptions::create(topology)
        .with_batch_size(batch_size)
        .with_hidden_activation_method(activation::method::tanh) 
        .with_output_activation_method(activation::method::sigmoid)
        .with_log_level(log_level)
        .with_learning_rate(learning_rate)
        .with_optimiser_type(OptimiserType::Adam)
        .with_hidden_layers(hidden_layers)
        .with_number_of_epoch(epochs)
        .with_enable_bptt(true)
        .with_number_of_threads(4)
        .with_bptt_max_ticks((int)sequence_length)
        .with_data_is_unique(false) // Generated data might have duplicates
        .with_clip_threshold(5.0)
        .with_shuffle_training_data(false)
        .with_shuffle_bptt_batches(true)
        .build();    
    NeuralNetwork nn(options);

    Logger::info("Training...");
    nn.train(train_inputs, train_outputs);

    Logger::info("Testing...");
    auto results = nn.think(test_inputs);

    int correct = 0;
    for(size_t i=0; i<test_samples; ++i)
    {
      double pred = results[i][0];
      double actual = test_outputs[i][0];
      bool pred_class = pred > 0.5;
      bool actual_class = actual > 0.5;

      if (pred_class == actual_class) correct++;

      if (i < 10)
      {
         Logger::info(" Sample ", i, ": Pred=", std::fixed, std::setprecision(4), pred, 
                      " Actual=", actual, " Correct=", (pred_class == actual_class ? "YES" : "NO"));
      }
    }

    double accuracy = (double)correct / test_samples * 100.0;
    Logger::info("Test Accuracy: ", std::fixed, std::setprecision(2), accuracy, "%");

    if (accuracy > 90.0)
    {
      Logger::info("SUCCESS: Model learned sentiment rules well.");
    }
    else
    {
      Logger::warning("FAILURE: Accuracy too low.");
    }

    TEST_END
    std::cout << std::endl;
  }
};
