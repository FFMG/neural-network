#pragma once
#include "../neuralnetwork.h"
#include "../neuralnetworkoptions.h"
#include "helper.h"
#include <iostream>
#include <vector>

inline void lstm_test()
{
  // 1. Define topology: 1 input, 10 LSTM hidden, 1 output
  std::vector<unsigned> topology = { 4, 10, 1 };

  // 2. Configure options
  auto options = NeuralNetworkOptions::create(topology)
    .with_learning_rate(0.01)
    .with_number_of_epoch(500)
    .with_batch_size(1)
    .with_enable_bptt(true)
    .with_bptt_max_ticks(5); // Sequence length

  // Set the hidden layer to LSTM
  std::vector<LayerDetails> hidden_layers;
  hidden_layers.emplace_back(
    LayerDetails::LayerType::Lstm, 
    10, 
    activation(activation::method::tanh, 0.0), 
    0.0, // dropout
    0.01, // weight decay
    OptimiserType::Adam, 
    0.9
  );
  options.with_hidden_layers(hidden_layers);

  NeuralNetwork nn(options);

  // 3. Prepare training data (Simple sequence: x_t = t/10, y_t = (t+1)/10)
  // Sequence: [0.1], [0.2], [0.3], [0.4] -> Target: [0.5]
  std::vector<double> sequence = { 0.1, 0.2, 0.3, 0.4 };
  std::vector<std::vector<double>> inputs = { sequence };
  std::vector<std::vector<double>> targets = { { 0.5 } };

  // 4. Train
  std::cout << "Training LSTM on simple sequence..." << std::endl;
  nn.train(inputs, targets);

  // 5. Test
  std::vector<double> test_input = { 0.1, 0.2, 0.3, 0.4 };
  auto prediction = nn.think(test_input);

  std::cout << "Prediction for next value after [0.1, 0.2, 0.3, 0.4]: " << prediction[0] << " (Expected ~0.5)" << std::endl;

  // 6. Test Serialization
  std::cout << "Testing serialization..." << std::endl;
  std::string model_path = "lstm_model.nn";
  // save it
  NeuralNetworkSerializer::save(nn, model_path);

  auto loaded_nn = NeuralNetworkSerializer::load(model_path);
  if (loaded_nn != nullptr)
  {
    auto loaded_prediction = loaded_nn->think(test_input);
    std::cout << "Loaded model prediction: " << loaded_prediction[0] << std::endl;
    if (std::abs(prediction[0] - loaded_prediction[0]) < 1e-9)
    {
      std::cout << "SUCCESS: Serialization verified. Predictions match." << std::endl;
    }
    else
    {
      std::cout << "FAILURE: Serialization failed. Predictions differ!" << std::endl;
    }
    delete loaded_nn;
  }
  else
  {
    std::cout << "FAILURE: Could not load the model!" << std::endl;
  }
}
