// neuralnetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "neuralnetwork.h"
#include "neuralnetworkserializer.h"

#include <iomanip>
#include <vector>

NeuralNetwork* create_neural_network()
{
  std::vector<unsigned> topology = {3,2,1};
  return new NeuralNetwork(topology, activation::sigmoid_activation, 0.15);
}

void train_neural_network( NeuralNetwork& nn, unsigned epoch)
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
  nn.train(training_inputs, training_outputs, epoch);
  std::cout << std::endl;

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

int main()
{
  // the file we will be loading from
  const char* file_name = "./xor.nn";
  const unsigned epoch = 10000;

  // assume that it does not ei
  auto nn = NeuralNetworkSerializer::load(file_name);
  if( nullptr == nn )
  {
    // we need to create it
    nn = create_neural_network();

    // train it
    train_neural_network(*nn, epoch);

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

  std::cout << "Error: " << nn->get_error() << std::endl;

  std::cout << "Output After Training:" << std::endl;
  std::cout << std::fixed << std::setprecision(10);

  // or we can train with a single inut
  // we know that the output only has one value.
  auto t1 = nn->think({ 0, 0, 1 });
  std::cout << t1.front() << std::endl;//  should be close to 0

  // or we can train with a single inut
  // we know that the output only has one value.
  auto t2 = nn->think({ 1, 1, 1 });
  std::cout << t2.front() << std::endl; //  should be close to 1

  delete nn;
  return 0;
}
