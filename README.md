# neural-network

## What is it?

This project aims to create a very simple neural network in c++

## How to use

### Multiple Hiden layers

```c++
#include <iostream>
#include "neuralnetwork_layer.h"
...

int main()
{
  std::vector<std::vector<double>> training_inputs = {
    {0, 0, 1},
    {1, 1, 1},
    {1, 0, 1},
    {0, 1, 1}
  };

  // output dataset
  std::vector<std::vector<double>> training_outputs = {
      {0}, {1}, {1}, {0}
  };

  auto* nnl = new NeuralNetworkLayer({ 3,4,1 }, activation::sigmoid_activation);
  nnl->train(training_inputs, training_outputs, 10000);

  std::vector<std::vector<double>> inputs = {
      {0, 0, 1},
      {1, 1, 1},
      {1, 0, 1},
      {0, 1, 1}
  };

  std::cout << std::endl;

  auto outputs = nnl->think(inputs);
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

  std::cout << "Output After Training:" << std::endl;
  std::cout << std::fixed << std::setprecision(10);
  std::cout << nnl->think({ 0, 0, 1 })[0] << std::endl;
  std::cout << nnl->think({ 1, 1, 1 })[0] << std::endl;

  delete nnl;

  return 0;
}
```

### Single Network

```c++
#include <iostream>
#include "neuralnetwork.h"
...

int main()
{
  std::vector<std::vector<double>> training_inputs = {
    {0, 0, 1},
    {1, 1, 1},
    {1, 0, 1},
    {0, 1, 1}
  };

  // output dataset
  std::vector<std::vector<double>> training_outputs = {
      {0}, {1}, {1}, {0}
  };

  auto* nn = new NeuralNetwork(3, activation::sigmoid_activation);
  nn->train(training_inputs, training_outputs, 10000);

  std::vector<std::vector<double>> inputs = {
      {0, 0, 1},
      {1, 1, 1},
      {1, 0, 1},
      {0, 1, 1}
  };

  std::cout << std::endl;

  auto outputs = nn->think(inputs);
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

  std::cout << "Output After Training:" << std::endl;
  std::cout << std::fixed << std::setprecision(10);
  std::cout << nn->think({ 0, 0, 1 }) << std::endl;
  std::cout << nn->think({ 1, 1, 1 }) << std::endl;

  delete nn;

  return 0;
}
```
