# neural-network

## What is it?

This project aims to create a very simple neural network in c++ without any external libraries.

It is not fast, it is not very good ... but it does the work and it shows you how it works.

Look at the code ...

## How to contribute

If you spot anything wrong, please open a new issue ... as I said, I am still learning myself and I am only on my second or third epoch .... (that a NN joke ... I am sorry).

## How to use

### Multiple Hidden layers

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

## Activation methods

The availables activation methods are:

* sigmoid
* tanh
* relu
* leakyRelu
* PRelu

## Data Normalisation

While the classes do not force you to normalise your data ... I strongly suggest you do :)

Normalise the input output between -1 and 1 or 0 and 1

This will save you a lot of headache ...