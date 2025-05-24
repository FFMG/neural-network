# neural-network

## What is it?

This project aims to create a very simple neural network in c++ without any external libraries.

It is not fast, it is not very good ... but it does the work and it shows you how it works.

Look at the code ...

## How to contribute

If you spot anything wrong, please open a new issue ... as I said, I am still learning myself and I am only on my second or third epoch .... (that a NN joke ... I am sorry).

## How to use

### XOR example with multiple hidden layers

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

  // the topology we create this NN with is
  // 3 input network, a hidden layer with 4 neuron and 1 output layer.
  auto* nnl = new NeuralNetwork({ 3,4,1 }, activation::sigmoid_activation);
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

### Save/Load the Neural Network

You might want to save your Neural Network for later use ...

You can use `NeuralNetworkSerializer::save( ... )` to save a trained network and `NeuralNetworkSerializer::load(...)` to reuse it.

```c++
#include <iostream>
#include "neuralnetwork.h"
#include "neuralnetworkserializer.h"
...

int main()
{
  // create the NN
  std::vector<unsigned> topology = {3,2,1};
  auto nn = new NeuralNetwork(topology, activation::sigmoid_activation);

  // train it 
  ...

  // save it
  NeuralNetworkSerializer::save(*nn, "myfile.nn");

  // then you can load it
  auto nn_saved = NeuralNetworkSerializer::load("myfile.nn");

  //  ... use it ...

  delete nn_saved;

  delete nn;
  return 0;
}
```

#### Trainning function variables

```c++
train(
  const std::vector<std::vector<double>>& training_inputs, 
  const std::vector<std::vector<double>>& training_outputs, 
  int number_of_epoch, 
  int batch_size = -1, 
  bool data_is_unique = true, 
  const std::function<bool(int, NeuralNetwork&)>& progress_callback = nullptr
  );
```

* training_inputs: this is a vector of one or more `doubles` that will be used for each epoch as input data.
* training_output: this is a vector of one or more `doubles` that will be used for each epoch as expected output.
* number_of_epoch: number of epoch we will be training for, the more epoch the better the learning, (unless your data is garbage)
* data_is_unique = true: In some cases all the input/ouput data is unique and must be used for training, if this is set to false, then some random data will *not* be used for training but will rather be used for error checking, (how well the model is trained).
* batch_size=-1: if -1 then all the data will be used per epoch. otherwise muliple threads will be used for training.
* progress_callback=null_ptr: the callback function, so you can tell how far the training is (see below).

##### Trainning callback method

You can use a tainning callback method to see how the NN is training ...

Simply pass a function pointer to the train method and it will be called after each percent change.
The value is between 0% and 100%

0% and 100% are always called.

```c++
#include <iostream>
#include "neuralnetwork.h"
...

void show_progress_bar(int progress, double error)
{
  int barWidth = 50;
  int pos = barWidth * (progress / 100.0);

  std::cout << "[";
  for (int i = 0; i < barWidth; ++i) 
  {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "] " << progress << " %(error:" << error << ")   \r";
  std::cout.flush();
}

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

  auto* nn = new NeuralNetwork({1, 4, 1}, activation::sigmoid_activation);
  nn->train(training_inputs, training_outputs, 10000, -1, show_progress_bar);

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
