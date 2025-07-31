# neural-network

## What is it?

This project aims to create a very simple neural network in c++ without any external libraries.

It is not fast, it is not very good ... but it does the work and it shows you how it works.

Look at the code ...

## How to contribute

If you spot anything wrong, please open a new issue ... as I said, I am still learning myself and I am only on my second or third epoch .... (that a NN joke ... I am sorry).

## How to use

### Activation methods.

* linear
* sigmoid
* tanh
* relu
* leakyRelu
* PRelu
* selu
* swish
* gelu
* mish
* elu

### Optimiser

* None
* SGD
* Adam
* AdamW
* Nadam
* NadamW

#### Not supported (yet)

* Momentum
* Nesterov
* RMSProp
* AdaGrad
* AdaDelta
* AMSGrad
* LAMB
* Lion

### Residual layer

You can use [residual layers](https://en.wikipedia.org/wiki/Residual_neural_network) during training, for that you simply need to define a jump

For example if you want to "jump" ahead by 3 layers then simply define it in your configuration.

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      ...
      .with_residual_layer_jump(3)
      ...
      .build();
```

Of course, the default is 0, (no jump)

### norm-based gradient clipping

The Neural Network uses [norm-based gradient clipping](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) with a default value of 1.0

The default value is betweem 0.5 and 2.0 depending on the number of hidden network.

For very deep networks it might be best to set a value of ~5.0

```cpp
    std::vector<unsigned> topology = {2, 8, 8, 8, 8, 1};

    auto options = NeuralNetworkOptions::create(topology)
      ...
      .with_clip_threshold(2.0)
      ...
      .build();
```

Effectively all the gradients will be "clipped" to within the threshold to prevent exploding gradients.

### Dropout (Dilution)

You can define one of more hidden layer to have a [dropout (or dilution)](https://en.wikipedia.org/wiki/Dilution_(neural_networks)) rate.

The rate must be between 0.0 and 1.0 and you must have the exact number of dropout defined for each hidden layer, (or none).

```cpp
    std::vector<unsigned> topology = {2, 8, 8, 8, 8, 1};
    std::vector<double> dropout = { 0.0, 0.0, 0.2, 0.0 };

    auto options = NeuralNetworkOptions::create(topology)
      ...
      .with_dropout(dropout)
      ...
      .build();
```

The default is to have no dilution, 0.0, for all the hidden layers.

### Examples

#### XOR example with multiple hidden layers

See the example in the `./example/xor.h` file.

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

  // just log information
  auto logger = Logger(Logger::LogLevel::Information);

  // the topology we create this NN with is
  // 3 input network, a hidden layer with 4 neuron and 1 output layer.
  auto options = NeuralNetworkOptions::create({ 3,4,1 })
    .with_batch_size(batch_size)
    .with_hidden_activation_method(activation::method::sigmoid)
    .with_output_activation_method(activation::method::sigmoid)
    .with_logger(logger)
    .with_learning_rate(0.1)
    .with_number_of_epoch(10000).build();

  auto* nnl = new NeuralNetwork(options);
  nnl->train(training_inputs, training_outputs);

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

#### Three bit parity

See the example in the `./example/threebitparity.h` file.

Similar example to the xor sample, but it uses batches.


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
  // just log information
  auto logger = Logger(Logger::LogLevel::Information);

  // create the NN
  auto options = NeuralNetworkOptions::create({ 3,2,1 })
  .with_batch_size(batch_size)
  .with_hidden_activation_method(activation::method::sigmoid)
  .with_output_activation_method(activation::method::sigmoid)
  .with_logger(logger)
  .with_learning_rate(0.1)
  .with_number_of_epoch(10000).build();
  auto nn = new NeuralNetwork(options);

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
  const std::vector<std::vector<double>>& training_outputs
  );
```

* training_inputs: this is a vector of one or more `doubles` that will be used for each epoch as input data.
* training_output: this is a vector of one or more `doubles` that will be used for each epoch as expected output.

##### Trainning callback method

You can use a tainning callback method to see how the NN is training ...

The helper method is to help you control how much data you want to pull while training.

Training is very CPU bound so it helps to limit what is being calculated per epoch.

You can use the `NeuralNetworkHelper& nn` to calculate the metrics.

```c++

// how far along are we?
auto current_epoch_number = nn.epoch();
auto total_number_of_epoch = nn.number_of_epoch();

auto metrics = nn.calculate_forecast_metrics{ NeuralNetworkOptions::ErrorCalculation::rmse, NeuralNetworkOptions::ForecastAccuracy::mape});

// use the rmse error and mape

metrics = nn.calculate_forecast_metrics({NeuralNetworkOptions::ErrorCalculation::huber_loss, NeuralNetworkOptions::ForecastAccuracy::smape});

// use the hubber loss error and smape
```

**NB:** Remember that calling those methods will impact training, so limit the number of calls to something once every 10 epoch, (for example).


```c++
#include <iostream>
#include "neuralnetwork.h"
...

void show_progress_bar(NeuralNetworkHelper& nn)
{
  // you can use the values to display pretty things.
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

  auto logger = Logger(Logger::LogLevel::Information);
  auto options = NeuralNetworkOptions::create({1, 4, 1})
    .with_batch_size(batch_size)
    .with_hidden_activation_method(activation::method::sigmoid)
    .with_output_activation_method(activation::method::sigmoid)
    .with_logger(logger)
    .with_learning_rate(0.1)
    .with_number_of_epoch(10000)
    .with_progress_callback(show_progress_bar).build();

  auto* nn = new NeuralNetwork(options);
  nn->train(training_inputs, training_outputs);

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

## Neural Network Options

You can create your Neural network with one or more of the following options, the only required value is the topology.
But some values don't really make sense to be left as default, (learning rate and epoch for example).

```c++
auto options = NeuralNetworkOptions::create({1, 4, 1}).build();
...

```

* hidden_activation_method[=sigmoid]
* output_activation_method[=sigmoid]
* learning_rate[=0.15]: The starting learning rate.
* number_of_epoch[=1000]: The number of epoch we want to train for. This value is really specific to your data.
* batch_size[=1]: The default number of batches we want to split the epochs in.
* data_is_unique[=true]: By default we assume that the input data is unique and cannot be split for in-batch validation and final error validation.
* progress_callback[=null]: The callback.
* logger[=none]: Your logger.
* number_of_threads[=0]: The number of threads to use durring batch training, (0 means we will use the number of CPU -1)
* learning_rate_decay_rate[=0.0]: durring training we will slowly decay the learning rate. The default is no change, and 0.5 would mean a 50% drop over the training. The number must be between 0 and 1
* adaptive_learning_rate[=false]: If we want to use adaptive learning or not, (help prevent explosion and so on).
* optimiser_type[=SGD]: The optimiser we will use during training.
* learning_rate_restart_rate[=1%] and learning_rate_restart_boost[=1]: Every 'x'% we will boost the learning rate by a factor of 'y', (the default is no boost as the boost is 1 ... and 1*LR=LR)
* residual_layer_jump[=-1] if you are using residual layer connections, this is the jump back value.
* dropout[={}]: you can set a dropout rate for one or more of your hidden layers.
* clip_threshold[=1.0]: if the gradient goes outside this value then it is clipped.

Remember to call `.build()` to create your option as it does error checking.

## Data Normalisation

While the classes do not force you to normalise your data ... I strongly suggest you do :)

Normalise the input output between -1 and 1 or 0 and 1 and make sure that you use the proper activation method for your data.

This will save you a lot of headache ...

## Error Functions

After training you can get the calculated error as well as the mean absolute percentage error


```c++
...
auto logger = Logger(Logger::LogLevel::Information);
auto options = NeuralNetworkOptions::create({1, 4, 1})
  .with_batch_size(batch_size)
  .with_hidden_activation_method(activation::method::sigmoid)
  .with_output_activation_method(activation::method::sigmoid)
  .with_logger(logger)
  .with_learning_rate(0.1)
  .with_number_of_epoch(10000).build();

auto* nn = new NeuralNetwork(options);

...
auto error_types = {NeuralNetworkOptions::ErrorCalculation::huber_loss, 
 NeuralNetworkOptions::ErrorCalculation::rmse };
auto errors = nn->calculate_forecast_metrics( error_types);

// errors[0] = huber_loss
// errors[1] = rmse
...
```

### Error Calculations

  * NeuralNetworkOptions::ErrorCalculation::none
  * NeuralNetworkOptions::ErrorCalculation::huber_loss
  * NeuralNetworkOptions::ErrorCalculation::mae
  * NeuralNetworkOptions::ErrorCalculation::mse
  * NeuralNetworkOptions::ErrorCalculation::rmse
  * NeuralNetworkOptions::ErrorCalculation::mape
  * NeuralNetworkOptions::ErrorCalculation::smape