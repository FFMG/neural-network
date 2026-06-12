# neural-network

## What is it?

This is a lightweight Feedforward and Recurrent Neural Network library written in modern C++ with a primary goal: to be an educational tool. It is built entirely from scratch with zero external dependencies (except for optional charting), making it easy to compile, run, and understand.

While not focused on high performance, it provides a clean implementation of the core mechanics of training and inference, including advanced features like Backpropagation Through Time (BPTT), AdamW/NadamW optimizers, and post-training temperature calibration.

## How to use

### Namespace

All classes, structures, and functions of the core neural network library are wrapped in the `myoddweb::nn` namespace. 

To use the library, you can import the namespace:

```cpp
using namespace myoddweb::nn;
```

Or reference the types explicitly:

```cpp
myoddweb::nn::NeuralNetworkOptions options = ...
myoddweb::nn::NeuralNetwork nn(options);
```

### Activation methods

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
* softmax

### Optimizers

* None
* SGD
* Adam
* AdamW
* Nadam
* NadamW
* Adagrad
* RMSProp

#### Not supported (yet)

* Nesterov
* AdaDelta
* AMSGrad
* LAMB
* Lion

## Options

The following sections describe the various configuration options available when building a network using `NeuralNetworkOptions`.

### Hidden Layers

The hidden layer configuration allows you to define the architecture of your network's trunk.

* **Layer type:** 
  * `FF`: Standard feed-forward layer.
  * `Elman`: Simple recurrent layer.
  * `Gru`: Gated recurrent unit layer.
  * `Lstm`: Long Short-Term Memory layer.
* **Layer size:** Number of neurons in the hidden layer.
* **Activation:** The activation object (method, alpha, and temperature).
* **Weight Decay:** Regularization strength.
* **Dropout:** Percentage of neurons to randomly drop during training (0.0 to 1.0).
* **Optimiser:** Each layer can optionally have its own optimizer configuration.

```cpp
    std::vector<unsigned> topology = {2, 8, 8, 8, 8, 1};
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Lstm, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95),
      LayerDetails(Layer::Architecture::Lstm, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95),
      LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.01), 0.2, 0.05, OptimiserType::AdamW, 0.95),
      LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95),
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_clip_threshold(2.0)
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(60)
      .build();
```

### Multi Output Layers (Branched)

Multi Output Layers allow the network to split from a central trunk into multiple independent paths (branches), each with its own hidden layers and output configuration.

```cpp
    // Trunk topology: 3 inputs, 4 hidden and 5 total outputs (2 + 3)
    std::vector<unsigned> topology = { 3, 4, 5 };
    
    std::vector<MultiOutputLayerDetails> multi_output_layer_details;

    // Branch 1: Shallow path, 2 outputs
    MultiOutputLayerDetails b1
    (
      { LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::tanh, 0.01), 0.0, 0.01, OptimiserType::NadamW, 0.95) },
      OutputLayerDetails(2, activation(activation::method::tanh, 0.01), ErrorCalculation::type::mse, EvaluationConfig(), 0.0, OptimiserType::NadamW, 0.95)
    );
    multi_output_layer_details.push_back(b1);

    // Branch 2: Deeper path, 3 outputs (Softmax)
    MultiOutputLayerDetails b2
    (
      {
        LayerDetails(Layer::Architecture::FF, 16, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::NadamW, 0.95),
        LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::NadamW, 0.95)
      },
      OutputLayerDetails(3, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::NadamW, 0.95)
    );
    multi_output_layer_details.push_back(b2);

    auto options = NeuralNetworkOptions::create(topology)
      .with_hidden_layers({ LayerDetails(Layer::Architecture::Gru, 4, activation(activation::method::tanh, 0.01)) })
      .with_output_layer_details(multi_output_layer_details)
      .build();
```

### Residual Layers

You can use residual layers to "jump" connections across layers:

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_residual_layer_jump(2)
      .build();
```

### Gradient Clipping

Norm-based gradient clipping is enabled by default to prevent exploding gradients, especially in RNNs:

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_clip_threshold(1.5)
      .build();
```

### Data Shuffling and BPTT

When training recurrent networks (RNN, GRU, LSTM), the order of samples is critical for learning temporal dependencies. The library provides two levels of shuffling:

*   **`shuffle-training-data` (Global Shuffling):** If set to `true`, the raw input samples are randomized *before* sequences are formed. 
    *   **WARNING:** This should be set to `false` when using recurrent layers, as it destroys the chronological order of the data, making it impossible for the network to learn time-based patterns.
*   **`shuffle-bptt-batches` (Sequence Shuffling):** If set to `true`, the library first creates contiguous "blocks" of data (of size `bptt_max_ticks`) where the internal chronological order is preserved. It then shuffles the order of these *blocks*.
    *   **RECOMMENDED:** This is the preferred way to shuffle recurrent data. It ensures the GRU/LSTM sees valid timelines within each batch while preventing the model from over-fitting to the global sequence of the dataset.

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_shuffle_training_data(false) // Keep chronological for RNNs
      .with_shuffle_bptt_batches(true)  // Shuffle blocks for better generalization
      .with_enable_bptt(true)
      .with_bptt_max_ticks(24)
      .build();
```

### Learning Rate Strategies

The library supports various strategies to manage learning rate dynamics:

*   **Warmup:** Linearly (or geometrically) increases the rate from a starting value to the target rate over a percentage of the total epochs.
*   **Exponential Decay:** Reduces the learning rate by a fixed decay factor after each epoch.
*   **Smooth Cosine Boosts (Restarts):** Periodically boosts the learning rate using a smooth cosine staircase to help the model escape local minima.
*   **Adaptive Learning Rate:** Dynamically adjusts the learning rate based on recent error trends. It detects states like `Plateauing`, `Improving`, or `Exploding` and adjusts the rate accordingly.

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_learning_rate(0.001)
      .with_learning_rate_warmup(0.0001, 0.05) // Start at 0.0001, reach target at 5% of training
      .with_learning_rate_decay_rate(0.985)    // Decay factor applied per epoch
      .with_learning_rate_boost_rate(0.2, 0.1) // Boost by 10% every 20% of training epochs
      .with_adaptive_learning_rates(true)      // Enable dynamic error-based adjustment
      .build();
```

### Dropout

Individual layers can have dropout applied via `LayerDetails`. During training, neurons are randomly deactivated according to the dropout rate, and the remaining activations are scaled by `1 / (1 - rate)` to maintain the expected sum. Dropout is automatically disabled during inference (`think`).

```cpp
    LayerDetails hl(Layer::Architecture::FF, 64, activation(activation::method::relu, 0.01), 0.25); // 25% dropout
```

### General Training Options

These options control the overall execution of the training process:

*   **`number_of_epoch`:** Total number of training iterations over the dataset.
*   **`batch_size`:** Number of samples processed before internal gradient updates are applied.
*   **`number_of_threads`:** Controls multi-threaded execution for GEMM and layer operations.
*   **`progress_callback`:** A lambda or function called after each epoch to monitor error metrics and progress.
*   **`has_bias`:** Global toggle to enable or disable bias neurons for all layers.

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_number_of_epoch(5000)
      .with_batch_size(32)
      .with_number_of_threads(8)
      .with_has_bias(true)
      .with_progress_callback([](NeuralNetworkHelper& helper) {
          Logger::info("Epoch: ", helper.epoch(), " Error: ", helper.error());
          return true; // Return false to stop training early
      })
      .build();
```

### Inference Temperature Calibration

For classification tasks using Softmax, the network automatically optimizes the inference temperature ($T$) post-training using a calibration set to ensure well-calibrated probability outputs.

## Examples

### XOR

```cpp
  auto options = NeuralNetworkOptions::create({ 3, 4, 1 })
    .with_output_layer_details(1, activation(activation::method::sigmoid, 0.1), ErrorCalculation::type::mse, OptimiserType::AdamW, 0.95)
    .with_learning_rate(0.01)
    .with_number_of_epoch(1000)
    .build();

  NeuralNetwork nn(options);
  nn.train(training_inputs, training_outputs);
  auto output = nn.think({0, 0, 1});
```

### Persistence

```cpp
  NeuralNetworkSerializer::save(nn, "model.nn");
  auto loaded_nn = NeuralNetworkSerializer::load("model.nn");
```

## Error Calculations

* `huber_loss`
* `huber_direction_loss`
* `mae`
* `mse`
* `rmse`
* `directional_accuracy`
* `cross_entropy`
* `bce_loss`
* `directional_confidence_score`
* `prediction_coverage`

## Performance Optimization (SIMD)

To achieve high throughput during training and inference, this library leverages **Advanced Vector Extensions 2 (AVX2)** intrinsics for core mathematical operations (GEMM, dot products, and optimizer updates).

To enable these optimizations, ensure your compiler is configured to target the AVX2 instruction set:

*   **MSVC (Visual Studio):** Set `Enable Enhanced Instruction Set` to `Advanced Vector Extensions 2 (/arch:AVX2)` in the project properties.
*   **GCC / Clang:** Use the `-mavx2 -mfma` flags during compilation.

For more information on AVX2, see the [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) or [Wikipedia](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions).

## Repository Layout

*   `\include\neuralnetwork\`: The stand-alone core neural network library (including `/layers/`, `/helpers/`, and `/common/` subdirectories).
*   `\examples\`: Standalone example implementations, runner (`main.cpp`), and the main Visual Studio solution (`neuralnetwork.sln`).
*   `\tests\`: Comprehensive unit test suite.

## Building and Running

1.  Open `examples/neuralnetwork.sln` in Visual Studio 2022.
2.  Select `neuralnetwork` (to run the examples) or `neuralnetwork_tests` (to run unit tests) as the startup project.
3.  Build and run using the IDE.

## Technical Stack

* **Language:** C++17/C++20
* **Build Tool:** Visual Studio 2022
* **Dependencies:** Zero external dependencies for core logic.
