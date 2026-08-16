# Python Bindings for neural-network

This subdirectory contains the source files, project files, and scripts necessary to build and run native Python bindings for the `myoddweb::nn` neural network library using `pybind11`.

---

## Using the API

The Python bindings expose the C++ API in a clean, Pythonic wrapper inside the `neuralnetwork` module.

### Enums

*   `nn.ActivationMethod`: Supported activation function methods.
    *   `Linear`: No activation (identity).
    *   `Sigmoid`: Standard logistic sigmoid function.
    *   `Tanh`: Hyperbolic tangent function.
    *   `Relu`: Rectified Linear Unit.
    *   `LeakyRelu`: Leaky Rectified Linear Unit.
    *   `PRelu`: Parametric Rectified Linear Unit.
    *   `Selu`: Scaled Exponential Linear Unit.
    *   `Swish`: Swish activation function.
    *   `Mish`: Mish activation function.
    *   `Gelu`: Gaussian Error Linear Unit.
    *   `Elu`: Exponential Linear Unit.
    *   `Softmax`: Softmax activation function.
*   `nn.OptimiserType`: Gradient descent optimisers.
    *   `SGD`: Stochastic Gradient Descent.
    *   `Momentum`: Stochastic Gradient Descent with Momentum.
    *   `Nesterov`: Nesterov Accelerated Gradient.
    *   `RMSProp`: Root Mean Squared Propagation.
    *   `Adam`: Adaptive Moment Estimation.
    *   `AdamW`: Adam with decoupled Weight decay.
    *   `AdaGrad`: Adaptive Gradient algorithm.
    *   `AdaDelta`: Extension of AdaGrad that seeks to reduce its aggressive learning rate decay.
    *   `Nadam`: Nesterov-accelerated Adam.
    *   `NadamW`: Nadam with decoupled Weight decay.
    *   `AMSGrad`: Variant of Adam using the maximum of past squared gradients.
    *   `LAMB`: Layer-wise Adaptive Moments optimizer for Batch training.
    *   `Lion`: EvoLved Sign Momentum (Lion) optimizer.
    *   `None_`: Disabled optimiser.
*   `nn.ErrorCalculationType`: Error evaluation functions.
    *   `None_`: Disabled error calculation.
    *   `HuberLoss`: Huber Loss.
    *   `HuberDirectionLoss`: Directional-aware Huber Loss.
    *   `MAE`: Mean Absolute Error.
    *   `MSE`: Mean Squared Error.
    *   `RMSE`: Root Mean Squared Error.
    *   `NRMSE`: Normalized Root Mean Squared Error.
    *   `MAPE`: Mean Absolute Percentage Error.
    *   `SMAPE`: Symmetric Mean Absolute Percentage Error.
    *   `WAPE`: Weighted Absolute Percentage Error.
    *   `DirectionalAccuracy`: Directional accuracy of prediction.
    *   `BCELoss`: Binary Cross Entropy Loss.
    *   `CrossEntropy`: Categorical Cross Entropy Loss.
    *   `LogCosh`: Logarithm of the hyperbolic cosine of the prediction error.
    *   `DirectionalConfidenceScore`: Confidence score of directional movement.
    *   `PredictionCoverage`: Ratio of valid predictions.
*   `nn.LayerArchitecture`: Core layer architectures.
    *   `None_`: Untyped architecture.
    *   `FF`: Feed-Forward (Dense / Fully Connected) layer.
    *   `Elman`: Elman Recurrent Neural Network (RNN) layer.
    *   `Gru`: Gated Recurrent Unit (GRU) layer.
    *   `Lstm`: Long Short-Term Memory (LSTM) layer.
    *   `MultiOutput`: Container for multiple parallel output layers.
    *   `AttentionPool`: Additive (Bahdanau-style) attention pooling over a preceding `Gru`/`Lstm` layer's BPTT window (see "Attention Pooling" below).
*   `nn.LayerRole`: Structural role of a layer.
    *   `Input`: Network input layer.
    *   `Hidden`: Network hidden layer.
    *   `Output`: Network output layer.
    *   `MultiOutput`: Network multi-output layer.
*   `nn.LogLevel`: Logger verbosity levels.
    *   `Trace`, `Debug`, `Info`, `Warning`, `Error`, `Panic`, `None_`.

### Functions and Classes

*   `nn.Logger`: Global logging interface.
    *   `set_level(level)`: Sets the current logging level.
    *   `get_level()`: Returns the current logging level.
    *   `trace(*args)`, `debug(*args)`, `info(*args)`, `warning(*args)`, `error(*args)`, `panic(*args)`: Logs a formatted string message.
*   `nn.Activation`: Activation function configuration.
    *   `Activation(method, alpha, temperature=1.0)`: Constructor.
    *   `activate(val)`: Evaluates the function at value `val`.
    *   `activate_derivative(val, active_val)`: Evaluates the derivative of the function.
    *   `method_to_string()`: Returns the string representation of the activation method.
    *   Properties: `method` (read-only), `alpha` (read-only), `inference_temperature` (read/write).
*   `nn.EvaluationConfig`: Configuration parameters for metrics evaluation.
    *   `EvaluationConfig(neutral_tolerance, confidence_threshold, huber_delta, direction_lambda, use_direction_penalty, cross_entropy_lambda, epsilon)`: Constructor.
    *   Properties: `neutral_tolerance`, `confidence_threshold`, `huber_delta`, `direction_lambda`, `use_direction_penalty`, `cross_entropy_lambda`, `epsilon` (all read-only).
*   `nn.LayerDetails`: Specifications for configuring a hidden layer.
    *   `LayerDetails(architecture, size, activation, dropout, weight_decay, optimiser_type, momentum, use_layer_normalisation=False, attention_hidden_size=0)`: Constructor. `use_layer_normalisation` enables recurrent-state Layer Normalization (see "Layer Normalization" below) and is only valid for `Gru`/`Lstm` architectures. `attention_hidden_size` sets the internal scoring-projection width for `AttentionPool` layers (see "Attention Pooling" below) and must be non-zero exactly when `architecture` is `AttentionPool`.
    *   Properties: `architecture`, `size`, `activation`, `dropout`, `weight_decay`, `optimiser_type`, `momentum`, `use_layer_normalisation`, `attention_hidden_size` (all read-only).
*   `nn.OutputLayerDetails`: Specifications for configuring the output layer.
    *   `OutputLayerDetails(size, activation, error_type, evaluation_config, weight_decay, optimiser_type, momentum)`: Constructor.
    *   Properties: `size`, `activation`, `output_error_calculation_type`, `error_evaluation_config`, `weight_decay`, `optimiser_type`, `momentum` (all read-only).
*   `nn.MultiOutputLayerDetails`: Specifications for configuring multiple output layers.
    *   `MultiOutputLayerDetails(hidden_layers, output_details)`: Constructor.
    *   Properties: `hidden_layers`, `output_details` (all read-only).
*   `nn.StochasticWeightAveragingDetails`: Specifications for configuring Stochastic Weight Averaging (SWA).
    *   `StochasticWeightAveragingDetails(swa_enabled, swa_start_percent, swa_update_percent)`: Constructor.
    *   Properties: `enabled` / `swa_enabled`, `start_percent` / `swa_start_percent`, `update_percent` / `swa_update_percent` (all read-only).
*   `nn.NeuralNetworkHelperMetrics`: Pair of metric values and their evaluation types.
    *   Properties: `error` (float), `error_type` (nn.ErrorCalculationType).
*   `nn.NeuralNetworkHelper`: Tracking helper passed to the progress callback.
    *   Properties: `learning_rate`, `number_of_epoch`, `epoch`, `percent_complete`, `sample_size`.
    *   `calculate_forecast_metric(error_type)`: Calculates forecast metric for the default output layer.
    *   `calculate_forecast_metrics(error_types, in_sample=True)`: Calculates list of forecast metrics for the default output layer.
*   `nn.NeuralNetworkOptions`: Builder for model options.
    *   `NeuralNetworkOptions.create(topology)`: Static builder factory. Returns an options builder instance.
    *   Builder Methods: `with_has_bias`, `with_output_layer_details`, `with_number_of_epoch`, `with_batch_size`, `with_data_is_unique`, `with_progress_callback`, `with_number_of_threads`, `with_learning_rate`, `with_learning_rate_decay_rate`, `with_learning_rate_warmup`, `with_learning_rate_boost_rate`, `with_adaptive_learning_rates`, `with_hidden_layers`, `with_residual_layer_jump`, `with_clip_threshold`, `with_shuffle_training_data`, `with_shuffle_bptt_batches`, `with_bptt_supervise_last_step_only`, `with_enable_bptt`, `with_bptt_max_ticks`, `with_update_training_monitor_percent`, `with_stochastic_weight_averaging(swa_details)` / `with_stochastic_weight_averaging(swa_enabled, swa_start_percent, swa_update_percent)`, `with_final_error_calculation_types`, `with_log_level`, `with_seed(seed)`.
    *   Properties / Methods: `stochastic_weight_averaging()` (returns `StochasticWeightAveragingDetails`), `seed()` (returns `Optional[int]`; `None` unless a seed was set).
    *   `build()`: Finalises and returns the immutable options object.
*   `nn.NeuralNetwork`: Core neural network model.
    *   `NeuralNetwork(options)`: Constructor.
    *   `train(inputs, outputs)`: Runs training on the provided datasets.
    *   `think(inputs)`: Performs prediction/inference. Accepts single or multiple input rows.
    *   `get_topology()`: Returns the list of layer sizes.
    *   `calculate_forecast_metric(...)`, `calculate_forecast_metrics(error_types, in_sample=True)`: Computes model forecast error metrics.
    *   `get_learning_rate()`, `get_temperature()`, `get_inference_temperature()`, `get_percent_complete()`, `has_training_data()`, `options()`.
*   `nn.NeuralNetworkSerializer`: Serialisation and deserialisation utilities.
    *   `save(net, filepath)`: Static method to save a network instance to a JSON file.
    *   `load(filepath)`: Static method to load a network instance from a JSON file.

### Attention Pooling

`nn.LayerArchitecture.AttentionPool` provides additive (Bahdanau-style) attention pooling over the full BPTT window of a preceding `Gru` or `Lstm` hidden layer:
* Must immediately follow a `Gru` or `Lstm` hidden layer.
* `LayerDetails`' `size` must equal the preceding recurrent layer's hidden size (pooling never changes dimensionality).
* `use_layer_normalisation` must be `False`.
* `attention_hidden_size` must be non-zero (sets the internal scoring-projection width).
* Requires `with_enable_bptt(True)`.

### Examples

Standalone python examples are located in the [examples/](examples/) folder.

#### XOR Classification (`examples/xor.py`)

A classic non-linearly separable classification example solving the XOR problem.

**What it does:**
- Constructs a 3-layer Feed-Forward network (`[3, 2, 1]`) with Sigmoid activation.
- Trains the network on the XOR truth table using Stochastic Gradient Descent (SGD).
- Evaluates predicted probabilities against expected binary XOR outputs and verifies convergence.

```python
import neuralnetwork as nn

# Configure topology: 3 inputs (2 data + 1 bias), 2 hidden neurons, 1 output neuron
topology = [3, 2, 1]

# Build options with Sigmoid activation and MSE loss
options = (
    nn.NeuralNetworkOptions.create(topology)
    .with_batch_size(1)
    .with_learning_rate(0.1)
    .with_number_of_epoch(5000)
    .build()
)

net = nn.NeuralNetwork(options)
net.train([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
          [[0.0], [1.0], [1.0], [0.0]])

print("Prediction for [1, 0]:", net.think([1.0, 0.0, 1.0])[0])
```

Run command:
```bash
python python/examples/xor.py
```

#### Multi-Output Layer (`examples/multi_output.py`)

Demonstrates a network with multiple output layers trained concurrently for different learning targets (similar to the C++ `examples/multi_output.h`).

**What it does:**
- Constructs a network with 1 input feature, 2 hidden layers of 8 neurons each, and 2 distinct output heads (`[1, 8, 8, 2]`).
- **Head 1 (Classification)**: Uses Sigmoid activation to classify whether input $x > 0$.
- **Head 2 (Regression)**: Uses Tanh activation to estimate the continuous function $\tanh(x)$.
- Trains on 500 synthetic data points using the NadamW optimiser and tests inference predictions across positive and negative inputs.

```python
import neuralnetwork as nn

# Define parallel output layers: Classification (Sigmoid) + Regression (Tanh)
output_layers = [
    nn.OutputLayerDetails(1, nn.Activation(nn.ActivationMethod.Sigmoid, 1.0), nn.ErrorCalculationType.MSE, nn.EvaluationConfig(), 0.001, nn.OptimiserType.NadamW, 0.99),
    nn.OutputLayerDetails(1, nn.Activation(nn.ActivationMethod.Tanh, 1.0), nn.ErrorCalculationType.MSE, nn.EvaluationConfig(), 0.001, nn.OptimiserType.NadamW, 0.9)
]

options = (
    nn.NeuralNetworkOptions.create([1, 8, 8, 2])
    .with_output_layer_details(output_layers)
    .with_learning_rate(0.01)
    .with_number_of_epoch(1000)
    .build()
)
```

Run command:
```bash
python python/examples/multi_output.py
```

#### General Example (`examples/example.py`)

Illustrates full network configuration, custom Python progress callbacks, model training, inference, and saving/loading model files (`nn.NeuralNetworkSerializer`).

Run command:
```bash
python python/examples/example.py
```

---

## Assumptions & Prerequisites

To build and run the Python bindings, you must have the following installed on your system:

1.  **Python 3.x** (64-bit recommended, version 3.6 or higher).
2.  **C++ Compiler**:
    *   **Windows:** Visual Studio 2022 (with the **Desktop development with C++** workload).
    *   **Linux / macOS:** GCC (g++ version 9 or higher) or Clang (version 10 or higher) supporting C++17.
3.  **pybind11**: Install the pybind11 Python package:
    *   ```bash
        pip install pybind11
        ```

---

## Building the Module

### Building with Visual Studio (Windows)

1.  Open the solution file [neuralnetwork_py.sln](neuralnetwork_py.sln) in Visual Studio 2022.
2.  Configure the environment variable `PYTHON_HOME` on your computer pointing to your Python installation directory (e.g. `C:\Users\<Name>\AppData\Local\Programs\Python\Python312` or `C:\Program Files\Python312`). This enables MSBuild to find Python's headers and library files.
3.  Set the build configuration to **Release** and platform to **x64**.
4.  Build the solution (`Ctrl+Shift+B` or right-click the project -> **Build**).
5.  This generates `neuralnetwork.pyd` inside the `x64/Release/` folder.

### Building with GCC/Clang (Linux / macOS)

Run the following command in a terminal within the `python` subdirectory to compile the pybind11 module:

```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC -I../include \
    $(python3 -m pybind11 --includes) \
    bindings.cpp \
    ../include/neuralnetwork/common/activation.cpp \
    ../include/neuralnetwork/layers/attentionpoollayer.cpp \
    ../include/neuralnetwork/layers/elmanrnnlayer.cpp \
    ../include/neuralnetwork/layers/fflayer.cpp \
    ../include/neuralnetwork/layers/ffoutputlayer.cpp \
    ../include/neuralnetwork/layers/grurnnlayer.cpp \
    ../include/neuralnetwork/layers/lstmlayer.cpp \
    ../include/neuralnetwork/layers/layer.cpp \
    ../include/neuralnetwork/layers/layers.cpp \
    ../include/neuralnetwork/libraries/TinyJSON.cpp \
    ../include/neuralnetwork/neuralnetwork.cpp \
    ../include/neuralnetwork/helpers/neuralnetworkhelper.cpp \
    ../include/neuralnetwork/helpers/neuralnetworkserializer.cpp \
    ../include/neuralnetwork/neuron.cpp \
    -o neuralnetwork$(python3-config --extension-suffix)
```

---

## Using / Running the Compiled Module

Once built, make sure the generated output file (`neuralnetwork.pyd` on Windows, or `neuralnetwork.so` on Unix-like platforms) is in your Python path or in the `python/` directory.

Run any of the example scripts from the repository root:

```bash
python python/examples/xor.py
python python/examples/multi_output.py
python python/examples/example.py
```

---

## Folder Layout

*   `bindings.cpp`: C++ source file defining the `pybind11` wrapper layer, mapping C++ types, enums, and classes into Python.
*   `neuralnetwork_py.vcxproj`: Visual Studio C++ project file configured to build a dynamic library output with a `.pyd` file extension.
*   `neuralnetwork_py.vcxproj.filters`: Project filters mapping for Solution Explorer organization.
*   `neuralnetwork_py.sln`: Main Visual Studio solution.
*   `import_check.py`: A lightweight validation script that verifies the binary module loads, initializes enums, and starts up correctly (used in CI).
*   `examples/`: Folder containing Python example scripts:
    *   `examples/xor.py`: Classic XOR classification example with output validation.
    *   `examples/multi_output.py`: Multi-output example with classification (Sigmoid) and regression (Tanh) heads.
    *   `examples/example.py`: General Python script illustrating options configuration, progress callbacks, training, inference, and serialization.

