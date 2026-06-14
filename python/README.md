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
    *   `LayerDetails(architecture, size, activation, dropout, weight_decay, optimiser_type, momentum)`: Constructor.
    *   Properties: `architecture`, `size`, `activation`, `dropout`, `weight_decay`, `optimiser_type`, `momentum` (all read-only).
*   `nn.OutputLayerDetails`: Specifications for configuring the output layer.
    *   `OutputLayerDetails(size, activation, error_type, evaluation_config, weight_decay, optimiser_type, momentum)`: Constructor.
    *   Properties: `size`, `activation`, `output_error_calculation_type`, `error_evaluation_config`, `weight_decay`, `optimiser_type`, `momentum` (all read-only).
*   `nn.MultiOutputLayerDetails`: Specifications for configuring multiple output layers.
    *   `MultiOutputLayerDetails(hidden_layers, output_details)`: Constructor.
    *   Properties: `hidden_layers`, `output_details` (all read-only).
*   `nn.NeuralNetworkHelperMetrics`: Pair of metric values and their evaluation types.
    *   Properties: `error` (float), `error_type` (nn.ErrorCalculationType).
*   `nn.NeuralNetworkHelper`: Tracking helper passed to the progress callback.
    *   Properties: `learning_rate`, `number_of_epoch`, `epoch`, `percent_complete`, `sample_size`.
    *   `calculate_forecast_metric(error_type)`: Calculates forecast metric for the default output layer.
    *   `calculate_forecast_metrics(error_types)`: Calculates list of forecast metrics for the default output layer.
*   `nn.NeuralNetworkOptions`: Builder for model options.
    *   `NeuralNetworkOptions.create(topology)`: Static builder factory. Returns an options builder instance.
    *   Builder Methods: `with_has_bias`, `with_output_layer_details`, `with_number_of_epoch`, `with_batch_size`, `with_data_is_unique`, `with_progress_callback`, `with_number_of_threads`, `with_learning_rate`, `with_learning_rate_decay_rate`, `with_learning_rate_warmup`, `with_learning_rate_boost_rate`, `with_adaptive_learning_rates`, `with_hidden_layers`, `with_residual_layer_jump`, `with_clip_threshold`, `with_shuffle_training_data`, `with_shuffle_bptt_batches`, `with_enable_bptt`, `with_bptt_max_ticks`, `with_update_training_monitor_percent`, `with_final_error_calculation_types`, `with_log_level`.
    *   `build()`: Finalises and returns the immutable options object.
*   `nn.NeuralNetwork`: Core neural network model.
    *   `NeuralNetwork(options)`: Constructor.
    *   `train(inputs, outputs)`: Runs training on the provided datasets.
    *   `think(inputs)`: Performs prediction/inference. Accepts single or multiple input rows.
    *   `get_topology()`: Returns the list of layer sizes.
    *   `calculate_forecast_metric(...)`, `calculate_forecast_metrics(...)`: Computes model forecast error metrics.
    *   `get_learning_rate()`, `get_temperature()`, `get_inference_temperature()`, `get_percent_complete()`, `has_training_data()`, `options()`.
*   `nn.NeuralNetworkSerializer`: Serialisation and deserialisation utilities.
    *   `save(net, filepath)`: Static method to save a network instance to a JSON file.
    *   `load(filepath)`: Static method to load a network instance from a JSON file.

### Example

Below is a complete example showing how to configure, train, evaluate, and save/load a neural network to solve the XOR classification problem.

```python
import os
import sys
import neuralnetwork as nn

# 1. Configure Topology (2 inputs, 2 hidden neurons, 1 output)
topology = [2, 2, 1]

# 2. Configure Hidden Layer (Feed-Forward, Sigmoid, SGD)
hidden_activation = nn.Activation(nn.ActivationMethod.Sigmoid, 1.0)
hidden_layers = [
    nn.LayerDetails(
        nn.LayerArchitecture.FF, 
        2, 
        hidden_activation, 
        0.0, 0.0, 
        nn.OptimiserType.SGD, 
        0.99
    )
]

# 3. Configure Output Layer (Sigmoid, MSE, SGD)
out_activation = nn.Activation(nn.ActivationMethod.Sigmoid, 1.0)
out_layer = nn.OutputLayerDetails(
    topology[-1], 
    out_activation, 
    nn.ErrorCalculationType.MSE,
    nn.EvaluationConfig(),
    0.0, 
    nn.OptimiserType.SGD, 
    0.99
)

# 4. Optional Progress Callback (stops training if callback returns False)
def on_progress(helper):
    if helper.epoch % 100 == 0:
        nn.Logger.info(f"Epoch: {helper.epoch:4d} | Percent Complete: {helper.percent_complete * 100:5.1f}%")
    return True

# 5. Build Options
options = nn.NeuralNetworkOptions.create(topology) \
    .with_batch_size(1) \
    .with_hidden_layers(hidden_layers) \
    .with_output_layer_details(out_layer) \
    .with_learning_rate(0.1) \
    .with_number_of_epoch(5000) \
    .with_log_level(nn.LogLevel.Info) \
    .with_progress_callback(on_progress) \
    .build()

# 6. Instantiate and Train the Network
net = nn.NeuralNetwork(options)

training_inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]
training_outputs = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]

nn.Logger.info("Starting training...")
net.train(training_inputs, training_outputs)

# 7. Run Inference
nn.Logger.info("Evaluating predictions:")
for inputs, expected in zip(training_inputs, training_outputs):
    outputs = net.think(inputs)
    nn.Logger.info(f"Input: {inputs} | Expected: {expected[0]} | Predicted: {outputs[0]:.4f}")

# 8. Save and Reload Model
model_path = "xor_model.json"
nn.Logger.info(f"Saving model to {model_path}...")
nn.NeuralNetworkSerializer.save(net, model_path)

nn.Logger.info(f"Loading model from {model_path}...")
loaded_net = nn.NeuralNetworkSerializer.load(model_path)
loaded_output = loaded_net.think([1.0, 0.0])
nn.Logger.info(f"Loaded model prediction on [1.0, 0.0]: {loaded_output[0]:.4f}")
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

Once built, make sure the generated output file (`neuralnetwork.pyd` on Windows, or `neuralnetwork.so` on Unix-like platforms) is in your Python path or in the same directory as your Python script.

Run the example script:

```bash
python example.py
```

This will run the full XOR classification training pipeline and verify that the Python bindings are functioning correctly.

---

## Folder Layout

*   `bindings.cpp`: C++ source file defining the `pybind11` wrapper layer, mapping C++ types, enums, and classes into Python.
*   `neuralnetwork_py.vcxproj`: Visual Studio C++ project file configured to build a dynamic library output with a `.pyd` file extension.
*   `neuralnetwork_py.vcxproj.filters`: Project filters mapping for Solution Explorer organization.
*   `neuralnetwork_py.sln`: Main Visual Studio solution.
*   `example.py`: Python script illustrating options configuration, model instantiation, callback monitoring, training, inference, and serialization.
*   `import_check.py`: A lightweight validation script that verifies the binary module loads, initializes enums, and starts up correctly (used in CI).
