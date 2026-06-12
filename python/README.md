# Python Bindings for neural-network

This subdirectory contains the Visual Studio 2022 C++ project and code necessary to build and run native Python bindings for the `myoddweb::nn` neural network library using `pybind11`.

---

## Folder Layout

*   `bindings.cpp`: C++ source file defining the `pybind11` wrapper layer, mapping C++ types, enums, and classes into Python.
*   `neuralnetwork_py.vcxproj`: Visual Studio C++ project file configured to build a dynamic library output with a `.pyd` file extension.
*   `neuralnetwork_py.vcxproj.filters`: Project filters mapping for Solution Explorer organization.
*   `neuralnetwork_py.sln`: Main Visual Studio solution.
*   `example.py`: Python script illustrating options configuration, model instantiation, callback monitoring, training, inference, and serialization.
*   `import_check.py`: A lightweight validation script that verifies the binary module loads, initializes enums, and starts up correctly (used in CI).

---

## Build Prerequisites

1.  **Visual Studio 2022** with the **Desktop development with C++** workload installed.
2.  **Python 3.x** (64-bit version recommended, matching the target build platform).
3.  **pybind11**: Install the header-only package via `pip`:
    ```bash
    pip install pybind11
    ```
4.  **Environment Setup**:
    To allow Visual Studio to locate Python headers (`python.h`) and libraries (`python3x.lib`), you should configure the `PYTHON_HOME` environment variable on your computer:
    *   Set `PYTHON_HOME` to your Python installation folder (e.g. `C:\Users\<Name>\AppData\Local\Programs\Python\Python312` or `C:\Program Files\Python312`).
    *   *Alternatively*, if you have Visual Studio's **Python development** workload installed, MSBuild will automatically search for the default Python paths.

---

## Building the Module

1.  Open the solution file [neuralnetwork_py.sln](file:///H:/projects/github/neural-network2/python/neuralnetwork_py.sln) in Visual Studio 2022.
2.  Set the solution build configuration to:
    *   **Configuration:** `Release`
    *   **Platform:** `x64`
3.  Build the solution (`Ctrl+Shift+B` or right-click the project -> **Build**).
4.  This compiles the library and generates `neuralnetwork.pyd` inside the `python/x64/Release/` folder.

---

## Running the Python Example

Once the `.pyd` module is built, you can run the example script:

```bash
python example.py
```

This script trains a neural network on the XOR classification task and validates predictions.

---

## API Usage Reference

The Python bindings expose the C++ API in a clean, Pythonic wrapper inside the `neuralnetwork` module:

### Enums
*   `nn.ActivationMethod`: `Linear`, `Sigmoid`, `Tanh`, `Relu`, `LeakyRelu`, `PRelu`, `Selu`, `Swish`, `Mish`, `Gelu`, `Elu`, `Softmax`.
*   `nn.OptimiserType`: `SGD`, `Momentum`, `Nesterov`, `RMSProp`, `Adam`, `AdamW`, `AdaGrad`, `AdaDelta`, `Nadam`, `NadamW`, `AMSGrad`, `LAMB`, `Lion`, `None`.
*   `nn.ErrorCalculationType`: `None`, `HuberLoss`, `HuberDirectionLoss`, `MAE`, `MSE`, `RMSE`, `BCELoss`, `CrossEntropy`, etc.

### Core Classes & Properties
*   `nn.Activation(method, alpha, temperature=1.0)`: Represents the activation configuration.
*   `nn.NeuralNetworkOptions.create(topology)`: Returns a builder options object. Exposes builder methods such as:
    *   `.with_number_of_epoch(500)`
    *   `.with_learning_rate(0.15)`
    *   `.with_batch_size(4)`
    *   `.with_progress_callback(callback_function)` (allows native Python functions to serve as callbacks during training).
*   `nn.NeuralNetwork(options)`: Core model executor. Methods:
    *   `.train(inputs, outputs)`: Input arguments are standard Python nested lists of floats (e.g. `[[0, 0, 1], [1, 0, 1]]`).
    *   `.think(inputs)`: Returns predictions as Python lists.
*   `nn.NeuralNetworkSerializer`: Exposes static methods:
    *   `nn.NeuralNetworkSerializer.save(net, filepath)`
    *   `nn.NeuralNetworkSerializer.load(filepath)` (returns a deserialized network instance).
