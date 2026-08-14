"""
Multi-Output Neural Network Example

This example demonstrates a neural network configured with multiple parallel
output layers using different activation functions and target objectives:

1. Classification Output (Sigmoid): Predicts whether input x > 0 (1.0 for positive, 0.0 for non-positive).
2. Regression Output (Tanh): Predicts the continuous hyperbolic tangent value tanh(x).

Network Architecture:
- Topology: [1, 8, 8, 2] (1 input feature, two hidden layers of 8 neurons each, 2 total outputs)
- Hidden layers: 2 Feed-Forward layers of 8 neurons each with Tanh activation and NadamW optimiser
- Output Layer 1: Sigmoid activation, MSE error calculation (Classification task)
- Output Layer 2: Tanh activation, MSE error calculation (Regression task)
"""

import math
import os
import random
import sys

# Add the directory containing the compiled .pyd module to the import search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PYTHON_DIR)
sys.path.append(os.path.join(PYTHON_DIR, "x64", "Release"))

try:
    import neuralnetwork as nn
    print("Successfully imported neuralnetwork library!")
except ImportError as e:
    print(f"Error: Could not import neuralnetwork extension module: {e}")
    print("Please make sure you have built the solution in Release/x64 first.")
    sys.exit(1)


def create_neural_network(log_level):
    """
    Configures and creates a multi-output neural network instance.
    
    Topology: 1 input -> Hidden [8, 8] -> 2 outputs (1 Sigmoid, 1 Tanh)
    """
    topology = [1, 8, 8, 2]

    # Configure two hidden layers with Tanh activation and NadamW optimiser
    hidden_activation = nn.Activation(nn.ActivationMethod.Tanh, 1.0)
    hidden_layers = [
        nn.LayerDetails(
            nn.LayerArchitecture.FF,
            8,                      # 8 neurons
            hidden_activation,      # Tanh activation
            0.0,                    # Dropout rate
            0.5,                    # Weight decay
            nn.OptimiserType.NadamW,# NadamW optimiser
            0.9                     # Momentum factor
        ),
        nn.LayerDetails(
            nn.LayerArchitecture.FF,
            8,                      # 8 neurons
            hidden_activation,      # Tanh activation
            0.0,                    # Dropout rate
            0.5,                    # Weight decay
            nn.OptimiserType.NadamW,# NadamW optimiser
            0.9                     # Momentum factor
        )
    ]

    # Define multiple output layers:
    # Output Layer 1: Sigmoid activation for classification (Is positive?)
    # Output Layer 2: Tanh activation for regression (tanh(x) value)
    output_layers = [
        nn.OutputLayerDetails(
            1,                                      # 1 neuron
            nn.Activation(nn.ActivationMethod.Sigmoid, 1.0),
            nn.ErrorCalculationType.MSE,
            nn.EvaluationConfig(),
            0.001,                                  # Weight decay
            nn.OptimiserType.NadamW,
            0.99                                    # Momentum factor
        ),
        nn.OutputLayerDetails(
            1,                                      # 1 neuron
            nn.Activation(nn.ActivationMethod.Tanh, 1.0),
            nn.ErrorCalculationType.MSE,
            nn.EvaluationConfig(),
            0.001,                                  # Weight decay
            nn.OptimiserType.NadamW,
            0.9                                     # Momentum factor
        )
    ]

    # Build options configuration
    options = (
        nn.NeuralNetworkOptions.create(topology)
        .with_batch_size(16)
        .with_hidden_layers(hidden_layers)
        .with_output_layer_details(output_layers)
        .with_learning_rate(0.01)
        .with_number_of_epoch(1000)
        .with_log_level(log_level)
        .build()
    )

    return nn.NeuralNetwork(options)


def generate_data(count):
    """
    Generates synthetic dataset for training.
    
    Inputs: Random values x in the range [-2.0, 2.0]
    Outputs: [1.0 if x > 0 else 0.0 (Classification), math.tanh(x) (Regression)]
    """
    inputs = []
    outputs = []

    random.seed(42)  # Seed for reproducible synthetic data generation

    for _ in range(count):
        x = random.uniform(-2.0, 2.0)
        y_class = 1.0 if x > 0 else 0.0
        y_reg = math.tanh(x)

        inputs.append([x])
        outputs.append([y_class, y_reg])

    return inputs, outputs


def run_validation_tests(net):
    """
    Runs evaluation test cases against expected targets and reports status.
    """
    nn.Logger.info("Running validation tests...")

    test_cases = [
        {"input": 1.5, "exp_class": 1.0, "exp_reg": math.tanh(1.5)},
        {"input": -1.5, "exp_class": 0.0, "exp_reg": math.tanh(-1.5)},
        {"input": 0.5, "exp_class": 1.0, "exp_reg": math.tanh(0.5)},
        {"input": -0.5, "exp_class": 0.0, "exp_reg": math.tanh(-0.5)}
    ]

    all_passed = True
    tolerance = 0.15

    for test in test_cases:
        result = net.think([test["input"]])
        got_class = result[0]
        got_reg = result[1]

        class_ok = abs(got_class - test["exp_class"]) < tolerance
        reg_ok = abs(got_reg - test["exp_reg"]) < tolerance

        status_class = "[OK]" if class_ok else "[FAIL]"
        status_reg = "[OK]" if reg_ok else "[FAIL]"

        nn.Logger.info(
            f"Input: {test['input']:5.2f} | "
            f"Class: {got_class:6.4f} (exp {test['exp_class']:.1f}) {status_class} | "
            f"Reg: {got_reg:6.4f} (exp {test['exp_reg']:6.4f}) {status_reg}"
        )

        if not class_ok or not reg_ok:
            all_passed = False

    return all_passed


def run_multi_output_example():
    nn.Logger.info("=== Multi-Output Layer (Sigmoid + Tanh) Example ===")

    # 1. Instantiate neural network
    net = create_neural_network(nn.LogLevel.Info)

    # 2. Generate training data (500 samples)
    training_inputs, training_outputs = generate_data(500)

    # 3. Train network
    nn.Logger.info("Training network with multiple output layers...")
    net.train(training_inputs, training_outputs)

    # 4. Run validation tests and display results
    success = run_validation_tests(net)

    if success:
        nn.Logger.info("*********************************")
        nn.Logger.info("*      OVERALL STATUS: SUCCESS  *")
        nn.Logger.info("*********************************")
    else:
        nn.Logger.error("*********************************")
        nn.Logger.error("*      OVERALL STATUS: FAILURE  *")
        nn.Logger.error("*********************************")


if __name__ == '__main__':
    run_multi_output_example()
