"""
XOR (Exclusive OR) Neural Network Example

This example demonstrates how to solve the classic non-linearly separable
XOR problem using the `neuralnetwork` Python library.

Network Topology:
- Input layer: 2 inputs (x1, x2) — the internal bias neuron is added automatically
- Hidden layer: 4 neurons with Sigmoid activation
- Output layer: 1 neuron with Sigmoid activation and MSE error calculation

The XOR function truth table:
- Input [0.0, 0.0] -> Output [0.0]
- Input [0.0, 1.0] -> Output [1.0]
- Input [1.0, 0.0] -> Output [1.0]
- Input [1.0, 1.0] -> Output [0.0]
"""

import os
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


def run_xor():
    nn.Logger.info("=== Running Python XOR Example ===")

    # 1. Define network topology: 2 input features, 8 hidden neurons, 1 output neuron
    topology = [2, 8, 1]

    # 2. Configure hidden layer (Feed-Forward architecture with Sigmoid activation and Adam optimiser)
    hidden_activation = nn.Activation(nn.ActivationMethod.Sigmoid, 1.0)
    hidden_layers = [
        nn.LayerDetails(
            nn.LayerArchitecture.FF,
            8,                         # 8 neurons in hidden layer
            hidden_activation,         # Sigmoid activation function
            0.0,                       # Dropout rate (0.0 = disabled)
            0.0,                       # Weight decay
            nn.OptimiserType.Adam,     # Adam optimiser for fast and stable convergence
            0.9                        # Momentum factor
        )
    ]

    # 3. Configure output layer (Sigmoid activation, Mean Squared Error loss)
    output_activation = nn.Activation(nn.ActivationMethod.Sigmoid, 1.0)
    output_layer = nn.OutputLayerDetails(
        topology[-1],                  # 1 output neuron
        output_activation,             # Sigmoid activation
        nn.ErrorCalculationType.MSE,    # Mean Squared Error calculation
        nn.EvaluationConfig(),         # Evaluation configuration defaults
        0.0,                           # Weight decay
        nn.OptimiserType.Adam,         # Adam optimiser
        0.9                            # Momentum factor
    )

    # 4. Optional progress callback to monitor training epochs
    def on_progress(helper):
        if helper.epoch % 500 == 0:
            nn.Logger.info(f"Epoch: {helper.epoch:4d} | Complete: {helper.percent_complete * 100:5.1f}%")
        return True  # Return True to continue training

    # 5. Build network configuration options
    options = (
        nn.NeuralNetworkOptions.create(topology)
        .with_batch_size(1)
        .with_hidden_layers(hidden_layers)
        .with_output_layer_details(output_layer)
        .with_learning_rate(0.1)
        .with_number_of_epoch(3000)
        .with_enable_bptt(False)
        .with_shuffle_training_data(True)
        .with_data_is_unique(True)
        .with_log_level(nn.LogLevel.Info)
        .with_progress_callback(on_progress)
        .build()
    )

    # 6. Instantiate the neural network
    net = nn.NeuralNetwork(options)

    # 7. Define training inputs (2 XOR inputs) and expected outputs
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

    # 8. Train the model
    nn.Logger.info("Training XOR model...")
    net.train(training_inputs, training_outputs)

    # 9. Evaluate model predictions against expected targets
    nn.Logger.info("Evaluating XOR predictions:")
    all_passed = True
    tolerance = 0.15

    for inputs, expected in zip(training_inputs, training_outputs):
        outputs = net.think(inputs)
        predicted = outputs[0]
        exp_val = expected[0]
        passed = abs(predicted - exp_val) < tolerance

        if not passed:
            all_passed = False

        status = "[OK]" if passed else "[FAIL]"
        nn.Logger.info(
            f"Input: {inputs} | Expected: {exp_val:.1f} | "
            f"Predicted: {predicted:.4f} {status}"
        )

    if all_passed:
        nn.Logger.info("XOR Example Completed Successfully!")
    else:
        nn.Logger.error("XOR Example Failed to Converge!")
        sys.exit(1)


if __name__ == '__main__':
    run_xor()
