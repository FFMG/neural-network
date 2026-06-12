import os
import sys

# Add the directory containing the compiled .pyd module to the import search path
# The module is built in x64/Release by MSBuild in python/x64/Release/
module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "x64", "Release")
sys.path.append(module_dir)

try:
    import neuralnetwork as nn
    print("Successfully imported neuralnetwork library!")
    print(f"Docstring: {nn.__doc__}")

    # Check activation enum and class instantiation
    print("Verifying ActivationMethod...")
    sigmoid_method = nn.ActivationMethod.Sigmoid
    print(f"Sigmoid enum value: {sigmoid_method}")

    print("Instantiating Activation class...")
    act = nn.Activation(sigmoid_method, 1.0)
    print(f"Activation instantiation success: method={act.method}, alpha={act.alpha}")

    # Verify Options
    print("Verifying NeuralNetworkOptions creation...")
    options = nn.NeuralNetworkOptions.create([3, 2, 1])
    print("NeuralNetworkOptions created successfully!")

    print("All checks passed successfully!")
    sys.exit(0)
except Exception as e:
    print(f"Verification failed: {e}", file=sys.stderr)
    sys.exit(1)
