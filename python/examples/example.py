import os
import sys

# Add the directory containing the compiled .pyd module to the import search path
# (Assuming the module is built in x64/Release or placed in the parent python folder)
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

def run_general_example():
    nn.Logger.info("--- Running General Python Example ---")
    
    # 1. Configure the network topology and options (2 inputs, 8 hidden, 1 output)
    topology = [2, 8, 1]
    
    # Expose hidden layer details
    hidden_activation = nn.Activation(nn.ActivationMethod.Sigmoid, 1.0)
    hidden_layers = [
        nn.LayerDetails(
            nn.LayerArchitecture.FF,
            8,
            hidden_activation,
            0.0,
            0.0,
            nn.OptimiserType.Adam,
            0.9
        )
    ]

    # Example of an AttentionPool layer: it must immediately follow a Gru/Lstm
    # layer, its own size must match that layer's hidden size, and it needs a
    # non-zero attention_hidden_size (the internal scoring-projection width).
    # Not wired into this XOR example's (non-recurrent) topology below - shown
    # here purely to demonstrate the LayerDetails shape end-to-end.
    attention_hidden_layers = [
        nn.LayerDetails(
            nn.LayerArchitecture.Gru,
            8,
            hidden_activation,
            0.0,
            0.0,
            nn.OptimiserType.Adam,
            0.9,
            False,
            0
        ),
        nn.LayerDetails(
            nn.LayerArchitecture.AttentionPool,
            8,
            nn.Activation(nn.ActivationMethod.Linear, 0.0),
            0.0,
            0.0,
            nn.OptimiserType.Adam,
            0.9,
            False,
            16
        )
    ]
    nn.Logger.debug(f"Example AttentionPool hidden layers configured: {len(attention_hidden_layers)} layers.")
    
    # Expose individual output layer settings
    out_activation = nn.Activation(nn.ActivationMethod.Sigmoid, 1.0)
    out_layer = nn.OutputLayerDetails(
        topology[-1], 
        out_activation, 
        nn.ErrorCalculationType.MSE,
        nn.EvaluationConfig(),
        0.0, 
        nn.OptimiserType.Adam, 
        0.9
    )
    
    # Create the builder options
    options = nn.NeuralNetworkOptions.create(topology) \
        .with_batch_size(1) \
        .with_hidden_layers(hidden_layers) \
        .with_output_layer_details(out_layer) \
        .with_learning_rate(0.1) \
        .with_number_of_epoch(3000) \
        .with_enable_bptt(False) \
        .with_shuffle_training_data(True) \
        .with_data_is_unique(True) \
        .with_log_level(nn.LogLevel.Debug) \
        .with_stochastic_weight_averaging(nn.StochasticWeightAveragingDetails(True, 0.75, 0.05)) \
        .build()
        
    # Optional progress callback (defined in Python!)
    def on_progress(helper):
        if helper.epoch % 500 == 0:
            nn.Logger.debug(f"Epoch: {helper.epoch:3d} | Complete: {helper.percent_complete * 100:5.1f}%")
        return True # Return False to stop training early
        
    options = options.with_progress_callback(on_progress)
    
    # 2. Instantiate the network
    net = nn.NeuralNetwork(options)
    
    # 3. Define the training dataset (XOR classification)
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
    
    # 4. Train the model
    nn.Logger.info("Training model...")
    net.train(training_inputs, training_outputs)
    
    # 5. Evaluate the model (Inference)
    nn.Logger.info("Evaluating predictions:")
    for inputs, expected in zip(training_inputs, training_outputs):
        outputs = net.think(inputs)
        nn.Logger.info(f"Input: {inputs} | Expected: {expected[0]} | Predicted: {outputs[0]:.4f}")
        
    # 6. Serialise and deserialise model
    model_path = "python_xor_model.json"
    nn.Logger.info(f"Saving model to: {model_path}...")
    nn.NeuralNetworkSerializer.save(net, model_path)
    
    nn.Logger.info(f"Loading model from: {model_path}...")
    loaded_net = nn.NeuralNetworkSerializer.load(model_path)
    
    # Test loaded model
    test_input = [1.0, 0.0]
    loaded_output = loaded_net.think(test_input)
    nn.Logger.info(f"Loaded model prediction on {test_input}: {loaded_output[0]:.4f}")
    
    if os.path.exists(model_path):
        os.remove(model_path)

if __name__ == '__main__':
    run_general_example()
