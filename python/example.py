import os
import sys

# Add the directory containing the compiled .pyd module to the import search path
# (Assuming the module is built in x64/Release or placed in this folder)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "x64", "Release"))

try:
    import neuralnetwork as nn
    print("Successfully imported neuralnetwork library!")
except ImportError as e:
    print(f"Error: Could not import neuralnetwork extension module: {e}")
    print("Please make sure you have built the solution in Release/x64 first.")
    sys.exit(1)

def run_xor_example():
    nn.Logger.info("--- Running XOR Example ---")
    
    # 1. Configure the network topology and options
    topology = [3, 2, 1]
    
    # Expose hidden layer details
    hidden_activation = nn.Activation(nn.ActivationMethod.Sigmoid, 1.0)
    hidden_layers = [
        nn.LayerDetails(
            nn.LayerArchitecture.FF, 
            2, 
            hidden_activation, 
            0.0, 
            0.0, 
            nn.OptimiserType.SGD, 
            0.99
        )
    ]
    
    # Expose individual output layer settings
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
    
    # Create the builder options
    options = nn.NeuralNetworkOptions.create(topology) \
        .with_batch_size(1) \
        .with_hidden_layers(hidden_layers) \
        .with_output_layer_details(out_layer) \
        .with_learning_rate(0.1) \
        .with_learning_rate_warmup(0.01, 0.075) \
        .with_learning_rate_decay_rate(0.0)  \
        .with_learning_rate_boost_rate(0.25, 0.05) \
        .with_number_of_epoch(5000) \
        .with_log_level(nn.LogLevel.Debug) \
        .build()
        
    # Optional progress callback (defined in Python!)
    def on_progress(helper):
        if helper.epoch % 50 == 0:
            nn.Logger.debug(f"Epoch: {helper.epoch:3d} | Complete: {helper.percent_complete * 100:5.1f}%")
        return True # Return False to stop training early
        
    options = options.with_progress_callback(on_progress)
    
    # 2. Instantiate the network
    net = nn.NeuralNetwork(options)
    
    # 3. Define the training dataset (XOR problem with bias/3rd input)
    training_inputs = [
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ]
    
    training_outputs = [
        [0.0],
        [0.0],
        [1.0],
        [1.0]
    ]
    
    # 4. Train the model
    nn.Logger.info("Training model...")
    net.train(training_inputs, training_outputs)
    
    # 5. Evaluate the model (Inference)
    nn.Logger.info("Evaluating predictions:")
    for inputs, expected in zip(training_inputs, training_outputs):
        outputs = net.think(inputs)
        nn.Logger.info(f"Input: {inputs[:2]} | Expected: {expected[0]} | Predicted: {outputs[0]:.4f}")
        
    # 6. Serialise and deserialise model
    model_path = "python_xor_model.json"
    nn.Logger.info(f"Saving model to: {model_path}...")
    nn.NeuralNetworkSerializer.save(net, model_path)
    
    nn.Logger.info(f"Loading model from: {model_path}...")
    loaded_net = nn.NeuralNetworkSerializer.load(model_path)
    
    # Test loaded model
    test_input = [1.0, 0.0, 1.0]
    loaded_output = loaded_net.think(test_input)
    nn.Logger.info(f"Loaded model prediction on {test_input[:2]}: {loaded_output[0]:.4f}")

if __name__ == '__main__':
    run_xor_example()
