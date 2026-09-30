"""
Gated Residual Network (GRN) & Attention Interpretability Example

This example demonstrates how to combine Multi-Head Self-Attention with a
Gated Residual Network (GRN) layer for temporal sequence forecasting, and how to
extract attention matrices for model interpretability.

Key components demonstrated:
1. Multi-Head Self-Attention Layer (`LayerDetails.create_self_attention`)
2. Gated Residual Network Layer (`LayerDetails.create_grn`) featuring:
   - Dense intermediate layer with ELU activation
   - Gated Linear Unit (GLU) gating
   - Residual skip connection
   - Layer Normalisation
3. BPTT (Backpropagation Through Time) sequence learning
4. Temporal Attention Weight extraction (`get_attention_weights` and `get_mean_attention_weights`)
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


def run_grn_attention_example():
    nn.Logger.info("=== Running GRN & Attention Interpretability Example ===")

    sequence_length = 5
    feature_dim = 4
    hidden_dim = 4
    num_heads = 2
    ffn_dim = 8

    # 1. Define topology: 4 inputs -> 4 Self-Attention -> 4 GRN -> 1 Output
    topology = [feature_dim, hidden_dim, hidden_dim, 1]

    # 2. Configure hidden layers:
    # Layer 1: Multi-Head Self-Attention
    attention_act = nn.Activation(nn.ActivationMethod.Linear)
    attention_layer = nn.LayerDetails.create_self_attention(
        size=hidden_dim,
        number_of_heads=num_heads,
        feed_forward_hidden_size=ffn_dim,
        activation=attention_act,
        dropout=0.0,
        weight_decay=1e-4,
        optimiser_type=nn.OptimiserType.AdamW,
        momentum=0.9,
        use_layer_norm=True
    )

    # Layer 2: Gated Residual Network (GRN)
    grn_act = nn.Activation(nn.ActivationMethod.Elu)
    grn_layer = nn.LayerDetails.create_grn(
        size=hidden_dim,
        feed_forward_hidden_size=ffn_dim,
        activation=grn_act,
        dropout=0.0,
        weight_decay=1e-4,
        optimiser_type=nn.OptimiserType.AdamW,
        momentum=0.9,
        use_layer_norm=True
    )

    hidden_layers = [attention_layer, grn_layer]

    # 3. Configure output layer (Linear regression with Huber Loss)
    eval_cfg = nn.EvaluationConfig(
        neutral_tolerance=0.0,
        confidence_threshold=0.0,
        huber_delta=1.0,
        direction_lambda=0.0,
        use_direction_penalty=False,
        cross_entropy_lambda=1.0,
        epsilon=1e-12,
        label_smoothing=0.0,
        quantiles=[0.5],
        transaction_cost_penalty=0.0,
        sortino_target_return=0.0
    )
    output_act = nn.Activation(nn.ActivationMethod.Linear)
    output_layer = nn.OutputLayerDetails(
        1,
        output_act,
        nn.ErrorCalculationType.HuberLoss,
        eval_cfg,
        1e-4,
        nn.OptimiserType.AdamW,
        0.9
    )

    # 4. Build options with BPTT enabled for temporal sequence processing
    options = (
        nn.NeuralNetworkOptions.create(topology)
        .with_hidden_layers(hidden_layers)
        .with_output_layer_details([output_layer])
        .with_learning_rate(0.01)
        .with_number_of_epoch(200)
        .with_batch_size(1)
        .with_enable_bptt(True)
        .with_bptt_max_ticks(sequence_length)
        .with_shuffle_bptt_batches(False)
        .with_bptt_supervise_last_step_only(False)
        .with_seed(42)
        .with_log_level(nn.LogLevel.Warning)
        .build()
    )

    # 5. Instantiate network
    net = nn.NeuralNetwork(options)
    print("Neural Network created with Self-Attention and GRN layers.")

    # 6. Generate synthetic temporal sequence data
    # Each sample is a sequence of length 5 with 4 features; target is a linear combination
    training_inputs = []
    training_targets = []
    for step in range(sequence_length):
        val = float(step + 1) * 0.2
        training_inputs.append([val, val * 0.5, val * 1.5, val * 2.0])
        training_targets.append([val * 1.2])

    print(f"Training network on sequence of {sequence_length} timesteps...")
    net.train(training_inputs, training_targets)
    print("Training complete.")

    # 7. Perform inference
    print("\nRunning inference on the test sequence...")
    net.set_attention_capture(True)
    predictions = net.think(training_inputs)
    for t, (inp, pred, targ) in enumerate(zip(training_inputs, predictions, training_targets)):
        print(f"  Step {t}: Target = {targ[0]:.4f}, Prediction = {pred[0]:.4f}")

    # 8. Extract and inspect attention weights for interpretability
    # Layer 1 is the Self-Attention layer
    mean_attention = net.get_mean_attention_weights(layer_index=1, batch_index=0)
    print("\n=== Mean Attention Weights (Averaged across heads) ===")
    print("Row = Query timestep, Col = Key timestep")
    print("     " + " ".join([f"K_{k:<6}" for k in range(len(mean_attention))]))
    for q_idx, row in enumerate(mean_attention):
        row_str = " ".join([f"{w:7.4f}" for w in row])
        print(f"Q_{q_idx}: {row_str}")

    # Inspect per-head attention weights
    head_attention = net.get_attention_weights(layer_index=1, batch_index=0)
    num_heads_extracted = len(head_attention)
    print(f"\nExtracted attention matrices for {num_heads_extracted} attention heads.")
    for h in range(num_heads_extracted):
        print(f"\n--- Head {h + 1} Attention Matrix ---")
        for q_idx, row in enumerate(head_attention[h]):
            row_str = " ".join([f"{w:7.4f}" for w in row])
            print(f"  Q_{q_idx}: {row_str}")

    print("\n=== GRN & Attention Interpretability Example Complete ===")


if __name__ == "__main__":
    run_grn_attention_example()
