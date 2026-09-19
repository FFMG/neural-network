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
* quickGelu
* mish
* elu
* softmax

### Optimizers

* None
* SGD
* Adam (Standard Adam does not apply weight decay; if `weight_decay > 0` is configured, a warning is logged. Use `AdamW` if decoupled weight decay is desired.)
* AdamW (Adam with decoupled weight decay)
* Nadam
* NadamW
* Adagrad
* RMSProp
* Lion

#### Not supported (yet)

* Nesterov
* AdaDelta
* AMSGrad
* LAMB

## Python Bindings

Python bindings are available for the `myoddweb::nn` library, allowing you to configure, train, and run models natively in Python. The bindings are powered by `pybind11`.

For detailed API documentation, prerequisites, and instructions on how to build and run the module, see [python/README.md](python/README.md).

Standalone Python examples are located in [python/examples/](python/examples/):
- **XOR Classification (`python/examples/xor.py`)**: Classic non-linearly separable XOR problem using Feed-Forward layers and Sigmoid activation.
- **Multi-Output Layer (`python/examples/multi_output.py`)**: Parallel multi-output model performing joint classification (Sigmoid) and regression (Tanh).
- **General Example (`python/examples/example.py`)**: Comprehensive demonstration of configuration options, progress monitoring callbacks, and model serialization.
- **Reinforcement Learning (`python/examples/tic_tac_toe.py`)**: Policy-gradient (REINFORCE) agent learning Tic-Tac-Toe via advantage rewards and playing against a Random opponent.

### Python Quickstart Example

```python
import neuralnetwork as nn

# 1. Configure the network architecture
topology = [3, 2, 1]

# 2. Set options (learning rate, epochs, etc.)
options = nn.NeuralNetworkOptions.create(topology) \
    .with_batch_size(1) \
    .with_learning_rate(0.1) \
    .with_number_of_epoch(1000) \
    .build()

# 3. Create the model
net = nn.NeuralNetwork(options)

# 4. Train the network
inputs = [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
outputs = [[0.0], [1.0], [1.0], [0.0]]
net.train(inputs, outputs)

# 5. Predict
prediction = net.think([1.0, 0.0, 1.0])
print(f"Prediction: {prediction[0]:.4f}")
```


## Options

The following sections describe the various configuration options available when building a network using `NeuralNetworkOptions`.

### Hidden Layers

The hidden layer configuration allows you to define the architecture of your network's trunk.

* **Layer type:** 
  * `FF`: Standard feed-forward layer.
  * `Elman`: Simple recurrent layer.
  * `Gru`: Gated recurrent unit layer.
  * `Lstm`: Long Short-Term Memory layer.
  * `AttentionPool`: Additive (Bahdanau-style) attention pooling over a preceding `Gru`/`Lstm` layer's BPTT window (see "Attention Pooling" below).
  * `Tcn`: Dilated causal 1D convolution ("Temporal Convolutional Network" block) over a window of preceding timesteps (see "TCN" below).
  * `SelfAttention`: Multi-head causal self-attention plus a position-wise feed-forward sub-block (see "Self-Attention" below).
* **Layer size:** Number of neurons in the hidden layer.
* **Activation:** The activation object (method, alpha, and temperature).
* **Weight Decay:** Regularization strength.
* **Dropout:** Percentage of neurons to randomly drop during training (0.0 to 1.0).
* **Optimiser:** Each layer can optionally have its own optimizer configuration.

```cpp
    std::vector<unsigned> topology = {2, 8, 8, 8, 8, 1};
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Lstm, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0),
      LayerDetails(Layer::Architecture::Lstm, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0),
      LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.01), 0.2, 0.05, OptimiserType::AdamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0),
      LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0),
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
      { LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::tanh, 0.01), 0.0, 0.01, OptimiserType::NadamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0) },
      OutputLayerDetails(2, activation(activation::method::tanh, 0.01), ErrorCalculation::type::mse, EvaluationConfig(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.0, { 0.5 }, 0.0, 0.0), 0.0, OptimiserType::NadamW, 0.95)
    );
    multi_output_layer_details.push_back(b1);

    // Branch 2: Deeper path, 3 outputs (Softmax with Label Smoothing)
    MultiOutputLayerDetails b2
    (
      {
        LayerDetails(Layer::Architecture::FF, 16, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::NadamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0),
        LayerDetails(Layer::Architecture::FF, 8, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::NadamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0)
      },
      OutputLayerDetails(3, activation(activation::method::softmax, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(0.0, 0.0, 1.0, 0.0, false, 1.0, 1e-12, 0.1, { 0.5 }, 0.0, 0.0), 0.0, OptimiserType::NadamW, 0.95)
    );
    multi_output_layer_details.push_back(b2);

    auto options = NeuralNetworkOptions::create(topology)
      .with_hidden_layers({ LayerDetails(Layer::Architecture::Gru, 4, activation(activation::method::tanh, 0.01), 0.0, 0.01, OptimiserType::NadamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0) })
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
*   **`bptt-supervise-last-step-only` (Last Step Supervision Only):** If set to `true`, only the final time step ($t = \text{bptt\_max\_ticks} - 1$) of each sequence block is supervised with target outputs during training.
    *   **USE CASE:** Ideal for sequence-to-one forecasting tasks (e.g. predicting the next price or direction after observing $T$ historical ticks). The recurrent layer consumes all $T$ input ticks to warm up its hidden state context, but loss and gradient backpropagation are calculated exclusively from the final prediction step.

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_shuffle_training_data(false) // Keep chronological for RNNs
      .with_shuffle_bptt_batches(true)  // Shuffle blocks for better generalization
      .with_enable_bptt(true)
      .with_bptt_max_ticks(24)
      .with_bptt_supervise_last_step_only(true) // Supervise only the final tick of each sequence
      .build();
```

### Learning Rate Strategies

The library supports various strategies to manage learning rate dynamics:

*   **Warmup:** Linearly (or geometrically) increases the rate from a starting value to the target rate over a percentage of the total epochs.
*   **Exponential Decay:** Reduces the learning rate by a fixed decay factor after each epoch.
*   **Smooth Cosine Boosts (Restarts):** Periodically boosts the learning rate using a smooth cosine staircase to help the model escape local minima.
*   **Cosine Annealing with Warm Restarts (SGDR):** Anneals the learning rate along a half-cosine curve with configurable initial cycle period ($T_0$), geometric period multiplier ($T_{\text{mult}}$), minimum learning rate floor ($\eta_{\min}$), and peak restart decay factor ($\gamma$).
*   **Adaptive Learning Rate:** Dynamically adjusts the learning rate based on recent error trends. It detects states like `Plateauing`, `Improving`, or `Exploding` and adjusts the rate accordingly.

```cpp
    // Example 1: Warmup with Cosine Annealing and Warm Restarts (SGDR)
    auto options_sgdr = NeuralNetworkOptions::create(topology)
      .with_learning_rate(0.001)
      .with_learning_rate_warmup(0.0001, 0.05) // Start at 0.0001, reach target at 5% of training
      .with_cosine_annealing_warm_restarts(true, 20, 2.0, 0.00001, 0.9) // T_0=20, T_mult=2.0, eta_min=1e-5, gamma=0.9
      .build();

    // Example 2: Warmup with Exponential Decay, Boosts, and Error-Adaptive Rates
    auto options_adaptive = NeuralNetworkOptions::create(topology)
      .with_learning_rate(0.001)
      .with_learning_rate_warmup(0.0001, 0.05) // Start at 0.0001, reach target at 5% of training
      .with_learning_rate_decay_rate(0.985)    // Decay factor applied per epoch
      .with_learning_rate_boost_rate(0.2, 0.1) // Boost by 10% every 20% of training epochs
      .with_adaptive_learning_rates(true)      // Enable dynamic error-based adjustment
      .build();
```

### Dynamic / External Learning Rate Control

Callers driving external training schedules (such as reinforcement learning episode loops, decay schedules, or online policy-gradient updates via `train_with_advantages()`) can forcefully set the learning rate at runtime via `NeuralNetwork::set_learning_rate(double)`:

*   **Runtime Override:** Calling `nn.set_learning_rate(lr)` forcefully overrides the learning rate used by subsequent `train()` and `train_with_advantages()` calls, bypassing the internal epoch/warmup/cosine-annealing/decay scheduler entirely.
*   **State Querying:** Query `nn.get_learning_rate()` to retrieve the current active learning rate, and `nn.has_learning_rate_override()` to check whether an external override is active.
*   **Restoring Default Schedule:** Calling `nn.set_learning_rate(0.0)` clears the active override and restores the original options-based scheduler.
*   **Validation:** Negative, non-finite (`NaN`/`Inf`) values are rejected with a warning log, leaving the current rate unchanged.

```cpp
    NeuralNetwork nn(options);

    // Dynamic RL episode-based decay loop
    for (int episode = 0; episode < 1000; ++episode)
    {
      const double episode_lr = 0.01 * std::pow(0.995, episode);
      nn.set_learning_rate(episode_lr);
      nn.train_with_advantages(inputs, targets, advantages);
    }

    // Clear override and restore options-configured scheduler
    nn.set_learning_rate(0.0);
```

### Dropout

Individual layers can have dropout applied via `LayerDetails`. During training, neurons are randomly deactivated according to the dropout rate, and the remaining activations are scaled by `1 / (1 - rate)` to maintain the expected sum. Dropout is automatically disabled during inference (`think`).

```cpp
    LayerDetails hl(Layer::Architecture::FF, 64, activation(activation::method::relu, 0.01), 0.25, 0.0, OptimiserType::None, 0.0, false, 0, 0, 0, 0, 0, 0, 0); // 25% dropout
```

### Reproducibility (Seed)

By default, every run of this library is non-deterministic: weight initialization, dropout masking, and data/BPTT-batch shuffling are all drawn from `std::random_device`, so two "identical" configs will diverge run to run. `NeuralNetworkOptions::with_seed(seed)` pins all three to a single `uint32_t` seed, so that an identical config plus an identical seed reproduces byte-identical training runs — a prerequisite for trusting any single-variable comparison between configs, rather than a coin flip against run-to-run noise.

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_seed(42) // opt-in; omit (or pass std::nullopt) for today's non-deterministic behaviour
      .build();
```

The seed is opt-in and fully backward compatible: `with_seed(std::nullopt)` (the default) preserves the exact non-deterministic behaviour of every prior release. When set, a single base seed derives independent, non-colliding sub-seeds per layer, per weight, and per neuron via a fast splitmix64-style mixer (`Rng::derive`) — so, unlike naively reusing one raw seed everywhere, no two weights or dropout streams collapse onto the same value. The seed is persisted by `NeuralNetworkSerializer::save`/`load`, along with the trained gain/bias values.

**Known limitation:** dropout determinism is keyed off `(seed, layer, neuron, batch_index, timestep)`, not an epoch counter. With shuffling enabled (the default), the training example occupying a given batch slot changes every epoch, so this is not an issue in practice. If shuffling is disabled for a run (`shuffle_training_data(false)` with unique/non-shuffled data), the same physical sample will receive the same dropout mask on every epoch.

### Layer Normalization

`Gru` and `Lstm` hidden layers can opt into recurrent-state Layer Normalization via the trailing `use_layer_normalisation` flag on `LayerDetails` (`false` when disabled, `true` when enabled). It normalizes the state each layer actually carries across timesteps — the blended hidden state for `Gru`, the cell state for `Lstm` — with its own learnable per-neuron gain (initialized to `1.0`) and bias (initialized to `0.0`), targeting the unstable activation scale that recurrent nets can build up over a long BPTT window. It is not available on `FF`/`Elman` layers. The flag, like the rest of a hidden layer's configuration, is persisted by `NeuralNetworkSerializer::save`/`load`, along with the trained gain/bias values.

```cpp
    LayerDetails hl(Layer::Architecture::Gru, 32, activation(activation::method::tanh, 0.0), 0.0, 0.01, OptimiserType::AdamW, 0.95, true, 0, 0, 0, 0, 0, 0, 0); // Layer Normalisation enabled
```

### Attention Pooling

Recurrent layers (`Gru`/`Lstm`/`Elman`) compress a whole BPTT window into a single fixed-size hidden state, and every downstream layer normally only ever sees the *last* timestep of it. An `AttentionPool` hidden layer instead sits immediately after a `Gru` or `Lstm` layer, consumes the full T-timestep hidden-state sequence, learns a per-timestep additive (Bahdanau-style) attention weight, and produces a single pooled context vector — letting the network learn which past ticks matter most instead of always using the final one.

`AttentionPool` layers have some structural constraints, all enforced (panic on violation) by `Layer::create_hidden_layer`:
*   Must immediately follow a `Gru` or `Lstm` hidden layer (not `Elman`, not `FF`).
*   `LayerDetails`' `size` must equal the preceding recurrent layer's hidden size — pooling never changes dimensionality.
*   `use_layer_normalisation` must be `false`.
*   `attention_hidden_size` (the internal scoring-projection width) must be non-zero.
*   Residual connections (`residual_layer_number >= 0`) are not supported.
*   Requires `with_enable_bptt(true)`.

```cpp
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Gru, 32, activation(activation::method::tanh, 0.0), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0),
      LayerDetails(Layer::Architecture::AttentionPool, 32, activation(activation::method::linear, 0.0), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 16, 0, 0, 0, 0, 0, 0), // 16-wide attention scoring projection
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(24)
      .build();
```

The layer's own trained weights (the scoring projection and scoring vector) are persisted by `NeuralNetworkSerializer::save`/`load`, along with the `attention_hidden_size` configuration.

**Known limitation:** because `AttentionPool`'s own backward pass has to fully chain-rule its attention math itself (rather than a simple weight-matrix multiply), it relies on the same "direct gradient injection" mechanism already used for recurrent-layer stacking. When the layer directly above `AttentionPool` (typically the output layer) also uses that mechanism, gradient flow through it is affected by the same pre-existing identity-proxy limitation described in the [1.1.21] Known Issues — matching the layer-size requirement above (`layer.size` equal to the preceding recurrent layer's hidden size) keeps this shape-compatible enough for gradients to flow, but this is inherited scope, not fixed by this feature.

### TCN

A `Tcn` hidden layer applies a dilated causal 1D convolution: for every output timestep `t` it gathers the `kernel_size` dilated input taps `{X[t - j*dilation] : j = 0..kernel_size-1}` (zero-padded where `t - j*dilation < 0`) into one flat vector and applies a single dense affine map + activation — letting the network look at a whole window of past ticks at once through widening receptive fields, rather than threading a single compressed hidden state forward the way `Elman`/`Gru`/`Lstm` do. It is always strictly causal (it never looks ahead) with no configuration flag to disable this. Unlike `AttentionPool`, a `Tcn` layer may follow any preceding layer type — including being the very first hidden layer — and may change channel width between its input and output. Stacking a dilation schedule (e.g. `1, 2, 4, 8`) is done the same way `Gru`/`Lstm` layers are already stacked: add multiple `Tcn` entries to `hidden_layers`, each with a different `dilation`.

`Tcn` layers have some structural constraints, all enforced (panic on violation) by `Layer::create_hidden_layer` or `NeuralNetworkOptions::build()`:
*   `kernel_size` must be non-zero.
*   `dilation` must be non-zero.
*   `use_layer_normalisation` must be `false` (not supported).
*   Requires `with_enable_bptt(true)`.
*   The layer's receptive field (`1 + (kernel_size - 1) * dilation`) must not exceed `bptt_max_ticks`.

Unlike `AttentionPool`, a `Tcn` layer accepts the library's existing external residual-connection mechanism (`with_residual_layer_jump`) — a jump of `1` gives the classic per-block TCN skip connection.

```cpp
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Tcn, 32, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 0, 3, 1, 0, 0, 0, 0), // kernel_size=3, dilation=1
      LayerDetails(Layer::Architecture::Tcn, 32, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 0, 3, 2, 0, 0, 0, 0), // kernel_size=3, dilation=2
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_hidden_layers(hidden_layers)
      .with_residual_layer_jump(1)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(24)
      .build();
```

The layer's own trained weights (reusing the same dense weight/bias arrays every hidden layer already has) are persisted by `NeuralNetworkSerializer::save`/`load`, along with the `kernel_size`/`dilation` configuration.

### Self-Attention

A `SelfAttentionLayer` hidden layer (`Layer::Architecture::SelfAttention`) is a small Transformer encoder block: it adds a fixed (non-learned) sinusoidal positional encoding to the preceding layer's T-timestep window, computes Q/K/V projections, runs causally-masked scaled dot-product attention independently per head, projects the concatenated heads back to width, adds that as a residual (optionally through LayerNorm), runs a position-wise feed-forward sub-block, and adds *that* as a second residual (optionally through a second LayerNorm). These two residuals are internal to the block's own math and always on - they are unrelated to the external `residual_layer_number`/`residual_projector` mechanism (also supported here, forwarded to the base class exactly like `FF`/`Gru`/`Lstm`/`Tcn`).

`SelfAttention` layers have some structural constraints, all enforced (panic on violation) by `Layer::create_hidden_layer` or `NeuralNetworkOptions::build()`:
*   `number_of_heads` must be non-zero and must evenly divide `LayerDetails`' `size`.
*   `feed_forward_hidden_size` must be non-zero.
*   `LayerDetails`' `size` must equal the size of the layer it attends over (no dimension-changing projection in v1).
*   `use_layer_normalisation` IS supported (deliberate divergence from `AttentionPool`, which forbids it) - it controls the two internal LayerNorms described above.
*   Requires `with_enable_bptt(true)` and `bptt_max_ticks() > 1`.

Unlike `AttentionPool`, a `SelfAttention` layer has no previous-architecture restriction - it may be the very first hidden layer, or follow anything.

```cpp
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::SelfAttention, 32, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95, true, 0, 0, 0, 4, 64, 0, 0), // 4 heads, feed_forward_hidden_size=64, LayerNorm enabled
    };

    auto options = NeuralNetworkOptions::create(topology)
      .with_hidden_layers(hidden_layers)
      .with_enable_bptt(true)
      .with_bptt_max_ticks(24)
      .build();
```

The layer's own trained weights (Q/K/V/output projections, the feed-forward sub-block's two dense layers, and - when enabled - both LayerNorms' gain/bias) are persisted by `NeuralNetworkSerializer::save`/`load`, along with the `number_of_heads`/`feed_forward_hidden_size` configuration.

`SelfAttentionLayer` and `LayerDetails` report `is_recurrent() == true` to ensure correct gradient propagation in recurrent pipelines. Multi-threaded execution is handled through encapsulated worker tasks (`self_attention_forward_task`, `self_attention_finish_hidden_gradients_task`, `self_attention_grad_calc_task`) and thread-local gradient accumulators, maintaining exact determinism across any thread count.

**Known limitation:** unlike every other layer in this library, `SelfAttention`'s per-batch-item scratch (the per-head attention-score matrix) scales `O(T^2)` in the window length `T`, not linearly - budget `bptt_max_ticks` accordingly for this layer type.

### Embedding Layer (Categorical Entity Embeddings)

An `EmbeddingLayer` (`Layer::Architecture::Embedding`) maps discrete categorical integer IDs (e.g., day-of-week, ticker asset ID, market regime category) into continuous dense embedding vectors of dimension $D$.

For $K$ categorical input features and embedding dimension $D$, the layer outputs $K \times D$ continuous values:
* `vocabulary_size`: The maximum number of discrete categories $V$ (IDs in range $[0, V-1]$).
* `embedding_dimension`: The dense embedding vector dimension $D$.
* `size`: Must equal $K \times D$.

```cpp
    // Example: 2 categorical inputs mapped to 8-dimensional embeddings -> layer size = 16
    std::vector<LayerDetails> hidden_layers = {
      LayerDetails(Layer::Architecture::Embedding, 16, activation(activation::method::linear, 0.0), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 0, 0, 0, 0, 0, 100, 8), // vocab_size=100, embed_dim=8
      LayerDetails(Layer::Architecture::FF, 32, activation(activation::method::relu, 0.01), 0.0, 0.01, OptimiserType::AdamW, 0.95, false, 0, 0, 0, 0, 0, 0, 0),
    };

    auto options = NeuralNetworkOptions::create({ 2, 16, 32, 1 })
      .with_hidden_layers(hidden_layers)
      .build();
```

The embedding lookup weights ($V \times D$) are trained via backpropagation and persisted by `NeuralNetworkSerializer::save`/`load`.

### Stochastic Weight Averaging (SWA)

Run-to-run noise (best epoch, peak accuracy, trajectory shape) makes it hard to tell whether a change to the network genuinely helped or the run just got lucky. SWA reduces that variance, and often yields a small free accuracy improvement, by periodically snapshotting the trained weight *values* once training has reached its stable plateau and averaging them together into the final model — no separate ensemble to store or run at inference time.

Once training reaches `swa_start_percent` of `number_of_epoch`, a snapshot of the current weights is folded into a running average every `swa_update_percent` of `number_of_epoch` (same cadence semantics as `update_training_monitor_percent`). At the end of `train()`, if at least one snapshot was taken, the averaged weights **replace** the network's trained weights before final metrics/temperature calibration are computed — so the deployed model is the averaged one. If SWA is disabled, or no snapshot ever fires (e.g. very short training runs), this is a no-op.

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_stochastic_weight_averaging(StochasticWeightAveragingDetails(true, 0.75, 0.02)) // enabled, start at 75% of epochs, update every 2%
      // or using the helper overload:
      // .with_stochastic_weight_averaging(true, 0.75, 0.02)
      .build();
```

These fields are encapsulated in `StochasticWeightAveragingDetails` and persisted by `NeuralNetworkSerializer::save`/`load`.

### Lookahead Optimizer Wrapper

The Lookahead optimizer wrapper (Zhang et al., 2019) wraps any inner base optimizer (AdamW, SGD, RAdam, Lion, NadamW, etc.) by maintaining two sets of weights: **fast weights** and **slow weights**.

The fast weights are updated iteratively by the base optimizer for $k$ batches (`synchronisation_period`). Every $k$ steps, the slow weights interpolate toward the fast weights with step size $\alpha$ (`slow_weights_step_size`), and the fast weights are synchronized back to the updated slow weights:
$$\phi \leftarrow \phi + \alpha (\theta - \phi), \quad \theta \leftarrow \phi$$

This stabilizes training, reduces variance across noisy minibatches, and improves convergence across diverse loss surfaces.

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_lookahead(LookaheadDetails(true, 5, 0.5)) // enabled, sync every 5 steps, slow step size 0.5
      // or using the helper overload:
      // .with_lookahead(true, 5, 0.5)
      .build();
```

Lookahead is fully orthogonal to other training strategies and can seamlessly co-exist with Stochastic Weight Averaging (SWA), Cosine Annealing with Warm Restarts, and residual connections. The configuration is encapsulated in `LookaheadDetails` and persisted by `NeuralNetworkSerializer::save`/`load`.

### Feed-Forward & Output Layers (`FFLayer` and `FFOutputLayer`)

The library provides high-performance dense and output layer implementations:

*   **`FFLayer` (Dense / Feed-Forward Layer):**
    *   **Forward Feed:** Computes pre-activation sums $Z = X W + b$ (plus optional residual skip connection $X_{res}$) and activations $A = \sigma(Z)$. Accelerated using hand-tuned AVX2 SIMD GEMM kernels processing 4, 2, or 1 batch item per iteration.
    *   **Fast Backward Pass:** Automatically maintains a pre-transposed weight cache ($W^T$) synchronized on initialization and weight updates. Backpropagation through next hidden layers uses forward GEMM kernels on $W^T$, guaranteeing contiguous sequential memory access and eliminating cache misses from strided column reads.
    *   **Inverted Dropout:** Drops neurons during training with probability $p$ and scales remaining activations by $1 / (1 - p)$. The cached binary mask is recorded in the cell state to scale gradients during backpropagation without recalculating random masks.
    *   **Thread Safety & Scaling:** Employs aligned thread-local gradient accumulators (`thread_ff_grad_accumulators`) to completely eliminate cache line false sharing during multi-threaded batch backpropagation.
*   **`FFOutputLayer` (Compound / Multi-Head Output Layer):**
    *   Derives from `FFLayer` and `OutputLayer`, serving as the primary output projection layer.
    *   **Multi-Head Architecture:** Supports splitting output neurons across multiple independent heads, each with its own activation function, loss metric (MSE, RMSE, Huber, Log-Cosh, BCE, Cross-Entropy, Quantile, Sharpe, Sortino), weight decay, and optimizer.
    *   **Derivative Optimization:** Automatically detects canonical loss-activation pairings (e.g. Softmax with Cross-Entropy, Sigmoid with Binary Cross-Entropy) where the loss delta $\delta = \hat{y} - y$ directly represents $\frac{\partial L}{\partial Z}$, skipping redundant activation derivative evaluations.
    *   **Exact Chain Rule Scaling:** For general activations (such as `tanh` with MSE), exact analytical derivatives are applied ($\frac{\partial L}{\partial Z} = \delta \odot \sigma'(Z)$). When soft logit capping is active ($C > 0$), gradients are scaled by $(1 - \tanh^2(Z / C))$ via vectorized AVX2/FMA SIMD instructions.
    *   **Isolated Sequence Losses:** Sharpe and Sortino ratio loss contexts are evaluated per training example without cross-sample return or position leakage.

### Hyperbolic Tangent (`tanh`) Activation

The `tanh` activation method maps real inputs to the $(-1, 1)$ interval:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2}{1 + e^{-2|x|}} - 1$$

*   **Exact Analytical Derivative:**
    $$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = 1 - y^2$$
    During backpropagation through hidden layers (`FFLayer`) and output layers (`FFOutputLayer`), the derivative is computed in $O(1)$ time directly from the saved post-activation output values $y$, eliminating expensive transcendental evaluations. When dropout is enabled, it falls back to raw pre-activations to ensure numerical integrity.
*   **AVX2 & FMA SIMD Acceleration:**
    *   Vectorized batch activation (`simd::tanh_activate`) processes 4 doubles per vector cycle using a degree-6 polynomial minimax approximation of $\exp(u)$ with Newton-Raphson reciprocal division.
    *   Vectorized derivative (`simd::tanh_derivative`) uses fused negative multiply-add (`_mm256_fnmadd_pd`) when `SIMD_FMA_ENABLED` is available to compute $1.0 - y^2$ in a single hardware cycle.
    *   Vectorized `tanh_pd` uses fused multiply-subtract (`_mm256_fmsub_pd`) for $(2 \cdot \text{rcp} - 1.0)$ computation.
    *   Batched Log-Cosh loss deltas ($\frac{1}{N}\tanh(\hat{y} - y)$) in `Layer::calculate_log_cosh_error_deltas` are accelerated via AVX2 SIMD `simd::tanh_pd`.

### General Training Options

These options control the overall execution of the training process:

*   **`number_of_epoch`:** Total number of training iterations over the dataset.
*   **`batch_size`:** Number of samples processed before internal gradient updates are applied.
*   **`number_of_threads`:** Controls multi-threaded execution for GEMM and layer operations.
*   **`progress_callback`:** A lambda or function called after each epoch to monitor error metrics and progress.
*   **`has_bias`:** Global toggle to enable or disable bias neurons for all layers.
*   **`log_training_info`:** Toggle to enable or disable printing training statistics/configurations to the log output at the start of training (defaults to `true`).

```cpp
    auto options = NeuralNetworkOptions::create(topology)
      .with_number_of_epoch(5000)
      .with_batch_size(32)
      .with_number_of_threads(8)
      .with_has_bias(true)
      .with_log_training_info(true)
      .with_progress_callback([](NeuralNetworkHelper& helper) {
          Logger::info("Epoch: ", helper.epoch(), " Error: ", helper.error());
          return true; // Return false to stop training early
      })
      .build();
```

### Inference Temperature Calibration

For classification tasks using Softmax, the network automatically optimizes the inference temperature ($T$) post-training using a calibration set to ensure well-calibrated probability outputs.

### Policy Advantage Training & Entropy Regularisation

The library supports policy gradient reinforcement learning via `train_with_advantages`:

*   **Reward-Weighted Gradient Scaling:** Scales cross-entropy output gradients by scalar advantage estimates ($A_t$), accelerating actions with positive advantages and depressing actions with negative advantages.
*   **Entropy Regularisation (`with_entropy_coefficient`):** Prevents premature policy collapse and sustains exploration across the discrete action space by adding an entropy bonus loss term:
    $$L(\theta) = - \mathbb{E}[A(s, a) \log \pi_\theta(a|s)] - \beta H(\pi_\theta(\cdot|s))$$
    where $H(\pi) = -\sum_k p_k \ln p_k$ and the analytical gradient with respect to output logit $z_k$ is:
    $$\frac{\partial(-\beta H)}{\partial z_k} = \beta \cdot p_k (\ln p_k + H)$$
    When probabilities approach a uniform distribution ($p_k = 1/N$), the entropy gradient vanishes ($\ln(1/N) + \ln N = 0$). When the policy collapses toward a deterministic action, the gradient pushes probabilities back toward uniformity.
*   **Persisted & Configurable:** Configured through `options.with_entropy_coefficient(beta)` (defaults to `0.0`, disabled), serialised/deserialised seamlessly via `NeuralNetworkSerializer`, and exposed in Python bindings.

### High-Performance Memory Allocator (`mimalloc`)

The library integrates [Microsoft's mimalloc](https://github.com/microsoft/mimalloc/) concurrent memory allocator:

*   **Thread-Local Free Lists:** Eliminates heap lock contention across multi-threaded batch operations (`TaskQueue`, multi-threaded GEMM, and parallel backward passes).
*   **Lock-Free Cross-Thread Deallocation:** Enables worker threads to pass and free gradient and hidden state buffers seamlessly without stalling other threads.
*   **Cache Locality:** Segregates allocations across 64KB pages to maximise L1/L2 cache hit rates and minimise TLB misses.
*   **AVX2 Alignment:** `AlignedAllocator` utilises `mi_malloc_aligned` for zero-overhead 32-byte alignment.
*   **Configurable & Optional:**
    *   Enabled by default via CMake: `ENABLE_MIMALLOC=ON` (defines `MYODDWEB_USE_MIMALLOC=1`).
    *   Can be disabled via CMake: `-DENABLE_MIMALLOC=OFF` to fall back cleanly to standard CRT (`_aligned_malloc` on Windows, `posix_memalign` on POSIX, and default C++ `new`/`delete`).


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
* `quantile_loss` (Pinball loss for single or multi-quantile regression)
* `sharpe_ratio_loss` (Negative Sharpe ratio loss for trading return optimization)
* `sortino_ratio_loss` (Negative Sortino ratio loss penalizing downside volatility)

### Calculating Metrics

You can calculate error metrics for the network's predictions using the `calculate_forecast_metrics` or `calculate_forecast_metrics_all_layers` methods:

```cpp
  // Calculate forecast metrics for the default output layer.
  // The 'in_sample' parameter defaults to true (evaluating on training data). Pass false to evaluate on validation/testing data.
  std::vector<NeuralNetworkHelperMetrics> metrics = nn.calculate_forecast_metrics({ ErrorCalculation::type::rmse }, /*in_sample=*/true);

  // Each NeuralNetworkHelperMetrics provides:
  // - error(): the metric ratio or loss value (double)
  // - error_type(): the evaluated ErrorCalculation::type
  // - numerator(): optional count of matching/confident occurrences (std::optional<size_t>) for ratio-based metrics
  // - denominator(): optional total count of evaluated samples (std::optional<size_t>) for ratio-based metrics
  // For continuous loss metrics (e.g., MSE, MAE, Huber), numerator() and denominator() return std::nullopt.

  // 'force_checking_indexes' can be configured model-wide via NeuralNetworkOptions::with_force_checking_indexes(true/false)
  // (default: false) or overridden per-call via std::optional<bool> force_checking_indexes (default: std::nullopt).
  // When resolved to true with in_sample = false, it evaluates against checking_indexes() (validation set) even after training completes.
  auto all_metrics = nn.calculate_forecast_metrics_all_layers(
    { ErrorCalculation::type::directional_accuracy, ErrorCalculation::type::prediction_coverage },
    /*in_sample=*/false,
    /*force_checking_indexes=*/true
  );
```

## Reinforcement Learning

In addition to traditional supervised training over fixed datasets (`train`), the library supports on-policy **Reinforcement Learning** via policy gradients (REINFORCE) using `train_with_advantages`.

### How It Works

* **Supervised Learning (`train`)** minimises empirical loss with respect to static target labels across fixed epochs, schedules learning rates, and caches epoch validation errors.
* **Reinforcement Learning (`train_with_advantages`)** performs a single on-policy update over trajectories collected during interaction with an environment.
* **Advantage Scaling**: For each sample $b$ in a batch, the output layer's prediction delta ($\hat{y} - y_{target}$) is scaled directly by the scalar advantage $A_b$:
  $$\nabla_\theta \mathcal{L}_{RL} = A_b \cdot \nabla_\theta \mathcal{L}_{CE}$$
  * A **positive advantage** ($A > 0$) increases the probability of taking the chosen action.
  * A **negative advantage** ($A < 0$) penalises the chosen action and decreases its probability.
  * A **zero advantage** ($A = 0$) leaves the weights unchanged.
* Hidden layer gradients are computed via standard backpropagation from the advantage-scaled output gradients, updating all upstream layers proportionally.
* Sub-batches are chunked and executed according to `options.batch_size()`.
* Model weights and optimiser velocity/momentum states persist across consecutive updates, allowing the agent to continuously learn online or episode-by-episode.

> [!NOTE]
> `train_with_advantages` requires a single output layer (multi-output architectures are not supported). Typically, a `Softmax` output head paired with `CrossEntropy` loss is used for discrete action spaces, with one-hot encoded action targets.

### How to Trigger in C++

```cpp
#include "neuralnetwork.h"
#include "neuralnetworkoptions.h"

using namespace myoddweb::nn;

// 1. Configure a policy network (e.g., 9 board inputs, 36 hidden units, 9 discrete actions)
NeuralNetworkOptions options = NeuralNetworkOptions::create({ 9, 36, 9 })
  .with_hidden_layers({
    LayerDetails(Layer::Architecture::FF, 36, activation(activation::method::relu, 0.0), 0.0, 0.0, OptimiserType::Adam, 0.9, false, 0, 0, 0, 0, 0, 0, 0)
  })
  .with_output_layer_details(
    OutputLayerDetails(9, activation(activation::method::softmax, 0.0, 1.0), ErrorCalculation::type::cross_entropy, EvaluationConfig(), 0.0, OptimiserType::Adam, 0.9)
  )
  .with_learning_rate(0.01)
  .with_batch_size(16)
  .build();

NeuralNetwork nn(options);

// 2. Collect states, chosen one-hot actions, and outcome advantages from an episode:
std::vector<std::vector<double>> states = { /* state at step 0 */, /* state at step 1 */ };
std::vector<std::vector<double>> actions = { /* one-hot action 0 */, /* one-hot action 1 */ };
std::vector<double> advantages = { 1.0, 1.0 }; // e.g. +1.0 for win, -1.0 for loss, +0.2 for draw

// 3. Trigger reinforcement learning policy update
nn.train_with_advantages(states, actions, advantages);
```

### How to Trigger in Python

```python
import neuralnetwork as nn

# 1. Build policy network
options = (
    nn.NeuralNetworkOptions.create([9, 36, 9])
    .with_hidden_layers([
        nn.LayerDetails(nn.LayerArchitecture.FF, 36, nn.Activation(nn.ActivationMethod.Relu, 0.0),
                        0.0, 0.0, nn.OptimiserType.Adam, 0.9)
    ])
    .with_output_layer_details(
        nn.OutputLayerDetails(9, nn.Activation(nn.ActivationMethod.Softmax, 0.0, 1.0),
                              nn.ErrorCalculationType.CrossEntropy, nn.EvaluationConfig(),
                              0.0, nn.OptimiserType.Adam, 0.9)
    )
    .with_learning_rate(0.01)
    .with_batch_size(16)
    .build()
)

net = nn.NeuralNetwork(options)

# 2. Collect trajectory and compute advantages
states = [[0.0] * 9]                # Initial empty board
actions = [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]  # Chose centre cell (4)
advantages = [1.0]                  # Positive reward / advantage

# 3. Update network policy
net.train_with_advantages(states, actions, advantages)
```

See [python/examples/tic_tac_toe.py](python/examples/tic_tac_toe.py) for a complete working implementation where an agent learns Tic-Tac-Toe and plays against a Random opponent, and [python/examples/gridworld.py](python/examples/gridworld.py) for an obstacle-avoiding navigation agent with visual ASCII path and policy map displays.

## Performance Optimization (SIMD)

To achieve high throughput during training and inference, this library leverages **Advanced Vector Extensions 2 (AVX2)** intrinsics for core mathematical operations (GEMM, dot products, and optimizer updates).

To enable these optimizations, ensure your compiler is configured to target the AVX2 instruction set:

*   **MSVC (Visual Studio):** Set `Enable Enhanced Instruction Set` to `Advanced Vector Extensions 2 (/arch:AVX2)` in the project properties.
*   **GCC / Clang:** Use the `-mavx2 -mfma` flags during compilation.

For more information on AVX2, see the [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) or [Wikipedia](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions).

## Repository Layout

*   `\include\neuralnetwork\`: The stand-alone core C++ neural network library (including `/layers/`, `/helpers/`, and `/common/` subdirectories).
*   `\examples\`: Standalone example implementations, runner (`main.cpp`), and the main Visual Studio solution (`neuralnetwork.sln`).
*   `\tests\`: Comprehensive unit test suite.
*   `\python\`: Pybind11-based Visual Studio 2022 solution (`neuralnetwork_py.sln`) and Python usage examples in `\python\examples\`.

## Building and Running

1.  Open `examples/neuralnetwork.sln` in Visual Studio 2022.
2.  Select `neuralnetwork` (to run the examples) or `neuralnetwork_tests` (to run unit tests) as the startup project.
3.  Build and run using the IDE.

## Technical Stack

* **Language:** C++17/C++20
* **Build Tool:** Visual Studio 2022
* **Dependencies:** Zero external dependencies for core logic.
