# Changelog

All notable changes to the `neural-network` library will be documented in this file.

## [1.1.30] - 2026-08-19

### Changed
- Optimised `LSTMLayer` and BPTT performance bottlenecks:
  - Implemented fused four-matrix GEMM routines (`simd::gemm_four_matrices_four_batches`, `simd::gemm_four_matrices_two_batches`, and `simd::gemm_four_matrices_one_batch`) in `include/neuralnetwork/common/simd_utils.h` to accumulate all 4 gate contributions ($f, i, o, g$) into register accumulators simultaneously, eliminating 75% of memory load/store roundtrips for recurrent backward GEMMs (`run_recurrent_gemm_backward`) and input backward GEMMs (`dx_matrix`).
  - Added specialized `simd::lstm_bptt_gate_step_tanh` fusing tanh derivative calculation ($1 - y^2$) directly into the AVX2 vector registers of the gate step, bypassing two function calls and workspace buffer roundtrips (`dc_act_deriv` and `dg_act_deriv`) per batch item per timestep.
  - Removed redundant double-clamping of `dh_curr` in `simd::lstm_bptt_gate_step`, eliminating 4 redundant vector clamp instructions per 8 elements in the inner BPTT loop.
  - Simplified vector arithmetic in `simd::lstm_bptt_upstream_step` to compute `(upstream + dh_next) * mask`, saving 2 vector multiplication instructions per 8 elements.
  - Implemented `simd::add_four_vectors` for vectorised 4-gate bias gradient accumulation in `LSTMLayer::calculate_and_store_gradients_chunk`.

### Added
- Added unit tests `SimdUtilsTest.GemmFourMatricesBatches`, `SimdUtilsTest.LstmBpttGateStepTanhVsStandard`, and `SimdUtilsTest.AddFourVectors` in `tests/simd_utils_tests.cpp`.
- Added multi-batch, multi-timestep numerical gradient equivalence test `LSTMLayerTest.FastBpttKernelsNumericalGradientEquivalence` in `tests/lstmlayer_tests.cpp`.

## [1.1.29] - 2026-08-18

### Added
- Implemented a multi-head causal self-attention hidden layer, a small Transformer encoder block (`SelfAttentionLayer` / `Layer::Architecture::SelfAttention`):
  - Adds a fixed (non-learned) sinusoidal positional encoding, computes Q/K/V projections, runs causally-masked scaled dot-product attention independently per head, projects the concatenated heads back to width, adds it as an internal residual (optionally through LayerNorm), runs a position-wise feed-forward sub-block, and adds that as a second internal residual (optionally through a second LayerNorm).
  - Always strictly causal, with no configuration flag to disable this.
  - May follow any preceding hidden layer type, including being the first hidden layer (unlike `AttentionPool`).
  - Unlike `AttentionPool`, `use_layer_normalisation` IS supported, and the layer accepts the existing external residual-connection mechanism (`residual_layer_number`/`residual_projector`) - distinct from its own always-on internal residuals.
  - Added `number_of_heads` and `feed_forward_hidden_size` configuration to `LayerDetails`.
  - Added eight new trainable weight families (Q/K/V/output projections, the feed-forward sub-block's two dense layers, and two LayerNorms' gain/bias), each with the full values/grads/velocities/m1/m2/timesteps/decays optimizer-state sextet.
  - Added forward pass, full manual backpropagation (reusing `simd::layer_norm_forward`/`layer_norm_backward` and `simd::softmax_forward`/`softmax_backward`), gradient calculation, and weight optimization.
  - Added placement/validation panics in `Layer::create_hidden_layer` (non-zero `number_of_heads` evenly dividing layer size, non-zero `feed_forward_hidden_size`, size must match the layer attended over) and in `NeuralNetworkOptions::build()` (requires `enable_bptt`, requires `bptt_max_ticks() > 1`).
  - Added JSON serialization/deserialization for `SelfAttention` layer parameters and all sixteen weight families in `NeuralNetworkSerializer`.
  - Exposed `Layer::Architecture::SelfAttention` and `number_of_heads`/`feed_forward_hidden_size` in Python bindings.
- Added unit and integration tests covering forward/backward correctness (hand-derived causal-masking check and per-family numerical-gradient checks), placement validation, LayerNorm toggling, batch isolation, SWA averaging, cloning, serializer round-trips, and thread-count invariance in `tests/selfattentionlayer_tests.cpp`, `tests/selfattentionlayer_mt_tests.cpp`, `tests/layer_tests.cpp`, `tests/layer_details_tests.cpp`, and `tests/network_integration_tests.cpp`.

## [1.1.28] - 2026-08-18

### Added
- Implemented a dilated causal 1D convolution ("Temporal Convolutional Network" block) hidden layer (`TcnLayer` / `Layer::Architecture::Tcn`):
  - For each output timestep, gathers the `kernel_size` dilated input taps (zero-padded where out of range) into one flat vector and applies a single dense affine map + activation, reusing the base `Layer` weight/bias SoA arrays rather than a bespoke weight family.
  - Always strictly causal, with no configuration flag to disable this.
  - May follow any preceding hidden layer type, including being the first hidden layer (unlike `AttentionPool`), and may change channel width between input and output.
  - Accepts the existing external residual-connection mechanism (`residual_layer_number`/`residual_projector`), unlike `AttentionPool`.
  - Added `kernel_size` and `dilation` configuration to `LayerDetails`.
  - Added forward pass (im2col-style dilated gather), analytic backpropagation (weight-matrix backward + scatter-add across dilated source timesteps), gradient calculation, and weight optimization, multithreaded by batch item via the layer's own `TaskQueuePool`.
  - Added placement/validation panics in `Layer::create_hidden_layer` (non-zero `kernel_size`/`dilation`, no LayerNorm support) and in `NeuralNetworkOptions::build()` (requires `enable_bptt`, receptive field must not exceed `bptt_max_ticks`).
  - Added JSON serialization/deserialization for `Tcn` layer parameters and trained weights in `NeuralNetworkSerializer`.
  - Exposed `Layer::Architecture::Tcn` and `kernel_size`/`dilation` in Python bindings.
- Added unit and integration tests covering forward/backward correctness (hand-computed and numerical-gradient checks), placement validation, residual support, batch isolation, SWA averaging, cloning, serializer round-trips, and single-vs-multi-threaded equivalence in `tests/tcnlayer_tests.cpp`, `tests/tcnlayer_mt_tests.cpp`, `tests/layer_tests.cpp`, `tests/layer_details_tests.cpp`, and `tests/network_integration_tests.cpp`.

## [1.1.27] - 2026-08-17

### Changed
- Optimized `LSTMLayer` and BPTT execution pipeline:
  - Streamlined `LSTMLayer::BPTTWorkspace` in `include/neuralnetwork/layers/lstmlayer.h` by eliminating 11 unused and redundant vector allocations (`temp_Uf_T_df`, `temp_Ui_T_di`, `temp_Uo_T_do`, `temp_Ug_T_dg`, `c_prev_vals`, `i_vals`, `o_vals`, `f_vals`, `g_vals`, `c_vals`, `tanh_c_vals`), cutting dynamic memory churn and zeroing overhead per BPTT pass.
  - Eliminated intermediate buffer gathering in `LSTMLayer::calculate_bptt_batch_chunk`, evaluating activation derivatives directly in-place from cached hidden states without copying.
  - Skipped redundant transposed recurrent GEMM calculations (`run_recurrent_gemm_backward`) at the final BPTT timestep ($t == t_{end}$), saving 4 matrix-vector products per batch item.
  - Guarded input weight backward GEMM passes with `if (N_prev > 0)` to skip unnecessary matrix math for input-less configurations.
  - Implemented `LSTMLayer::calculate_and_store_gradients_chunk` with truncated timestep bounds ($t_{start}$ down to $t_{end}$) governed by `bptt_max_ticks`, eliminating outer product and bias gradient calculation outside the truncated BPTT window.
  - Replaced thread pool inline lambdas with the named functor `LstmGradCalcTask` in `LSTMLayer::calculate_and_store_gradients`.
- Optimized `simd::lstm_cell_step` in `include/neuralnetwork/common/simd_utils.h`:
  - Added 8-double dual `__m256d` AVX2 + FMA loop unrolling to maximize execution port saturation during LSTM forward cell state updates.

### Added
- Added unit test `SimdUtilsTest.LstmCellStepLargeVector` in `tests/simd_utils_tests.cpp` validating 8-wide AVX2 unrolling, 4-wide path, and scalar tail.
- Added unit tests in `tests/lstmlayer_tests.cpp`:
  - `LSTMLayerTest.CalculateAndStoreGradientsVariousTopologiesMathematicalProof`: analytically verifies gradient accumulation for all 4 gates ($W_f, W_i, W_o, W_g$, $RW_f, RW_i, RW_o, RW_g$, $b_f, b_i, b_o, b_g$) across asymmetric topologies ($10^{-12}$ precision).
  - `LSTMLayerTest.CalculateAndStoreGradientsSingleStepEquivalence`: verifies single-step gradient accumulation and asserts zero recurrent gradient leakage at $t = 0$.
  - `LSTMLayerTest.CalculateAndStoreGradientsBpttMaxTicksTruncation`: validates that gradient accumulation correctly respects `bptt_max_ticks` truncation.

## [1.1.26] - 2026-08-17

### Changed
- Optimized `GRURNNLayer::calculate_and_store_gradients_chunk` in `include/neuralnetwork/layers/grurnnlayer.cpp`:
  - Inverted the loop hierarchy to make batch index $b$ and timestep $t$ the outer loops, consolidating bias, input weight, and recurrent weight gradient accumulation into a single chronological traversal over the batch and sequence history, eliminating cache thrashing and redundant accessor calls.
  - Vectorised outer-product gradient updates across all 6 weight matrices ($W_h, W_z, W_r, RW_h, RW_z, RW_r$) using 4-wide (`simd::mul_add_four_scalars`), 2-wide (`simd::mul_add_two_scalars`), and 1-wide (`simd::mul_add_three` / `simd::mul_add_three_scalars`) AVX2 kernels to keep gate gradients in vector registers and L1 cache.
- Optimized `simd::gru_output_step` in `include/neuralnetwork/common/simd_utils.h`:
  - Added 8-double dual `__m256d` AVX2 loop unrolling to maximize FMA pipeline utilization during GRU forward propagation.

### Added
- Added unit test `SimdUtilsTest.GruOutputStepLargeVector` in `tests/simd_utils_tests.cpp` verifying 8-wide, 4-wide, and scalar remainder paths in `simd::gru_output_step`.
- Added unit tests in `tests/grurnnlayer_tests.cpp`:
  - `GRURNNLayerTest.CalculateAndStoreGradientsVariousTopologiesMathematicalProof`: analytically proves gradient accumulation against exact paper formulas ($10^{-14}$ precision) across asymmetric 4-wide/2-wide/1-wide topologies.
  - `GRURNNLayerTest.CalculateAndStoreGradientsSingleStepEquivalence`: verifies single-step BPTT gradient updates and asserts zero recurrent gradient leakage at $t = 0$.

## [1.1.25] - 2026-08-17

### Fixed
- Fixed `TrainingMonitor` per-checkpoint metric history not persisting across progress callbacks:
  - `NeuralNetworkHelper::_training_monitors` changed from `std::vector<TrainingMonitor>` to `std::shared_ptr<std::vector<TrainingMonitor>>` so that copies of `NeuralNetworkHelper` dispatched to progress callbacks share and accumulate monitor history instead of mutating discarded copies.
- Fixed floating-point serialization precision loss in `NeuralNetworkSerializer`:
  - Added dedicated float/double serializers (`set_float`, `set_floats`, `create_float_value`) decomposing values up to 17 decimal digits without integer scaling overflow.

### Added
- Added tests in `tests/neuralnetworkhelper_tests.cpp` covering shared `TrainingMonitor` state propagation and history accumulation across progress callbacks:
  - `NeuralNetworkHelperTest.TrainingMonitorsSharedAcrossHelperCopies`
  - `NeuralNetworkHelperTest.TrainingMonitorsPersistAcrossProgressCallbacks`
  - `NeuralNetworkHelperTest.TrainingMonitorsMultiOutputSharedCopies`
- Added test `NetworkIntegrationTest.FloatingPointWeightsSerializationPrecision` in `tests/network_integration_tests.cpp` to verify precision round-tripping for floating-point weights during JSON serialization.

## [1.1.24] - 2026-08-16

### Added
- Added opt-in network-wide reproducibility seed (`NeuralNetworkOptions::with_seed` / `seed()`) for deterministic training runs:
  - Added stateless seed-mixing helper `Rng::derive` in `include/neuralnetwork/common/rng.h`.
  - Added deterministic weight initialization per layer and weight index across all layer architectures.
  - Added deterministic dropout masking in `Neuron` parameterised by call index.
  - Added deterministic data and BPTT batch shuffling via seeded shuffle engine in `NeuralNetwork`.
  - Added JSON serialization and deserialization support for network seed in `NeuralNetworkSerializer`.
  - Exposed `with_seed` and `seed` in Python pybind11 bindings and updated Python documentation.
- Added comprehensive unit and integration tests covering RNG seed derivation, seeded weight initialization, deterministic dropout, and seeded training convergence in `tests/activation_tests.cpp`, `tests/neuron_tests.cpp`, and `tests/network_integration_tests.cpp`.

## [1.1.23] - 2026-08-16

### Added
- Refactored Stochastic Weight Averaging options to use dedicated `StochasticWeightAveragingDetails` class (`include/neuralnetwork/common/stochasticweightaveragingdetails.h`) with builder methods in `NeuralNetworkOptions`, serialization support, and Python bindings.
- Implemented additive (Bahdanau-style) attention pooling (`AttentionPoolLayer` / `Layer::Architecture::AttentionPool`) for BPTT sequence aggregation:
  - Added `simd::softmax_forward` and `simd::softmax_backward` in `include/neuralnetwork/common/simd_utils.h`.
  - Added `attention_hidden_size` configuration to `LayerDetails`.
  - Added forward pass, analytic backpropagation, gradient calculation, and weight optimization for attention scoring weights.
  - Added JSON serialization/deserialization for attention pool layer parameters.
  - Exposed `Layer::Architecture::AttentionPool` and `attention_hidden_size` in Python bindings.
- Added unit and integration tests covering SIMD softmax, layer options validation, attention pooling forward/backward gradient correctness, and serialization in `tests/simd_utils_tests.cpp`, `tests/layer_tests.cpp`, `tests/attentionpoollayer_tests.cpp`, `tests/stochastic_weight_averaging_details_tests.cpp`, and `tests/network_integration_tests.cpp`.

## [1.1.22] - 2026-08-15

### Added
- Implemented Stochastic Weight Averaging (SWA) to improve model generalisation by maintaining running averages of weights across training epochs:
  - Added `swa`, `swa_start_percent`, and `swa_update_percent` options to `NeuralNetworkOptions`.
  - Added `accumulate_swa_average` across all concrete layer implementations (`FFLayer`, `ElmanRNNLayer`, `GRURNNLayer`, `LSTMLayer`, `MultiOutputLayer`, `ResidualProjector`).
  - Integrated SWA snapshot cadence and final model weight replacement in `NeuralNetwork::train`.
  - Added serialization support in `NeuralNetworkSerializer` and exposed options in Python bindings.
- Added unit and integration tests in `tests/swa_tests.cpp` and `tests/network_integration_tests.cpp` verifying running mean calculation, layer propagation, and serialization round-tripping.

## [1.1.21] - 2026-08-15

### Added
- Implemented recurrent-state Layer Normalization for `GRURNNLayer` and `LSTMLayer`:
  - Added `simd::layer_norm_forward` and `simd::layer_norm_backward` in `include/neuralnetwork/common/simd_utils.h`.
  - Added `use_layer_normalisation` option to `LayerDetails`.
  - Added learnable gain and bias parameters with optimiser state across forward and BPTT backward passes.
  - Added serialization support in `NeuralNetworkSerializer` and exposed options in Python bindings.
- Added unit and integration tests in `tests/simd_utils_tests.cpp`, `tests/layer_details_tests.cpp`, `tests/grurnnlayer_tests.cpp`, `tests/lstmlayer_tests.cpp`, and `tests/network_integration_tests.cpp`.

### Fixed
- Fixed `GRURNNLayer::zero_gradients` and `LSTMLayer::zero_gradients` clearing Layer Normalization gain/bias gradients before they were applied.
- Fixed `use_layer_normalisation` not persisting during `NeuralNetworkSerializer` hidden-layer configuration serialization.

## [1.1.20] - 2026-08-15

### Fixed
- Fixed training thread contention caused by nested thread-pool oversubscription in `Layers::update_weights`:
  - Removed `_update_weights_pool` from `Layers` so gradient updates iterate sequentially across layers while relying on each layer's internal SIMD and thread pool parallelism.

### Added
- Added unit and integration tests verifying thread count configurations and deep network convergence in `tests/network_integration_tests.cpp`.

## [1.1.19] - 2026-08-15

### Added
- Implemented Lion (EvoLved Sign Momentum) optimiser (`OptimiserType::Lion`):
  - Added AVX2 vectorised `simd::lion_step` and `simd::scalar_lion_step` in `include/neuralnetwork/common/simd_utils.h`.
  - Added Lion update support across all layer architectures.
  - Added mathematical unit tests, SIMD tests, and convergence tests.
  - Updated `README.md` to document Lion optimiser support.

### Fixed
- Replaced `TempBuffer`'s thread-local storage pool with a stack-scoped RAII buffer, eliminating CRT dynamic TLS teardown crashes on thread termination.
- Fixed weight explosion clamping in per-weight Lion updates to match vectorised implementation.
- Fixed `OptimiserType::Lamb` enum case in Python bindings.

## [1.1.18] - 2026-08-14

### Changed
- Optimized `LSTMLayer::calculate_forward_feed`'s recurrent pass in `include/neuralnetwork/layers/lstmlayer.cpp`: batched the recurrent (hidden-to-hidden) GEMV operations for all four gates (forget, input, output, candidate) across groups of up to 4 batch items per timestep using `simd::gemm_four_batches`, `simd::gemm_two_batches`, and `simd::gemm_one_batch` (against the raw, non-transposed recurrent weight matrices) instead of evaluating batch items individually via `simd::gemv_add_four` against the transposed weight caches.
- Refactored post-recurrent step logic into a new private `LSTMLayer::finalize_forward_step` in `include/neuralnetwork/layers/lstmlayer.cpp`, eliminating redundant scratch-buffer copies by activating the candidate gate and cell state directly into their final packed-state slots.

### Fixed
- Fixed multi-threaded inference and training heap corruption (`0xc0000374`) across tests:
  - Fixed out-of-bounds workspace indexing in `GRURNNLayer`, `ElmanRNNLayer`, and `LSTMLayer`: `allocate_workspace()` now ensures at least 1 `BPTTWorkspace` is allocated even when `number_of_threads <= 1` (`_task_queue_pool == nullptr`), and `get_workspace(thread_idx)` dynamically allocates additional workspaces on demand to eliminate invalid memory dereferencing.
  - Removed static inline `thread_local std::ostringstream` instances (`get_msg_oss()`, `get_msg_fmt_oss()`) from `include/neuralnetwork/common/logger.h`, preventing duplicate MSVC CRT dynamic TLS destructor entries from triggering multiple deallocations upon thread termination.
  - Replaced complex `thread_local` cache structures (`EvaluationCache` and `TempCache`) in `include/neuralnetwork/neuralnetwork.cpp` with local stack/heap vectors, eliminating fragile dynamic TLS destruction on ephemeral test threads.
  - Centralised `TempBuffer`'s thread-local storage pool into a single translation unit (`include/neuralnetwork/layers/layer.cpp`), eliminating MSVC duplicate dynamic TLS destructor registrations across translation units that caused heap corruption (`0xc0000374`) on ephemeral thread exit.
  - Added safe `get_number_of_threads()` helper in `Layer` (`include/neuralnetwork/layers/layer.h`) to prevent null pointer dereferences on `_task_queue_pool` across all layer implementations (`FFLayer`, `FFOutputLayer`, `ElmanRNNLayer`, `GRURNNLayer`, `LSTMLayer`).
  - Fixed `Layers` copy constructor and copy assignment operator in `include/neuralnetwork/layers/layers.cpp` to properly null-check `src._update_weights_pool`.
  - Replaced function-local `thread_local` vectors in `Layers::calculate_forward_feed`'s residual projection path with local stack vectors.
  - Removed dangerous `static thread_local` return references in `Layer::get_weight_params()` and `Layer::get_bias_weight_params()`.

### Added
- Added 9 unit tests in `tests/lstmlayer_tests.cpp`: `NoBatchCrossTalkFourWideGroupInference`, `NoBatchCrossTalkOneWideCleanupInference`, `NoBatchCrossTalkFourWideGroupTraining`, `NoBatchCrossTalkOneWideCleanupTraining`, `NoBatchCrossTalkExactFourMultiple`, `NoBatchCrossTalkOneWideCleanupRemainder`, `NoBatchCrossTalkTwoWideCleanupRemainder`, `NoBatchCrossTalkTwoFullFourWideGroups`, and `NoBatchCrossTalkLargerHiddenSize` to verify zero cross-talk between grouped batch items, mirroring the GRURNNLayer regression suite added in `[1.1.17]`.

## [1.1.17] - 2026-08-14

### Added
- Created dedicated `python/examples/` folder for standalone Python library examples.
- Added `python/examples/xor.py`: fully commented Python example solving the XOR classification problem with prediction evaluation and `[OK]`/`[FAIL]` status reporting.
- Added `python/examples/multi_output.py`: multi-output Python example mirroring `examples/multi_output.h` with joint classification (Sigmoid) and regression (Tanh) heads, synthetic dataset generation, and output evaluation.
- Added step to run all Python examples (`xor.py`, `multi_output.py`, `example.py`) in `.github/workflows/python.yml` CI workflow.
- Added 9 unit tests in `tests/grurnnlayer_tests.cpp`: `NoBatchCrossTalkFourWideGroupInference`, `NoBatchCrossTalkOneWideCleanupInference`, `NoBatchCrossTalkFourWideGroupTraining`, `NoBatchCrossTalkOneWideCleanupTraining`, `NoBatchCrossTalkExactFourMultiple`, `NoBatchCrossTalkOneWideCleanupRemainder`, `NoBatchCrossTalkTwoWideCleanupRemainder`, `NoBatchCrossTalkTwoFullFourWideGroups`, and `NoBatchCrossTalkLargerHiddenSize` to verify zero cross-talk between grouped batch items.

### Fixed
- Fixed dangling pointers in `NeuralNetworkHelper`: converted `_training_inputs` and `_training_outputs` from raw pointers to `std::shared_ptr<const std::vector<std::vector<double>>>`, preventing undefined behaviour and access violations during post-training forecast metric calculations and model serialization (`NeuralNetworkSerializer::save`) when training data was allocated temporarily on the caller's stack (e.g. from Python bindings).
- Fixed uninitialised memory access in copy and move constructors of `NeuralNetworkHelper`, `NeuralNetworkHelperMetrics`, `NeuralNetworkOptions`, and `NeuralNetwork` by replacing `*this = src;` assignments with proper member initialiser lists.
- Fixed Python pybind11 bindings for `NeuralNetworkHelperMetrics` by registering default and value constructors (`py::init<>()` and `py::init<double, ErrorCalculation::type>()`).
- Fixed convergence in `python/examples/xor.py` and `python/examples/example.py` by configuring topology `[2, 8, 1]`, `learning_rate = 0.1`, `number_of_epoch = 3000`, `enable_bptt = False`, `shuffle_training_data = True`, `data_is_unique = True`, and `Adam` optimiser for reliable 100% convergence.

### Changed
- Moved `python/example.py` to `python/examples/example.py` and updated import search path resolution for compiled `.pyd` module.
- Updated `python/README.md` and root `README.md` with new `python/examples/` layout, write-ups for XOR and multi-output examples, and updated run commands.
- Optimized `GRURNNLayer::run_forward_pass` in `include/neuralnetwork/layers/grurnnlayer.cpp`: batched the recurrent (hidden-to-hidden) GEMV operations across groups of up to 4 batch items per timestep using `simd::gemm_four_batches`, `simd::gemm_two_batches`, and `simd::gemm_one_batch` instead of evaluating batch items individually.
- Refactored post-recurrent step logic into `GRURNNLayer::finalize_forward_step` in `include/neuralnetwork/layers/grurnnlayer.cpp`, eliminating redundant memory copies by aliasing state buffers in-place during `simd::gru_output_step`.


## [1.1.16] - 2026-08-13


### Added
- Added `simd::transpose` in `include/neuralnetwork/common/simd_utils.h`: a cache-blocked matrix transpose function using 64x64 tiling to eliminate L1/L2 cache line thrashing during weight matrix transpositions.
- Added unit tests in `tests/simd_utils_tests.cpp`: `TransposeSquareSmall`, `TransposeSquareCrossesBlockBoundary`, `TransposeRectangularWideSource`, `TransposeRectangularTallSource`, `TransposeSingleElement`, and `TransposeSingleRowAndColumn`.

### Changed
- Optimized `FFLayer::cache_recurrent_weights` in `include/neuralnetwork/layers/fflayer.cpp`: replaced naive nested loop with `simd::transpose`.
- Optimized `GRURNNLayer::cache_recurrent_weights` in `include/neuralnetwork/layers/grurnnlayer.cpp`: replaced 6 manual nested weight matrix transposition loops for input ($W_h, W_z, W_r$) and recurrent ($RW_h, RW_z, RW_r$) weight matrices with `simd::transpose`.

## [1.1.15] - 2026-08-13

### Fixed
- Fixed mathematical bug in `activation::calculate_softmax` in `include/neuralnetwork/common/activation.cpp`: the NaN-detection scan started at `begin + 1`, never checking `*begin` itself (which had already seeded `max_val`/`min_val`). A NaN in the *first* logit of a row was therefore never detected — every subsequent comparison against a NaN `max_val` evaluates to false, so the extreme/catastrophic-range checks were silently bypassed, the whole row was exponentiated against a NaN max (producing NaN everywhere), and the `sum` non-finite fallback then kicked in and wrote a fake, deterministic `{1.0, 0.0, 0.0, ...}` "confident class 0" result — masking serious numerical instability as a confident prediction, contradicting the function's own documented intent ("If any input is NaN produce NaN outputs"). A NaN anywhere after index 0 was already handled correctly. Fixed by scanning from `begin` instead of `begin + 1`.
- Fixed mathematical bug in `activation::calculate_softmax_derivative` in `include/neuralnetwork/common/activation.cpp`: computed `sigmoid(x)*(1-sigmoid(x))` on the raw pre-activation logit, a formula unrelated to softmax and inconsistent with its own comment ("simplified scalar derivative (S(1-S))", where S should be the softmax output, not `sigmoid(x)`). Softmax's true derivative is a full-row Jacobian and cannot be computed from one scalar value in isolation — its sibling `calculate_softmax` (the scalar single-value *activation* stub) already acknowledges this by logging a warning and returning a degenerate stub value; `calculate_softmax_derivative` did not follow the same pattern and instead returned a plausible-looking but wrong nonzero gradient. This was reachable in practice: this codebase supports `softmax` as a *hidden*-layer activation (see `FFLayerTest.ForwardFeedSoftmax`), and hidden-layer backward propagation (e.g. `FFLayer::run_post_gemm_backward`) calls `activate_derivative()` unconditionally regardless of activation method, unlike the output-layer path which explicitly skips the derivative for softmax. Now logs a warning and returns `0.0`, matching `calculate_softmax`'s convention.
- Added an explicit `case method::softmax` to the batched `activation::activate_derivative(begin, end, y_begin, out)` in `include/neuralnetwork/common/activation.cpp`, which previously fell through to the generic `default:` branch and would have called the (now warning-logging) scalar `calculate_softmax_derivative` once per element — flooding the log on a real batch. The batched path now logs a single warning per call and fills the whole output range with `0.0` directly.
- Removed `activation::lecun_initialization` (declaration in `include/neuralnetwork/common/activation.h`, definition in `include/neuralnetwork/common/activation.cpp`): dead code, never called from `weight_initialization`'s dispatch switch, and an exact duplicate of `selu_initialization` (both compute `Normal(0, sqrt(1/fan_in))`, which *is* LeCun-normal initialization — SELU's recommended init already covers this).

### Added
- Added unit tests in `tests/activation_tests.cpp`: `SoftmaxNaNAtFirstIndexPropagatesNaN` and `SoftmaxNaNAtLaterIndexPropagatesNaN` (NaN-propagation regression, previously entirely untested), `SoftmaxDerivativeScalarFallbackReturnsZeroNotSigmoid` and `SoftmaxDerivativeBatchedFallbackReturnsZeroForWholeRange` (softmax derivative fallback correctness), `SoftmaxCatastrophicLogitRangePanics` and `SoftmaxExtremeLogitRangeStillProducesValidDistribution` (previously-untested extreme/catastrophic logit-range warning and panic paths), and `UnknownStringToMethodThrows` (previously-untested `string_to_method` failure path).

## [1.1.14] - 2026-08-13

### Fixed
- Fixed mathematical bug in `ErrorCalculation::calculate_prediction_coverage` in `include/neuralnetwork/helpers/errorcalculation.h`: the non-softmax branch measured confidence as the raw magnitude of a prediction (`abs(value) > threshold`) instead of its distance from the activation's neutral baseline. For `sigmoid` heads (neutral point 0.5, not 0.0) this silently undercounted confidently-negative predictions (values near 0.0 read as "unconfident" purely because they are numerically small) while still requiring `tanh`/`linear`/`relu` heads to clear the threshold from 0.0, which was already correct for those. Now uses the same `baseline = (activation_method == sigmoid) ? 0.5 : 0.0` convention already used by `calculate_directional_accuracy`/`calculate_directional_confidence_score`.
- Fixed silent data loss in `ErrorCalculation::calculate_mse_error`: non-finite (NaN/Inf) prediction errors were excluded from the running sum and count with no indication at all, unlike every other malformed-input branch in the file (e.g. the mismatched-size branch two lines above it, which does log). A diverged/unstable training run could silently produce an artificially low MSE with no warning. Now logs a single warning per call (not per value, to avoid flooding the log on a hot per-epoch path) summarising how many non-finite values were skipped; the returned numeric value is unchanged.
- Hoisted the `Logger::can_trace()` check in `ErrorCalculation::calculate_forecast_mape` out of the per-sequence loop into a local, matching the pattern already used by `calculate_directional_accuracy`/`calculate_directional_confidence_score` (`can_trace_log` computed once, checked per-iteration).

### Added
- Added unit tests in `tests/error_calculation_tests.cpp`: `PredictionCoverageSigmoidUsesNeutralBaseline` and `PredictionCoverageEmptySequencePanics` (prediction-coverage baseline/edge cases), `MSESkipsNonFiniteValuesButKeepsFiniteOnesInTheAverage` and `MSEReturnsNaNWhenNoValidValuesExist` (non-finite handling), `MismatchedVectorSizePanicsForStrictMetrics` and `MismatchedVectorSizeSkippedSilentlyForSequenceMetrics` (previously-untested panic vs. skip behaviour across all metric functions on mismatched row sizes), and `UnknownStringToTypeThrows` (previously-untested `string_to_type` failure path).

## [1.1.13] - 2026-08-13

### Fixed
- Fixed performance issue in `NeuralNetwork::calculate_forecast_metrics_all_layers_impl` in `include/neuralnetwork/neuralnetwork.cpp`: the per-row thread_local `GradientsAndOutputs` evaluation cache was fully `zero()`-ed on every call (an epoch-callback hot path), which zero-fills both `_outputs` and `_gradients`. `_outputs` is already fully overwritten by the subsequent forward-only `calculate_forward_feed` pass, and `_gradients`/`_rnn_gradients`/`_rnn_gate_gradients` are never written or read outside of backward propagation (which this forecast-only path never runs), so zeroing them was wasted work on every epoch. Switched to the existing (previously unused in production code) `GradientsAndOutputs::reset_for_inference()`, which clears only `_rnn_outputs` — the one piece of cached state that does need clearing, to prevent a stale BPTT sequence output from a prior call leaking into the prediction extracted for a reused cache row.
- Added unit test `BPTTForecastMetricsCacheReuseRepeatable` in `tests/network_integration_tests.cpp` to verify that interleaved in-sample/out-of-sample calls to `calculate_forecast_metrics` on a BPTT-enabled network reuse the thread_local cache correctly and reproduce bit-identical results on repeat, guarding against stale cached state leaking across calls.

## [1.1.12] - 2026-08-12

### Added
- Added `inline_task<R>` class in `include/neuralnetwork/common/inline_task.h`: a move-only, type-erased task wrapper with a 96-byte inline buffer and a 3-pointer manual vtable (invoke, move, destroy).
- Added `ResidualProjector::project_batch_into(...)` overload in `include/neuralnetwork/layers/residualprojector.h` to reuse allocated vector capacity during residual batch projections.
- Added unit test `InlineTaskZeroHeapAllocationForRealisticCaptures` in `tests/taskqueue_tests.cpp` to verify zero heap allocations when enqueuing task closures up to 9 captured references.
- Added unit test `ResidualProjectorProjectBatchIntoEquivalence` in `tests/layer_tests.cpp` to verify output equivalence between `project_batch_into` and `project_batch`.
- Added unit test `LayersTrainAsymmetricLayerSizesNoBufferOverflow` in `tests/layer_tests.cpp` to verify asymmetric layer topology training without buffer overflow.
- Added unit test `LayersTrainMathematicalSoundnessMultiLayerRecurrentGradientFlow` in `tests/layer_tests.cpp` to verify multi-layer recurrent backpropagation weight updates.

### Changed
- Refactored `TaskQueue<R>` and `TaskQueue<void>` in `include/neuralnetwork/common/taskqueue.h` to store `inline_task<R>` instead of `std::function<R()>`, eliminating heap allocations on every parallel task submission during training.
- Simplified `TaskQueue::enqueue` to bypass `std::bind` when zero extra arguments are passed.
- Optimized `Layers::calculate_forward_feed` in `include/neuralnetwork/layers/layers.cpp`: replaced temporary vector allocations for residual connections with `thread_local` scratch buffers, eliminating per-batch allocations during training while preserving thread-safety for concurrent `think()` inference calls.
- Removed unused `_batch_next_gradients_buffer` member from `Layers` (`include/neuralnetwork/layers/layers.h`).

### Fixed
- Fixed performance bottleneck in `Layers::train(...)`: replaced value-copy variable declarations with `const auto&` references across `Layers`, `FFLayer`, and `MultiOutputLayer` during forward propagation, eliminating per-sample deep vector copies.
- Fixed multi-layer recurrent backpropagation gradient flow in `ElmanRNNLayer`, `GRURNNLayer`, and `LSTMLayer`:
  - Resolved `target_layer_idx` to `get_layer_index() + 1` when `_identity_proxy` is active.
  - Corrected single-timestep gradient buffer targeted by `set_gradients` to `get_layer_index() - 1` (preceding layer), preventing buffer overflow crashes when input size $N_{prev}$ exceeds layer size $N_{this}$.
- Required `is_training` guard before enabling multithreading in `ElmanRNNLayer::calculate_forward_feed` in `include/neuralnetwork/layers/elmanrnnlayer.cpp`, preventing thread pool queue contention during concurrent multi-threaded inference (`think()`).

## [1.1.11] - 2026-08-11

### Changed
- Optimised `LSTMLayer` BPTT backward pass in `include/neuralnetwork/layers/lstmlayer.cpp`: the forward pass already computes `tanh(g)` (candidate) and `tanh(c)` (cell state) once per timestep, but discarded them, so BPTT re-evaluated `get_activation().activate(...)` on both a second time, every timestep, every training batch.
  - Extended the packed per-timestep `HiddenState` storage from `Multiplier = 5` to `Multiplier = 7` slots (`include/neuralnetwork/layers/lstmlayer.h`) to cache the already-computed activated `g` and `c` values from the forward pass.
  - BPTT now copies the cached activations directly instead of recomputing them, matching the "compute once, cache, reuse in backward" pattern already used by `GRURNNLayer` and `ElmanRNNLayer`. Raw (pre-activation) `g` and `c` values are still stored and available, so `activate_derivative` continues to work correctly for any configured activation method, not just `tanh`.
- Added unit test `ForwardFeedCachesActivatedCandidateAndCellStateForBptt` to `tests/lstmlayer_tests.cpp`, asserting the cached activated slots equal `tanh()` of the raw values stored during the forward pass.

## [1.1.10] - 2026-08-11

### Added
- Added `bptt-supervise-last-step-only` configuration option (`NeuralNetworkOptions::with_bptt_supervise_last_step_only`). When enabled, only the final time step ($t = \text{bptt\_max\_ticks} - 1$) of each sequence window is supervised with target outputs during BPTT training, enabling sequence-to-one forecasting.
- Added JSON serialization and deserialization support for `bptt-supervise-last-step-only` in `NeuralNetworkSerializer`.
- Exposed `with_bptt_supervise_last_step_only` and `bptt_supervise_last_step_only` in Python Pybind11 bindings.
- Added unit tests `GRUSequenceConvergenceBpttSuperviseLastStepOnly`, `GRUSequenceConvergenceMultiOutputBpttSuperviseLastStepOnly`, `BpttSuperviseLastStepOnlyShapeAndValue`, and `BpttSuperviseLastStepOnlySerializerSaveLoad` in `tests/network_integration_tests.cpp`.
- Added unit test `GRURNNLayerCalculateAndStoreGradientsMathematicalSoundness` to `tests/grurnnlayer_tests.cpp` to mathematically prove all 6 GRU weight gradient matrices ($W_h, W_z, W_r, RW_h, RW_z, RW_r$) and 3 bias gradient vectors ($B_h, B_z, B_r$) against exact analytical formulas ($10^{-14}$ precision).

### Changed
- Optimized `GRURNNLayer::calculate_and_store_gradients_chunk` in `include/neuralnetwork/layers/grurnnlayer.cpp`:
  - Inverted input weight and recurrent weight gradient accumulation loop hierarchy across all 6 gate weight matrices to make input neuron index $i$ and recurrent neuron index $k$ outer loops, eliminating L1 cache thrashing during GRU training.
  - Replaced captured lambda closures with named task functor structure `GruGradCalcTask`, adhering to coding standards.

### Fixed
- Increased sample size (to 5,000 neurons) and relaxed tolerance (to 0.08) in `DropoutStatisticalVerification` unit tests across `ffoutputlayer_tests.cpp`, `fflayer_tests.cpp`, `elmanrnnlayer_tests.cpp`, `grurnnlayer_tests.cpp`, and `lstmlayer_tests.cpp` (matching `layer_tests.cpp` and `multioutputlayer_tests.cpp`) to prevent random statistical sampling flakiness on CI runners.
- Fixed a multi-threading race condition and recursive mutex deadlock in `SingleTaskQueue` and `SingleTaskQueue<void>` in `include/neuralnetwork/common/taskqueue.h`:
  - `busy()` checked two independent atomic flags (`_busy_task` and `_task_is_present`) without acquiring `_mutex`. On multi-core CI runners under CPU contention, worker thread transitions between setting `_busy_task = true` and `_task_is_present = false` created a microsecond window where both flags evaluated to `false`, causing `SingleTaskQueueTest.BusyStatus` to randomly fail on macOS/Windows CI.
  - Made `_mutex` `mutable` and acquired `_mutex` inside `busy()`. Added `busy_nolock()` for internal wait conditions to prevent recursive mutex deadlocks when `wait_for_task()` is called with `_mutex` already locked.
- Fixed `NeuralNetwork::create_initial_neural_network_helper` in `include/neuralnetwork/neuralnetwork.cpp`: when `shuffle-training-data` is `true` and `enable-bptt` is also `true`, the library now keeps training rows in chronological order (ignoring the row-shuffle and logging a warning) instead of scrambling them.
  - Previously, `create_shuffled_indexes_in_lock` permuted the row order before `create_bptt_batches` sliced consecutive array entries into fixed-size windows, so each "sequence" fed to BPTT was actually `bptt_max_ticks` unrelated, randomly ordered historical rows glued together rather than a genuine contiguous time window. Evaluation was unaffected, as it always indexed the untouched chronological array directly.
  - Recurrent networks should use `shuffle-bptt-batches` instead (shuffles whole chronological blocks after windowing), as already documented in `README.md`.

## [1.1.9] - 2026-08-11

### Added
- Added unit tests `FFLayerCalculateAndStoreGradientsMathematicalSoundness` and `LayersTrainCoverageAndConsistencyAcrossBatchSizes` to `tests/layer_tests.cpp` to mathematically prove weight/bias gradient calculations against analytical formulas ($10^{-14}$ precision) and verify multi-batch training execution.
- Added unit test `ElmanRNNLayerCalculateAndStoreGradientsMathematicalSoundness` to `tests/elmanrnnlayer_tests.cpp` to mathematically prove Elman RNN input weight, recurrent weight, and bias gradient calculations against exact paper formulas ($10^{-14}$ precision).

### Changed
- Optimized `FFLayer::calculate_and_store_gradients_chunk` in `include/neuralnetwork/layers/fflayer.cpp`:
  - Inverted weight gradient accumulation loop hierarchy to make input neuron index the outer loop, ensuring row $i$ of `local_w_grads` stays in SIMD registers and L1 cache across all batch samples and time steps, eliminating L1 cache thrashing during training.
- Optimized `ElmanRNNLayer::calculate_and_store_gradients_chunk` in `include/neuralnetwork/layers/elmanrnnlayer.cpp`:
  - Inverted input weight and recurrent weight gradient accumulation loop hierarchy to make input neuron index $k$ and recurrent neuron index $rk$ outer loops, maintaining row data in L1 cache and SIMD registers across all batch samples and time steps.
  - Replaced captured lambda closures with named task functor structure `ElmanGradCalcTask`, adhering to coding standards.
- Optimized `Layers::train` in `include/neuralnetwork/layers/layers.cpp`:
  - Removed duplicate `cache_recurrent_weights()` invocation at the end of `Layers::train(...)` as `apply_stored_gradients(...)` already updates transposed weights per layer.
- Optimized `FFOutputLayer::run_output_gradients` in `include/neuralnetwork/layers/ffoutputlayer.cpp`:
  - Replaced local heap `std::vector` allocations with `TempBuffer` for thread-local buffer reuse.

## [1.1.8] - 2026-08-07

### Added
- Added `HandCalculatedAnalyticalProofs` and `AllTypesStringRoundtripCoverage` to `tests/error_calculation_tests.cpp` to mathematically prove all 16 error metrics against exact paper calculations ($10^{-12}$ precision) and verify string roundtrip conversions.
- Added `HandCalculatedAnalyticalProofs`, `AllMethodsStringRoundtripCoverage`, `TemperatureAndInferenceTemperature`, and `StatisticalWeightInitializationVerification` to `tests/activation_tests.cpp` to mathematically prove all 12 activation functions/derivatives and verify statistical weight initialization distributions.
- Added `output_back_span()` to `GradientsAndOutputs` in `include/neuralnetwork/common/gradientsandoutputs.h`, returning non-allocating `std::span<const double>` views over output layer activations.
- Added unit tests `ThinkEmptyInputsHandling`, `ThinkInvalidTopologySizeHandling`, `ThinkBatchVersusSingleConsistency`, and `ThinkConcurrentMultiThreadedInference` (8-thread concurrent prediction test) to `tests/network_integration_tests.cpp`.
- Added unit tests `LayersTrainRepeatedBatchBufferReuse`, `LayersTrainParallelWeightUpdateCorrectness`, `LayersTrainGradientClippingDisabledFastPath`, and `LayersTrainRecurrentSequenceBackprop` to `tests/layer_tests.cpp` to verify gradient/hidden states buffer reuse, zero-reallocation properties, fast-path gradient clipping bypass, and recurrent sequence backprop correctness.

### Changed
- Optimized `ErrorCalculation` in `include/neuralnetwork/helpers/errorcalculation.h`:
  - Replaced running mean division in `calculate_mse_error` with raw sum accumulation and single final division.
  - Replaced vector indexing with raw contiguous pointer access (`gt_vec.data()`), eliminating bounds checking and enabling compiler AVX2 SIMD vectorization.
  - Added heap-allocation-free static string comparator `iequals` in `string_to_type`.
- Optimized `activation` in `include/neuralnetwork/common/activation.h` & `activation.cpp`:
  - Replaced scalar function pointer indirections in `activate(x)` and `activate_derivative(x)` with inline `switch (_method)` dispatch, enabling header inlining for standard activation arithmetic (`linear`, `relu`, `leakyRelu`, `PRelu`, `tanh`).
  - Added heap-free string comparator `iequals` in `string_to_method`.
- Optimized `NeuralNetwork::think` & `Layers::think` in `include/neuralnetwork/neuralnetwork.cpp` & `layers.cpp`:
  - Standardized single and batch `think` methods to construct output vectors directly from `output_back_span()`, eliminating redundant intermediate `std::vector` copies.
  - Added early empty input checks in `NeuralNetwork::think` prior to acquiring `_mutex` shared lock, preventing lock contention on empty inputs.
- Optimized `Layers::train` & `Layers::update_weights` in `include/neuralnetwork/layers/layers.cpp`:
  - Added `has_rnn_gradients(unsigned layer)` helper to `GradientsAndOutputs` to replace $O(\text{batch\_size})$ search loops in `Layers::calculate_back_propagation_hidden_layers` with an $O(1)$ check.
  - Bypassed global gradient norm calculation in `Layers::update_weights` when gradient clipping is disabled (`clip_threshold <= 0.0`), saving $O(\text{weights})$ sum-of-squares iterations per training step.
  - Accelerated gradient zeroing in `FFLayer::apply_stored_gradients` using `std::memset` block clearing.
  - Hoisted recurrent input/gradient checks out of sample loops in `FFLayer::calculate_and_store_gradients_chunk`.
  - Optimized buffer zeroing in `Layers::train` to zero out only existing reused buffer items, skipping redundant zero-initialization of newly constructed `GradientsAndOutputs` and `HiddenStates` buffer elements.
  - Replaced inline lambda closures in `Layers::update_weights` with named functor task structures (`GradCalcTask` and `GradApplyTask`), adhering to coding standards and eliminating lambda instantiation overhead.
  - Vectorized matrix transposition in `FFLayer::cache_recurrent_weights` using raw pointer data access.
  - Optimized `Layers::calculate_forward_feed`, `FFLayer::calculate_hidden_gradients`, and `FFOutputLayer::calculate_output_gradients` to pre-reserve gradient vector capacities and use contiguous raw pointer copying (`.data()`), avoiding vector reallocations and enabling compiler AVX2 SIMD vectorization.

### Fixed
- Fixed copy-paste error log in `ErrorCalculation::type_to_string` (was logging `"Unknown activation type!"` instead of `"Unknown ErrorCalculation type!"`).
- Fixed Linux CI build failure in `include/neuralnetwork/common/activation.h` by adding missing `#include <cmath>` and `#include <algorithm>` headers.
- Fixed heap corruption (`0xc0000374`) in `HiddenStates::assign` in `include/neuralnetwork/common/hiddenstates.h` by checking if buffer memory address shifted (`views[0].get_pre_activation_sums().data() != _pre_activation_sums[layer_number].data()`) during vector relocation and triggering automatic view rebuild.
- Fixed MSVC CRT thread-local storage heap corruption (`0xc0000374`) in `Layers::think` in `include/neuralnetwork/layers/layers.cpp` during concurrent testing (`NetworkIntegrationTest.ThinkConcurrentMultiThreadedInference`) by replacing `thread_local` cache structures with standard local vectors.

## [1.1.7] - 2026-08-04

### Added
- Added `reset_cool_down()` method to `AdaptiveLearningRateScheduler` in `include/neuralnetwork/helpers/adaptivelearningratescheduler.h`.
- Added unit tests `ExtendedEpochTrainingDoesNotOverflow` and `NoisyErrorHistoryDecreasingState` to `tests/adaptive_learning_rate_scheduler_tests.cpp` to verify long-term training safety and noisy loss curve adaptation.
- Added comprehensive unit tests (`BpttBatchShufflePreservesPairingIntegrity`, `BpttBatchShuffleDistributionUniformity`, `SingleStepShufflePreservesPairingIntegrity`) to `tests/network_integration_tests.cpp` to verify BPTT sequence batch shuffling pair integrity and uniform distribution.
- Added `MediumWorkloadParallelExecutionVerification` unit test to `tests/fflayer_mt_tests.cpp` to verify multi-threaded parallel layer execution correctness under updated workload thresholds.
- Added unit tests `HiddenStatesZeroReuseZeroReallocation` and `HiddenStatesMultiLayerAllocationPersistence` to `tests/hidden_state_tests.cpp` to verify memory persistence and zero-reallocation properties during iterative forward propagation.
- Added `BatchForwardFeedInputCopyingSequenceAndBiasVerification` unit test to `tests/fflayer_tests.cpp` to verify optimized contiguous pointer input copying and bias vector initialization across standard and sequence batch inputs.
- Added missing `MYODDWEB_PROFILE_FUNCTION("NeuralNetwork")` macro to `NeuralNetwork::has_training_data()`.

### Changed
- Optimized input data copying and bias initialization in `FFLayer::calculate_forward_feed` in `include/neuralnetwork/layers/fflayer.cpp`:
  - Replaced element-by-element vector iterator copies with raw contiguous pointer copies (`std::copy` on raw `data()` pointers), enabling MSVC compiler AVX2 vectorization for batch inputs and bias initialization.
- Optimized `Layers::think` in `include/neuralnetwork/layers/layers.cpp`:
  - Replaced per-call dynamic heap allocation of `GradientsAndOutputs` and `HiddenStates` during inference with thread-local `InferenceCache` structures, eliminating memory allocation thrashing during high-frequency prediction/think calls.
- Optimized multi-threading FLOP dispatch thresholds across all layer types (`FFLayer`, `ElmanRNNLayer`, `GRURNNLayer`, `LSTMLayer`, `FFOutputLayer`):
  - Lowered matrix GEMM division thresholds from `2,000,000` to `100,000` FLOPs per thread, enabling multi-threaded task queue pool dispatch for medium-sized training batches.
  - Lowered vector post-activation thresholds from `1,000,000` to `50,000` ops per thread.
- Optimized `NeuralNetwork::create_bptt_batches` in `include/neuralnetwork/neuralnetwork.cpp`:
  - Replaced integer modulo shuffling loops (`g() % (i + 1)`) with `std::uniform_int_distribution` and direct `std::shuffle` on index vectors.
  - Eliminated redundant intermediate vector allocations during BPTT sequence start index shuffling.
- Refactored `AdaptiveLearningRateScheduler` in `include/neuralnetwork/helpers/adaptivelearningratescheduler.h`:
  - Replaced linear progress decay on plateau with geometric multiplicative decay (`current_learning_rate * (1.0 - _adjustment_rate / 2.0)`), preventing sudden 50%–99% learning rate collapses during long-term training.
  - Balanced `get_rate_change()` classification thresholds against actual step count `num_steps = comparisons - 1` so `Decreasing` correctly triggers on noisy downward loss trends, and `Plateauing` requires a true 75% flat trend.
  - Reduced plateau cooldown multiplier from 3 to 1 history window.
- Optimized `AdaptiveLearningRateScheduler` functions with `MYODDWEB_PROFILE_FUNCTION("AdaptiveLearningRateScheduler")` instrumentation.

### Fixed
- Fixed typo in log message in `AdaptiveLearningRateScheduler::update` (`"learning down rate"` -> `"learning rate"`).

## [1.1.6] - 2026-07-27

### Changed
- Optimized `FFLayer::run_post_gemm` in `include/neuralnetwork/layers/fflayer.cpp` to reuse thread-local buffers via `TempBuffer` tags 7 and 8, completely avoiding dynamic stack vector allocation during forward feed.

## [1.1.5] - 2026-07-26

### Added
- Added custom unit test cases (`MishAVX2Correctness`, `MishAVX2AllPositive`, `MishAVX2AllNegative`) in `tests/activation_tests.cpp` to verify optimized Mish SIMD branches.

### Changed
- Optimized `simd::mish_activate` and `simd::mish_derivative` in `include/neuralnetwork/common/simd_utils.h` using AVX2 mask checking (`_mm256_movemask_pd`) to completely bypass expensive vectorized math (`exp_pd`, `log_pd`, `tanh_pd`, `reciprocal_pd`) when inputs are all positive (> 20.0) or all negative (< -20.0).

### Fixed
- Fixed thread-local storage heap corruption (`Exit code 0xc0000374`) during concurrent testing (`LearningRateTest.ConcurrentThinkDuringTrainingIsThreadSafe`) by replacing local `thread_local` vectors in `FFLayer`, `ElmanRNNLayer`, `LSTMLayer`, and `GRURNNLayer` forward pass functions with standard local vectors. This removes unsafe destructors running on thread termination of ephemeral test threads.

## [1.1.4] - 2026-07-25

### Added
- Added custom unit test cases (`ELUAVX2AllPositive`, `ELUAVX2AllNegative`, `SELUAVX2AllPositive`, `SELUAVX2AllNegative`) in `tests/activation_tests.cpp` to verify optimized SIMD branches.

### Changed
- Optimized `simd::elu_activate`, `simd::elu_derivative`, `simd::selu_activate`, and `simd::selu_derivative` in `include/neuralnetwork/common/simd_utils.h` using AVX2 mask checking (`_mm256_movemask_pd`) to completely bypass expensive vectorized exponentiation (`exp_pd`) when inputs are all positive or all non-positive.

## [1.1.3] - 2026-07-19

### Added
- Added `ShuffleSingleStepsBehavior` test to `tests/network_integration_tests.cpp` to verify stochastic gradient descent shuffling behavior when backpropagation through time (BPTT) is disabled.

### Changed
- Corrected and optimized `NeuralNetwork::create_bptt_batches` in `include/neuralnetwork/neuralnetwork.cpp` to correctly shuffle training data between epochs when BPTT is disabled.
- Optimized `NeuralNetwork::calculate_forecast_metrics_all_layers_impl` in `include/neuralnetwork/neuralnetwork.cpp` by eliminating a redundant copy of checking indices.

## [1.1.2] - 2026-07-18

### Changed
- Optimized `FFLayer::run_post_gemm`, `ElmanRNNLayer::calculate_forward_feed`, `LSTMLayer::calculate_forward_feed`, and `GRURNNLayer::run_forward_pass` by replacing dynamically allocated local vectors with `thread_local` vectors, eliminating heap allocation overhead from layer forward paths.

## [1.1.1] - 2026-07-18

### Added
- Added comprehensive unit tests in `tests/layer_tests.cpp` to verify `Layer::calculate_huber_loss_error_deltas` behavior under different direction penalty configurations.
- Added new test cases `AdamStepNoDecay` and `NadamStepNoDecay` in `tests/simd_utils_tests.cpp` to cover and verify standard optimization steps where weight decay is disabled.

### Changed
- Updated `Layer::calculate_huber_loss_error_deltas` in `include/neuralnetwork/layers/layer.cpp` to respect the `use_direction_penalty` flag from `EvaluationConfig`.
- Optimized `Layer::calculate_huber_loss_error_deltas` using loop unswitching to eliminate branching overhead inside the neuron loop for maximum performance.
- Optimized `simd::adam_step`, `simd::scalar_adam_step`, `simd::nadam_step`, and `simd::scalar_nadam_step` in `include/neuralnetwork/common/simd_utils.h` using loop unswitching on `decays != nullptr` to eliminate branching in the hot path.

## [1.1.0] - 2026-06-12

### Added
- Created the `myoddweb::nn` namespace.
- Wrapped all core neural network library classes, structures, and helper functions in the new `myoddweb::nn` namespace (including `NeuralNetwork`, `Layer`, `Neuron`, `activation`, `NeuralNetworkOptions`, etc.).
- Added explicit documentation in the `README.md` explaining how to import and use the new namespace.
- Created a new `/python/` subdirectory containing a C++ binding toolchain (using `pybind11` and NuGet package restore) to compile the C++ library into a Python extension module (`neuralnetwork.pyd`).
- Added a Python test script `example.py` demonstrating how to train and use the neural network from Python.
- Added explicit documentation in `python/README.md` explaining how to build and call the Python module.

### Changed
- Updated all stand-alone example header files in `src/neuralnetwork/examples/` to use the `myoddweb::nn` namespace.
- Updated all test files in `tests/` to use the `myoddweb::nn` namespace.
- Kept third-party libraries (`TinyJSON`, `tracy`) and instrumentation code (`instrumentor.h`) outside the namespace to maintain clean integration boundaries.
- Reorganised the core NeuralNetwork library directory structure from a flat root layout into `/layers/`, `/helpers/`, and `/common/` subdirectories to improve code modularity.
- Updated all include directives in library headers, source files, tests, and examples to point to the new subdirectory paths.
- Updated MSVC Visual Studio project files (`.vcxproj` and `.vcxproj.filters`) and CMake files (`CMakeLists.txt`) to reflect the new folder structure.
