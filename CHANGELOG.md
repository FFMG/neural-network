# Changelog

All notable changes to the `neural-network` library will be documented in this file.

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
