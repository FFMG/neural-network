# Changelog

All notable changes to the `neural-network` library will be documented in this file.

## [1.1.65] - 2026-09-19

### Added
- Integrated Microsoft's `mimalloc` high-performance memory allocator (v2.1.7):
  - Vendored static `mimalloc` in [`include/neuralnetwork/libraries/mimalloc`](./include/neuralnetwork/libraries/mimalloc).
  - Added global `new` and `delete` replacement override via [`include/neuralnetwork/common/mimalloc_override.cpp`](./include/neuralnetwork/common/mimalloc_override.cpp) when `MYODDWEB_USE_MIMALLOC` is defined.
  - Integrated `mi_malloc_aligned` and `mi_free` into [`include/neuralnetwork/common/aligned_allocator.h`](./include/neuralnetwork/common/aligned_allocator.h) for SIMD-aligned allocations (`AlignedVector`).
  - Added `ENABLE_MIMALLOC` CMake option (default: `ON`) in [`tests/CMakeLists.txt`](./tests/CMakeLists.txt) defining `MYODDWEB_USE_MIMALLOC=1` and linking against `mimalloc-static`.
  - Added dedicated unit tests in [`tests/mimalloc_tests.cpp`](./tests/mimalloc_tests.cpp) covering runtime allocator detection, alignment verification across 16/32/64/128-byte boundaries, `AlignedVector` container integration, and multi-threaded concurrent allocation stress.
  - Added dedicated CI workflow in [`.github/workflows/tests-no-mimalloc.yml`](./.github/workflows/tests-no-mimalloc.yml) validating compilation and unit tests with `-DENABLE_MIMALLOC=OFF` across Windows, Ubuntu, and macOS runners.
  - Configured Visual Studio project files ([`examples/neuralnetwork.vcxproj`](./examples/neuralnetwork.vcxproj) and [`tests/neuralnetwork_tests.vcxproj`](./tests/neuralnetwork_tests.vcxproj)) with `mimalloc` include paths, single-source compilation (`static.c`), and `mimalloc_override.cpp`.
  - Documented `mimalloc` configuration, architecture, and build toggles in [`README.md`](./README.md).

### Fixed
- Fixed sequential test interaction in [`tests/output_layer_details_tests.cpp`](./tests/output_layer_details_tests.cpp) and [`tests/layer_details_tests.cpp`](./tests/layer_details_tests.cpp) where `NeuralNetworkOptions::build()` in earlier tests set `Logger::LogLevel::None`, suppressing warning log captures in `AdamWeightDecayWarning`. Tests now explicitly scope and restore `Logger::LogLevel::Warning`.
- Fixed false-positive GCC 13 `-Wfree-nonheap-object` compiler error under `-Werror` on Linux by computing smoothed vectors directly in `ErrorCalculation::smooth_labels` ([`include/neuralnetwork/helpers/errorcalculation.h`](./include/neuralnetwork/helpers/errorcalculation.h)) without temporary span indirection, and added `-Wno-free-nonheap-object` for GCC in [`tests/CMakeLists.txt`](./tests/CMakeLists.txt).

## [1.1.64] - 2026-09-18

### Added
- Added entropy regularisation for policy gradient advantage training (`train_with_advantages`) with Softmax output layers to prevent premature policy collapse and maintain exploration:
  - Added `NeuralNetworkOptions::with_entropy_coefficient(double)` and `entropy_coefficient()` in [`include/neuralnetwork/neuralnetworkoptions.h`](./include/neuralnetwork/neuralnetworkoptions.h), defaulting to `0.0` (disabled).
  - Added validation ensuring `entropy_coefficient` is non-negative and finite in `NeuralNetworkOptions::build()`.
  - Added analytical entropy regularisation gradient $\frac{\partial(-\beta H)}{\partial z_k} = \beta \cdot p_k (\ln p_k + H)$ in `Layers::calculate_back_propagation_output_layer_with_advantages` in [`include/neuralnetwork/layers/layers.cpp`](./include/neuralnetwork/layers/layers.cpp) across feed-forward and recurrent BPTT time steps.
  - Added serialisation and deserialisation support for `"entropy-coefficient"` in [`include/neuralnetwork/helpers/neuralnetworkserializer.cpp`](./include/neuralnetwork/helpers/neuralnetworkserializer.cpp).
  - Exposed `with_entropy_coefficient` and `entropy_coefficient` in Python bindings ([`python/bindings.cpp`](./python/bindings.cpp)) and documented in [`python/README.md`](./python/README.md).
  - Added comprehensive unit tests in [`tests/neuralnetwork_advantage_training_tests.cpp`](./tests/neuralnetwork_advantage_training_tests.cpp) covering regression safety, probability dispersion toward uniform distribution, linear scaling, non-Softmax isolation, serialisation persistence, input validation, and BPTT recurrent layers.

## [1.1.63] - 2026-09-17

### Added
- Added comprehensive unit test coverage for `tanh`, `FFLayer`, and `FFOutputLayer`:
  - Added `SimdUtilsTanhActivateAndDerivative` in [`tests/simd_utils_tests.cpp`](./tests/simd_utils_tests.cpp): verifies AVX2 SIMD `simd::tanh_activate` and `simd::tanh_derivative` across varied buffer sizes (aligned, unaligned, tail elements), covering both raw pre-activations and cached post-activations (`y_begin`).
  - Added `CalculateHiddenGradientsTanh` in [`tests/fflayer_tests.cpp`](./tests/fflayer_tests.cpp): verifies hidden layer backpropagation through `FFLayer` with `tanh` activation ($g_{next} W_{next}^T \odot (1 - \tanh^2(z))$).
  - Added `ForwardFeedTanhMultiNeuronMultiBatch` in [`tests/fflayer_tests.cpp`](./tests/fflayer_tests.cpp): verifies multi-neuron, multi-batch forward feed through `FFLayer` with `tanh` against exact analytical calculations.
  - Added `CalculateOutputGradientsTanhMSE` in [`tests/ffoutputlayer_tests.cpp`](./tests/ffoutputlayer_tests.cpp): verifies exact analytical output gradients on `FFOutputLayer` with `tanh` activation and MSE loss without dropout ($dE/dz = \frac{\hat{y} - y}{N} \cdot (1 - \hat{y}^2)$).
  - Added `TanhBoundaryAndExtremeValues` in [`tests/activation_tests.cpp`](./tests/activation_tests.cpp): verifies exact zero, mathematical symmetry ($\tanh(-x) = -\tanh(x)$, $\tanh'(-x) = \tanh'(x)$), small-value stability ($10^{-8}$), and extreme saturation ($\pm 50$, $\pm 100$).
- Updated documentation across [`README.md`](./README.md) and [`python/README.md`](./python/README.md).

### Optimised
- Optimised `simd::tanh_derivative` in [`include/neuralnetwork/common/simd_utils.h`](./include/neuralnetwork/common/simd_utils.h): accelerated vectorized AVX2 derivative computation with fused negative multiply-add (`_mm256_fnmadd_pd`) when `SIMD_FMA_ENABLED` is available, computing $1.0 - y^2$ in a single hardware cycle.
- Optimised `simd::tanh_pd` in [`include/neuralnetwork/common/simd_utils.h`](./include/neuralnetwork/common/simd_utils.h): accelerated vector $\tanh$ evaluation with fused multiply-subtract (`_mm256_fmsub_pd`) under `SIMD_FMA_ENABLED`.
- Vectorized `Layer::calculate_log_cosh_error_deltas` in [`include/neuralnetwork/layers/layer.cpp`](./include/neuralnetwork/layers/layer.cpp) with AVX2 SIMD using `simd::tanh_pd`.
- Vectorized soft logit capping gradient scaling in `FFOutputLayer::run_output_gradients` in [`include/neuralnetwork/layers/ffoutputlayer.cpp`](./include/neuralnetwork/layers/ffoutputlayer.cpp) with AVX2 SIMD and FMA using `simd::tanh_pd`.

## [1.1.62] - 2026-09-15

### Added
- Added clean, fully working Gridworld reinforcement learning example in [`python/examples/gridworld.py`](./python/examples/gridworld.py):
  - Demonstrates discrete policy gradient training with `NeuralNetwork::train_with_advantages` in a 4x4 obstacle grid.
  - Implements action masking for boundary and obstacle avoidance, with discounted advantage scaling rewarding path efficiency.
  - Features formatted ASCII grid path visualisation, step-by-step navigation transition logs, and full learned policy map rendering across all accessible grid cells.
  - Added support for `GRIDWORLD_EPISODES` environment variable override for custom training or CI execution.
- Added GitHub Actions CI workflow in [`.github/workflows/gridworld.yml`](./.github/workflows/gridworld.yml) to build the Python binding and run the GridWorld RL example automatically.
- Updated documentation across [`README.md`](./README.md) and [`python/README.md`](./python/README.md).

## [1.1.61] - 2026-09-14

### Added
- Added dynamic runtime learning rate control on [`NeuralNetwork`](./include/neuralnetwork/neuralnetwork.h) via `set_learning_rate(double)` and `has_learning_rate_override()`:
  - Allows callers (e.g. reinforcement learning episode loops or custom outer schedulers) to forcefully override the learning rate used by subsequent `train()` and `train_with_advantages()` calls.
  - Calling `set_learning_rate(lr)` with a positive rate marks the override active and completely bypasses internal epoch warmup, cosine-annealing, and decay schedulers.
  - Calling `set_learning_rate(0.0)` clears any active override and restores default options-based scheduling.
  - Added input validation guarding against negative and non-finite (`NaN`/`Inf`) values with warning logs, preserving existing rates safely.
  - Ensured thread safety with `std::shared_lock<std::shared_mutex>` in `get_learning_rate()` and `has_learning_rate_override()`, matching `std::unique_lock` in `set_learning_rate()`.
- Exposed Python bindings in [`python/bindings.cpp`](./python/bindings.cpp):
  - Added `set_learning_rate(learning_rate)`, `has_learning_rate_override()`, and read/write `learning_rate` property on `NeuralNetwork`.
- Added unit tests in [`tests/learning_rate_tests.cpp`](./tests/learning_rate_tests.cpp):
  - `SetLearningRateGetterSetterAndOverride`: Validates getter/setter, override flag, 0.0 reset, and rejection of negative/NaN/Inf values.
  - `SetLearningRateOverridesTrainWithAdvantages`: Confirms `train_with_advantages()` respects forced learning rates instead of resetting to options.
  - `SetLearningRateOverridesTrainEpochsAndBypassesScheduler`: Validates that `train()` epoch iterations bypass warmup and cosine decay when overridden.
  - `SetLearningRateThreadSafety`: Validates concurrent read/write access across multiple threads.
- Updated documentation in [`README.md`](./README.md) and [`python/README.md`](./python/README.md).

### Fixed
- Fixed bug in `NeuralNetwork::train_with_advantages` where `_learning_rate` was unconditionally overwritten by `_options.learning_rate()`, ensuring runtime overrides are preserved.
- Fixed `NeuralNetwork::train` unconditionally resetting `_learning_rate` to `_options.learning_rate()` and recalculating scheduler rates when an override is active.

## [1.1.60] - 2026-09-13

### Added
- Added Soft Logit Capping ($z_i' = C \cdot \tanh(z_i / C)$) for Softmax activation in [`activation`](./include/neuralnetwork/common/activation.h) and [`activation.cpp`](./include/neuralnetwork/common/activation.cpp):
  - Strictly bounds pre-activation logits within $(-C, C)$, preventing logit range explosion during reinforcement learning and advantage training.
  - When $C \le 50$, the maximum span cannot exceed $2C \le 100$, cleanly eliminating runaway logit warnings ($> 200$) and panics ($> 1000$) through sound mathematical formulation.
  - Added `logit_cap` parameter and getter/setter (`get_logit_cap()`, `set_logit_cap()`) on [`activation`](./include/neuralnetwork/common/activation.h).
- Added exact chain-rule gradient scaling in [`FFOutputLayer::run_output_gradients`](./include/neuralnetwork/layers/ffoutputlayer.cpp):
  - When cross-entropy loss skips explicit derivative computation, output gradients $\delta_k = y_k - t_k$ are scaled by $\left(1 - \tanh^2(z_k / C)\right)$ when `logit_cap > 0.0`, ensuring exact backward pass gradients that vanish as logits approach saturation.
- Added serialization and deserialization support for `"activation-logit-cap"` across [`NeuralNetworkSerializer`](./include/neuralnetwork/helpers/neuralnetworkserializer.cpp):
  - Serialises and deserialises `logit_cap` for output layers, hidden layers, activation helpers, and multi-output layer configurations with exact backward compatibility (defaults missing keys to `0.0`).
- Exposed Python bindings for `logit_cap` property and constructors on `Activation` in [`python/bindings.cpp`](./python/bindings.cpp).
- Added unit tests:
  - `SoftLogitCappingGetterSetter`, `SoftLogitCappingStrictlyBoundsExtremeLogits`, and `SoftLogitCappingApproximatesLinearForSmallLogits` in [`tests/activation_tests.cpp`](./tests/activation_tests.cpp).
  - `CalculateOutputGradientsCESoftLogitCapping` in [`tests/ffoutputlayer_tests.cpp`](./tests/ffoutputlayer_tests.cpp).

### Fixed
- Fixed artificial underflow probability floor in softmax by expanding `LOGIT_CLAMP` from `30.0` to `500.0` in [`activation::calculate_softmax`](./include/neuralnetwork/common/activation.cpp):
  - The previous clamp at $30.0$ prevented probabilities from falling below $e^{-30} \approx 10^{-14}$, maintaining non-zero gradients indefinitely and causing Adam optimizer updates to drive unregularised weights and biases towards infinity.

## [1.1.59] - 2026-09-13

### Fixed
- Fixed critical mathematical bug where momentum and Adam first-moment `beta1` were dropped (forced to `0.0`) on [`FFOutputLayer`](./include/neuralnetwork/layers/ffoutputlayer.h):
  - In [`Layer::apply_update_to_vector`](./include/neuralnetwork/layers/layer.h) and [`Layer::apply_update_to_vector_internal`](./include/neuralnetwork/layers/layer.cpp), added support for an explicit `momentum_override` parameter.
  - In [`FFOutputLayer::apply_stored_gradients`](./include/neuralnetwork/layers/ffoutputlayer.cpp), passed `detail.get_momentum()` to ensure SGD, Adam, AdamW, Nadam, NadamW, Lion, and RAdam receive the head's configured momentum and first-moment decay factor.
- Fixed potential crash / undefined behaviour on empty hidden states or zero batch size across all gradient functions in [`FFLayer`](./include/neuralnetwork/layers/fflayer.cpp) and [`FFOutputLayer`](./include/neuralnetwork/layers/ffoutputlayer.cpp):
  - Added defensive validation guarding against `batch_hidden_states.empty()` and missing layer indices before indexing time steps, backed by `size()` and `empty()` helper methods on [`HiddenStates`](./include/neuralnetwork/common/hiddenstates.h).
- Fixed null pointer vulnerability in bias gradient accumulation in [`FFLayer::calculate_and_store_gradients_chunk`](./include/neuralnetwork/layers/fflayer.cpp) by verifying `g_base != nullptr` prior to SIMD vector accumulation.
- Fixed cross-example transaction cost leakage in [`FFOutputLayer::calculate_output_metrics`](./include/neuralnetwork/layers/ffoutputlayer.cpp) for Sharpe ratio and Sortino ratio losses:
  - Sequences are now evaluated per batch item before pooling returns, ensuring initial step positions are compared to zero rather than the previous batch item's final position.

### Performance
- Added fast-path contiguous vectorized updates in [`FFOutputLayer::apply_stored_gradients`](./include/neuralnetwork/layers/ffoutputlayer.cpp):
  - When output heads share identical optimizer types and momentums (the standard configuration), the layer performs a single contiguous vectorized update across all weights and biases rather than looping $N_{\text{inputs}}$ times over small slices.
- Optimized static-context sequence weight gradient accumulation in [`FFLayer::calculate_and_store_gradients_chunk`](./include/neuralnetwork/layers/fflayer.cpp) for recurrent sequence training (BPTT):
  - For items with static inputs across timesteps (`x_stride == 0`), sequence gradients are pre-accumulated across timesteps once per item, collapsing the $T$-timestep loop into a single SIMD fused multiply-add per input group.
- Optimized buffer copying in [`FFOutputLayer::run_output_gradients`](./include/neuralnetwork/layers/ffoutputlayer.cpp):
  - Replaced vector `assign` reallocations with fixed-capacity pre-sized buffers and direct memory copy (`std::memcpy`).
- Eliminated redundant full-vector scaling in [`FFLayer::calculate_and_store_gradients`](./include/neuralnetwork/layers/fflayer.cpp) when `batch_size == 1`.
- Replaced lambda thread pool task with named functor `FfOutputGradientsTask` in [`FFOutputLayer::calculate_output_gradients`](./include/neuralnetwork/layers/ffoutputlayer.cpp) adhering to coding standards.

### Added
- Added Python binding default parameter values for [`OutputLayerDetails`](./python/bindings.cpp) constructor (`error_evaluation_config`, `weight_decay`, `optimiser_type`, and `momentum`).
- Added comprehensive unit tests in [`tests/ffoutputlayer_tests.cpp`](./tests/ffoutputlayer_tests.cpp):
  - `OutputLayerAppliesMomentumWithSGD`: Verifies momentum is retained and correctly applied on output layers.
  - `OutputLayerApplyStoredGradientsFastPathEquivalence`: Confirms numerical equivalence between fast-path contiguous updates and multi-head slice updates.
  - `OutputLayerMetricsSharpeSortinoNoCrossBatchLeakage`: Confirms Sharpe ratio metrics do not subtract transaction costs across batch item boundaries.
  - `OutputLayerEmptyHiddenStatesDefensive`: Confirms zero batch size and empty hidden states do not crash.
- Added comprehensive unit tests in [`tests/fflayer_tests.cpp`](./tests/fflayer_tests.cpp):
  - `EmptyHiddenStatesDefensive`: Verifies empty hidden states safety across all FFLayer gradient methods.
  - `StaticContextSequenceBPTTWeightGradientEquivalence`: Validates pre-summed sequence gradient math for static inputs under BPTT.
  - `WeightDecayWithAdamW`: Confirms decoupled weight decay on FFLayer with AdamW.

## [1.1.58] - 2026-09-12

### Added
- Added warning logs and documentation clarifying that [`OptimiserType::Adam`](./include/neuralnetwork/common/optimiser.h) deliberately does not apply weight decay:
  - [`LayerDetails`](./include/neuralnetwork/layers/layerdetails.h) and [`OutputLayerDetails`](./include/neuralnetwork/layers/outputlayerdetails.h) constructors now log a [`Logger::warning`](./include/neuralnetwork/common/logger.h) if configured with `OptimiserType::Adam` and `weight_decay > 0`, informing the user that standard Adam ignores weight decay and recommending [`OptimiserType::AdamW`](./include/neuralnetwork/common/optimiser.h) for decoupled weight decay.
  - Added unit test `ApplyUpdateToWeightAdamIgnoresDecay` in [`tests/layer_optimizer_tests.cpp`](./tests/layer_optimizer_tests.cpp) confirming scalar updates ignore weight decay for `OptimiserType::Adam` alongside vectorised `ApplyUpdateToVectorAdamIgnoresDecay`.
  - Added unit tests `AdamWeightDecayWarning`, `AdamZeroWeightDecayNoWarning`, and `AdamWWeightDecayNoWarning` in [`tests/layer_details_tests.cpp`](./tests/layer_details_tests.cpp) and [`tests/output_layer_details_tests.cpp`](./tests/output_layer_details_tests.cpp).
  - Updated examples [`examples/addingproblem.h`](./examples/addingproblem.h) and [`examples/lstm_multi.h`](./examples/lstm_multi.h) to use `OptimiserType::AdamW` where weight decay is configured.
  - Updated [`README.md`](./README.md) and [`python/README.md`](./python/README.md) to highlight that standard Adam does not apply weight decay and warns when `weight_decay > 0`.

## [1.1.57] - 2026-09-10

### Added
- Added dual-temperature activation controls and runtime temperature setters:
  - Added [`activation::set_temperature`](./include/neuralnetwork/common/activation.h) and [`activation::set_inference_temperature`](./include/neuralnetwork/common/activation.h).
  - Added `set_temperature` to [`Layer`](./include/neuralnetwork/layers/layer.h), [`Layers`](./include/neuralnetwork/layers/layers.h), [`MultiOutputLayer`](./include/neuralnetwork/layers/multioutputlayer.h), [`OutputLayerDetails`](./include/neuralnetwork/layers/outputlayerdetails.h), and [`NeuralNetworkOptions`](./include/neuralnetwork/neuralnetworkoptions.h).
  - Added `get_inference_temperature()`, `set_temperature()`, and `set_inference_temperature()` to [`NeuralNetwork`](./include/neuralnetwork/neuralnetwork.h) supporting both single-head and indexed multi-output heads.
  - Exposed Python bindings for `get_inference_temperature`, `set_temperature`, `set_inference_temperature` on `NeuralNetwork`, and `temperature` / `inference_temperature` properties on `Activation` in [`python/bindings.cpp`](./python/bindings.cpp).
- Added comprehensive temperature and optimiser unit tests in [`tests/temperature_and_optimiser_tests.cpp`](./tests/temperature_and_optimiser_tests.cpp):
  - `ActivationTemperatureClampingAndValidation`: Verifies negative, zero, sub-epsilon (< 1e-6), and non-finite (`NaN`/`Inf`) temperatures are clamped to 1e-6 in constructors and runtime setters.
  - `SoftmaxInferenceTemperatureControlsSharpness`: Mathematically verifies entropy modulation across low ($T = 0.2$), standard ($T = 1.0$), and high ($T = 5.0$) temperatures.
  - `SoftmaxDualTemperatureTrainingVsInference`: Validates decoupling high training exploration ($T_{\text{train}} = 2.0$) from sharp inference exploitation ($T_{\text{infer}} = 0.2$).
  - `NeuralNetworkInferenceTemperatureRuntimeAdjustment`: Tests dynamic runtime modification of inference temperature on a trained network.
  - `MultiOutputLayerIndependentTemperatures`: Tests independent temperature configurations and runtime setters across multiple output heads.
  - `RLAdvantageTrainingWithExplorationVsExploitationTemperature`: Verifies policy gradient advantage updates with dual temperatures.
  - `ExtremeInferenceTemperatureStability`: Verifies numerical stability under extreme temperatures ($10^{-6}$ argmax and $100.0$ uniform).
  - `JsonSerializationRoundTripAllOptimisersAndTemperatures`: Verifies round-trip serialization and inference consistency across all seven optimisers (`AdamW`, `Adam`, `SGD`, `Nadam`, `NadamW`, `Lion`, `RAdam`) and dual temperatures.
- Added comprehensive unit tests for `Logger` in [`tests/logger_tests.cpp`](./tests/logger_tests.cpp):
  - `MinimumLevelFiltersLowerSeverityMessages`: Verifies log level filtering per severity threshold.
  - `PanicAlwaysThrowsEvenWhenLoggingIsFullyDisabled`: Confirms `Logger::panic` always throws `std::runtime_error` regardless of whether logging is set to `None`.
  - `WarningErrorAndPanicFlushImmediately`: Verifies immediate buffer flush (`std::cout.flush()`) on Warning, Error, and Panic.
  - `TraceDebugAndInfoDoNotFlushImmediately`: Confirms Trace, Debug, and Info remain buffered without forcing immediate syncs.
  - `ConcurrentLoggingDoesNotInterleaveMessages`: Multi-threaded test verifying that concurrent writes from 8 parallel threads do not interleave mid-line characters.
  - `LevelStringRoundTripIsCaseInsensitive`: Verifies case-insensitive string parsing and serialization of all log levels.
  - `MultiLineMessageIndentsSubsequentLines`: Validates vertical indentation of multi-line log messages aligning underneath the log tag.
  - `LazyLoggingCallableOnlyEvaluatedWhenSeverityEnabled`: Ensures expensive callable/lambda arguments are never evaluated when their severity level is disabled.
  - `FactoryFormatsMixedTypesAndVectors`: Verifies `Logger::factory` string formatting with mixed primitive types and vectors.
  - `AllLogLevelsCanBeSetAndRetrieved`: Validates getting and setting every log level in `Logger::LogLevel`.
  - `AtomicLevelConcurrentReadAndWriteStress`: Stress tests concurrent level modifications and reads across multiple writer and reader threads.
- Added on-policy Reinforcement Learning support via policy gradients (REINFORCE) with `NeuralNetwork::train_with_advantages` and `Layers::train_with_advantages`:
  - Output-layer delta scaling: Multiplies output error deltas directly by per-sample scalar advantages before hidden-layer backpropagation, scaling all upstream hidden-layer weight and bias updates proportionally.
  - Sub-batching support: Automatically chunks trajectory updates according to `options.batch_size()`.
  - Guarded single output layer constraint: Enforces single output head (throws explanatory error if multi-output head is used).
  - Optimiser persistence: Preserves layer optimiser velocity/momentum states across consecutive calls for online and episodic learning.
- Added Python bindings for Reinforcement Learning:
  - Bound `NeuralNetwork::train_with_advantages` in [`python/bindings.cpp`](./python/bindings.cpp) with GIL release guard (`py::call_guard<py::gil_scoped_release>()`).
- Added Tic-Tac-Toe Reinforcement Learning example in [`python/examples/tic_tac_toe.py`](./python/examples/tic_tac_toe.py):
  - Self-training agent learning Tic-Tac-Toe using policy gradients with scalar rewards (+1.0 for win, +0.2 for draw, -1.0 for loss).
  - Masked action sampling over valid board cells.
  - Post-training evaluation of 100 matches against a Random opponent with win/draw/loss statistics.
  - Step-by-step visual demonstration match rendering the 3x3 board at each turn.
- Added unit tests in [`tests/neuralnetwork_advantage_training_tests.cpp`](./tests/neuralnetwork_advantage_training_tests.cpp):
  - `PositiveAdvantageIncreasesTakenActionProbability`: Verifies positive advantage increases the chosen action's probability.
  - `NegativeAdvantageDecreasesTakenActionProbability`: Verifies negative advantage decreases the chosen action's probability.
  - `ZeroAdvantageLeavesOutputWeightsUnchanged`: Verifies zero advantage produces no change to output weights.
  - `ZeroAdvantageLeavesHiddenWeightsUnchanged`: Verifies zero advantage leaves hidden-layer weights strictly unchanged.
  - `AdvantageMagnitudeScalesOutputWeightDeltaLinearly`: Verifies output weight updates scale linearly with advantage magnitude.
  - `AdvantageMagnitudeScalesHiddenWeightDeltaLinearly`: Verifies hidden-layer weight updates scale linearly with advantage magnitude.
  - `HandlesMoreExamplesThanConfiguredBatchSizeByChunking`: Verifies sub-batch chunking across uneven batch boundaries.
  - `EmptyInputsLeavesWeightsUnchanged`: Verifies empty input vectors are safely handled without throwing or altering weights.
  - `MismatchedInputSizesThrows`: Verifies validation error when input, target, or advantage vector lengths mismatch.
  - `MultiOutputLayerHeadThrows`: Verifies validation error when multi-output heads are used.
  - `SerializerSavesAndLoadsAdvantageTrainedNetwork`: Verifies that saving and loading an advantage-trained network preserves all weights and inference outputs identically.
  - `WorksWithAdamOptimiser`: Verifies policy gradient updates operate correctly under the Adam optimiser.
  - `MismatchedInputDimensionsThrows`: Verifies validation error when input sample feature dimension does not match input layer topology.
  - `MismatchedActionTargetDimensionsThrows`: Verifies validation error when action target dimension does not match output layer topology.
  - `NonFiniteAdvantageThrows`: Verifies `NaN` and `Inf` advantage values throw `std::runtime_error`.
  - `ContinuousActionRegressionPolicyWithMSE`: Verifies advantage scaling on continuous action / regression policies using Linear outputs and MSE loss.
  - `RecurrentLSTMPolicyNetwork`: Verifies advantage-weighted policy gradient updates on recurrent LSTM networks with BPTT.
  - `ExtremeAdvantageScalingNumericalStability`: Verifies that extreme advantage magnitudes ($\pm 100.0$) maintain numerical stability and valid probabilities without producing `NaN` or exploding weights under gradient clipping.
  - `NonFiniteInputThrows`: Verifies `NaN` and `Inf` values inside input feature vectors throw `std::runtime_error`.
  - `NonFiniteActionTargetThrows`: Verifies `NaN` and `Inf` values inside action target vectors throw `std::runtime_error`.
  - `NegativeActionTargetWithSoftmaxThrows`: Verifies negative action target values throw `std::runtime_error` when output activation is Softmax.
  - `ZeroSumActionTargetWithSoftmaxThrows`: Verifies all-zero action target vectors throw `std::runtime_error` when output activation is Softmax.
  - `SoftmaxTemperatureScalingInPolicyNetwork`: Verifies policy gradient learning with exploration temperature scaling ($\tau \neq 1.0$).
  - `LabelSmoothingWithAdvantageTraining`: Verifies advantage-weighted policy gradient updates with Cross-Entropy label smoothing.
  - `FFTanhHiddenLayerWithAdamWAndSoftmax`: Verifies advantage-weighted policy gradient training on a Feed Forward network with Tanh hidden layers, AdamW optimiser, and Softmax output.
  - `FFTanhOutputLayerWithAdamWAndMSE`: Verifies continuous action policy training with Tanh output activation, MSE loss, and AdamW optimiser under advantage scaling.
  - `AdamWDecoupledWeightDecayWithAdvantageTraining`: Verifies that under zero advantage ($A_t = 0$), AdamW applies decoupled weight decay ($w \leftarrow w(1 - \eta \lambda)$) strictly to weights while biases remain unaffected.
  - `TanhOutputTargetOutOfRangeThrows`: Verifies validation rejection when action targets exceed the reachable range of Tanh output layers ($[-1.0, 1.0]$).
  - `FFTanhWithNegativeAdvantageDecreasesProbability`: Verifies negative advantage updates penalise chosen actions under Tanh hidden layers and AdamW optimiser.
- Added GitHub Actions workflow in [`.github/workflows/tic_tac_toe.yml`](./.github/workflows/tic_tac_toe.yml) to automatically compile the Python bindings and execute the Tic-Tac-Toe Reinforcement Learning example in CI.
- Added Reinforcement Learning section to [`README.md`](./README.md) and [`python/README.md`](./python/README.md).

### Optimised
- Removed a couple of throw std::* and replaced by Logger::panic 
- Enhanced thread safety and durability in [`Logger`](./include/neuralnetwork/common/logger.h):
  - Made `_min_level` atomic (`std::atomic<LogLevel>`) with relaxed memory ordering to prevent data races during concurrent logging and runtime level changes across threads.
  - Added compile-time `static_assert` guarantees verifying exact tag string lengths match `TagLen` across all build configurations.
  - Serialised console writes with a static mutex (`output_mutex()`) to eliminate torn or interleaved characters during parallel multi-threaded execution.
  - Added immediate `std::cout.flush()` for Warning, Error, and Panic messages to ensure critical diagnostics are not lost during abnormal termination or unhandled exceptions.
- Optimised softmax normalization in [`activation::calculate_softmax`](./include/neuralnetwork/common/activation.cpp):
  - Replaced scalar probability division loop with AVX2 vectorised [`simd::mul_scalar`](./include/neuralnetwork/common/simd_utils.h) multiplying by reciprocal sum ($1 / \sum e^z$).
- Optimised advantage gradient scaling in [`Layers::calculate_back_propagation_output_layer_with_advantages`](./include/neuralnetwork/layers/layers.cpp):
  - Hoisted output neuron count, recurrent layer detection (`has_rnn`), and RNN gradient size queries outside the per-sample batch loop.
  - Replaced scalar advantage multiplication loops with AVX2 vectorised [`simd::mul_scalar`](./include/neuralnetwork/common/simd_utils.h).
  - Added identity bypass (`advantage == 1.0`) to avoid redundant scaling arithmetic and memory writes.
  - Added zero-advantage fast path (`advantage == 0.0`) using `std::memset` to rapidly zero out gradients.
- Optimised trajectory training concurrency in [`NeuralNetwork::train_with_advantages`](./include/neuralnetwork/neuralnetwork.cpp):
  - Replaced per-sub-batch mutex acquisition with a single exclusive lock over the entire trajectory batch loop, eliminating lock contention and ensuring atomic rollout updates across threads.
  - Standardised iterator parameter passing by value in [`Layers::train_with_advantages`](./include/neuralnetwork/layers/layers.h) to eliminate unnecessary references and allow direct rvalue passing.

### Fixed
- Fixed activation temperature validation and clamping:
  - Guarded against $\le 0.0$, sub-epsilon, and non-finite (`NaN`/`Inf`) values in [`activation::activation`](./include/neuralnetwork/common/activation.cpp) and runtime setters, clamping both training and inference temperatures to $10^{-6}$ to prevent division by zero and `NaN` propagation.
- Fixed untrained model serialization panic:
  - Corrected `_learning_rate` initialisation in [`NeuralNetwork::NeuralNetwork`](./include/neuralnetwork/neuralnetwork.cpp) from `0.0` to `options.learning_rate()`, ensuring untrained networks serialize valid learning rates and prevent deserialization corruption panics.
  - Updated `_learning_rate` in [`NeuralNetwork::train`](./include/neuralnetwork/neuralnetwork.cpp) to track the final learning rate at the end of training.
- Added robust upfront input, target, and mathematical validation in [`NeuralNetwork::train_with_advantages`](./include/neuralnetwork/neuralnetwork.cpp):
  - Pre-emptive multi-output check before batch allocation or forward computation.
  - Rigorous per-sample dimension checks ensuring training inputs match input topology (or valid BPTT multiples).
  - Rigorous per-sample dimension checks ensuring action targets match output topology (or valid BPTT multiples).
  - Element-wise validation ensuring all values in `training_inputs`, `training_action_targets`, and `training_advantages` are finite (`std::isfinite`), preventing `NaN` or `Inf` corruption.
  - Domain validation for Softmax policies ensuring action target probabilities are non-negative ($y_j \ge 0$) with non-zero sum ($\sum_j y_j > 0$).
  - Target range validation for Tanh output layers, rejecting any action target values outside $[-1.0, 1.0]$ to prevent vanishing gradients and unreachable target explosion.

## [1.1.56] - 2026-09-09

### Fixed
- Fixed GCC `-Werror=maybe-uninitialized` build error in `GRURNNLayer::finalize_forward_step`: Zero-initialised stack buffer `h_hat_final_stack` to prevent false-positive uninitialised warnings under strict compiler settings.

### Added
- Added comprehensive unit test coverage and mathematical verification for weight decay across all optimisers and layer architectures:
  - Vectorised optimiser weight decay tests in [`tests/layer_optimizer_tests.cpp`](./tests/layer_optimizer_tests.cpp):
    - `ApplyUpdateToVectorSGDWithDecay`: Verifies SGD L2 regularisation ($g_{\text{eff}} = g + \lambda w$, $v = \mu v + g_{\text{eff}}$, $w = w - \eta v$) with zero and non-zero momentum across both AVX2 SIMD loops and scalar tails.
    - `ApplyUpdateToVectorAdamWWithDecay`: Verifies decoupled weight decay ($w \leftarrow w(1 - \eta \lambda) - \eta \cdot \text{step}$) in AdamW.
    - `ApplyUpdateToVectorAdamIgnoresDecay`: Confirms standard Adam ignores weight decay, applying updates purely based on gradients.
    - `ApplyUpdateToVectorNadamWWithDecay`: Verifies decoupled weight decay with Nesterov momentum in NadamW.
    - `ApplyUpdateToVectorNadamIgnoresDecay`: Confirms standard Nadam ignores weight decay.
    - `ApplyUpdateToVectorLionWithDecay`: Verifies decoupled weight decay ($w \leftarrow w(1 - \eta \lambda) - \eta \operatorname{sign}(c)$) in Lion.
    - `ApplyUpdateToVectorRAdamWithDecay`: Verifies decoupled weight decay in RAdam.
    - `ApplyUpdateToVectorBiasExclusionAllOptimizers`: Verifies that biases (`is_bias = true`) are strictly excluded from weight decay across all optimisers (SGD, Adam, AdamW, Nadam, NadamW, Lion, RAdam).
  - Dedicated layer-level weight decay test suite in [`tests/weight_decay_tests.cpp`](./tests/weight_decay_tests.cpp):
    - `FFLayer`: Initialisation of `_w_decays` and `_b_decays`, exact mathematical decay under AdamW and SGD ($\mu = 0$), and bias stability.
    - `FFOutputLayer`: Independent per-head weight decays (multiple heads with different $\lambda$ values decaying at their specific rates) and bias stability.
    - `ElmanRNNLayer`: Concurrent decoupled decay of input weights `_w` and recurrent weights `_rw` with bias stability.
    - `LSTMLayer`: Verification that all 8 weight matrices (input and recurrent for candidate, forget, input, and output gates) decay correctly, while all 4 gate biases and recurrent LayerNorm gain and bias parameters are preserved.
    - `GRURNNLayer`: Verification that all 6 weight matrices (candidate, update gate, reset gate) decay correctly, while biases and LayerNorm gain/bias parameters are preserved.
    - `TCNLayer`: Verification that causal dilated convolution weights decay correctly while biases remain unaffected.
    - `EmbeddingLayer`: Verification that embedding vocabulary lookup weights decay at the specified rate.
    - `AttentionPoolLayer`: Verification that attention projection weights `_wa` and context vector `_v` decay at the specified rate while attention bias `_ba` is preserved.
    - `SelfAttentionLayer`: Verification that all projection weights (`_wq`, `_wk`, `_wv`, `_wo`) and feed-forward weights (`_ff1_w`, `_ff2_w`) decay correctly, while biases and post-attention/post-FF LayerNorm gain and bias parameters are preserved.
    - `MultiOutputLayer`: Verification of weight decay propagation into branch hidden and output layers during `apply_stored_gradients`.
    - `ResidualProjector`: Verification of L2 weight decay during `apply_weight_gradient`.

## [1.1.55] - 2026-09-09

### Optimised
- Optimised `GRURNNLayer` gradient accumulation, BPTT execution, and state blending:
  - Zero-copy gradient accumulation for worker thread 0 in `GRURNNLayer::calculate_and_store_gradients`: Worker thread 0 now accumulates directly into member gradient vectors (`_w_grads`, `_rw_grads`, `_z_w_grads`, `_z_rw_grads`, `_r_w_grads`, `_r_rw_grads`, `_b_grads`, `_z_b_grads`, `_r_b_grads`), allocating auxiliary accumulators only for threads $1 \dots T-1$ and saving redundant buffer allocations, zeroing, and merge operations.
  - Multi-vector SIMD reduction in `GRURNNLayer::calculate_and_store_gradients`: Merged auxiliary thread accumulators into the member gradient vectors using `simd::accumulate_four_vectors` and `simd::accumulate_two_vectors`, cutting memory traffic and passes across the 6 weight matrices and 3 bias vectors.
  - Vectorised state blending with dropout in `GRURNNLayer::finalize_forward_step`: Pre-computed dropout-scaled candidate states into a stack buffer (for $N_{\text{this}} \le 128$) and called `simd::gru_output_step` to blend states using AVX2 and FMA instructions for both dropout and non-dropout training paths.
  - Fast path for single-step gradient accumulation in `GRURNNLayer::calculate_and_store_gradients_chunk`: When $num\_time\_steps == 1$, bypassed recurrent weight loops entirely (since recurrent transitions require $t \ge 1$), and eliminated timestep loops and stride calculations for input weights and biases.
  - Zero-scalar bypass in `GRURNNLayer::calculate_and_store_gradients_chunk`: Skipped SIMD scalar multiplications and additions when input neurons ($x_0 = x_1 = x_2 = x_3 = 0.0$) or recurrent hidden states ($hp_0 = hp_1 = hp_2 = hp_3 = 0.0$) are zero.
  - Guarded multi-threading dispatch to require `batch_size > 1` in `calculate_forward_feed`, `calculate_hidden_gradients`, and `calculate_and_store_gradients`.

### Fixed
- Fixed heap buffer overflow and sequence length detection in `GRURNNLayer::calculate_forward_feed`:
  - Added division-by-zero guards `N_prev > 0` and `N_this > 0`.
  - Scanned all batch items to determine the maximum sequence length $T_{\text{max}}$ before allocating `flattened_batch_inputs`, preventing buffer overflow when item 0 is static ($T=1$) and subsequent items contain sequences ($T > 1$).
  - Added sequence length fallback to `batch_hidden_states[0].at(get_layer_index()).size()`.
  - Safely broadcast single-step items across multi-step batch sequences.
- Fixed direct gradients zeroing bug in `GRURNNLayer::calculate_bptt_batch_chunk`:
  - Added static helper `get_output_grads_span` to inspect `rnn_gradients(target_layer_idx)`, `gradients(target_layer_idx)`, `rnn_gradients(this_layer_idx)`, and `gradients(this_layer_idx)`. Prevents upstream sequence gradients from being discarded or resolving to `nullptr` when the downstream layer is absent or gradients are assigned to the current layer.
- Fixed sequence residual connection handling in `GRURNNLayer::finalize_forward_step`:
  - Supported full-sequence residual tensors (`batch_residual_output_values[b].size() == num_time_steps * N_this`), slicing per timestep rather than only supporting single-step residuals.
- Fixed input resolution per batch item in `GRURNNLayer::calculate_and_store_gradients_chunk`:
  - Resolved `prev_outputs_rnn` and `prev_outputs_std` independently for each batch item without an item's static input being discarded when another item has sequence input.

### Added
- Added unit tests in `tests/grurnnlayer_tests.cpp`:
  - `GRURNNLayerTest.DirectGradientsFallbackToLayerIndexRnnGradients`: Verifies that sequence gradients placed at `get_layer_index()` flow through BPTT when the downstream layer is absent.
  - `GRURNNLayerTest.ForwardFeedFullSequenceResiduals`: Verifies that full sequence residual connections ($T \times N$) slice and accumulate per timestep.
  - `GRURNNLayerTest.ForwardFeedPartialSequenceBroadcastingSafety`: Verifies mixed single-step and sequence items across batches do not overflow heap buffer and broadcast correctly.
  - `GRURNNLayerTest.SingleStepFastPathGradientEquivalence`: Verifies single-step fast path matches exact analytical gradients.
  - `GRURNNLayerTest.DropoutBPTTConsistencyMultiBatch`: Verifies dropout masking and backprop scaling consistency across multi-batch sequences.

## [1.1.54] - 2026-09-08

### Optimised
- Optimised `FFLayer` gradient accumulation, memory caching, and multi-threading efficiency:
  - Multi-vector SIMD vector accumulation in `include/neuralnetwork/common/simd_utils.h`: Added `simd::accumulate_four_vectors(x0, x1, x2, x3, y, n)` ($y += x_0 + x_1 + x_2 + x_3$) and `simd::accumulate_two_vectors(x0, x1, y, n)` ($y += x_0 + x_1$) with AVX2 vectorization and scalar fallbacks.
  - Multi-vector worker thread gradient accumulation merge in `FFLayer::calculate_and_store_gradients`: Merges worker thread accumulators in 4-way and 2-way vector passes via `accumulate_four_vectors` and `accumulate_two_vectors`, halving memory traffic on `_w_grads` and `_b_grads`.
  - Vectorized bias accumulation in `FFLayer::calculate_and_store_gradients_chunk`: Merged incoming gradients across batch items (single-step) and timesteps (sequence steps) in 4-way and 2-way SIMD passes, replacing sequential element-wise additions.
  - Fast path for single-step weight gradient accumulation in `FFLayer::calculate_and_store_gradients_chunk`: When `num_time_steps == 1`, bypasses the inner timestep loop and stride calculations, directly streaming single-step item inputs and gradients into weight gradient accumulators.
  - Static sequence input optimization in `FFLayer::calculate_and_store_gradients_chunk`: When sequence input and gradient strides are zero (`x_stride == 0 && g_stride == 0`), scales inputs once by `num_time_steps`, skipping $T$ repetitive FMADD operations per sequence item.
  - Parallelized chunk-level memory zeroing in `FFLayer::calculate_hidden_gradients`: Disabled synchronous zero-init of `flattened_this_grads_buffer` on the main thread and moved chunk zeroing via `std::memset` into `run_backward_chunk` across thread workers, warming L1/L2 caches immediately prior to backward GEMM.
  - Small-Buffer Optimization (SBO) for sequence dropout output buffer in `FFLayer::run_post_gemm`: Added stack buffer `output_row_stack` (`std::array<double, 128>`) for sequence outputs when $num\_time\_steps \times N_{\text{this}} \le 128$, eliminating heap allocation for typical sequence lengths.
  - Hoisted invariants: Hoisted `get_neurons()` and pre-calculated default dropout scale in `FFLayer::run_post_gemm`. Guarded multi-threading dispatch to require `batch_size > 1` in `calculate_hidden_gradients_from_output_gradients` and `calculate_and_store_gradients`.

### Fixed
- Fixed sequence length detection in `FFLayer::calculate_forward_feed`: Added division-by-zero guard `N_prev > 0` before calculating `num_time_steps`, and added sequence length fallback to `batch_hidden_states[0].at(get_layer_index()).size()`, ensuring sequence outputs are correctly shaped when previous layer inputs are static or broadcast.
- Fixed direct gradient fallback in `FFLayer::calculate_hidden_gradients_from_output_gradients`: Created static helper `get_output_grads_span` to inspect `rnn_gradients(next_layer_idx)`, `gradients(next_layer_idx)`, `rnn_gradients(this_layer_idx)`, and `gradients(this_layer_idx)`. Prevents sequence gradients from being discarded or zeroed out when the downstream layer is absent.
- Fixed sequence residual connection addition in `FFLayer::run_post_gemm`: Added support for full-sequence residual tensors (`residual.size() == num_time_steps * N_this`), slicing per timestep rather than only supporting single-step residuals.

### Added
- Added unit tests in `tests/simd_utils_tests.cpp`:
  - `SimdUtilsTest.AccumulateFourAndTwoVectorsEquivalence`: Verifies AVX2 vectorized 4-way and 2-way vector accumulation against scalar references across varying vector sizes.
- Added unit tests in `tests/fflayer_tests.cpp`:
  - `FFLayerTest.DirectGradientsFallbackToLayerIndexRnnGradients`: Verifies fallback to `rnn_gradients(this_layer_idx)` when downstream layer is absent and `batch_output_gradients` is empty.
  - `FFLayerTest.ForwardFeedFullSequenceResiduals`: Verifies that full sequence residual connections ($T \times N$) are sliced and accumulated correctly across each timestep.
  - `FFLayerTest.ForwardFeedSequenceDetectionFromHiddenStates`: Verifies correct sequence length detection and output sizing from hidden states when input activations are static.
  - `FFLayerTest.SingleStepFastPathGradientEquivalence`: Verifies that single-step fast path gradient accumulation strictly matches reference gradient calculations.

## [1.1.53] - 2026-09-08

### Optimised
- Optimised `FFLayer` compute throughput, cache locality, and multi-threading efficiency:
  - Fused chunk execution in `FFLayer::calculate_forward_feed` and `FFLayer::calculate_hidden_gradients`: Consolidated bias initialisation, GEMM, and post-GEMM activation/dropout into a single task pass (`run_forward_chunk`), and backward GEMM with post-GEMM derivative accumulation into a single task pass (`run_backward_chunk`). This halves thread synchronization barriers and keeps freshly computed GEMM outputs hot in L1/L2 cache for immediate activation.
  - Zero-copy gradient accumulation for worker thread 0 in `FFLayer::calculate_and_store_gradients`: Worker thread 0 now accumulates directly into `_w_grads` and `_b_grads`, allocating auxiliary accumulator buffers only for threads $1 \dots T-1$ and eliminating an entire redundant buffer allocation, zeroing, and SIMD vector merge pass.
  - Redundant FMADD zero-scalar bypass in `FFLayer::calculate_and_store_gradients_chunk`: Added checks for zero input blocks ($x_0 = x_1 = x_2 = x_3 = 0.0$), skipping useless SIMD load, FMADD, and write-backs across `num_outputs` doubles for inactive or dropped-out neurons.
  - Small-Buffer Optimization (SBO) for activation derivatives and dropout masks: Stack-allocated `mask_buf` in `run_post_gemm` and `deriv_buf` in `run_post_gemm_backward` via `std::array<double, 64>` when $N_{\text{this}} \le 64$, removing heap allocation overhead for typical layer widths.
  - Replaced lambda dispatch with named functor structs (`FfForwardTask`, `FfBackwardTask`, `FfPostGemmBackwardTask`, `FfGradsTask`).
  - Added const accessor `get_thread_grad_accumulators()` in `FFLayer`.

### Fixed
- Fixed uninitialised memory bug in `FFLayer::calculate_forward_feed`: When previous layer RNN outputs were static or single-step (`rnn_in.size() == N_prev`) while `num_time_steps > 1`, now properly broadcasts across all timesteps $t \in [0, num\_time\_steps)$ and zero-fills partial sequences, preventing uninitialised heap memory from feeding into GEMM.
- Fixed out-of-bounds memory read bug in `FFLayer::calculate_and_store_gradients_chunk`: When sequence inputs or incoming gradients had fewer elements than $num\_time\_steps \times N$ (e.g. single-step outputs), strides are safely set to 0 to broadcast step 0 across all timesteps without indexing past buffer bounds.
- Fixed code formatting in `FFLayer::apply_stored_gradients` to enforce Allman brace blocks for all conditional statements.

### Added
- Added unit tests in `tests/fflayer_tests.cpp`:
  - `FFLayerTest.ForwardFeedPartialSequenceBroadcastingSafety`: Verifies safe broadcasting and lack of uninitialised memory when `rnn_in` is single-step and sequence length is 3.
  - `FFLayerTest.GradientsSingleStepBroadcastSafety`: Verifies that single-step inputs and gradients with multi-step sequence lengths accumulate safely without buffer overruns.
  - `FFLayerTest.GradientsZeroScalarSkipEquivalence`: Verifies that zero-scalar skipping in gradient accumulation produces identical mathematical results.
  - `FFLayerTest.MultiThreadedForwardAndBackwardEquivalence`: Verifies exact numerical equivalence between single-threaded (1 thread) and multi-threaded (4 threads) forward feed, backward pass, and gradient storage.

## [1.1.52] - 2026-09-07

### Optimised
- Optimised `GRURNNLayer` cache locality, memory footprint, and backpropagation compute efficiency:
  - Cache-locality loop inversion in `GRURNNLayer::calculate_and_store_gradients_chunk`: Inverted the gradient accumulation loops for candidate, update, and reset gates so outer loops iterate over input neurons $i$ and recurrent neurons $k$ with 4-way unrolling, keeping weight gradient accumulator rows resident in L1 cache while streaming across all batch items and timesteps.
  - One-pass vector addition in SIMD: Added `simd::add_three_vectors(x0, x1, x2, y0, y1, y2, n)` with AVX2 and scalar fallback in `include/neuralnetwork/common/simd_utils.h` to accumulate candidate, update, and reset gate bias gradients in a single memory pass.
  - Optimised `BPTTWorkspace::resize`: Removed unnecessary zero-fill on transient scratch buffers (`rnn_grad_matrix`, `dx_matrix`, `chunk_dz`, `chunk_dr`, `chunk_dh_hat`, `chunk_dh_prev_accum`, `h_hat_vals`, `temp_Uh_T_dh_hat`, `dh_hat_pre_deriv_buf`, `h_hat_pre_buf`, `h_hat_val_buf`, `ln_dy_buf`, `ln_dx_buf`, `ln_zero_buf`), reducing allocation and memory-zeroing overhead between training iterations.
  - Stack-allocated batch item hoisting in `GRURNNLayer::calculate_and_store_gradients_chunk` and `calculate_bptt_batch_chunk`: Hoisted batch item extraction and hidden state pointers into stack buffers (`std::array<..., 64>`), eliminating heap allocations on chunk worker dispatches for standard batches.
  - Fast contiguous SIMD GEMM backpropagation in `calculate_bptt_batch_chunk`: Added contiguous SIMD GEMM path using `W_next_T` when the downstream layer is an `FFLayer`, and added identity bypass (`std::memcpy`) when `next_layer` is `this` or `_identity_proxy`.
  - Transposed weights accessors: Added public const accessors `get_rw_values_T()`, `get_z_rw_values_T()`, `get_r_rw_values_T()`, `get_w_values_T()`, `get_z_w_values_T()`, `get_r_w_values_T()`.
  - Replaced `std::copy` and `std::fill` with `std::memcpy` and `std::memset` across `calculate_forward_feed`, `pre_calculate_gates`, `run_forward_pass`, `finalize_forward_step`, `zero_gradients`, `calculate_and_store_gradients`, and `cache_recurrent_weights`.
  - Replaced lambda functions with named functor structs (`GruPreCalculateGatesTask`, `GruRunForwardPassTask`, `GruBpttTask`) and helper method `apply_gradient_update`.

### Fixed
- Fixed truncated BPTT stale memory bug in `GRURNNLayer::calculate_bptt_batch_chunk`: When truncated BPTT is active (`t_end > 0`), explicitly zeroed `workspace.rnn_grad_matrix` and `workspace.dx_matrix` for timesteps $t \in [0, t\_end)$, preventing uninitialised or stale gradients from corrupting sequence gradients.
- Fixed multi-batch partial RNN input handling in `GRURNNLayer`: In `GRURNNLayer::calculate_and_store_gradients`, computed `any_has_rnn_input` across the entire batch upfront rather than per-chunk, ensuring thread-chunk boundary invariance when handling mixed sequence inputs and preventing corrupted multi-timestep input weight gradients.
- Fixed transposed weights cache freshness:
  - In `GRURNNLayer::accumulate_swa_average_impl`, added call to `cache_recurrent_weights()` to ensure transposed recurrent and input weights reflect the new SWA weights.
  - In `GRURNNLayer::update_lookahead_slow_weights_impl`, `FFLayer::update_lookahead_slow_weights_impl`, and `LSTMLayer::update_lookahead_slow_weights_impl`, added `cache_recurrent_weights()` on the slow layer (`this`) in addition to `other`.
- Verified mathematical validity of dropout in `GRURNNLayer`: Candidate gate activation derivatives use unscaled activation values $\hat{h}_{\text{raw}} \in [-1, 1]$ before inverted dropout, ensuring $1 - \hat{h}^2$ derivative remains bounded and mathematically exact, and incoming gradients are properly masked and scaled by $1 / (1 - p)$.

### Added
- Added unit tests in `tests/grurnnlayer_tests.cpp`:
  - `GRURNNLayerTest.BackpropFromFFLayerWithTransposedWeightsEquivalence`: Verifies numerical equivalence between `FFLayer` contiguous transposed GEMM and standard transposed GEMM backpropagation into GRU.
  - `GRURNNLayerTest.IdentityProxyBypassEquivalence`: Verifies numerical equivalence between `is_identity` direct memcpy bypass and general matrix multiplication.
  - `GRURNNLayerTest.TruncatedBpttZeroesUnprocessedTimesteps`: Verifies that timesteps $[0, t\_end)$ are strictly zeroed under truncated BPTT.
  - `GRURNNLayerTest.BatchItemWithoutPrevLayerInputStillAccumulatesBiasAndRecurrentGradients`: Verifies that an item with missing previous-layer input still contributes bias and recurrent gradients without crashing or corrupting input weight gradients.
  - `GRURNNLayerTest.AnyHasRnnInputIsComputedOverWholeBatchNotPerChunk`: Verifies that `any_has_rnn_input` evaluated at batch level properly gates single-step fallback accumulation for input weights without affecting bias or recurrent gradients.
  - `GRURNNLayerTest.CalculateAndStoreGradientsChunkLargeBatchHeapGrowth`: Tests chunk batch sizes exceeding 64 items to verify heap fallback logic for batch item and pointer arrays.
  - `GRURNNLayerTest.ThreadedGradientAccumulationEquivalence`: Verifies exact gradient accumulation equivalence between single-threaded (1 thread) and multi-threaded (4 threads) dispatch.
  - `GRURNNLayerTest.SwaAndLookaheadWeightCaching`: Verifies that `accumulate_swa_average_impl` and `update_lookahead_slow_weights_impl` refresh transposed weight caches for both slow and fast layers.
  - `GRURNNLayerTest.RecurrentAndInputWeightsFiniteDifferenceMultiBatchNumericalEquivalence`: Verifies analytical gradients for all 6 weight matrices against two-sided finite difference numerical approximations.
  - `GRURNNLayerTest.DropoutMathematicalSoundnessInBPTT`: Verifies that dropout mask scaling and candidate activation derivatives remain within theoretical bounds during training.

## [1.1.51] - 2026-09-05

### Optimised
- Optimised `LSTMLayer` cache locality and backpropagation compute efficiency:
  - Cache-locality loop inversion in `LSTMLayer::calculate_and_store_gradients_chunk`: Inverted the accumulation loops so outer loops tile over input neurons $k$ and recurrent neurons $rk$ with 4-way unrolling, keeping weight gradient accumulator rows resident in L1 cache while accumulating across all batch samples and timesteps.
  - Fast-path backpropagation from `FFLayer` into `LSTMLayer` in `LSTMLayer::calculate_bptt_batch_chunk`: When the downstream layer is an `FFLayer` with an active transposed weight cache (`_w_values_T`), backpropagation uses contiguous SIMD micro-kernels (`simd::gemm_four_batches`, `simd::gemm_two_batches`, and `simd::gemm_one_batch`) instead of transposed dot products.
  - Stack-allocated batch item hoisting in `LSTMLayer::calculate_and_store_gradients_chunk` and `calculate_bptt_batch_chunk`: Hoisted batch item extraction and hidden state pointers into stack buffers (`std::array<..., 64>`), eliminating heap allocations on chunk worker dispatches for standard batches.
  - Replaced `std::copy`, `std::copy_n`, and `std::fill` with `std::memcpy` and `std::memset` across `zero_gradients`, `calculate_output_gradients`, `calculate_hidden_gradients`, and `finalize_forward_step`.
  - Replaced scalar nested transposition loops in `LSTMLayer::cache_recurrent_weights` with vectorised `simd::transpose` for all 4 recurrent weight matrices and 4 input weight matrices.
  - Added identity proxy GEMM bypass in `LSTMLayer::calculate_bptt_batch_chunk`: when backpropagating through `_identity_proxy` (e.g. from `calculate_hidden_gradients_from_output_gradients`), replaces $O(T \cdot N^2)$ matrix-vector multiplications against identity weights with $O(T \cdot N)$ direct `std::copy_n`.
  - Replaced shared thread-0 `workspace.deltas_buf` allocation in `LSTMLayer::calculate_output_gradients` with stack-scoped `TempBuffer<double, 15> deltas_buf(0)`, avoiding workspace pollution and improving cache locality.
- Optimised `FFLayer` and `FFOutputLayer` computational throughput and memory efficiency:
  - Replaced heap-allocated `std::vector<batch_item_info>` in `FFLayer::calculate_and_store_gradients_chunk` with stack-allocated `std::array<batch_item_info, 64>` (falling back to heap only for chunk sizes exceeding 64), avoiding vector allocation churn on every training thread chunk.
  - Extended single-batch input bypass in `FFLayer::calculate_forward_feed` to cover single-timestep batches (`num_time_steps == 1`), eliminating redundant input matrix allocations and copies when evaluating individual samples.
  - Pre-allocated `rnn_grads_row` to sequence length in `FFLayer::run_post_gemm_backward` and added a fast-path for `linear` activations that replaces derivative evaluations and 3-vector multiplications with direct vector copies or single mask multiplications.
  - Eliminated duplicate implementations of `calculate_forward_feed` and `run_post_gemm` in `FFOutputLayer`, allowing it to inherit `FFLayer`'s multi-threaded GEMM and post-GEMM dispatch, input buffer bypass, and zero-copy activation paths.
  - Cached Sharpe and Sortino loss head presence via `_has_sharpe_sortino_heads` in `FFOutputLayer`, avoiding allocation of `PerHeadStepContexts` and execution of `calculate_sharpe_sortino_context` on every training batch for standard regression and classification heads.
  - Optimised `FFOutputLayer::run_output_gradients` by hoisting dropout checks, bypassing `cell_state_values` reads and redundant 1.0 multiplications when dropout is inactive, and adding a direct copy fast-path for linear output activations.
  - Accurately pre-reserved unrolled sequence metrics containers in `FFOutputLayer::calculate_output_metrics` based on total sample steps to avoid reallocations.
  - Replaced `std::fill` and `std::copy_n` with `std::memset` and `std::memcpy` in `calculate_hidden_gradients`, `calculate_hidden_gradients_from_output_gradients`, `calculate_and_store_gradients`, `run_post_gemm`, and `run_post_gemm_backward`.
  - Replaced `std::fill` on `rnn_grads_row` and `deltas` with `std::memset` in `FFOutputLayer::run_output_gradients`, and replaced `std::copy_n` with `std::memcpy` for output gradient assignments.
- Optimised `ResidualProjector` batch projection and SWA updates:
  - Replaced scalar and element-wise projection loops across `project(const std::vector<double>&)`, `project(const double*, double*)`, `project_batch`, and `project_batch_into` with AVX2/FMA register-blocked SIMD micro-kernels (`simd::gemm_four_batches`, `simd::gemm_two_batches`, and `simd::gemm_one_batch`). This processes 4 batch samples simultaneously in AVX2 registers, loading weights once per 4 samples and eliminating repeated memory writebacks.
  - Added overloaded `ResidualProjector::project_batch_into(const std::vector<std::vector<double>>&, std::vector<std::vector<double>>&)` and flat pointer `ResidualProjector::project_batch(const double*, double*, size_t)` to eliminate dynamic allocations of pointer vectors during batch projection.
  - Hoisted residual input and output buffers outside the forward feed layer loop in `NeuralNetwork::calculate_forward_feed`, reusing allocated memory with `project_batch_into` on every training batch.
  - Vectorised `ResidualProjector::accumulate_swa_average` using `simd::swa_step` with precalculated `alpha = 1.0 / denom`, replacing the element-wise division loop.
  - Reused pre-allocated `WeightParam` vector rows in `ResidualProjector::get_weight_params()` to eliminate allocation churn on dirty cache reads.
- Added vectorised `simd::swa_step` and scalar fallback in `include/neuralnetwork/common/simd_utils.h`, and vectorised `Layer::swa_average_into` in `include/neuralnetwork/layers/layer.h` to accelerate SWA updates across all layers.
- Implemented Rule-of-Five copy and move assignment operators for `ResidualProjector` (`operator=(const ResidualProjector&)` and `operator=(ResidualProjector&&)`) with deadlock-free `std::scoped_lock` on `_cache_mutex`.
- Added `#if VALIDATE_DATA == 1` bounds checking in `ResidualProjector::apply_weight_gradient`.

### Fixed
- Fixed missing brace in `FFLayer::run_post_gemm` after `batch_hidden_states` check that erroneously nested forward pass execution inside `!batch_hidden_states.empty()`. Forward pass now properly computes outputs during inference when `batch_hidden_states` is empty.
- Fixed gradient retrieval fallback in `FFLayer::calculate_hidden_gradients_from_output_gradients`: Added fallback to `get_gradients(get_layer_index())` when `get_gradients(get_layer_index() + 1)` is empty, ensuring correct direct gradient propagation.
- Fixed cross-timestep dropout mask persistence in `FFLayer::run_post_gemm` by re-initialising `mask_buf` inside the timestep loop.
- Fixed index selection in `NeuralNetwork::calculate_forecast_metrics_all_layers_impl` at final training checkpoint:
  - At the final training checkpoint, the helper passed to `progress_callback` was not yet pushed to `_neural_network_helpers`, causing `_neural_network_helpers.back()` to resolve to the penultimate epoch and evaluate against `checking_indexes()` instead of `final_check_indexes()`.
  - Resolved by passing the calling `NeuralNetworkHelper` directly via `NeuralNetwork::calculate_forecast_metrics_all_layers_for_helper`.
- Fixed uninitialised / stale memory leak in `LSTMLayer::calculate_bptt_batch_chunk` under truncated BPTT (`bptt_max_ticks > 0`):
  - When truncated BPTT is configured, the BPTT loop stops at `t_end > 0`, leaving timesteps $[0, t\_end - 1]$ in `workspace.dx_matrix` and `workspace.rnn_grad_matrix` unpopulated. Upstream layers copying sequence gradients then received uninitialised memory or stale gradients from prior iterations.
  - Added explicit zeroing of `dx_matrix` and `rnn_grad_matrix` for the truncated range $[0, t\_end)$ whenever `t_end > 0`.
- Fixed BPTT sequence gradient propagation bug in `FFLayer::calculate_hidden_gradients`:
  - `calculate_hidden_gradients` previously read only single-timestep gradients from `next_layer` via `get_gradients`, ignoring `get_rnn_gradients`. In multi-timestep sequences ($T > 1$), this caused `FFLayer` to treat the final timestep gradient as a single-element gradient and broadcast it across all timesteps, discarding the true per-timestep backpropagated gradients.
  - Resolved by querying `get_rnn_gradients` first across both single-item bypass and batch processing paths.
- Fixed stale transposed weight cache in `FFOutputLayer`:
  - `FFOutputLayer::apply_stored_gradients` updated `_w_values` but omitted calling `cache_recurrent_weights()`. Consequently, `_w_values_T` remained frozen with initial weights, causing upstream layers backpropagating through `FFOutputLayer` via `get_w_values_T()` to backpropagate against stale pre-update weights.
  - Resolved by invoking `cache_recurrent_weights()` at the end of `FFOutputLayer::apply_stored_gradients` and in `FFLayer::accumulate_swa_average_impl`.
- Fixed Clang compiler error in `ResidualProjectorTest.CopyAndMoveAssignmentOperators`: Suppressed `-Wself-assign-overloaded` warning when explicitly validating self-assignment operator correctness by assigning through reference under diagnostic pragmas.
- Fixed intermittent timing failure in `NeuralNetworkHelperTest.TrainingMonitorsPersistAcrossProgressCallbacks`: Provided adequate sample and epoch breathing room for the asynchronous worker thread to process progress callbacks, and aligned checkpoint assertion to `>= 3` matching the minimum evaluation window.
- Verified mathematical validity of dropout in `ResidualProjector`, `LSTMLayer`, `GRURNNLayer`, `FFLayer`, and `FFOutputLayer`:
  - Output dropout is applied strictly to activations with mask $m \in \{0, \frac{1}{1-p}\}$. Incoming gradients are scaled by the dropout mask $m$ during backpropagation, and non-linear activation derivatives are computed from pre-activation values $z$ without inverted dropout corruption.

### Added
- Added unit tests in `tests/lstmlayer_tests.cpp`:
  - `LSTMLayerTest.BackpropFromFFLayerWithTransposedWeightsEquivalence`: Verifies numerical equivalence between `FFLayer` contiguous transposed GEMM and standard transposed GEMM backpropagation.
  - `LSTMLayerTest.ThreadedGradientAccumulationEquivalence`: Verifies exact gradient accumulation equivalence between single-threaded and multi-threaded dispatch with the L1-cache tiled outer loop.
  - `LSTMLayerTest.ResidualCandidateGateInjectionAcrossTimesteps`: Verifies multi-step candidate gate residual injection against analytical values.
  - `LSTMLayerTest.ResidualWithDropoutAndInferenceEquivalence`: Verifies forward and backward stability under residual projection with dropout in training and inference modes.
  - `LSTMLayerTest.TruncatedBpttZeroesUnprocessedTimesteps`: Verifies that timesteps $[0, t\_end - 1]$ are strictly zeroed when `bptt_max_ticks > 0`.
  - `LSTMLayerTest.IdentityProxyBypassEquivalence`: Verifies that `calculate_hidden_gradients_from_output_gradients` via `_identity_proxy` bypass matches direct identity GEMM backpropagation.
  - `LSTMLayerTest.DropoutWithTanhActivationDerivative`: Verifies that cached candidate and cell state activations remain strictly in $[-1, 1]$ under dropout without corruption from the inverted dropout mask.
  - `LSTMLayerTest.RecurrentWeightsFiniteDifferenceWithDropout`: Verifies analytical vs finite-difference numerical gradients for recurrent weights under deterministic dropout.
  - `LSTMLayerTest.CalculateAndStoreGradientsChunkLargeBatchHeapGrowth`: Verifies dynamic heap expansion in `calculate_and_store_gradients_chunk` for batch sizes exceeding the 64-item stack buffer threshold ($B = 70$).
  - `LSTMLayerTest.RecurrentAndInputWeightsFiniteDifferenceMultiBatchNumericalEquivalence`: Verifies analytical vs finite-difference numerical gradients for recurrent and candidate input weights across multiple batch sequences ($B = 2, T = 3$).
  - `LSTMLayerTest.BatchItemWithoutPrevLayerInputStillAccumulatesBiasAndRecurrentGradients`: Verifies that batch samples lacking sequence inputs from the preceding layer still properly accumulate bias and recurrent weight gradients without corrupting or skipping the item.
- Added unit tests in `tests/fflayer_tests.cpp`:
  - `FFLayerTest.DirectGradientsFallbackToLayerIndex`: Verifies fallback gradient retrieval at `get_layer_index()` when `get_layer_index() + 1` is empty.
  - `FFLayerTest.DirectGradientsFallbackToLayerIndexMultiBatch`: Verifies fallback gradient retrieval across multi-sample batches ($B > 1$) in `calculate_hidden_gradients_from_output_gradients`.
  - `FFLayerTest.ForwardFeedEmptyHiddenStatesInference`: Verifies forward feed execution during inference with empty hidden states.
  - `FFLayerTest.CalculateAndStoreGradientsLargeBatchHeapGrowth`: Verifies dynamic heap expansion in `calculate_and_store_gradients_chunk` for batches exceeding 64 items.
  - `FFLayerTest.RecurrentSequenceGradientBPTT`: Verifies that multi-timestep sequence gradients from recurrent next layers propagate accurately through `FFLayer` without being overwritten by broadcast final-step gradients.
  - `FFLayerTest.RecurrentSequenceGradientBPTT_GeneralLoopMultiBatch`: Verifies multi-timestep sequence gradient propagation across multiple batch items ($B > 1$) through the general processing loop.
  - `FFLayerTest.WeightGradientsMultiTimestepAccumulation`: Verifies exact analytical $X^T G$ weight gradient and bias gradient accumulation across multi-timestep sequences.
- Added unit tests in `tests/ffoutputlayer_tests.cpp`:
  - `FFOutputLayerTest.SequenceMultiTimestepOutputGradients`: Verifies multi-timestep sequence output gradient computation across linear and sigmoid heads.
  - `FFOutputLayerTest.OutputGradientsSingleTargetBroadcastToLastStep`: Verifies single-target broadcast behavior to final timestep with 0-gradient padding for prior steps.
  - `FFOutputLayerTest.TransposedWeightsCacheUpdateOnApplyGradients`: Verifies that `_w_values_T` is accurately synchronised after gradient application in `FFOutputLayer`.
  - `FFOutputLayerTest.DropoutMultiTimestepGradientFlow`: Verifies multi-timestep forward feed dropout masking and corresponding backward gradient scaling across timesteps.
  - `FFOutputLayerTest.MultithreadedForwardWithResidualConsolidationEquivalence`: Verifies that multi-threaded chunked forward feed produces bit-identical outputs to single-threaded execution when residual connections are enabled on `FFOutputLayer`.
- Added unit tests in `tests/grurnnlayer_tests.cpp`:
  - `GRURNNLayerTest.ResidualCandidateGateInjectionAcrossTimesteps`: Verifies GRU candidate hidden state residual projection integration across timesteps.
  - `GRURNNLayerTest.ResidualWithDropoutAndInferenceEquivalence`: Verifies GRU forward and backward gradient stability under residual projection with dropout.
- Added comprehensive unit tests in `tests/residualprojector_tests.cpp`:
  - `ResidualProjectorTest.CopyAndMoveAssignmentOperators`: Verifies copy and move assignment semantics, deep copying, and self-assignment safety.
  - `ResidualProjectorTest.ProjectBatchMultiBatchSizesEquivalence`: Verifies numerical equivalence across batch sizes (1, 2, 3, 4, 7, 8, 15, 16, 33, 64) between single-sample projection, vector-of-vectors batch projection, raw pointer batch projection, and flat pointer buffer projection.
  - `ResidualProjectorTest.ProjectRawPointersBufferInto`: Verifies raw pointer buffer projection into pre-allocated memory and nullptr safety.
  - `ResidualProjectorTest.AccumulateSwaAverageVectorisedEquivalence`: Verifies vectorised SWA running mean against exact arithmetic mean across multiple snapshot projectors.
  - `ResidualProjectorTest.ZeroDimensionEdgeCases`: Verifies safe handling of 0 input size and 0 output size without errors or crashes.
- Added unit tests in `tests/neuralnetworkhelper_tests.cpp`:
  - `NeuralNetworkHelperTest.InCallbackForecastMetricsAtFinalCheckpointUsesFinalCheckIndexes`: Verifies that evaluating forecast metrics directly inside `progress_callback` at the final epoch uses `final_check_indexes` rather than falling back to `checking_indexes`.
- Added unit tests in `tests/swa_tests.cpp`:
  - `SwaTests.LayerRunningMeanRefreshesTransposedWeightsCache`: Verifies that `accumulate_swa_average` refreshes the cached transposed weight matrix `_w_values_T`.

## [1.1.50] - 2026-09-05

### Fixed
- Fixed mathematical bug in activation derivative calculation under dropout across `FFLayer`, `ElmanRNNLayer`, `TcnLayer`, `AttentionPoolLayer`, `EmbeddingLayer`, and `FFOutputLayer`:
  - When dropout was active, `activate_derivative` was receiving dropout-scaled output values `y_vals` ($m \odot f(z)$), corrupting derivative evaluation for non-linear activations ($\tanh$, $\text{sigmoid}$, $\text{elu}$, $\text{selu}$) and producing incorrect or negative derivatives ($1 - m^2 \tanh^2(z) < 0$).
  - Corrected backward pass across all affected layers by passing `nullptr` for `y_vals` when `has_dropout` is true, ensuring $f'(z)$ is evaluated mathematically accurately directly from pre-activation values $z$ before applying the dropout mask.

### Added
- Added unit tests in `tests/fflayer_tests.cpp`:
  - `FFLayerTest.DropoutWithTanhActivationDerivative`: Verifies analytical derivative calculation with $\tanh$ activation under dropout, ensuring derivatives remain strictly positive and match $g \cdot f'(z) \cdot \text{scale}$.
  - `FFLayerTest.DropoutWithSigmoidActivationDerivative`: Verifies analytical derivative calculation with sigmoid activation under dropout.
  - `FFLayerTest.DropoutMultiTimestepSequenceForwardFeed`: Verifies multi-timestep sequence execution with dropout across all sequence steps.
- Added unit tests in `tests/ffoutputlayer_tests.cpp`:
  - `FFOutputLayerTest.DropoutWithTanhActivationDerivative`: Verifies output gradient calculation with $\tanh$ activation under dropout.
- Added unit tests in `tests/elmanrnnlayer_tests.cpp`:
  - `ElmanRNNLayerTest.DropoutWithTanhActivationDerivative`: Verifies BPTT gate gradients under $\tanh$ activation with dropout.
- Added unit tests in `tests/tcnlayer_tests.cpp`:
  - `TcnLayerTest.DropoutNotInference`: Verifies dropout is completely bypassed during inference mode.
  - `TcnLayerTest.DropoutWithTanhActivationDerivative`: Verifies TCN gradient calculations under $\tanh$ activation with dropout.
- Added unit tests in `tests/attentionpoollayer_tests.cpp`:
  - `AttentionPoolLayerTest.DropoutNotInference`: Verifies pooling outputs remain unscaled and un-dropped during inference mode.
  - `AttentionPoolLayerTest.DropoutWithTanhActivationDerivative`: Verifies pooling backward gradients under $\tanh$ activation with dropout.
- Added unit tests in `tests/embeddinglayer_tests.cpp`:
  - `EmbeddingLayerTest.DropoutNotInference`: Verifies categorical embedding outputs during inference mode.
  - `EmbeddingLayerTest.DropoutWithTanhActivationDerivative`: Verifies embedding backward gradients under $\tanh$ activation with dropout.
- Added unit tests in `tests/selfattentionlayer_tests.cpp`:
  - `SelfAttentionLayerTest.DropoutNotInference`: Verifies that self-attention inference mode matches 0.0 dropout execution.
  - `SelfAttentionLayerTest.DropoutConsistencyVerification`: Verifies 100% dropout forward zeroing and backward gradient zeroing in self-attention layers.

## [1.1.49] - 2026-09-03

### Fixed
- Fixed segmentation fault in concurrent multi-threaded inference (`NetworkIntegrationTest.ThinkConcurrentMultiThreadedInference`):
  - Removed fragile `static thread_local AlignedVector` buffers from `FFLayer` and `ElmanRNNLayer` forward pass, post-GEMM, and gradient calculation functions, replacing them with standard stack-scoped `TempBuffer` allocations. This prevents unsafe TLS destructors from executing during thread termination when ephemeral worker threads exit.
  - Removed recursive profiling instrumentation (`MYODDWEB_PROFILE_FUNCTION`) from `AlignedAllocator` methods (`allocate`, `deallocate`, `construct`, `destroy`, `max_size`, and comparison operators), eliminating heap allocation via `std::stringstream` inside deallocator teardown paths and eliminating allocator lock contention.
  - Guarded `AlignedAllocator::allocate` against `n == 0` and `deallocate` against null pointers to ensure well-defined cross-platform behavior on Linux POSIX and Windows.

### Added
- Added comprehensive unit test suite in `tests/residualprojector_tests.cpp` covering all constructors, factory methods, single and batched vector projections, in-place projection buffer states, weight gradient application with decay and clipping, direct weight delta updates, lazy `WeightParam` caching and invalidation, running-mean SWA accumulation, and Lookahead slow weight updates.

## [1.1.48] - 2026-09-03

### Optimised
- Optimised `ElmanRNNLayer` memory allocations, cache locality, and compute efficiency:
  - Eliminated continuous per-iteration dynamic heap allocations via `TempBuffer` across forward feed and output gradient calculation by utilising aligned, thread-isolated scratch buffers (`static thread_local AlignedVector`).
  - Added single-sample zero-copy input sequence bypass in `calculate_forward_feed` when `batch_size == 1` and inputs are already contiguous.
  - Enabled multi-threaded execution during evaluation and inference in `calculate_forward_feed`.
  - Unrolled recurrent pass GEMM across batches 4-wide (`simd::gemm_four_batches`), 2-wide, and 1-wide using `_rw_values.data()`, while skipping $t = 0$ recurrent GEMM because $h_{-1} = \mathbf{0}$.
  - Added dropout bypass fast path in `calculate_forward_feed` when dropout is disabled (`!(is_training && get_dropout() > 0.0)`).
  - Optimised weight gradient accumulation in `calculate_and_store_gradients_chunk` with 4-wide unrolling via `simd::mul_add_four_scalars` (and 2-wide via `simd::mul_add_two_scalars`) for both input weights and recurrent weights, broadcasting gradient vector $g_t$ across 4 rows and reducing memory read traffic by 75%.
  - Added `std::span<double>` overload for `calculate_and_store_gradients_chunk` and replaced per-thread gradient vectors with `AlignedVector` accumulators in `_thread_grad_accumulators`.
  - Eliminated redundant buffer memset-zeroing in `BPTTWorkspace::resize` for buffers overwritten every tick (`rnn_grad_matrix`, `deriv_buf`, `pre_act_buf`, `hidden_val_buf`).
  - Vectorised `calculate_output_gradients` deltas subtraction using `simd::sub_vectors`.
  - Replaced scalar transpose loops with cache-blocked SIMD transpose (`simd::transpose`) in `cache_recurrent_weights()`.
  - Bypassed GEMM in `calculate_bptt_batch_chunk` when `next_layer` is `_identity_proxy` by directly copying gradients via `std::copy_n`.

### Added
- Added unit tests in `tests/elmanrnnlayer_tests.cpp`:
  - `ElmanRNNLayerTest.CalculateOutputGradientsFullSequenceEquivalence`: Verifies sequence output gradient deltas and last-tick gradient projection.
  - `ElmanRNNLayerTest.BatchedForwardFeedUnrollingEquivalence`: Verifies that 4-wide, 2-wide, and 1-wide unrolled batched forward pass matches single-sample execution.
  - `ElmanRNNLayerTest.ResidualConnectionsForward`: Verifies that residual connections are correctly accumulated into pre-activation sums before activation.
  - `ElmanRNNLayerTest.BPTTMaxTicksTruncation`: Verifies BPTT gradient propagation truncation when `bptt_max_ticks` is set.
  - `ElmanRNNLayerTest.RecurrentWeightsTransposedCacheAndAccessors`: Verifies SIMD transposed weight caching and recurrent weight accessors.
  - `ElmanRNNLayerTest.AdamOptimiserAndSettersCoverage`: Verifies Adam optimiser state updates, setters, getters, and cloning for recurrent weights.
  - `ElmanRNNLayerTest.SpanAndVectorOverloadEquivalence`: Verifies that `std::span` and `std::vector` overloads for gradient calculation produce identical results.

## [1.1.47] - 2026-09-03

### Optimised
- Optimised `FFLayer` buffer allocations and memory bandwidth across forward and backward passes:
  - Replaced per-call dynamic heap allocations with aligned thread-isolated scratch buffers (`thread_local AlignedVector`), eliminating continuous heap allocations and deallocations during training and evaluation iterations while preserving full thread safety for concurrent multi-threaded inference.
  - Added zero-copy contiguous sequence bypass in `calculate_forward_feed` when `batch_size == 1` and inputs are already contiguous in memory.
  - Added zero-copy contiguous gradient bypass in `calculate_hidden_gradients` and `calculate_hidden_gradients_from_output_gradients` when `batch_size == 1`.
  - Optimised weight gradient accumulation in `calculate_and_store_gradients_chunk` with 4-wide unrolling via `simd::mul_add_four_scalars` (and 2-wide via `simd::mul_add_two_scalars`), broadcasting gradient vector $g_t$ across 4 rows and reducing memory read traffic by 75%.
  - Added fast-path activation and direct output copying in `run_post_gemm` when dropout is disabled (`!(is_training && get_dropout() > 0.0)`), bypassing intermediate sequence buffer copies.
  - Added fast-path derivative multiplication in `run_post_gemm_backward` using `simd::mul_vectors` when `get_dropout() == 0.0`, eliminating mask loads and a redundant vector multiplication per neuron.
  - Enabled multi-threaded execution during evaluation and inference in `calculate_forward_feed` for large batch sizes.

### Added
- Added unit tests in `tests/fflayer_tests.cpp`:
  - `FFLayerTest.GradientAccumulationFourWideEquivalence`: Verifies 4-wide, 2-wide, and scalar unrolled gradient accumulation against ground-truth scalar computation across varying input and output dimensions.
  - `FFLayerTest.InferenceMultiThreadingConsistency`: Verifies that multi-threaded evaluation during inference produces identical results to single-threaded execution across multi-step sequences.
  - `FFLayerTest.SingleSampleContiguousBypassVerification`: Verifies consecutive zero-copy bypass execution across multiple sequence iterations with `batch_size == 1`.

## [1.1.46] - 2026-09-03

### Optimised
- Optimised `LSTMLayer` memory bandwidth and buffer allocations during backpropagation:
  - Eliminated redundant memset-zeroing in `bptt_workspace::resize` for intermediate output buffers (`rnn_grad_matrix`, `dx_matrix`, `chunk_df`, `chunk_di`, `chunk_do`, `chunk_dg`, `dh_curr`, `dc_act_deriv`, `dg_act_deriv`), calling `.resize(...)` directly and saving megabytes of cache-polluting writes per BPTT batch chunk.
  - Avoided zeroing layer normalisation buffers (`ln_dy_buf`, `ln_dx_buf`, `ln_dc_next_substitute_buf`, `ln_c_gain_grad_accum`, `ln_c_bias_grad_accum`) when `_use_layer_normalisation` is disabled.
  - Bypassed redundant input sequence copying and memory allocation in `calculate_forward_feed` and `pre_calculate_gates` when `batch_size == 1` and input is already a contiguous RNN sequence.
  - Eliminated redundant buffer memset zeroing in `calculate_output_gradients` by using `.resize(...)` on `workspace.deltas_buf` instead of `.resize_and_zero(...)`.
  - Added AVX2-vectorised `simd::scale_four_vectors` kernel to scale 4 gate matrices simultaneously (`_w_grads`, `_rw_grads`, `_b_grads`) in `calculate_and_store_gradients`, reducing loop passes and function call overhead by 75%.

### Fixed
- Fixed recurrent gradient accumulation across timesteps in `LSTMLayer::run_recurrent_gemm_backward`:
  - Zeroed `dh_next_batch` prior to `gemm_four_matrices_*` accumulation, preventing recurrent gradients from leaking and accumulating across sequence steps ($T > 2$).

### Added
- Added unit test `SimdUtilsTest.ScaleFourVectorsEquivalence` verifying AVX2-fused 4-vector scaling against scalar implementation across various vector lengths.
- Added unit test `LSTMLayerTest.RecurrentWeightsFiniteDifferenceNumericalEquivalence` confirming multi-step analytical recurrent gradients match numerical finite differences across all 4 gates.

## [1.1.45] - 2026-09-02

### Added
- Added `ErrorResult` structure returning metric `ratio` along with optional sample counts (`numerator` and `denominator`) across ratio-based error and metric functions:
  - `calculate_directional_accuracy`, `calculate_softmax_directional_accuracy`: reports matching direction count and total non-neutral sample count.
  - `calculate_directional_confidence_score`, `calculate_softmax_directional_confidence_score`: reports confident matching direction count and total confident sample count.
  - `calculate_prediction_coverage`, `calculate_softmax_prediction_coverage`: reports confident sequence count and total sequence count.
  - Non-ratio continuous loss functions (MSE, MAE, RMSE, NRMSE, Huber, Log-Cosh, BCE, Cross-Entropy, Quantile, Sharpe, Sortino) return `std::nullopt` for `numerator` and `denominator`.
- Extended `NeuralNetworkHelperMetrics` with `numerator()` and `denominator()` accessors, copy and move constructors/operators, and a 4-argument constructor accepting sample counts.
- Added `force_checking_indexes` configuration to `NeuralNetworkOptions`:
  - Added builder method `with_force_checking_indexes(bool)` and accessor `force_checking_indexes() const noexcept`.
  - Initialised in constructor, copy constructor, move constructor, and copy/move assignment operators.
  - Defaulted to `false` in `NeuralNetworkOptions::create(...)`.
- Added serialisation and deserialisation support for `"force-checking-indexes"` in `NeuralNetworkSerializer`.
- Updated `NeuralNetwork::calculate_forecast_metrics_all_layers` and `NeuralNetworkHelper::calculate_forecast_metrics`:
  - Updated `force_checking_indexes` parameter to `std::optional<bool> force_checking_indexes = std::nullopt`.
  - When omitted (`std::nullopt`), dynamically resolves to `options.force_checking_indexes()`.
  - When resolved to `true` with `in_sample = false`, forces evaluation against `checking_indexes()` even when `epoch >= number_of_epoch`, allowing consistent side-by-side metric comparison against validation batches without switching to the smaller `final_check_indexes()`.
- Fixed and enriched training logging:
  - In `NeuralNetwork::log_training_info`:
    - Added logging for `Force checking indexes`.
    - Added parity to single-output layer configuration logging to match multi-output layers (`epsilon`, `label-smoothing`, `quantiles`, `tran. cost penalty`, and `sortino target return`).
    - Added logging for `Lookahead` optimizer configuration (synchronisation period and slow weights step size).
    - Added logging for `Stochastic Weight Averaging` (start percent and update percent).
    - Added logging for random `Seed` when configured.
  - In `NeuralNetwork::train`:
    - Enriched post-training final error logs to print sample count ratios `(numerator/denominator)` whenever count details are available on ratio-based metrics.
- Added unit tests in `tests/error_calculation_tests.cpp`, `tests/neuralnetworkhelper_tests.cpp`, and `tests/network_integration_tests.cpp`:
  - `ErrorResultCountsForRatioMetrics`: Verifies accurate count derivation for directional accuracy, confidence score, and coverage metrics across linear, tanh, and softmax heads.
  - `ErrorResultNulloptForNonRatioMetrics`: Verifies `std::nullopt` counts across all continuous loss types.
  - `NeuralNetworkHelperMetricsNumeratorDenominatorLifecycle`: Verifies default, 2-argument, 4-argument construction, copy, and move semantics.
  - `ForceCheckingIndexesSelectsValidationSetAtFinalEpoch`: Verifies index set selection between `checking_indexes()` and `final_check_indexes()` via captured helper and `NeuralNetwork`.
  - `OptionsForceCheckingIndexesConfiguredTrueDefaultsToValidationSet`: Verifies that configuring `with_force_checking_indexes(true)` on options defaults post-training evaluation to `checking_indexes()`, and can be overridden with explicit `false`.
  - `ForceCheckingIndexesOptionAndSerialization`: Verifies model options configuration and JSON round-trip serialisation/deserialisation of `force_checking_indexes`.
- Updated Python bindings in `python/bindings.cpp` and documentation in `python/README.md` and `README.md`:
  - Exposed `numerator` and `denominator` read-only properties on `nn.NeuralNetworkHelperMetrics`.
  - Exposed `with_force_checking_indexes` and `force_checking_indexes` on `nn.NeuralNetworkOptions`.
  - Exposed `force_checking_indexes = None` default parameter in `calculate_forecast_metrics` and `calculate_forecast_metrics_all_layers`.

## [1.1.44] - 2026-09-01

### Optimised
- Optimised `TcnLayer` forward feed, backpropagation, and multi-threaded scaling:
  - Vectorised forward convolution taps using AVX2 GEMM kernels (`simd::gemm_four_matrices_one_batch`, `simd::gemm_two_matrices_one_batch`, `simd::gemm_one_matrix_one_batch`), processing up to 4 convolution taps in a single fused pass with column-wise AVX2 registers.
  - Eliminated per-batch heap memory allocations in forward feed and backward feed by introducing `tcn_workspace` with reusable `AlignedVector<double, 32>` buffers (`pre_act`, `output`, `mask`, `out_seq`, `deriv`, `d_pre_act`, `d_prev`, `raw_delta`).
  - Parallelised upstream raw delta computation directly within worker thread ranges, eliminating serial main-thread overhead and intermediate vector-of-vectors allocations.
  - Replaced inner loop backward dot products with vectorised `simd::gemv_add` across all dilated convolution taps.
  - Optimised weight gradient accumulation in `accumulate_gradients_range` with 4-way and 2-way unrolled rank-1 updates (`simd::mul_add_four_scalars`, `simd::mul_add_two_scalars`), reusing delta vectors across 4 channels in AVX registers and cutting delta memory loads by 75%.
  - Reused 32-byte `AlignedVector<double, 32>` thread accumulators and C++20 `std::span<double>` parameter passing for multi-threaded gradient accumulation.
- Optimised `LSTMLayer` forward feed, backpropagation, and multi-threaded scaling:
  - Eliminated heap memory reallocations during forward feed by embedding `forward_workspace` with reusable `AlignedVector` buffers inside `BPTTWorkspace` and reusing pre-allocated batch scratch buffers (`_forward_flattened_inputs`, `_forward_batch_pre_act`, `_forward_batch_output_sequences`).
  - Enabled multi-threading during evaluation/inference (`is_training = false`) in `calculate_forward_feed`, removing an artificial single-thread restriction for evaluation and prediction workloads.
  - Bypassed recurrent GEMM matrix multiplications on initial timestep ($t=0$) in `run_forward_pass` since the initial hidden state $h_{-1}$ is zero.
  - Parallelised input flattening and output sequence writeback directly within worker task slices (`pre_calculate_gates` and `run_forward_pass`), eliminating serial main-thread overhead and ensuring hot L1/L2 cache locality for subsequent GEMMs.
  - Replaced `std::vector<double>` in `thread_grad_accumulators` with 32-byte `AlignedVector<double, 32>` and updated `calculate_and_store_gradients_chunk` with C++20 `std::span<double>`, guaranteeing AVX2 alignment and zero-reallocation gradient accumulation.
  - Reused `deltas_buf` from workspace in `calculate_output_gradients`, avoiding intermediate heap vector allocations per batch element.
  - Streamlined `calculate_bptt_batch_chunk` and `calculate_and_store_gradients_chunk` by hoisting input shape checks and eliminating redundant buffer fills and conditional branches within inner timestep loops.
  - Converted lambda closures in `calculate_forward_feed`, `calculate_hidden_gradients`, `calculate_and_store_gradients`, and `apply_stored_gradients` to named private methods (`pre_calculate_gates`, `run_forward_pass`, `apply_gradient_update`) and functor task structs (`lstm_forward_precalc_task`, `lstm_forward_recurrent_task`, `lstm_bptt_chunk_task`, `lstm_grad_calc_task`).

### Added
- Added multi-threading invariance unit tests in `tests/tcnlayer_mt_tests.cpp`:
  - `InferenceForwardFeedMTConsistency`: Verifies multi-threaded inference consistency with `is_training = false` across odd batch sizes.
  - `SingleStepInferenceAndTrainingThreadCountInvariance`: Verifies forward feed, backward feed, and gradient calculation for single-step ($T=1$) sequences across 1, 2, 4, 8 threads.
  - `LargeKernelAndDilationThreadCountInvariance`: Verifies gradient invariance across thread counts for large kernel ($K=7$) and dilation ($D=4$) configurations.
- Added multi-threading invariance unit tests in `tests/lstmlayer_mt_tests.cpp`:
  - `OddBatchSizeAllGradsThreadCountInvariance`: Verifies that forward feed, hidden gradients, and all gate/recurrent weight and bias gradients are strictly invariant across thread counts (1, 2, 4, 8) with odd batch sizes.
  - `InferenceForwardFeedMTConsistency`: Verifies multi-threaded inference consistency with `is_training = false`.
  - `SingleStepInferenceAndTrainingThreadCountInvariance`: Verifies forward feed and gradient calculation across all thread counts for single-step ($T=1$) sequences.

## [1.1.43] - 2026-09-01

### Refactored
- Encapsulated `SelfAttentionLayer` internal multi-threading task structures and positional encoding:
  - Moved `add_positional_encoding` into `SelfAttentionLayer` as a `private static` member function with dedicated Tracy profiling instrumentation (`MYODDWEB_PROFILE_FUNCTION("SelfAttentionLayer")`).
  - Nested worker task functors (`self_attention_forward_task`, `self_attention_finish_hidden_gradients_task`, `self_attention_grad_calc_task`) and thread accumulators (`thread_grad_accumulators`, `grad_accumulators`) as `private` structs inside `SelfAttentionLayer`.
  - Moved multi-threading range processing methods (`process_forward_range`, `finish_hidden_gradients_range`, and `accumulate_gradients_range`) from `public:` to `private:`, purifying the public API surface of `SelfAttentionLayer`.
  - Added const accessor `get_pe_inv_denom()` on `SelfAttentionLayer` for inspecting precomputed positional encoding inverse frequency denominators.
  - Added `[[nodiscard]] inline bool is_recurrent() const noexcept` to `LayerDetails`.

### Optimized
- Optimized `TcnLayer` forward feed, backpropagation, and multi-threaded scaling:
  - Vectorised `process_forward_range` with `simd::mul_add` streaming contiguously across weight rows, eliminating strided cache misses and removing per-timestep `gathered` buffer zero-filling and memory copies.
  - Vectorised `finish_hidden_gradients_range` with `simd::dot_product`, skipping padded taps and eliminating intermediate `d_gathered` scratch vectors.
  - Vectorised `accumulate_gradients_range`, eliminating per-timestep zeroing and buffer allocations.
  - Implemented multi-threaded gradient accumulation in `calculate_and_store_gradients` using thread-local accumulators (`thread_tcn_grad_accumulators`) and parallel reduction via `simd::add_vectors`, resolving a major single-threaded performance bottleneck during batch training.
  - Replaced task queue lambda closures with private nested functor structs (`tcn_forward_task`, `tcn_finish_hidden_gradients_task`, `tcn_grad_calc_task`).
  - Added unit tests for dilation zero-padding on short sequences (`DilationLongerThanSequencePaddingBehavior`), multi-batch numerical soundness (`MultiBatchForwardAndBackwardNumericalSoundness`), and odd batch size thread-count invariance (`OddBatchSizeAllGradsThreadCountInvariance`).

### Added
- Extended Python bindings and API ergonomics:
  - Exposed `is_recurrent` read-only property on `LayerDetails` in Python bindings.
  - Added static convenience factory methods to `LayerDetails`: `create_ff`, `create_elman`, `create_gru`, `create_lstm`, `create_tcn`, `create_self_attention`, `create_attention_pool`, and `create_embedding` for simplified layer definition.
  - Added comprehensive multi-threading invariance tests (`OddBatchSizeAllGradsThreadCountInvariance`), positional encoding precomputation tests (`PositionalEncodingPrecomputationMatchesFormula`), and multi-timestep stability tests (`LargeDimensionMultiTimestepEquivalence`).

## [1.1.42] - 2026-08-30

### Fixed
- Fixed gradient backpropagation flow in multi-layer topologies where a recurrent/sequence layer precedes a feedforward or output layer:
  - Added `[[nodiscard]] virtual bool is_recurrent() const noexcept` to the `Layer` base class returning `true` for recurrent and sequence architectures (`Architecture::Elman`, `Architecture::Gru`, `Architecture::Lstm`, `Architecture::AttentionPool`, `Architecture::Tcn`, and `Architecture::SelfAttention`).
  - Corrected `Layers::calculate_back_propagation_hidden_layers` to inspect `hidden_1.is_recurrent()` rather than `has_rnn_gradients(next_layer_idx)`. This resolves an issue where standard feedforward/output layers that populated sequence gradients caused upstream layers to incorrectly bypass dense weight matrix backpropagation in favour of direct output gradient injection through an identity proxy.
  - Expanded `LayerTest.LayersTrainMathematicalSoundnessMultiLayerRecurrentGradientFlow` to verify gradient flow across Elman, GRU, and LSTM layers.

## [1.1.41] - 2026-08-25

### Added
- Implemented Differentiable Sharpe and Sortino Ratio Losses (`ErrorCalculation::type::sharpe_ratio_loss` and `ErrorCalculation::type::sortino_ratio_loss`):
  - Direct maximization of risk-adjusted returns (Sharpe ratio $S = \frac{\bar{R}}{\sigma}$ and Sortino ratio $S_{\text{Sortino}} = \frac{\bar{R} - \tau}{\sigma_d}$) as loss functions $L = -S$.
  - Exact analytic chain-rule batch gradient optimization $\frac{\partial L}{\partial \hat{y}_t} = w(R_t) \frac{\partial R_t}{\partial \hat{y}_t} + w(R_{t+1}) \frac{\partial R_{t+1}}{\partial \hat{y}_t}$ accounting for mean return and volatility / downside semi-variance.
  - Supports configurable transaction cost penalties ($c \ge 0.0$) with position delta derivatives $\text{sgn}(\hat{y}_t - \hat{y}_{t-1})$ and adjacent timestep cross-terms without inter-example coupling.
  - Supports configurable target return benchmark $\tau$ (`sortino_target_return`) for Sortino downside semi-variance.
  - Supports multi-asset portfolio outputs where portfolio return is averaged across asset positions.
  - Added helper types `PortfolioBatchStats`, `StepGradientContext` and calculation methods `calculate_sharpe_batch_stats`, `calculate_sortino_batch_stats`, `calculate_sharpe_ratio_step_weight`, `calculate_sortino_ratio_step_weight`.
  - Extended `EvaluationConfig` with `_transaction_cost_penalty` ($c \ge 0.0$) and `_sortino_target_return` ($\tau$), with non-defaulted constructor parameters, validation, and const getters.
  - Added JSON serialization and deserialization support for `"transaction-cost-penalty"` and `"sortino-target-return"` in `NeuralNetworkSerializer` with backwards compatibility for legacy model files (defaulting to 0.0).
  - Added fluent builder overloads in `NeuralNetworkOptions` (`with_output_layer_details` accepting `transaction_cost_penalty` and `sortino_target_return`).
  - Added Python bindings in `python/bindings.cpp` exposing `ErrorCalculationType.SharpeRatioLoss`, `ErrorCalculationType.SortinoRatioLoss`, and properties in `EvaluationConfig` and `NeuralNetworkOptions`.
  - Comprehensive unit, finite-difference gradient verification, multi-asset, serializer round-trip, and end-to-end feedforward and recurrent BPTT training convergence tests.

## [1.1.40] - 2026-08-25

### Added
- Implemented Quantile / Pinball Loss for Single and Multi-Quantile Regression (`ErrorCalculation::type::quantile_loss`):
  - Supports asymmetric pinball loss $L_q(y, \hat{y}) = \max(q(y - \hat{y}), (1 - q)(\hat{y} - y))$ for arbitrary target quantiles $q \in (0.0, 1.0)$.
  - Multi-quantile regression allows predicting prediction intervals and uncertainty bounds simultaneously (e.g. 10th percentile, median, 90th percentile: $q \in \{ 0.1, 0.5, 0.9 \}$).
  - Extended `EvaluationConfig` with `_quantiles` (`std::vector<double>`), 9-parameter constructor without default parameters, validation that every quantile satisfies $0.0 < q < 1.0$, and const accessor `quantiles()`.
  - Implemented `ErrorCalculation::calculate_quantile_loss` and backward gradient delta calculations in `Layer::calculate_quantile_loss_error_deltas`.
  - Added JSON serialization and deserialization support for `"quantiles"` in `NeuralNetworkSerializer` with backwards compatibility for legacy models (defaulting to median `[0.5]`).
  - Added fluent builder overloads in `NeuralNetworkOptions` (`with_output_layer_details` taking `quantiles` vector).
  - Added Python bindings in `python/bindings.cpp` exposing `ErrorCalculationType.QuantileLoss`, `ErrorCalculationType.PinballLoss`, and `quantiles` in `EvaluationConfig` and `NeuralNetworkOptions`.
  - Comprehensive unit, multi-quantile vector loss, asymmetric penalties, gradient deltas, serializer round-trip, and end-to-end training integration test suite in `tests/error_calculation_tests.cpp`, `tests/layer_tests.cpp`, and `tests/network_integration_tests.cpp`.

## [1.1.39] - 2026-08-24

### Added
- Extended `NeuralNetwork::log_training_info` to log Cosine Annealing with Warm Restarts configuration (enabled status, initial cycle period, cycle multiplier, minimum learning rate floor, and restart decay).
- Added schedule conflict detection warning in `NeuralNetworkOptions::build` when both adaptive learning rates and Cosine Annealing with Warm Restarts are simultaneously enabled.
- Added `NeuralNetworkOptions::adaptive_learning_rates()` accessor alias.

## [1.1.38] - 2026-08-23

### Added
- Implemented Lookahead Optimiser Wrapper (`LookaheadDetails`):
  - Wraps any base optimiser (AdamW, SGD, RAdam, Lion, NadamW, etc.) by maintaining slow weights $\phi$ and fast weights $\theta$.
  - Periodically interpolates slow weights towards fast weights every $k$ steps (`synchronisation_period` $\ge 1$) with slow weights step size $\alpha \in (0.0, 1.0]$: $\phi \leftarrow \phi + \alpha (\theta - \phi)$, then synchronizes fast weights $\theta \leftarrow \phi$.
  - Vectorised AVX2/FMA linear interpolation SIMD kernel `simd::lookahead_step` with fallback `simd::scalar_lookahead_step`.
  - Implemented `update_lookahead_slow_weights` and `update_lookahead_slow_weights_impl` across all layer architectures (`FFLayer`, `ElmanRNNLayer`, `GRURNNLayer`, `LSTMLayer`, `AttentionPoolLayer`, `TcnLayer`, `SelfAttentionLayer`, `EmbeddingLayer`, `ResidualProjector`, `MultiOutputLayer`, and `Layers`).
  - Integrated into `NeuralNetwork::train` with final synchronization of leftover steps before downstream evaluation.
  - Fluent builder methods `with_lookahead()` and validation in `NeuralNetworkOptions`.
  - JSON serialization and deserialization support in `NeuralNetworkSerializer` with full backwards compatibility.
  - Python bindings in `python/bindings.cpp` for `LookaheadDetails` and `NeuralNetworkOptions`.
  - Comprehensive unit, SIMD equivalence, layer interpolation, serializer round-trip, and end-to-end training tests in `tests/lookahead_tests.cpp` and `tests/simd_utils_tests.cpp`.

## [1.1.37] - 2026-08-23

### Added
- Implemented Cosine Annealing with Warm Restarts (SGDR) learning rate scheduler (`CosineAnnealingWarmRestartsDetails`):
  - Supports cyclic cosine learning rate decay with periodic restarts: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max, i} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)$.
  - Configurable initial cycle period ($T_0 \ge 1$), cycle period multiplier ($T_{\text{mult}} \ge 1.0$), minimum learning rate floor ($\eta_{\min} \ge 0.0$), and peak restart amplitude decay ($\gamma \in (0.0, 1.0]$).
  - Integrated into `NeuralNetworkOptions` with fluent builder overloads `with_cosine_annealing_warm_restarts()`, strict validation in `build()`, and harmonious coexistence with initial warmup schedules.
  - JSON model serialization and deserialization support in `NeuralNetworkSerializer` with full backwards compatibility for legacy model files.
  - Added Python bindings in `python/bindings.cpp` for `CosineAnnealingWarmRestartsDetails` and `NeuralNetworkOptions`.
  - Comprehensive unit and integration test suite in `tests/cosine_annealing_warm_restarts_tests.cpp` and `tests/network_integration_tests.cpp`.

## [1.1.36] - 2026-08-22

### Added
- Implemented RAdam (Rectified Adam) Optimiser (`OptimiserType::RAdam`):
  - Dynamically calculates the degrees of freedom of the approximated moving average ($\rho_t$).
  - Smoothly transitions between unadapted momentum SGD ($\rho_t \le 5$) and variance-rectified adaptive updates ($\rho_t > 5$) via rectification factor $r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}{(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}}$.
  - Full AVX2 SIMD vectorised acceleration with FMA3 (`_mm256_fmadd_pd`, `_mm256_fnmadd_pd`) in `simd::radam_step` and scalar fallback `simd::scalar_radam_step`.
  - Supports decoupled weight decay with configurable per-layer decays.
  - Seamless JSON serialization and deserialization in `NeuralNetworkSerializer` with full backwards and forwards compatibility.
  - Added Python bindings in `python/bindings.cpp` exposing `OptimiserType.RAdam`.
  - Comprehensive unit and integration test coverage across early-step unadapted momentum, tractable variance rectification, decoupled decay, SIMD equivalence across vector sizes, and end-to-end training.

## [1.1.35] - 2026-08-22

### Added
- Added Label Smoothing Regularisation ($\alpha \in [0.0, 1.0)$) across classification loss metrics and output layer gradient delta calculations:
  - Extended `EvaluationConfig` with `_label_smoothing` (private member, const accessor `label_smoothing()`, 8-parameter constructor without default parameters, and $[0.0, 1.0)$ range validation).
  - Added out-of-place and in-place `ErrorCalculation::smooth_labels()` helper functions for multi-class and binary targets.
  - Vectorised Cross-Entropy and Binary Cross-Entropy loss calculations in `ErrorCalculation` with fused label smoothing.
  - Vectorised AVX2 `_mm256_fmadd_pd` forward loss and backward output delta calculations in `Layer::calculate_cross_entropy_error_deltas` and `Layer::calculate_bce_error_deltas` with scalar fallback paths.
  - Added JSON serialization and deserialization support for `"label-smoothing"` in `NeuralNetworkSerializer` with backwards compatibility for legacy model files.
  - Added Python bindings in `python/bindings.cpp` for `EvaluationConfig` with `label_smoothing`.
  - Added comprehensive unit and integration tests in `tests/error_calculation_tests.cpp`, `tests/layer_tests.cpp`, and `tests/network_integration_tests.cpp`.

## [1.1.34] - 2026-08-22

### Added
- Added `EmbeddingLayer` architecture (`Layer::Architecture::Embedding`) for categorical entity embeddings:
  - Supports embedding lookup tables mapping categorical input feature indices to dense continuous vector representations ($V \times D$).
  - Full BPTT forward feed, recurrent outputs, activations, and inverted dropout support.
  - Multi-threaded gradient accumulation with `_thread_w_grads` and `_task_queue_pool`.
  - Added JSON serialization and deserialization support in `NeuralNetworkSerializer`.
  - Added Python bindings with `LayerArchitecture.Embedding`, `vocabulary_size`, and `embedding_dimension` in `python/bindings.cpp`.
  - Extended `LayerDetails` constructor and properties with `vocabulary_size` and `embedding_dimension`.
  - Added comprehensive test suite `EmbeddingLayerTest` in `tests/embeddinglayer_tests.cpp` covering construction, invalid parameter validation, hand-computed forward pass, index clamping, finite-difference gradient verification, multi-threaded accumulation equivalence, serializer round-trip, and end-to-end training convergence.

## [1.1.33] - 2026-08-22

### Added
- Added `QuickGELU` activation function ($f(x) = x \cdot \sigma(\alpha x) = \frac{x}{1 + e^{-\alpha x}}$ with default $\alpha = 1.702$):
  - Added `activation::method::quickGelu` to `activation::method` enum in `include/neuralnetwork/common/activation.h`.
  - Added scalar functions `calculate_quickGelu` and `calculate_quickGelu_derivative` in `include/neuralnetwork/common/activation.cpp`.
  - Added vectorised AVX2 kernels `simd::quick_gelu_pd`, `simd::quick_gelu_derivative_pd`, `simd::quick_gelu_activate`, and `simd::quick_gelu_derivative` in `include/neuralnetwork/common/simd_utils.h` with 8-wide unrolled SIMD vector loops.
  - Added He normal weight initialisation support for `quickGelu`.
  - Added Python binding enum export `ActivationMethod.QuickGelu` in `python/bindings.cpp`.
  - Updated documentation in `README.md` and `python/README.md`.
  - Added unit tests `ActivationTest.QuickGELU`, `ActivationTest.QuickGELUWithCustomAlpha`, `SimdUtilsTest.QuickGeluActivateAndDerivativeVsScalar`, and updated vectorized and string roundtrip test suites.

## [1.1.32] - 2026-08-22

### Changed
- Optimised `LSTMLayer` forward feed and BPTT backward pass for GELU activation:
  - Added fused vectorised AVX2 functions `simd::gelu_pd` and `simd::gelu_derivative_pd` in `include/neuralnetwork/common/simd_utils.h`.
  - Implemented register-fused `simd::lstm_forward_step_gelu` fusing gate sigmoid activations, candidate GELU activation, cell state update ($c_t = f \cdot c_{t-1} + i \cdot g$), cell state GELU activation, and hidden state output ($h_t = o \cdot \text{gelu}(c_t)$) in vector registers without intermediate memory roundtrips.
  - Implemented fused `simd::lstm_bptt_gate_step_gelu` integrating GELU derivative calculations directly into the AVX2 vector registers of the LSTM BPTT gate step, eliminating dynamic derivative buffer writing (`dc_act_deriv` and `dg_act_deriv`) and separate scalar passes.
  - Refactored `simd::gelu_activate` and `simd::gelu_derivative` to leverage `gelu_pd` and `gelu_derivative_pd` with 8-wide AVX2 unrolling.

### Added
- Added unit tests `SimdUtilsTest.LstmForwardStepGeluVsStandard` and `SimdUtilsTest.LstmBpttGateStepGeluVsStandard` in `tests/simd_utils_tests.cpp`.
- Added unit tests `LSTMLayerTest.GeluForwardFeedEquivalence` and `LSTMLayerTest.GeluBpttNumericalGradientEquivalence` in `tests/lstmlayer_tests.cpp`.

## [1.1.31] - 2026-08-21

### Changed
- Optimised `LSTMLayer` forward feed and gradient pipeline:
  - Implemented fused four-weight GEMM routines (`simd::gemm_four_weights_four_batches`, `simd::gemm_four_weights_two_batches`, and `simd::gemm_four_weights_one_batch`) in `include/neuralnetwork/common/simd_utils.h` to calculate pre-activations for all 4 gates ($f, i, o, g$) in parallel from shared input streams, eliminating 75% of input broadcast and memory stream overhead in `precalc_gates` and `recurrent_pass`.
  - Implemented single-pass register-fused `simd::lstm_forward_step_tanh` fusing gate sigmoid activations, candidate tanh activation, cell state update ($c_t = f \cdot c_{t-1} + i \cdot g$), cell state tanh activation, hidden state output ($h_t = o \cdot \tanh(c_t)$), and mask initialisation in vector registers without intermediate memory roundtrips in `finalize_forward_step`.
  - Added fast-path vectorised target gradient subtraction in `LSTMLayer::calculate_output_gradients` via `simd::sub_vectors`.
  - Vectorised multi-threaded gradient reduction in `LSTMLayer::calculate_and_store_gradients` using `simd::add_four_vectors` across all 4 gate weight, recurrent weight, and bias buffers.
  - Accelerated gradient norm calculation in `LSTMLayer::get_gradient_norm_sq()` using `simd::sum_sq_four`.

### Added
- Added unit tests `SimdUtilsTest.GemmFourWeightsBatches`, `SimdUtilsTest.LstmForwardStepTanhVsStandard`, and `SimdUtilsTest.SumSqFour` in `tests/simd_utils_tests.cpp`.
- Added forward pass equivalence and output gradient tests `LSTMLayerTest.ForwardFeedFusedEquivalence` and `LSTMLayerTest.OutputGradientsFusedEquivalence` in `tests/lstmlayer_tests.cpp`.

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
