# Changelog

All notable changes to the `neural-network` library will be documented in this file.

## [1.1.23] - 2026-08-16

### Added
- Refactored Stochastic Weight Averaging options to use a dedicated `StochasticWeightAveragingDetails` class (`include/neuralnetwork/common/stochasticweightaveragingdetails.h`):
  - Encapsulated `_swa_enabled`, `_swa_start_percent`, and `_swa_update_percent` into `StochasticWeightAveragingDetails` with full copy/move constructors, copy/move assignment operators, profiling instrumentation (`MYODDWEB_PROFILE_FUNCTION`), and const getters (`enabled()`/`swa_enabled()`, `start_percent()`/`swa_start_percent()`, `update_percent()`/`swa_update_percent()`).
  - Simplified `NeuralNetworkOptions`: replaced individual `with_swa(bool)`/`with_swa_start_percent(double)`/`with_swa_update_percent(double)` and `swa_start_percent()`/`swa_update_percent()` methods with `with_stochastic_weight_averaging(const StochasticWeightAveragingDetails&)` plus the helper `with_stochastic_weight_averaging(bool, double, double)`, with `NeuralNetworkOptions::stochastic_weight_averaging()` returning `const StochasticWeightAveragingDetails&`.
  - Updated `NeuralNetworkSerializer` (`include/neuralnetwork/helpers/neuralnetworkserializer.cpp`) save/load paths to work with `StochasticWeightAveragingDetails`.
  - Exposed `StochasticWeightAveragingDetails` and the updated `with_stochastic_weight_averaging` overloads and `stochastic_weight_averaging()` getter in the Python bindings (`python/bindings.cpp`), and updated `python/README.md` and `python/examples/example.py`.
  - Added unit test suite `tests/stochastic_weight_averaging_details_tests.cpp` (registered in `tests/CMakeLists.txt`) covering constructors, copy/move semantics, and options validation, and updated `tests/network_integration_tests.cpp`.
- Implemented additive (Bahdanau-style) attention pooling over the BPTT window, addressing plan item #2 in `can-you-please-write-agile-swan.md` (every recurrent architecture tested this session compresses the whole BPTT window into a single fixed-size state and only ever uses the last timestep for prediction, with no way to weight past ticks unequally):
  - Added `Layer::Architecture::AttentionPool` (`include/neuralnetwork/layers/layer.h`), with matching `architecture_to_string`/`architecture_from_string` cases. `Layer::create_hidden_layer` (`include/neuralnetwork/layers/layer.cpp`) gained a new required `previous_layer_architecture` parameter (inserted before the existing defaulted `residual_layer_number`/`residual_projector` params) and, for `AttentionPool`, panics unless: the previous layer is `Gru` or `Lstm` (**not** `Elman`); `LayerDetails::size` equals the previous layer's hidden size (pooling never changes dimensionality); `use_layer_normalisation` is `false`; `attention_hidden_size` is non-zero; and no residual connection is requested. `Layers::create_hidden_layer` (`include/neuralnetwork/layers/layers.cpp`) now passes `previous_layer.get_layer_architecture()` through; `MultiOutputLayer`'s branch-hidden-layer loop (`include/neuralnetwork/layers/multioutputlayer.h`) passes `Layer::Architecture::None` for a branch's first layer (an `AttentionPool` layer cannot currently be the first layer of a branch, since the branch constructor has no `Layer&` reference for whatever fed the trunk into that position — a known, documented limitation, not fixed here).
  - Added a new 9th constructor parameter, `attention_hidden_size` (unsigned), to `LayerDetails` (`include/neuralnetwork/layers/layerdetails.h`), threaded through copy/move ctor/assignment following `use_layer_normalisation`'s exact pattern, with a matching getter. This is the internal scoring-projection width, independent of the layer's own hidden size, and is meaningless (must be `0`) for every architecture except `AttentionPool`. Per this codebase's strict no-default-parameters convention, all ~50 existing `LayerDetails(...)` call sites across `examples/`, `tests/`, `python/examples/`, plus the two `LayerDetails` constructions inside `NeuralNetworkSerializer`/`NeuralNetworkOptions::create`, were updated to pass `0` explicitly.
  - Added `include/neuralnetwork/layers/attentionpoollayer.h`/`.cpp` (`AttentionPoolLayer`, inheriting `Layer` directly rather than `FFLayer`, since the T→1 sequence collapse and fixed-tanh scoring math don't fit `FFLayer`'s per-timestep-broadcast shape). Forward pass, per batch item, per timestep `t` (reading `h_t` from the preceding recurrent layer's full sequence via `previous_layer.get_rnn_outputs`): `e_t = h_t·W_a + b_a` (scoring projection, width `attention_hidden_size`), `u_t = tanh(e_t)`, `score_t = v·u_t`, `alpha_t = softmax_t(score_t)` (normalized across the whole BPTT window for that batch item), `context = Σ_t alpha_t·h_t`, then the layer's own configured activation + dropout are applied to `context` exactly like every other layer, written via `set_outputs` only (never `set_rnn_outputs`, since pooling collapses to a single "timestep"). Backward pass derives `dL/dW_a`/`dL/db_a`/`dL/dv` and, critically, `dL/dh_t` for every `t` (needed by the recurrent layer below) via full analytic backprop through the softmax and tanh, deposited through the exact same "direct gradient injection" mechanism (`GradientsAndOutputs::set_rnn_gradients` keyed by this layer's own index) that recurrent-layer-on-recurrent-layer stacking already relies on — requiring zero changes to `Layers::calculate_back_propagation_hidden_layers` or to `GRURNNLayer`/`LSTMLayer` themselves. Because `attention_hidden_size` is independently configurable and not tied to the layer's own neuron count, the scoring weights (`W_a`/`b_a`/`v`) cannot reuse the base `Layer` class's inherited `_w_values`/`_b_values` (whose size is locked to the layer's own input/output neuron count) — three new dedicated SoA weight groups (`_wa_*`/`_ba_*`/`_v_*`, each with the usual values/grads/velocities/m1/m2/timesteps/decays) were added instead, leaving the inherited arrays allocated (required by the base constructor) but intentionally always zero/unused — a small, bounded, documented memory overhead, not a correctness concern. Implements `accumulate_swa_average_impl`, `apply_stored_gradients`, `get_gradient_norm_sq`, `zero_gradients`, `clone`, and the standard two-phase gradient split (`calculate_hidden_gradients`/`calculate_hidden_gradients_from_output_gradients` produce and store the per-batch-item local delta plus the recurrent layer's input gradient; the later, separately-invoked `calculate_and_store_gradients` reads the stored delta back to accumulate the weight gradients), matching every other concrete `Layer` subclass's shape.
  - Added `simd::softmax_forward`/`simd::softmax_backward` (`include/neuralnetwork/common/simd_utils.h`), mirroring exactly how the `[1.1.21]` LayerNorm feature added `layer_norm_forward`/`layer_norm_backward` there: plain scalar loops (called once per batch item/layer, not once per weight), numerically-stable max-subtraction for the forward pass, and the standard softmax Jacobian-vector-product formula for the backward pass.
  - Added an `AttentionPool`-specific validation to `NeuralNetworkOptions::build()` (`include/neuralnetwork/neuralnetworkoptions.h`): panics if any hidden layer is `AttentionPool` and `enable_bptt` is `false`. Deliberately does **not** require `bptt_supervise_last_step_only`, since that is an orthogonal training-target-supervision concern — `AttentionPool`'s own output is single-timestep regardless of that setting.
  - Serialization (`include/neuralnetwork/helpers/neuralnetworkserializer.h`/`.cpp`): trained-weights path adds `create_attentionpoollayer`/`add_attentionpoollayer` (wired into `create_layers`'s type-string dispatch and `add_layer`'s `dynamic_cast` chain) saving/loading the `wa-*`/`ba-*`/`v-*` groups (7 keys each) plus the base `b-*` group, all via plain `.get<...>()` (no legacy files reference this brand-new layer type, so no `get_or` back-compat needed there). The separate config path (`add_hidden_layers`/`get_hidden_layers`, both the top-level and the `MultiOutputLayerDetails` branch-local call sites — the exact spot the `[1.1.21]` LayerNorm feature missed once and had to fix in a follow-up bullet) adds `attention-hidden-size` via `get_or<unsigned>(..., 0)`, since this field is now part of every architecture's shared `LayerDetails` config object and old saved models must still load with a sensible default. The partial/values-only `load_weights` reload path also gained matching optional `wa-*`/`ba-*`/`v-*` handling, mirroring the `[1.1.21]` `ln-h-gain-*`/`ln-h-bias-*` precedent, requiring new `set_wa_*`/`set_ba_*`/`set_v_*` setters on `AttentionPoolLayer`.
  - Exposed `Layer::Architecture::AttentionPool` and `LayerDetails`'s `attention_hidden_size` in the Python bindings (`python/bindings.cpp`), following the existing enum-value and trailing-constructor-argument patterns (`py::arg("attention_hidden_size") = 0`). Updated `python/README.md` (`nn.LayerArchitecture` enum list, `nn.LayerDetails` constructor doc) and `python/examples/example.py` (an illustrative `AttentionPool`-following-`Gru` `LayerDetails` snippet, demonstrating the required shape without needing to restructure the file's existing non-recurrent XOR training data).
  - Documented in `README.md` (new "Attention Pooling" subsection under `## Options`, between "Layer Normization" and "Stochastic Weight Averaging (SWA)", plus an `AttentionPool` bullet under "Hidden Layers"), including an explicit callout of the known interaction with the `[1.1.21]` Known Issue below.

### Known Issues
- `AttentionPool`'s own backward pass, like GRU/LSTM/Elman-on-recurrent-layer stacking, relies on `GradientsAndOutputs`'s "direct gradient injection" mechanism (`has_rnn_gradients`/`set_rnn_gradients`) for gradient flow between adjacent layers. Whenever the layer directly above `AttentionPool` (typically the output layer) has *also* already deposited into that same mechanism — which, per the existing `[1.1.21]` Known Issue, every `FFLayer`/`FFOutputLayer` unconditionally does regardless of whether it's actually in a recurrent context — `AttentionPool`'s own `calculate_hidden_gradients_from_output_gradients` (rather than the generic weight-matrix-multiply `calculate_hidden_gradients`) fires, inheriting the exact same pre-existing, documented limitation. `AttentionPool`'s mandatory "output size equals the preceding recurrent layer's hidden size" constraint keeps this shape-compatible enough for gradients to flow (mirroring the workaround already used by the `[1.1.21]`/`[1.1.22]` GRU/LSTM LayerNorm and SWA integration tests), but this is inherited, pre-existing scope — not introduced or fixed by this change. `AttentionPoolLayer`'s own attention forward/backward math is independently verified correct via direct numerical-gradient checks (bypassing this framework-level dispatch entirely) in `tests/attentionpoollayer_tests.cpp`.

### Test
- Added `tests/simd_utils_tests.cpp` coverage for the new primitives: `SoftmaxForwardSumsToOne`, `SoftmaxForwardUniformOnEqualScores`, `SoftmaxForwardSingleElement`, `SoftmaxForwardNumericallyStableOnLargeValues`, `SoftmaxBackwardMatchesNumericalGradient` (central finite-difference check, independent of the derivation itself), `SoftmaxBackwardZeroWhenUniformGradient`.
- Added `tests/layer_details_tests.cpp::LayerDetailsAttentionHiddenSizeField` and, to `tests/layer_tests.cpp`: `CreateHiddenLayerRejectsAttentionPoolOnNonRecurrentPrevious`, `CreateHiddenLayerRejectsAttentionPoolOnElman`, `CreateHiddenLayerRejectsAttentionPoolSizeMismatch`, `CreateHiddenLayerRejectsAttentionPoolWithLayerNorm`, `CreateHiddenLayerRejectsAttentionPoolZeroHiddenSize`, `CreateHiddenLayerRejectsAttentionPoolWithResidual`, `CreateHiddenLayerAcceptsAttentionPoolAfterGruOrLstm`.
- Added `tests/attentionpoollayer_tests.cpp` (new file): `Construction`, `ForwardHandComputedExample` (small 2-timestep, 1-wide-scoring example computed by hand), `WeightGradientsMatchNumericalGradient` (central finite-difference check of `_wa_grads`/`_ba_grads`/`_v_grads` against the real forward pass, re-run per perturbation), `InputGradientsMatchNumericalGradient` (same technique for the `dL/dh_t` sequence handed to the preceding recurrent layer), `CalculateHiddenGradientsMatchesDirectInjectionThroughIdentity` (proves the generic weight-matrix-multiply path and the direct-injection path agree when the weight matrix is the identity), `NoBatchCrossTalk`, `AccumulateSwaAverageMatchesDirectMean`, `CloneProducesIndependentCopy`. No `_mt_tests.cpp` variant was added since the implementation is deliberately single-threaded (kept simple given the BPTT window and attention-hidden-size involved are both small; see the Added section above).
- Added to `tests/network_integration_tests.cpp`: `GRUSequenceConvergenceAttentionPool`/`LSTMSequenceConvergenceAttentionPool` (full `NeuralNetwork::train` smoke tests: no throw, finite bounded predictions, and — for the GRU variant — the scoring vector `v` actually moves from its random initialization during training), `AttentionPoolSerializerSaveLoad` (weight round-trip for `wa`/`ba`/`v`, `EXPECT_NEAR` tolerance per the TinyJSON precedent, plus the `attention-hidden-size` config-path round-trip), `AttentionPoolRequiresBpttOptionValidation` (the new `NeuralNetworkOptions::build()` validation).

## [1.1.22] - 2026-08-15

### Added
- Implemented Stochastic Weight Averaging (SWA), addressing plan item #3 in `can-you-please-write-agile-swan.md` (run-to-run noise across nominally identical training runs, making it hard to distinguish a genuine architectural gain from noise):
  - Added three opt-in `NeuralNetworkOptions` fields (`include/neuralnetwork/neuralnetworkoptions.h`): `swa` (bool, default `false`), `swa_start_percent` (double, default `0.75` — fraction of total epochs after which snapshotting begins) and `swa_update_percent` (double, default `0.02` — cadence between snapshots, reusing the exact percent-of-epoch semantics already used by `update_training_monitor_percent` via `NeuralNetworkHelper::is_at_epoch_interval`). `build()` panics if `swa` is enabled with `swa_start_percent` outside `[0.0, 1.0)` or `swa_update_percent` outside `(0.0, 1.0]`.
  - Added a template-method pair on `Layer` (`include/neuralnetwork/layers/layer.h`): a non-virtual `accumulate_swa_average(const Layer& snapshot, size_t existing_swa_count)` that folds in the optional `_residual_projector`'s weights (via a matching new `ResidualProjector::accumulate_swa_average`, `include/neuralnetwork/layers/residualprojector.h`) and then dispatches to a new pure-virtual `accumulate_swa_average_impl(...)`, plus a protected static `Layer::swa_average_into(...)` helper implementing the standard incremental/running-mean update (`running_avg[i] += (snapshot[i] - running_avg[i]) / (existing_swa_count + 1)`) — so only one running-average `Layers` copy is ever held, never every snapshot simultaneously. This follows the exact same per-subclass-override shape already used by `Layer::get_gradient_norm_sq()` for gradient-clipping's global norm.
  - Overrode `accumulate_swa_average_impl` in every concrete `Layer` subclass that owns its own weight arrays — `FFLayer` (`_w_values`/`_b_values`, also covers `FFOutputLayer` by inheritance), `ElmanRNNLayer` (adds `_rw_values`), `GRURNNLayer` (base + `_z_w/_z_rw/_z_b` + `_r_w/_r_rw/_r_b` values, plus `_ln_h_gain/_ln_h_bias` values when `use_layer_normalisation` is enabled), `LSTMLayer` (base + `_f/_i/_o` gate `w/rw/b` values, plus `_ln_c_gain/_ln_c_bias` values when enabled), and `MultiOutputLayer` (recurses into every `_branches[i].layers[j]`, mirroring its existing `get_gradient_norm_sq()` branch-recursion). Added a no-op override to the internal `MultiInputProxyLayer` (no trainable weights) and to `test_helper::MockLayer` (`tests/test_helper.h`, required to keep the shared test double instantiable now that `Layer` gained a new pure virtual). Only weight *values* are ever averaged — gradients, velocities, moments, timesteps and decays (optimiser state) are never touched. The `_*_values_T` transposed BPTT-performance caches on `GRURNNLayer`/`LSTMLayer` are deliberately left untouched (derived, not canonical) and are regenerated afterwards via the existing `Layers::cache_recurrent_weights()`.
  - Added `Layers::accumulate_swa_average(const Layers& snapshot, size_t existing_swa_count)` (`include/neuralnetwork/layers/layers.cpp`), looping every layer index (skipping the input layer, matching the existing `get_total_weights()` loop bounds).
  - Wired into `NeuralNetwork::train()` (`include/neuralnetwork/neuralnetwork.cpp`): two new scratch members, `_swa_layers` (`std::unique_ptr<Layers>`) and `_swa_snapshot_count`, reset at the start of every `train()` call. Inside the epoch loop, once `epoch >= swa_start_epoch` and `base_helper->is_at_epoch_interval(swa_update_percent)` fires, either takes the first full snapshot (`std::make_unique<Layers>(_layers)`) or folds a new one into the running average. After the epoch loop ends — but before final metrics logging, and critically before `optimize_inference_temperature(...)` — the averaged `Layers` replaces `_layers` (`std::move`) and `cache_recurrent_weights()` is called to regenerate the transposed BPTT caches for the new averaged values. If training stops early via the progress callback, or no snapshot cadence ever fires, `_swa_layers` stays null and the swap-in is skipped gracefully. Because the averaged weights become the network's normal trained weights before `NeuralNetworkSerializer::save`/temperature calibration/`think()` ever see them, **no per-layer weight serialization changes were needed** — only the three new scalar options.
  - Serialization: `include/neuralnetwork/helpers/neuralnetworkserializer.cpp` saves/loads `swa-enabled`/`swa-start-percent`/`swa-update-percent` via `set_boolean`/`set_float` and `get_or<bool>`/`get_or<double>` with the same defaults as `NeuralNetworkOptions::create`, so models saved before this feature still load with SWA off.
  - Exposed `with_swa`/`with_swa_start_percent`/`with_swa_update_percent` in the Python bindings (`python/bindings.cpp`), following the existing one-line `.def("with_x", &NeuralNetworkOptions::with_x)` fluent-builder pattern. Updated `python/README.md`'s `NeuralNetworkOptions` builder-methods list and `python/examples/example.py` to exercise the new options end-to-end.
  - Documented in `README.md` (new "Stochastic Weight Averaging (SWA)" subsection under `## Options`, between "Layer Normalization" and "General Training Options").

### Test
- Added `tests/swa_tests.cpp` (new file, registered in `tests/CMakeLists.txt`): `LayerRunningMeanMatchesDirectArithmeticMean` (verifies the incremental running-mean update over 3 snapshots matches a direct arithmetic mean), `MockLayerAccumulateSwaAverageIsNoOp` (the new pure virtual doesn't break the shared `MockLayer` test double), `LayersAccumulateSwaAveragePropagatesAcrossLayers` (verifies `Layers::accumulate_swa_average` folds a snapshot into every layer, not just the first).
- Added to `tests/network_integration_tests.cpp`: `SwaOptionSerialization` (the three new options round-trip through save/load, following the `BpttSuperviseLastStepOnlySerializerSaveLoad` pattern), `SwaProducesAveragedWeightsDifferentFromBaseline` (two otherwise-identical, fully deterministic FF training runs differing only in `swa` enabled/disabled produce different, finite final weights — proves the swap-in actually replaces the trained weights rather than being a silent no-op), `GRUSequenceConvergenceSwa`/`LSTMSequenceConvergenceSwa` (smoke tests mirroring `GRUSequenceConvergenceLayerNorm`/`LSTMSequenceConvergenceLayerNorm`: full `NeuralNetwork::train`, no throw, finite bounded predictions), `SwaWithMultiOutputBranches` (smoke test mirroring `GRUSequenceConvergenceMultiOutputBpttSuperviseLastStepOnly`, verifying the branch-recursion path doesn't crash or hit a size mismatch).

## [1.1.21] - 2026-08-15

### Added
- Implemented recurrent-state Layer Normalization for `GRURNNLayer` and `LSTMLayer`, addressing plan item #1 in `can-you-please-write-agile-swan.md` (unstable internal activation scale across BPTT timesteps):
  - Added `simd::layer_norm_forward`/`simd::layer_norm_backward` in `include/neuralnetwork/common/simd_utils.h`: standard LayerNorm (population mean/variance, `eps=1e-5`, learnable per-element gain/bias), implemented as plain scalar loops with a division-by-zero safety guard against zero/near-zero gain values (called once per batch item/timestep/layer rather than once per weight, unlike the file's hand-vectorised AVX2 kernels). The backward pass recovers `a_hat` from the cached post-normalization output and gain/bias (`a_hat = (y - bias) / safe_gain`), needing only one extra cached scalar (`inv_std`) per call rather than the raw pre-normalization input.
  - Added a new opt-in `bool use_layer_normalisation` parameter to `LayerDetails` (`include/neuralnetwork/layers/layerdetails.h`), with a getter and full copy/move/assignment support. Conforming to this codebase's strict no-default-parameters convention, all `LayerDetails(...)` call sites across `examples/`, `tests/`, and `python/examples/` have been explicitly updated to specify `use_layer_normalisation` (`false` when disabled, `true` when enabled). `Layer::create_hidden_layer` (`include/neuralnetwork/layers/layer.cpp`) now panics if `use_layer_normalisation` is requested on a non-`Gru`/`Lstm` architecture.
  - `GRURNNLayer` (`include/neuralnetwork/layers/grurnnlayer.h`/`.cpp`) normalizes the blended hidden state `h_t = (1-z)⊙h_{t-1} + z⊙ĥ` in `finalize_forward_step`, in place, before it is cached and propagated — both the value fed downstream and the value carried forward as `h_{t-1}` are normalized. Added `_ln_h_gain_*`/`_ln_h_bias_*` SoA weight/optimiser-state members (values/grads/velocities/m1/m2/timesteps/decays, mirroring the existing z/r gate members), wired through the constructors (fresh-init and deserialization), copy/move/assignment, `calculate_and_store_gradients`/`apply_stored_gradients`/`zero_gradients`/`get_gradient_norm_sq`. `get_pre_activation_multiplier()` now returns a new `LayerNormMultiplier` (`Multiplier + 1`) when `use_layer_normalisation` is enabled, adding one packed-state slot to cache `inv_std`; disabled layers keep the original `Multiplier` and layout unchanged.
  - `LSTMLayer` (`include/neuralnetwork/layers/lstmlayer.h`/`.cpp`) normalizes the cell state `c_t = f⊙c_{t-1} + i⊙g` analogously in `finalize_forward_step`, immediately after `simd::lstm_cell_step` computes it and before it feeds `tanh(c_t) → h_t` or persists as `c_{t-1}`. Added the equivalent `_ln_c_gain_*`/`_ln_c_bias_*` members and wiring. Because LSTM's output-gate gradient depends on `dh_curr` directly (not solely through `dc`, unlike GRU's single-state case), the backward substitution solves for the `dc_next_in` value that makes `lstm_bptt_gate_step`'s internal `dc` recomputation equal the LayerNorm-adjusted gradient, without modifying the shared kernel.
  - Both layers thread the LayerNorm gradient through the existing two-phase BPTT design without modifying the shared `gru_bptt_gate_step`/`lstm_bptt_gate_step` SIMD kernels: in `calculate_bptt_batch_chunk` (`calculate_hidden_gradients`, Phase A, `const`), the combined external+recurrent gradient is passed through `simd::layer_norm_backward` and fed back into the unmodified kernel; per-thread-chunk gain/bias gradient accumulators were added to each layer's `BPTTWorkspace` (reusing the class's existing `mutable`-workspace pattern for const-method-safe accumulation) and merged into `_ln_h_gain_grads`/`_ln_c_gain_grads` etc. once every dispatched chunk completes.
  - Serialization support in `include/neuralnetwork/helpers/neuralnetworkserializer.cpp`: `add_grurnnlayer`/`add_lstmlayer` save a `use-layer-normalisation` flag plus the full gain/bias optimiser state when enabled; `create_grurnnlayer`/`create_lstmlayer` load them via `get_or<bool>("use-layer-normalisation", false)` so models saved before this feature still load; `load_weights` (the partial/values-only reload path) gained matching optional `ln-h-gain-values`/`ln-h-bias-values`/`ln-c-gain-values`/`ln-c-bias-values` handling.
  - Exposed `use_layer_normalisation` in the Python bindings (`python/bindings.cpp`): `LayerDetails`'s `py::init<...>` gained the trailing parameter (`py::arg("use_layer_normalisation") = false`) and a matching read-only property.
  - Documented in `README.md` (new "Layer Normalization" subsection under `## Options`) and `python/README.md` (extended `nn.LayerDetails` constructor/properties documentation).

### Fixed
- Fixed `GRURNNLayer::zero_gradients`/`LSTMLayer::zero_gradients` clobbering the LayerNorm gain/bias gradients: `zero_gradients()` runs at the start of `calculate_and_store_gradients` (which recomputes every other gradient from scratch each call), but LayerNorm's gain/bias gradients are fully computed earlier, in `calculate_hidden_gradients` (which already zeroes and fills them itself) — zeroing them again in between silently discarded them before `apply_stored_gradients` ever read them, so the LayerNorm gain/bias never actually updated during training despite correct forward/backward math. Caught by `NetworkIntegrationTest.GRUSequenceConvergenceLayerNorm`/`LSTMSequenceConvergenceLayerNorm` and `LayerNormGainBiasSerializerSaveLoad`, which assert the gain moves away from its `1.0` identity initialization after training.
- Fixed `use_layer_normalisation` not being persisted as part of a `NeuralNetworkOptions`/`LayerDetails` hidden-layer configuration: `NeuralNetworkSerializer::add_hidden_layers`/`get_hidden_layers` (`include/neuralnetwork/helpers/neuralnetworkserializer.cpp`) — the separate save/load path for the human-authored `hidden_layers()` config (architecture, size, activation, dropout, weight-decay, optimiser, momentum), distinct from the per-layer trained-weight save/load already covered above — did not read or write the flag, so it silently defaulted back to `false` after any `NeuralNetworkSerializer::save`/`load` round-trip even though a network's trained GRU/LSTM LayerNorm gain/bias weights themselves round-tripped correctly. Added `use-layer-normalisation` to both `add_hidden_layers` and both `get_hidden_layers` load sites (top-level hidden layers and the `MultiOutputLayerDetails` branch-local hidden layers), loaded via `get_or<bool>("use-layer-normalisation", false)` so older saved models still load.

### Known Issues
- Discovered (not fixed, out of scope for this change — pre-dates LayerNorm and affects gate weights too, not just LayerNorm parameters): when a `GRURNNLayer`/`ElmanRNNLayer`/`LSTMLayer` is followed directly by an `FFOutputLayer` doing per-timestep BPTT supervision, `calculate_hidden_gradients_from_output_gradients` (`include/neuralnetwork/layers/grurnnlayer.cpp`/`lstmlayer.cpp`/`elmanrnnlayer.cpp`) substitutes an `N_this x N_this` identity `_identity_proxy` layer as the `next_layer` argument passed into `calculate_hidden_gradients`. That function reads `N_next = next_layer.get_number_neurons()` from the proxy — i.e. the recurrent layer's *own* neuron count — rather than the real downstream layer's neuron count. Whenever the real next layer's neuron count differs from the recurrent layer's, the resulting stored-gradient size no longer matches `num_time_steps * N_next`, so `next_is_seq`/the single-step size check both fail and the upstream gradient contribution is silently dropped for every timestep — the affected gate weights (and, for `use_layer_normalisation`, the LayerNorm gain/bias) receive zero gradient and never update. This is masked in the pre-existing `GRUSequenceConvergence`/`LSTMSequenceConvergence` tests because they hand-seed near-exact-solution weights, so a lack of further updates doesn't change already-correct predictions. Worked around in the new LayerNorm integration tests below by matching the output layer's neuron count to the recurrent layer's hidden size, which keeps the (still incorrect, since it uses the proxy's identity weights rather than the real output layer's weights) substitution shape-compatible enough to exercise gradient flow end-to-end. Flagging for a follow-up fix.

### Test
- Added `tests/simd_utils_tests.cpp` coverage for the new primitive: `LayerNormForwardIdentityGainBias`, `LayerNormForwardAppliesGainAndBias`, `LayerNormForwardConstantInputUsesEpsilon`, `LayerNormForwardSingleElement`, `LayerNormBackwardMatchesReferenceFormula`, `LayerNormBackwardAccumulatesIntoExistingGradients`, `LayerNormBackwardMatchesNumericalGradient` (central-finite-difference check of the hand-derived backward formula, independent of the derivation itself), `LayerNormBackwardZeroOrNearZeroGain` (guarding against division by zero on zero/near-zero gain values).
- Added `tests/layer_details_tests.cpp::LayerDetailsUseLayerNormalisationFlag` and `tests/layer_tests.cpp::CreateHiddenLayerRejectsLayerNormOnFFAndElman`/`CreateHiddenLayerAcceptsLayerNormOnGruAndLstm`.
- Added to `tests/grurnnlayer_tests.cpp`: `LayerNormForwardNormalizesHiddenState`, `LayerNormDisabledMatchesUnnormalizedForwardFeed`, `LayerNormGainBiasGradientsMatchNumericalGradient` (central-finite-difference check through the full forward+BPTT pipeline), and LayerNorm-enabled variants of the existing `NoBatchCrossTalk*` batch-isolation regression pattern. Added `tests/grurnnlayer_mt_tests.cpp::LayerNormForwardAndBackwardMTConsistency` (single- vs multi-threaded gain/bias gradient equivalence, exercising the per-workspace accumulation/merge across multiple dispatched BPTT chunks).
- Added the mirrored set to `tests/lstmlayer_tests.cpp` and `tests/lstmlayer_mt_tests.cpp` for the cell-state variant.
- Added `tests/network_integration_tests.cpp::LayerNormGainBiasSerializerSaveLoad` (weight-value round-trip through save/load, unlike the existing option-only `BpttSuperviseLastStepOnlySerializerSaveLoad` pattern — compared with `EXPECT_NEAR`/tolerance rather than exact equality, since TinyJSON's decimal-text serialization of `double` values is not bit-for-bit round-trippable) and `GRUSequenceConvergenceLayerNorm`/`LSTMSequenceConvergenceLayerNorm` (full `NeuralNetwork::train` smoke tests: no throw, finite bounded predictions, and gain actually moves during training). Both integration tests use an output layer neuron count matched to the recurrent layer's hidden size to avoid the identity-proxy gradient-routing issue noted above.
- Added `tests/network_integration_tests.cpp::UseLayerNormalisationOptionSerialization`, following the `BpttSuperviseLastStepOnlySerializerSaveLoad` option-survives-round-trip pattern: asserts `use_layer_normalisation` set via `LayerDetails`/`NeuralNetworkOptions::with_hidden_layers` is still `true` on `loaded_nn->options().hidden_layers()[0]` after a save/load cycle, exercising the `add_hidden_layers`/`get_hidden_layers` fix above.

## [1.1.20] - 2026-08-15

### Fixed
- Fixed a training performance issue caused by nested thread-pool oversubscription in `Layers::update_weights` (`include/neuralnetwork/layers/layers.cpp`): gradient calculation and gradient application were dispatched per layer onto a `Layers`-owned `TaskQueuePool` (`_update_weights_pool`) whenever the combined workload crossed a threshold, while each dispatched layer's `calculate_and_store_gradients`/`apply_stored_gradients` could *also* re-enter that same layer's own internal `TaskQueuePool` for its per-batch chunking. With `number_of_threads` typically defaulting to `hardware_concurrency() - 1`, this created far more concurrently runnable OS threads than CPU cores on every batch of every epoch, causing contention rather than speedup.
  - Removed the `_update_weights_pool` member, its `GradCalcTask`/`GradApplyTask` dispatch structs, and all associated allocation/copy/move/assignment plumbing from `Layers` (`include/neuralnetwork/layers/layers.h`, `include/neuralnetwork/layers/layers.cpp`).
  - `Layers::update_weights` now always iterates layers sequentially for both gradient calculation and gradient application, relying solely on each layer's own already-tuned internal SIMD/thread-pool parallelism (unchanged) for a single level of parallelism instead of two nested ones.
  - `Layers::set_number_of_threads` no longer recreates the removed pool.

### Added
- Added `NetworkIntegrationTest.UpdateWeightsTouchesEveryLayerAcrossThreadCounts` in `tests/network_integration_tests.cpp`: trains a 3-hidden-layer FF network with `number_of_threads` set to 1, 2, and 8, and verifies every hidden and output layer's weights (and biases, where present) actually change after training, guarding the layer loop bounds in the simplified `Layers::update_weights`.
- Added `NetworkIntegrationTest.DeepNetworkConvergesWithExplicitThreadCount` in `tests/network_integration_tests.cpp`: reuses the known-good hand-set XOR weights from `XorFFConvergence` with `number_of_threads(4)` explicitly set, confirming training still converges correctly through the now-always-sequential `update_weights` loop.

## [1.1.19] - 2026-08-15

### Added
- Implemented the Lion (EvoLved Sign Momentum) optimiser:
  - Added `simd::lion_step` (AVX2/FMA vectorised implementation with branchless sign computation and decoupled weight decay) and `simd::scalar_lion_step` in `include/neuralnetwork/common/simd_utils.h`.
  - Added `OptimiserType::Lion` support in `Layer::apply_update_to_vector_internal` and `Layer::apply_update_to_weight` in `include/neuralnetwork/layers/layer.cpp`, enabling per-weight and vectorised updates across all layer architectures (`FFLayer`, `ElmanRNNLayer`, `GRURNNLayer`, `LSTMLayer`, `FFOutputLayer`, `MultiOutputLayer`).
  - Added mathematical unit tests in `tests/layer_optimizer_tests.cpp` (`ApplyUpdateToWeightLion`, `ApplyUpdateToWeightLionWithDecay`, `ApplyUpdateToWeightLionZeroGradient`, `ApplyUpdateToVectorLion`, `ApplyUpdateToWeightLionClampsExtremeValues`, `ApplyUpdateToWeightLionWithClipping`, `ApplyUpdateToVectorLionSkipsDecayForBias`).
  - Added SIMD unit tests in `tests/simd_utils_tests.cpp` (`ScalarLionStep`, `ScalarLionStepNoDecay`, `LionStep`, `LionStepNoDecay`, `LionStepWithClipping`, and `FmaEquivalenceVerify` Lion step verification).
  - Added string conversion and parsing unit tests in `tests/layer_tests.cpp` (`OptimiserTypeToString`, `StringToOptimiserType`).
  - Added integration test `XorFFConvergenceLion` in `tests/network_integration_tests.cpp`.

### Fixed
- Replaced `TempBuffer`'s thread-local storage pool (`ThreadBufferPool`) with a clean, stack-scoped RAII buffer in `include/neuralnetwork/common/tempbuffer.h` and `include/neuralnetwork/layers/layer.cpp`, eliminating all MSVC CRT dynamic TLS teardown crashes (`0xc0000374`) when worker/inference threads terminate.
- `Layer::apply_update_to_weight`'s per-weight Lion path now clamps the updated value to `[-100000, 100000]`, matching the hard explosion guard already applied by the vectorised `simd::lion_step` path, so the two code paths can no longer diverge under extreme values.
- Corrected `OptimiserType::LAMB` to `OptimiserType::Lamb` in Python pybind11 bindings (`python/bindings.cpp`).

### Documentation
- Updated `README.md` to list `Lion` among the supported optimisers.

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
