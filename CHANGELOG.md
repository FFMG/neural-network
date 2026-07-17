# Changelog

All notable changes to the `neural-network` library will be documented in this file.

## [1.1.1] - 2026-07-17

### Added
- Added comprehensive unit tests in `tests/layer_tests.cpp` to verify `Layer::calculate_huber_loss_error_deltas` behavior under different direction penalty configurations.

### Changed
- Updated `Layer::calculate_huber_loss_error_deltas` in `include/neuralnetwork/layers/layer.cpp` to respect the `use_direction_penalty` flag from `EvaluationConfig`.
- Optimized `Layer::calculate_huber_loss_error_deltas` using loop unswitching to eliminate branching overhead inside the neuron loop for maximum performance.

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
