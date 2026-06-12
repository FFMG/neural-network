# Changelog

All notable changes to the `neural-network` library will be documented in this file.

## [1.1.0] - 2026-06-12

### Added
- Created the `myoddweb::nn` namespace.
- Wrapped all core neural network library classes, structures, and helper functions in the new `myoddweb::nn` namespace (including `NeuralNetwork`, `Layer`, `Neuron`, `activation`, `NeuralNetworkOptions`, etc.).
- Added explicit documentation in the `README.md` explaining how to import and use the new namespace.

### Changed
- Updated all stand-alone example header files in `src/neuralnetwork/examples/` to use the `myoddweb::nn` namespace.
- Updated all test files in `tests/` to use the `myoddweb::nn` namespace.
- Kept third-party libraries (`TinyJSON`, `tracy`) and instrumentation code (`instrumentor.h`) outside the namespace to maintain clean integration boundaries.
- Reorganised the core NeuralNetwork library directory structure from a flat root layout into `/layers/`, `/helpers/`, and `/common/` subdirectories to improve code modularity.
- Updated all include directives in library headers, source files, tests, and examples to point to the new subdirectory paths.
- Updated MSVC Visual Studio project files (`.vcxproj` and `.vcxproj.filters`) and CMake files (`CMakeLists.txt`) to reflect the new folder structure.
