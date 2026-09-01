# Contributing to neural-network

Thanks for your interest in contributing! This project is primarily an
educational C++ neural network library, and contributions of all sizes —
bug fixes, new layer types, documentation, tests — are welcome.

Please also read our [Code of Conduct](CODE_OF_CONDUCT.md); it applies to
all project spaces (issues, pull requests, discussions).

## Before you start

* For a small fix (typo, obvious bug), feel free to open a pull request
  directly.
* For a larger change (new layer architecture, API change, behavioural
  change), please open an issue first to discuss the approach before
  investing significant time.

## Repository layout

* `include/neuralnetwork/` — the stand-alone core C++ library
  (`layers/`, `helpers/`, `common/`).
* `examples/` — standalone examples, `main.cpp` runner, and the main
  Visual Studio solution (`neuralnetwork.sln`).
* `tests/` — the unit test suite (CMake/CTest).
* `python/` — the pybind11 bindings, Visual Studio solution
  (`neuralnetwork_py.sln`), and Python examples.

## Building and testing

The maintainer develops exclusively with **Visual Studio 2022**; please
target that toolchain for any changes you submit.

* Core library / examples: open `examples/neuralnetwork.sln` in Visual
  Studio 2022 and build/run the `neuralnetwork` or `neuralnetwork_tests`
  project.
* Python bindings: see [python/README.md](python/README.md) for
  prerequisites and build instructions.

CI also builds and runs the test suite via CMake/CTest on Windows, Linux
and macOS (see `.github/workflows/tests.yml`), and builds/verifies the
Python bindings on Windows (see `.github/workflows/python.yml`). If you
have CMake/CTest available locally, you can reproduce the C++ CI job with:

```sh
cmake -B tests/build -S tests -DCMAKE_BUILD_TYPE=Release
cmake --build tests/build --config Release
ctest --test-dir tests/build -C Release --output-on-failure
```

Please make sure the test suite passes, and add or update unit tests in
`tests/` for any behavioural change or new feature.

## Coding conventions

This project follows a specific, consistently-applied style — please match
it rather than your own default style or an auto-formatter's output:

* **Indentation:** spaces only, never tabs (the `.sln` file is the one
  exception).
* **Braces:** Allman style — the opening brace goes on its own new line,
  and so does the closing brace (`else`, `catch`, `finally` each start on
  their own new line too).
* **Control flow:** `if`, `while`, and `for` must always use braces `{}`,
  even for a single-statement body.
* **Naming:**
  * Classes: `PascalCase` (e.g. `NeuralNetwork`) — except small
    classes/structs, which use `snake_case`; if you're unsure which applies,
    ask in the PR rather than guessing.
  * Methods/functions: `snake_case` (e.g. `train`, `think`).
  * Private members: prefixed with an underscore (e.g. `_learning_rate`).
* **Headers:** use `#pragma once` for include guards.
* **Error handling:** use `Logger` for logging, and standard exceptions
  for critical failures.
* **Member order:** `public`, then `protected`, then `private` — for both
  methods and variables. Variables should be `private` with a helper
  `const` accessor rather than public fields.
* **Lambdas:** avoid where possible; prefer named functions or functors.
* **`this`:** never use `this->function()`; call `function()` directly. If
  that would be ambiguous, rename the function rather than qualifying with
  `this->`.
* **Warnings:** all warnings are treated as errors — please build warning-free
  before submitting.

## Commit messages and pull requests

* Keep commits focused and description-worthy — explain *why*, not just
  *what*.
* Fill in the pull request template; link any related issue.
* Small, reviewable pull requests are much easier to merge than large ones.

## Reporting bugs / requesting features

Please use [GitHub Issues](https://github.com/FFMG/neural-network/issues),
including:

* A clear description of the problem or request.
* A minimal repro (topology/options snippet) for bugs, where possible.
* Your platform/compiler (this project is developed against Visual Studio
  2022, but CI also covers GCC/Clang on Linux and macOS).

## License

By contributing, you agree that your contributions will be licensed under
the project's [MIT License](LICENSE).
