# Gemini Project Context: neural-network

This document provides context and guidelines for interacting with the `neural-network` project.

My name is Florent

## Project Overview

The `neural-network` project is a lightweight, dependency-free Feedforward and Recurrent Neural Network library implemented in modern C++. It is designed for both educational purposes and practical application in financial market analysis and trading.

### Key Features
- **Neural Network Architecture:** Supports standard Feedforward (FF), Elman RNN, and Gated Recurrent Unit (GRU) layers.
- **Advanced Training:** Includes Adam, AdamW, Nadam, and NadamW optimizers, Backpropagation Through Time (BPTT), residual connections, gradient clipping, and dropout (dilution).
- **Learning Rate Strategies:** Implements warmup, exponential decay, periodic restarts with boosts, and adaptive learning rate scheduling.
- **Market Integration:** A dedicated `market` class handles technical indicators (RSI, Bollinger Bands, EMA, MACD, etc.) and data normalization for trading strategies.
- **Visualization:** A built-in `Chart` tool using the `olcPixelGameEngine` for real-time visualization of market data, predictions, and trade signals.
- **Persistence:** Supports saving and loading trained models via a custom serializer.
- **Zero Dependencies:** Built from scratch with minimal external dependencies (embedded `sqlite3`, `TinyJSON`, `olcPixelGameEngine`).

### Technology Stack
- **Language:** C++17/C++20
- **Platform:** Windows (Win32 API for screen metrics, MSVC for compilation)
- **Profiling:** Integrated with Tracy Profiler, disabled by default.

---

## Building and Running

### Prerequisites
- **Visual Studio 2022** (or compatible version) with C++ development workload.
- **Windows OS** (due to Win32 API usage in the charting tool).
- Try and support C++ 99 where possible, comment if not possible.

### Build Instructions
1.  Open `examples/neuralnetwork.sln` in Visual Studio.
2.  Select the desired configuration (Debug/Release) and platform (x64).
3.  Build the solution (`Ctrl+Shift+B`).

### Running the Project
- The primary entry point is `examples/main.cpp`.

### Examples
Multiple standalone examples are located in `examples/`, including:
- `xor.h`: Classic XOR problem.
- `threebitparity.h`: 3-bit parity check.
- `twomoon.h`: Two moons classification.
- `addingproblem.h`: RNN benchmark for long-term dependencies.

- Examples are commented out by default, the user will enable the example(s) needed where and when needed.

---

## Development Conventions

### Coding Style
- Use 2 spaces, never tab.
- **Naming:**
  - Classes: `PascalCase` (e.g., `NeuralNetwork`).
  - Methods/Functions: `snake_case` (e.g., `train`, `think`).
  - Private Members: Prefixed with an underscore (e.g., `_learning_rate`).
- **Headers:** Uses `#pragma once` for include guards.
- **Error Handling:** Uses `Logger` for logging and standard exceptions for critical failures.

### Architecture Patterns
- **Options Pattern:** `NeuralNetworkOptions` uses a builder-like pattern for configuration.
- **RAII:** Strict adherence to RAII for memory and resource management (though some raw pointers are used in legacy parts).
- **Thread Safety:** Uses `std::shared_mutex` for concurrent access to market data and network states.

### Logging and Profiling
- Use `Logger::info`, `Logger::debug`, `Logger::error`, etc., for all output.
- Every new function my start with MYODDWEB_PROFILE_FUNCTION("Name-Of-Class")
- If you see a function that does not have MYODDWEB_PROFILE_FUNCTION("Name-Of-Class") as a first line, please flag it to me.
- **important** The MACRO MYODDWEB_PROFILE_FUNCTION has no impact on performance, when I ask you to review my code for performance, ignore MYODDWEB_PROFILE_FUNCTION as it has no impact.

---

- When looking at the Layer Class, remember to look at all the derived classes.
- Do not worry about Git commits, I will deal will them.

## Key Files
- `include/neuralnetwork/neuralnetwork.h`: The main interface for the NN library.
- `examples/config.json`: Main configuration for the examples runner.
- `examples/strategy.json`: Trading strategy definitions.
