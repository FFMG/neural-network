cmake_minimum_required(VERSION 3.14)
project(NeuralNetworkTests CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For Windows: prevent gtest from overriding our compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Define source files from the main project (relative to this directory)
set(NN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/neuralnetwork")

set(NN_SOURCES
    "${NN_SOURCE_DIR}/activation.cpp"
    "${NN_SOURCE_DIR}/elmanrnnlayer.cpp"
    "${NN_SOURCE_DIR}/fflayer.cpp"
    "${NN_SOURCE_DIR}/ffoutputlayer.cpp"
    "${NN_SOURCE_DIR}/grurnnlayer.cpp"
    "${NN_SOURCE_DIR}/layer.cpp"
    "${NN_SOURCE_DIR}/layers.cpp"
    "${NN_SOURCE_DIR}/libraries/TinyJSON.cpp"
    "${NN_SOURCE_DIR}/libraries/tracy/TracyClient.cpp"
    "${NN_SOURCE_DIR}/lstmlayer.cpp"
    "${NN_SOURCE_DIR}/neuralnetwork.cpp"
    "${NN_SOURCE_DIR}/neuralnetworkhelper.cpp"
    "${NN_SOURCE_DIR}/neuralnetworkserializer.cpp"
    "${NN_SOURCE_DIR}/neuron.cpp"
)

# Define include directories
set(NN_INCLUDE_DIRS
    "${NN_SOURCE_DIR}"
    "${NN_SOURCE_DIR}/libraries"
)

# Create the test executable
add_executable(neuralnetwork_tests
    neuralnetwork.cpp
    activation_tests.cpp
    error_calculation_tests.cpp
    gradients_and_outputs_tests.cpp
    aligned_allocator_tests.cpp
    layers_and_neurons_container_tests.cpp
    hidden_state_tests.cpp
    weight_param_tests.cpp
    neuron_tests.cpp
    simd_utils_tests.cpp
    fflayer_tests.cpp
    fflayer_mt_tests.cpp
    grurnnlayer_tests.cpp
    grurnnlayer_mt_tests.cpp
    ffoutputlayer_tests.cpp
    ffoutputlayer_mt_tests.cpp
    layer_tests.cpp
    multioutputlayer_tests.cpp
    elmanrnnlayer_tests.cpp
    lstmlayer_tests.cpp
    layer_optimizer_tests.cpp
    layer_details_tests.cpp
    output_layer_details_tests.cpp
    ${NN_SOURCES}
)

# Set include directories for the test executable
target_include_directories(neuralnetwork_tests PRIVATE
    ${NN_INCLUDE_DIRS}
)

# Link Google Test
target_link_libraries(neuralnetwork_tests PRIVATE
    gtest_main
)

# Add preprocessor definitions to match the main project
target_compile_definitions(neuralnetwork_tests PRIVATE
    VALIDATE_DATA=1
    _CRT_SECURE_NO_WARNINGS
    # TRACY_ENABLE # Disabled by default as per user preference
    MYODDWEB_PROFILE=0
)

# Enable CTest (optional but good practice)
include(CTest)
add_test(NAME NeuralNetworkTests COMMAND neuralnetwork_tests)
