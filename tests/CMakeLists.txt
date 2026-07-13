cmake_minimum_required(VERSION 3.24)
project(NeuralNetworkTests LANGUAGES CXX)

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

# Option to enable Tracy Profiler
option(ENABLE_TRACY "Enable Tracy Profiler" OFF)

# Define source files from the main project (relative to this directory)
set(NN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include/neuralnetwork")

set(NN_SOURCES
    "${NN_SOURCE_DIR}/common/activation.cpp"
    "${NN_SOURCE_DIR}/layers/elmanrnnlayer.cpp"
    "${NN_SOURCE_DIR}/layers/fflayer.cpp"
    "${NN_SOURCE_DIR}/layers/ffoutputlayer.cpp"
    "${NN_SOURCE_DIR}/layers/grurnnlayer.cpp"
    "${NN_SOURCE_DIR}/layers/layer.cpp"
    "${NN_SOURCE_DIR}/layers/layers.cpp"
    "${NN_SOURCE_DIR}/libraries/TinyJSON.cpp"
    "${NN_SOURCE_DIR}/layers/lstmlayer.cpp"
    "${NN_SOURCE_DIR}/neuralnetwork.cpp"
    "${NN_SOURCE_DIR}/helpers/neuralnetworkhelper.cpp"
    "${NN_SOURCE_DIR}/helpers/neuralnetworkserializer.cpp"
    "${NN_SOURCE_DIR}/neuron.cpp"
)

if(ENABLE_TRACY)
    list(APPEND NN_SOURCES "${NN_SOURCE_DIR}/libraries/tracy/TracyClient.cpp")
endif()

# Define include directories
set(NN_INCLUDE_DIRS
    "${NN_SOURCE_DIR}"
    "${NN_SOURCE_DIR}/libraries"
)

# Create the test executable
add_executable(neuralnetwork_tests
    activation_tests.cpp
    adaptive_learning_rate_scheduler_tests.cpp
    aligned_allocator_tests.cpp
    elmanrnnlayer_mt_tests.cpp
    elmanrnnlayer_tests.cpp
    error_calculation_tests.cpp
    fflayer_mt_tests.cpp
    fflayer_tests.cpp
    ffoutputlayer_mt_tests.cpp
    ffoutputlayer_tests.cpp
    gradients_and_outputs_tests.cpp
    grurnnlayer_mt_tests.cpp
    grurnnlayer_tests.cpp
    hidden_state_tests.cpp
    layer_details_tests.cpp
    layer_optimizer_tests.cpp
    layer_tests.cpp
    layers_and_neurons_container_tests.cpp
    learning_rate_tests.cpp
    lstmlayer_mt_tests.cpp
    lstmlayer_tests.cpp
    multioutputlayer_tests.cpp
    network_integration_tests.cpp
    neuralnetwork_tests.cpp    
    neuralnetworkhelper_tests.cpp
    neuron_tests.cpp
    output_layer_details_tests.cpp
    simd_utils_tests.cpp
    simd_utils_mt_tests.cpp
    taskqueue_tests.cpp
    taskqueue_comprehensive_tests.cpp
    weight_param_tests.cpp
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

# Set strict warning levels
if(MSVC)
    target_compile_options(neuralnetwork_tests PRIVATE 
        /W4 
        /WX 
        /permissive-
        /w15038 # Enable C5038: member initialization order as a Level 1 warning
        /arch:AVX2
    )
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|AMD64|x86_64|x64|X64|i386|i686|x86")
        target_compile_options(neuralnetwork_tests PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter -mavx2 -mfma)
    else()
        target_compile_options(neuralnetwork_tests PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter)
    endif()
endif()

# Add preprocessor definitions to match the main project
target_compile_definitions(neuralnetwork_tests PRIVATE
    VALIDATE_DATA=1
    _CRT_SECURE_NO_WARNINGS
    MYODDWEB_PROFILE=0
)

if(ENABLE_TRACY)
    target_compile_definitions(neuralnetwork_tests PRIVATE TRACY_ENABLE)
endif()

include(GoogleTest)
include(CTest)

gtest_discover_tests(neuralnetwork_tests)
