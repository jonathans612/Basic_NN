cmake_minimum_required(VERSION 3.10)
project(BasicNeuralNetwork CXX)

# Enable C++17 or C++20 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Source files for the executable
set(SOURCE_FILES
    main.cpp
    Perceptron.cpp
)

# Create the executable
add_executable(BasicNeuralNetwork ${SOURCE_FILES})

# Optional: Set output directory
set_target_properties(BasicNeuralNetwork PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)