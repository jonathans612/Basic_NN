cmake_minimum_required(VERSION 3.10)
project(BasicNeuralNetwork CXX CUDA)

# --- Find CUDA Toolkit ---
# This command defines the "CUDA::cudart" target.
find_package(CUDAToolkit REQUIRED)

# --- Set Architectures ---
# This fixes the CMP0104 warning and tells nvcc to build for your native GPU.
set(CMAKE_CUDA_ARCHITECTURES "native")

# Enable C++17 or C++20 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Source files for the executable
set(SOURCE_FILES
    main.cpp
    NeuralNetwork_CUDA.cu
)

# Create the executable
add_executable(BasicNeuralNetwork ${SOURCE_FILES})

# Link the CUDA Runtime library
target_link_libraries(BasicNeuralNetwork PUBLIC CUDA::cudart) 

# Optional: Set output directory
set_target_properties(BasicNeuralNetwork PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)