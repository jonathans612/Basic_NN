cmake_minimum_required(VERSION 3.10)

# --- Add the toggle option (default is ON) ---
option(USE_CUDA "Enable GPU acceleration with CUDA" ON)

# --- Set project languages based on the option ---
if(USE_CUDA)
    project(BasicNeuralNetwork CXX CUDA)
    
    # --- Find CUDA Toolkit ---
    # This command defines the "CUDA::cudart" target.
    find_package(CUDAToolkit REQUIRED)

    # --- Set Architectures ---
    set(CMAKE_CUDA_ARCHITECTURES "native")
else()
    project(BasicNeuralNetwork CXX)
endif()

# Enable C++17 or C++20 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Conditionally select source files ---
# Start with the common file
set(SOURCE_FILES
    main.cpp
)

# Add the correct implementation file based on the flag
if(USE_CUDA)
    list(APPEND SOURCE_FILES NeuralNetwork_CUDA.cu)
else()
    list(APPEND SOURCE_FILES NeuralNetwork.cpp)
endif()

# Create the executable
add_executable(BasicNeuralNetwork ${SOURCE_FILES})

# --- Conditionally link libraries and set definitions ---
if(USE_CUDA)
    # Link the CUDA Runtime
    target_link_libraries(BasicNeuralNetwork PUBLIC CUDA::cudart)
    
    # Define the preprocessor flag so NeuralNetwork.h knows
    target_compile_definitions(BasicNeuralNetwork PRIVATE USE_CUDA_FLAG)
endif()

# Optional: Set output directory
set_target_properties(BasicNeuralNetwork PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)