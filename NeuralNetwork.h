// NeuralNetwork.h (Correct CUDA Version)
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept> // Already present, but needed for runtime_error
#include <string>    // <<< ADD THIS LINE for std::to_string

// --- I. Math Utilities ---

// 1. Sigmoid Activation Function (for hidden and output layers)
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// 2. Derivative of the Sigmoid (needed for Backpropagation)
inline double sigmoid_derivative(double x) {
    // Note: This calculation is often done using the function's output (y),
    // but here we use the input (x) for a clean utility function. 
    // We will calculate it using the output later in the main class for efficiency.
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// CUDA Error Checking macro (necessary for the header as well)
#define CUDA_CHECK(call)                                                          \
{                                                                                 \
    cudaError_t err = call;                                                       \
    if (err != cudaSuccess) {                                                     \
        throw std::runtime_error(std::string("CUDA Error at ") + __FILE__ + ":" + \
                                 std::to_string(__LINE__) + " - " +              \
                                 cudaGetErrorString(err));                        \
    }                                                                             \
}

// ----------------------------------------------------
// II. Neural Network Class Definition
// ----------------------------------------------------
class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate);
    ~NeuralNetwork(); // <<< Crucial for cleaning GPU memory

    std::vector<double> forward(const std::vector<double>& input);
    void train(const std::vector<double>& input, const std::vector<double>& target);

private:
    // --- Configuration and Sizes ---
    int input_size_;
    int hidden_size_;
    int output_size_;
    double learning_rate_;
    
    size_t w1_size_; 
    size_t w2_size_;

    // --- Host (CPU) Pointers (for initialization and final results) ---
    double* h_w1_;
    double* h_b1_;
    double* h_w2_;
    double* h_b2_;
    
    // --- Device (GPU) Pointers (The core memory) ---
    double* d_w1_; // Weights: Input -> Hidden
    double* d_b1_; // Biases: Hidden Layer
    double* d_w2_; // Weights: Hidden -> Output
    double* d_b2_; // Biases: Output Layer

    // --- Device (GPU) Activation/Cache Buffers ---
    double* d_hidden_output_; // Output of hidden layer (a_j)
    double* d_input_buffer_;  // Buffer to hold input vector
    double* d_target_buffer_; // Buffer to hold target vector

    // --- Private Helper Functions ---
    void initialize_host_weights();
    void cleanup_memory();
};

#endif // NEURALNETWORK_H