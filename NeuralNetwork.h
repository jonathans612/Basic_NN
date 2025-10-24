// NeuralNetwork.h
// Unified Header for CPU and GPU implementations

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

// This flag will be set by CMake
#if defined(USE_CUDA_FLAG)
    #include <string> // For std::to_string

    // CUDA Error Checking macro
    #define CUDA_CHECK(call)                                                          \
    {                                                                                 \
        cudaError_t err = call;                                                       \
        if (err != cudaSuccess) {                                                     \
            throw std::runtime_error(std::string("CUDA Error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " +              \
                                     cudaGetErrorString(err));                        \
        }                                                                             \
    }
#endif

// --- I. Math Utilities (Common to both) ---
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// ----------------------------------------------------
// II. Neural Network Class Definition
// ----------------------------------------------------
class NeuralNetwork {
public:
    // Public interface is identical for both versions
    NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate);
    ~NeuralNetwork();  // <<< For cleaning GPU memory
    std::vector<double> forward(const std::vector<double>& input);
    void train(const std::vector<double>& input, const std::vector<double>& target);

private:
    // --- Configuration (Common) ---
    int input_size_;
    int hidden_size_;
    int output_size_;
    double learning_rate_;

// Use preprocessor to swap private members
#if defined(USE_CUDA_FLAG)
    // --- Private Members for CUDA (GPU) Version ---
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
    double* d_output_buffer_; // Buffer for final network output

    // Private Helper Functions (for CUDA)
    void initialize_host_weights();
    void cleanup_memory();

#else
    // --- Private Members for Sequential (CPU) Version ---
    std::vector<std::vector<double>> w1_;
    std::vector<std::vector<double>> w2_;
    std::vector<double> b1_; 
    std::vector<double> b2_;

    std::vector<double> hidden_activations_;
    std::vector<double> hidden_outputs_;
    std::vector<double> output_activations_;

    // Private Helper Function (for CPU)
    void initialize_weights();
#endif
};

#endif // NEURALNETWORK_H