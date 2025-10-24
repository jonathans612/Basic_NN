// NeuralNetwork_CPU.cpp
// The sequential (CPU-only) implementation

#include "NeuralNetwork.h"
#include <numeric>
#include <stdexcept>

// Global random number generation setup
std::random_device rd_cpu;
std::mt19937 gen_cpu(rd_cpu());
std::uniform_real_distribution<> distrib_cpu(-0.5, 0.5); 

// ----------------------------------------------------
// Constructor
// ----------------------------------------------------
NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
      learning_rate_(learning_rate) {
    
    // Resize data structures
    w1_.resize(input_size, std::vector<double>(hidden_size));
    w2_.resize(hidden_size, std::vector<double>(output_size));
    b1_.resize(hidden_size);
    b2_.resize(output_size);
    
    // Resize intermediate storage
    hidden_activations_.resize(hidden_size);
    hidden_outputs_.resize(hidden_size);
    output_activations_.resize(output_size);

    // Initialize all weights and biases randomly
    initialize_weights();

    std::cout << "MLP (CPU) initialized: [" << input_size << " -> " << hidden_size << " -> " << output_size << "]" << std::endl;
}

// ----------------------------------------------------
// Destructor (empty for CPU version, but must be defined)
// ----------------------------------------------------
NeuralNetwork::~NeuralNetwork() {
    std::cout << "MLP (CPU) destroyed." << std::endl;
}

// ----------------------------------------------------
// Weight and Bias Initialization
// ----------------------------------------------------
void NeuralNetwork::initialize_weights() {
    for (int i = 0; i < input_size_; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            w1_[i][j] = distrib_cpu(gen_cpu);
        }
    }
    // Initialize B1 (Hidden Bias)
    for (int i = 0; i < hidden_size_; ++i) {
        b1_[i] = distrib_cpu(gen_cpu);
    }
    
    // Initialize W2 (Hidden to Output)
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < output_size_; ++j) {
            w2_[i][j] = distrib_cpu(gen_cpu);
        }
    }
    // Initialize B2 (Output Bias)
    for (int i = 0; i < output_size_; ++i) {
        b2_[i] = distrib_cpu(gen_cpu);
    }
}

// ----------------------------------------------------
// Forward Propagation
// ----------------------------------------------------
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    if (input.size() != input_size_) {
        throw std::runtime_error("Input vector size mismatch.");
    }

    // Input -> Hidden
    for (int j = 0; j < hidden_size_; ++j) {
        double sum = 0.0;
        // Calculate weighted sum (dot product of input vector and W1 column)
        for (int i = 0; i < input_size_; ++i) {
            sum += input[i] * w1_[i][j];
        }
        
        // Add bias
        sum += b1_[j];
        
        // Store net activation (z) for backprop
        hidden_activations_[j] = sum; 
        
        // Apply Sigmoid activation (a)
        hidden_outputs_[j] = sigmoid(sum);
    }

    // Hidden -> Output
    std::vector<double> output(output_size_);
    for (int k = 0; k < output_size_; ++k) {
        double sum = 0.0;
        // Calculate weighted sum (dot product of hidden outputs and W2 column)
        for (int j = 0; j < hidden_size_; ++j) {
            sum += hidden_outputs_[j] * w2_[j][k];
        }
        
        // Add bias
        sum += b2_[k];
        
        // Store net activation (z) for backprop
        output_activations_[k] = sum; 
        
        // Apply Sigmoid activation (a)
        output[k] = sigmoid(sum);
    }
    return output;
}

// ----------------------------------------------------
// Training Function (Backpropagation)
// ----------------------------------------------------
void NeuralNetwork::train(const std::vector<double>& input, const std::vector<double>& target) {
    if (target.size() != output_size_) {
        throw std::runtime_error("Target vector size mismatch in train.");
    }
    
    // --- 1. FORWARD PASS ---
    // The forward pass is run here to get outputs and cache activations
    std::vector<double> output = forward(input);

    // --- 2. BACKWARD PASS (Calculate Gradients) ---
    
    // Output Layer Error (d_k)
    // Error = target - output
    // For Sigmoid, derivative = output * (1 - output)
    std::vector<double> output_deltas(output_size_);
    for (int k = 0; k < output_size_; ++k) {
        double error = target[k] - output[k];
        double output_deriv = output[k] * (1.0 - output[k]); 
        output_deltas[k] = error * output_deriv;
    }

    // Hidden Layer Error (d_j)
    // Error = sum over k of (w2_jk * d_k)
    std::vector<double> hidden_deltas(hidden_size_);
    for (int j = 0; j < hidden_size_; ++j) {
        double weighted_output_delta_sum = 0.0;
        for (int k = 0; k < output_size_; ++k) {
            weighted_output_delta_sum += w2_[j][k] * output_deltas[k];
        }
        double hidden_deriv = hidden_outputs_[j] * (1.0 - hidden_outputs_[j]);
        hidden_deltas[j] = weighted_output_delta_sum * hidden_deriv;
    }

    // --- 3. WEIGHT AND BIAS UPDATE (Gradient Descent) ---
    
    // Update W2 and B2
    // Delta W2_jk = learning_rate * d_k * a_j
    // Delta B2_k = learning_rate * d_k
    for (int k = 0; k < output_size_; ++k) {
        // Update each weight W2_jk
        for (int j = 0; j < hidden_size_; ++j) {
            w2_[j][k] += learning_rate_ * output_deltas[k] * hidden_outputs_[j];
        }
        // Update bias B2_k
        b2_[k] += learning_rate_ * output_deltas[k];
    }
    
    // Update W1 and B1
    // Delta W1_ij = learning_rate * d_j * input_i
    // Delta B1_j = learning_rate * d_j
    for (int j = 0; j < hidden_size_; ++j) {
        // Update each weight W1_ij
        for (int i = 0; i < input_size_; ++i) {
            w1_[i][j] += learning_rate_ * hidden_deltas[j] * input[i];
        }
        // Update bias B1_j
        b1_[j] += learning_rate_ * hidden_deltas[j];
    }
}