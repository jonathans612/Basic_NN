// NeuralNetwork.cpp
#include "NeuralNetwork.h"
#include <numeric>
#include <stdexcept>

// Global random number generation setup
std::random_device rd;
std::mt19937 gen(rd());
// He/Xavier-like initialization (e.g., -0.5 to 0.5) is sufficient for a simple MLP
std::uniform_real_distribution<> distrib(-0.5, 0.5); 

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

    std::cout << "MLP initialized: [" << input_size << " -> " << hidden_size << " -> " << output_size << "]" << std::endl;
}

// ----------------------------------------------------
// Weight and Bias Initialization
// ----------------------------------------------------
void NeuralNetwork::initialize_weights() {
    // Initialize W1 (Input to Hidden)
    for (int i = 0; i < input_size_; ++i) {
        for (int j = 0; j < hidden_size_; ++j) {
            w1_[i][j] = distrib(gen);
        }
    }
    // Initialize B1 (Hidden Bias)
    for (int i = 0; i < hidden_size_; ++i) {
        b1_[i] = distrib(gen);
    }

    // Initialize W2 (Hidden to Output)
    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < output_size_; ++j) {
            w2_[i][j] = distrib(gen);
        }
    }
    // Initialize B2 (Output Bias)
    for (int i = 0; i < output_size_; ++i) {
        b2_[i] = distrib(gen);
    }
}
// ----------------------------------------------------
// Forward Propagation
// ----------------------------------------------------
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    if (input.size() != input_size_) {
        throw std::runtime_error("Input vector size mismatch.");
    }

    // --- 1. Input Layer to Hidden Layer ---
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

    // --- 2. Hidden Layer to Output Layer ---
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
// Training Function (Forward + Backpropagation + Update)
// ----------------------------------------------------
void NeuralNetwork::train(const std::vector<double>& input, const std::vector<double>& target) {
    if (target.size() != output_size_) {
        throw std::runtime_error("Target vector size mismatch in train.");
    }
    
    // --- 1. FORWARD PASS ---
    // The forward pass is run here to fill hidden_outputs_ and output_activations_
    std::vector<double> output = forward(input);

    // --- 2. BACKWARD PASS (Calculate Gradients) ---

    // A. Output Layer Error (d_k): 
    // Error = (Target - Output) * f'(Net_Input)
    // For Sigmoid: f'(z) = f(z) * (1 - f(z)) = a * (1 - a)
    std::vector<double> output_deltas(output_size_);
    for (int k = 0; k < output_size_; ++k) {
        double error = target[k] - output[k];
        
        // Output derivative: a_k * (1 - a_k)
        double output_deriv = output[k] * (1.0 - output[k]); 
        
        output_deltas[k] = error * output_deriv;
    }

    // B. Hidden Layer Error (d_j): 
    // Error = (Sum of w_jk * d_k) * f'(Net_Input_j)
    std::vector<double> hidden_deltas(hidden_size_);
    for (int j = 0; j < hidden_size_; ++j) {
        double weighted_output_delta_sum = 0.0;
        
        // Sum (w_jk * d_k) across all output neurons (k)
        for (int k = 0; k < output_size_; ++k) {
            weighted_output_delta_sum += w2_[j][k] * output_deltas[k];
        }

        // Hidden derivative: a_j * (1 - a_j)
        double hidden_deriv = hidden_outputs_[j] * (1.0 - hidden_outputs_[j]);
        
        hidden_deltas[j] = weighted_output_delta_sum * hidden_deriv;
    }


    // --- 3. WEIGHT AND BIAS UPDATE (Gradient Descent) ---

    // C. Update W2 (Hidden -> Output Weights) and B2 (Output Biases)
    // Delta_w_jk = learning_rate * d_k * a_j
    // Delta_b_k = learning_rate * d_k
    for (int k = 0; k < output_size_; ++k) {
        // Update W2
        for (int j = 0; j < hidden_size_; ++j) {
            w2_[j][k] += learning_rate_ * output_deltas[k] * hidden_outputs_[j];
        }
        // Update B2
        b2_[k] += learning_rate_ * output_deltas[k];
    }
    
    // D. Update W1 (Input -> Hidden Weights) and B1 (Hidden Biases)
    // Delta_w_ij = learning_rate * d_j * x_i
    // Delta_b_j = learning_rate * d_j
    for (int j = 0; j < hidden_size_; ++j) {
        // Update W1
        for (int i = 0; i < input_size_; ++i) {
            w1_[i][j] += learning_rate_ * hidden_deltas[j] * input[i];
        }
        // Update B1
        b1_[j] += learning_rate_ * hidden_deltas[j];
    }
}