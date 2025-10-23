// NeuralNetwork.h
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <iostream>
#include <random>

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

// ----------------------------------------------------
// II. Neural Network Class Definition
// ----------------------------------------------------
class NeuralNetwork {
public:
    // Constructor: Define network structure (e.g., [2, 4, 1] means 2 inputs, 4 hidden, 1 output)
    NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate);

    // Forward Propagation
    std::vector<double> forward(const std::vector<double>& input);

    // Training (Forward + Backpropagation + Weight Update)
    void train(const std::vector<double>& input, const std::vector<double>& target);

private:
    // Network structure and parameters
    int input_size_;
    int hidden_size_;
    int output_size_;
    double learning_rate_;

    // Weights: W1 (Input->Hidden), W2 (Hidden->Output)
    std::vector<std::vector<double>> w1_; // Matrix of size (input_size x hidden_size)
    std::vector<std::vector<double>> w2_; // Matrix of size (hidden_size x output_size)

    // Biases: B1 (Hidden), B2 (Output)
    std::vector<double> b1_; 
    std::vector<double> b2_;

    // Storage for intermediate results during forward pass (needed for backprop)
    std::vector<double> hidden_activations_; // Net input (z) to hidden layer
    std::vector<double> hidden_outputs_;     // Activation output (a) from hidden layer
    std::vector<double> output_activations_; // Net input (z) to output layer

    // Initialization helper
    void initialize_weights();
};

#endif // NEURALNETWORK_H