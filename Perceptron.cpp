// Perceptron.cpp
#include "Perceptron.h"
#include <iostream>
#include <numeric> // For std::inner_product

// Use a simple random number generator for initialization
std::random_device rd;
std::mt19937 gen(rd());
// Distribution for small random numbers (e.g., -0.5 to 0.5)
std::uniform_real_distribution<> distrib(-0.5, 0.5);

// ----------------------------------------------------
// Constructor
// ----------------------------------------------------
Perceptron::Perceptron(int num_inputs, double learning_rate)
    : learning_rate_(learning_rate), num_inputs_(num_inputs) {

    // Initialize weights randomly and the bias to a small random value
    for (int i = 0; i < num_inputs; ++i) {
        weights_.push_back(distrib(gen));
    }
    bias_ = distrib(gen);

    std::cout << "Perceptron initialized with " << num_inputs << " inputs." << std::endl;
}

// ----------------------------------------------------
// Private Helper: Weighted Sum
// ----------------------------------------------------
double Perceptron::weighted_sum(const std::vector<double>& inputs) const {
    // The sum is (w1*x1 + w2*x2 + ... + wn*xn) + bias
    return std::inner_product(inputs.begin(), inputs.end(), weights_.begin(), 0.0) + bias_;
}

// ----------------------------------------------------
// Private Helper: Activation Function (Unit Step)
// ----------------------------------------------------
int Perceptron::activation(double sum) const {
    // Strictly greater than zero
    return (sum >= 0.0) ? 1 : 0;

    // // Use a small epsilon (e.g., 1e-9) to handle floating-point
    // // arithmetic and ensure the boundary is not exactly at zero, which
    // // causes a prediction of 1 when the target is 0 for inputs (0,0), (0,1), etc.
    // const double EPSILON = 1e-9;
    
    // // Change the condition from (sum >= 0.0) to (sum > EPSILON)
    // return (sum > EPSILON) ? 1 : 0; 
}

// ----------------------------------------------------
// Forward Propagation (Prediction)
// ----------------------------------------------------
int Perceptron::predict(const std::vector<double>& inputs) const {
    if (inputs.size() != num_inputs_) {
        throw std::runtime_error("Input size mismatch in predict.");
    }
    return activation(weighted_sum(inputs));
}

// ----------------------------------------------------
// Training Function (Weight Update)
// ----------------------------------------------------
void Perceptron::train(const std::vector<double>& inputs, int target) {
    if (inputs.size() != num_inputs_) {
        throw std::runtime_error("Input size mismatch in train.");
    }

    // 1. Calculate the actual prediction
    int prediction = predict(inputs);

    // 2. Calculate the error
    int error = target - prediction;

    // 3. Update weights and bias (Perceptron Learning Rule)
    if (error != 0) {
        for (int i = 0; i < num_inputs_; ++i) {
            // New_weight = Old_weight + (learning_rate * error * input)
            weights_[i] += learning_rate_ * error * inputs[i];
        }
        // New_bias = Old_bias + (learning_rate * error)
        bias_ += learning_rate_ * error;
    }
}