// Perceptron.h
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <random>

class Perceptron {
public:
    // Constructor: Takes the number of inputs (N) and a learning rate (alpha)
    Perceptron(int num_inputs, double learning_rate);

    // Forward Propagation: Calculates the output (0 or 1) for a given input vector
    int predict(const std::vector<double>& inputs) const;

    // Training Function: Adjusts weights based on a single training example
    void train(const std::vector<double>& inputs, int target);

private:
    std::vector<double> weights_;
    double bias_;
    double learning_rate_;
    int num_inputs_;

    // Private helper function: Calculates the sum of (weight * input) + bias
    double weighted_sum(const std::vector<double>& inputs) const;

    // Activation function: The Unit Step function
    int activation(double sum) const;
};

#endif // PERCEPTRON_H