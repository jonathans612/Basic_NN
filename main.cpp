// main.cpp
#include "Perceptron.h"
#include <iostream>

// The training data for the AND gate (4 examples)
const std::vector<std::vector<double>> AND_INPUTS = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

const std::vector<int> AND_TARGETS = {
    0, // (0, 0) -> 0
    0, // (0, 1) -> 0
    0, // (1, 0) -> 0
    1  // (1, 1) -> 1
};

void test_perceptron() {
    // Perceptron for 2 inputs (x1, x2) and a learning rate of 0.2
    Perceptron model(2, 0.2);

    int max_epochs = 200;
    bool all_correct = false;

    // --- Training Loop ---
    for (int epoch = 1; epoch <= max_epochs && !all_correct; ++epoch) {
        int errors = 0;
        
        // Iterate over all training examples (inputs and targets)
        for (size_t i = 0; i < AND_INPUTS.size(); ++i) {
            const auto& inputs = AND_INPUTS[i];
            int target = AND_TARGETS[i];

            // Train on one example
            model.train(inputs, target);

            // Check if the current example is correctly predicted
            if (model.predict(inputs) != target) {
                errors++;
            }
        }

        std::cout << "Epoch " << epoch << ": Errors = " << errors << std::endl;

        if (errors == 0) {
            all_correct = true;
            std::cout << "\nTraining complete. Perceptron converged in " << epoch << " epochs." << std::endl;
        }
    }

    // --- Testing (Verification) ---
    std::cout << "\n--- Final Test Results (AND Gate) ---" << std::endl;
    for (size_t i = 0; i < AND_INPUTS.size(); ++i) {
        const auto& inputs = AND_INPUTS[i];
        int target = AND_TARGETS[i];
        int prediction = model.predict(inputs);

        std::cout << "(" << inputs[0] << ", " << inputs[1] << ") -> Target: "
                  << target << ", Prediction: " << prediction
                  << (prediction == target ? " (CORRECT)" : " (INCORRECT!)")
                  << std::endl;
    }
}

int main() {
    try {
        test_perceptron();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}