// main.cpp
#include "Perceptron.h"
#include <iostream>
#include <vector>
#include <string>

// ====================================================
// FLAG TO TOGGLE DATASET
// Set to 1 for XOR, Set to 0 for AND
// ====================================================
#define USE_XOR 1 
// ====================================================

// --- 1. Dataset Definitions ---
// The training data for the AND gate (4 examples)
const std::vector<std::vector<double>> AND_INPUTS = {
    {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
};
const std::vector<int> AND_TARGETS = {
    0, 0, 0, 1 
};

// The training data for the XOR gate
const std::vector<std::vector<double>> XOR_INPUTS = {
    {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
};
const std::vector<int> XOR_TARGETS = {
    0, 1, 1, 0 
};

// --- 2. Function to Get Active Data ---

void get_active_data(
    const std::vector<std::vector<double>>*& inputs, 
    const std::vector<int>*& targets, 
    std::string& title) 
{
    // Use the preprocessor to select the data based on the flag
    #if USE_XOR
        inputs = &XOR_INPUTS;
        targets = &XOR_TARGETS;
        title = "XOR Gate";
    #else
        inputs = &AND_INPUTS;
        targets = &AND_TARGETS;
        title = "AND Gate";
    #endif
}

// --- 3. Perceptron Test Function ---

void test_perceptron() {
    // Pointers to hold the active data and title
    const std::vector<std::vector<double>>* current_inputs;
    const std::vector<int>* current_targets;
    std::string problem_title;

    // Get the data based on the flag
    get_active_data(current_inputs, current_targets, problem_title);

    // Perceptron initialization
    Perceptron model(2, 0.2); // Using 0.2 learning rate, 2 inputs
    int max_epochs = 200;
    bool all_correct = false;

    // --- Training Loop ---
    for (int epoch = 1; epoch <= max_epochs && !all_correct; ++epoch) {
        int errors = 0;
        
        // Use the dereferenced pointers in the loop
        for (size_t i = 0; i < current_inputs->size(); ++i) {
            const auto& inputs = (*current_inputs)[i];
            int target = (*current_targets)[i];

            model.train(inputs, target);

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
    std::cout << "\n--- Final Test Results (" << problem_title << ") ---" << std::endl;
    for (size_t i = 0; i < current_inputs->size(); ++i) {
        const auto& inputs = (*current_inputs)[i];
        int target = (*current_targets)[i];
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
        // Keep the window open to read error message
        std::cout << "\nPress ENTER to exit..." << std::endl;
        std::cin.get(); 
        return 1;
    }
    // Keep the window open to read results
    std::cout << "\nPress ENTER to exit..." << std::endl;
    std::cin.get(); 
    return 0;
}