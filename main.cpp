// main.cpp
#include "NeuralNetwork.h"
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

// --- 3. Neural Network Test Function ---

// Rename function and update contents
void test_neural_network() { 
    // Pointers to hold the active data and title
    const std::vector<std::vector<double>>* current_inputs;
    const std::vector<int>* current_targets;
    std::string problem_title;
    
    get_active_data(current_inputs, current_targets, problem_title);

    int max_attempts = 10;
    int attempt_count = 0;
    bool perfect_solution_found = false;
    
    // --- Loop to guarantee a stable solution ---
    do {
        // Increment attempt counter and check limit
        attempt_count++;
        if (attempt_count > max_attempts) {
            std::cerr << "Error: Failed to find a perfect solution after " 
                      << max_attempts << " attempts. Stopping." << std::endl;
            return; // Exit function if max attempts reached
        }
        
        std::cout << "\n--- Starting Training Attempt " << attempt_count << " (" << problem_title << ") ---\n";

        // Re-initialize the model with new random weights/biases
        NeuralNetwork model(2, 3, 1, 0.3); 
        int max_epochs = 50000; 
    
    // Since the output is a continuous double (Sigmoid), we need a threshold
        const double THRESHOLD = 0.5;
        int convergence_epoch = -1;
        
        // --- Training Loop ---
        for (int epoch = 1; epoch <= max_epochs; ++epoch) {
            int errors = 0;
            
            for (size_t i = 0; i < current_inputs->size(); ++i) {
                const auto& inputs = (*current_inputs)[i];
            
            // The target must be a vector of doubles for the MLP
                std::vector<double> target_vec = { (double)(*current_targets)[i] }; 

            model.train(inputs, target_vec); // Train

            // Prediction/Error check using forward()
                double prediction = model.forward(inputs)[0];
                int binary_prediction = (prediction >= THRESHOLD) ? 1 : 0;
                int target = (int)target_vec[0];

                if (binary_prediction != target) {
                    errors++;
                }
            }

            if (epoch % 10000 == 0) { // Print progress less frequently for high epoch count
                std::cout << "Epoch " << epoch << ": Errors = " << errors << std::endl;
            }

            if (errors == 0 && epoch > 100) { // Check for convergence only after some training
                convergence_epoch = epoch;
                break;
            }
        }
        
        // --- Testing (Verification) ---
        int final_errors = 0;
        
        // Only run final test if training convergence was signaled
        if (convergence_epoch != -1) {
            for (size_t i = 0; i < current_inputs->size(); ++i) {
                const auto& inputs = (*current_inputs)[i];
                int target = (int)(*current_targets)[i];
                
                double prediction_val = model.forward(inputs)[0];
                int binary_prediction = (prediction_val >= THRESHOLD) ? 1 : 0;
                
                if (binary_prediction != target) {
                    final_errors++;
                }
            }
        } else {
             final_errors = 4; // Failed to converge within max_epochs
        }
        
        // --- Check for Perfect Success ---
        if (final_errors == 0) {
            perfect_solution_found = true;
            
            // Print the successful result
            std::cout << "\nTraining complete. Network converged in " << convergence_epoch 
                      << " epochs on attempt " << attempt_count << ".\n";
                      
            std::cout << "\n--- Final Test Results (" << problem_title << ") ---\n";
            // Print all successful test results...
            for (size_t i = 0; i < current_inputs->size(); ++i) {
                const auto& inputs = (*current_inputs)[i];
                int target = (int)(*current_targets)[i];
                
                double prediction_val = model.forward(inputs)[0];
                int binary_prediction = (prediction_val >= THRESHOLD) ? 1 : 0;

                std::cout << "(" << inputs[0] << ", " << inputs[1] << ") -> Target: "
                          << target << ", Prediction: " << binary_prediction
                          << " (Raw: " << prediction_val << ")" 
                          << (binary_prediction == target ? " (CORRECT)" : " (INCORRECT!)")
                          << std::endl;
            }
        } else {
             std::cout << "Attempt " << attempt_count << " failed final test with " 
                       << final_errors << " errors. Retrying...\n";
        }
        
    } while (!perfect_solution_found);
}

// Update the function call in main()
int main() {
    try {
        // Change the call to the new function name
        test_neural_network(); 
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        std::cout << "\nPress ENTER to exit..." << std::endl;
        std::cin.get(); 
        return 1;
    }
    std::cout << "\nPress ENTER to exit..." << std::endl;
    std::cin.get(); 
    return 0;
}