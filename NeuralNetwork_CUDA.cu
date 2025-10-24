// NeuralNetwork_CUDA.cu
#include "NeuralNetwork.h"
#include <cuda_runtime.h>
#include <numeric>

// Global random number generation setup (for initialization)
// This only runs on the Host (CPU)
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> distrib(-0.5, 0.5); 

// -------------------------------------------------------------------
// --- DEVICE (GPU) UTILITIES ---
// -------------------------------------------------------------------
__device__ double sigmoid_device(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// -------------------------------------------------------------------
// --- KERNEL 1: Forward Pass (Input -> Hidden) ---
// Each thread handles one hidden neuron (j)
// -------------------------------------------------------------------
__global__ void forward_kernel_layer1(
    const double* d_input,
    const double* d_w1, 
    const double* d_b1,
    double* d_hidden_output, // Stores the activated output (a_j)
    int input_size, 
    int hidden_size) 
{
    int j = threadIdx.x; // Thread index = hidden neuron index
    if (j < hidden_size) {
        double sum = 0.0;
        
        // Calculate weighted sum (dot product)
        // W1 is accessed via 1D index: (i * hidden_size + j)        
        for (int i = 0; i < input_size; ++i) {
            sum += d_input[i] * d_w1[i * hidden_size + j];
        }
        sum += d_b1[j];
        d_hidden_output[j] = sigmoid_device(sum);
    }
}

// -------------------------------------------------------------------
// --- KERNEL 2: Forward Pass (Hidden -> Output) ---
// Each thread handles one output neuron (k)
// -------------------------------------------------------------------
__global__ void forward_kernel_layer2(
    const double* d_hidden_output,
    const double* d_w2,
    const double* d_b2,
    double* d_output, // Stores the final output (a_k) 
    int hidden_size,
    int output_size)
{
    int k = threadIdx.x; // Thread index = output neuron index
    if (k < output_size) {
        double sum = 0.0;
        
        // Calculate weighted sum (dot product)
        // W2 is accessed via 1D index: (j * output_size + k)
        for (int j = 0; j < hidden_size; ++j) {
            sum += d_hidden_output[j] * d_w2[j * output_size + k];
        }
        sum += d_b2[k];
        d_output[k] = sigmoid_device(sum);
    }
}

// -------------------------------------------------------------------
// --- KERNEL 3: Backprop & Update (Output Layer: W2 & B2) ---
// Calculates output deltas (d_k) and updates W2 and B2.
// Each thread handles one output neuron (k).
// -------------------------------------------------------------------
__global__ void backprop_update_layer2(
    const double* d_target,
    const double* d_output,
    const double* d_hidden_output,
    double* d_w2,
    double* d_b2,
    double* d_output_deltas, // Stores d_k for the next kernel
    double learning_rate,
    int hidden_size,
    int output_size)
{
    int k = threadIdx.x; // Output neuron index
    if (k < output_size) {
        // A. Calculate Output Delta (d_k)
        double error = d_target[k] - d_output[k];
        double output_deriv = d_output[k] * (1.0 - d_output[k]); // Sigmoid derivative
        double delta_k = error * output_deriv;
        
        d_output_deltas[k] = delta_k; // Save delta for layer 1
        
        // B. Update W2 (Delta_w_jk = learning_rate * d_k * a_j)
        for (int j = 0; j < hidden_size; ++j) {
            d_w2[j * output_size + k] += learning_rate * delta_k * d_hidden_output[j];
        }

        // C. Update B2 (Delta_b_k = learning_rate * d_k)
        d_b2[k] += learning_rate * delta_k;
    }
}

// -------------------------------------------------------------------
// --- KERNEL 4: Backprop & Update (Hidden Layer: W1 & B1) ---
// Calculates hidden deltas (d_j) and updates W1 and B1.
// Each thread handles one hidden neuron (j).
// -------------------------------------------------------------------
__global__ void backprop_update_layer1(
    const double* d_input,
    double* d_w1, // W1 is read and written in this kernel
    double* d_b1, // B1 is written in this kernel
    const double* d_w2, // Needed to back-propagate the error
    const double* d_hidden_output, // Needed for derivative
    const double* d_output_deltas,
    double learning_rate,
    int input_size,
    int hidden_size,
    int output_size)
{
    int j = threadIdx.x; // Hidden neuron index
    if (j < hidden_size) {
        // A. Calculate Hidden Delta (d_j)
        double weighted_output_delta_sum = 0.0;
       
        // Sum (w_jk * d_k) across all output neurons (k)
        for (int k = 0; k < output_size; ++k) {
            weighted_output_delta_sum += d_w2[j * output_size + k] * d_output_deltas[k];
        }

        // Hidden derivative: a_j * (1 - a_j)
        double a_j = d_hidden_output[j];
        double hidden_deriv = a_j * (1.0 - a_j);

        double delta_j = weighted_output_delta_sum * hidden_deriv;
        
        // B. Update W1 (Delta_w_ij = learning_rate * d_j * x_i)
        for (int i = 0; i < input_size; ++i) {
            d_w1[i * hidden_size + j] += learning_rate * delta_j * d_input[i];
        }
        
        // C. Update B1 (Delta_b_j = learning_rate * d_j)
        d_b1[j] += learning_rate * delta_j;
    }
}

// ===================================================================
// --- HOST FUNCTION IMPLEMENTATIONS (Memory Management & Launch) ---
// ===================================================================

// --- (MEMORY MANAGEMENT HELPERS) ---

// Helper function to initialize weights on the host (CPU)
void NeuralNetwork::initialize_host_weights() {
    h_w1_ = new double[w1_size_];
    h_b1_ = new double[hidden_size_];
    h_w2_ = new double[w2_size_];
    h_b2_ = new double[output_size_];

    for (size_t i = 0; i < w1_size_; ++i) { h_w1_[i] = distrib(gen); }
    for (int i = 0; i < hidden_size_; ++i) { h_b1_[i] = distrib(gen); }
    for (size_t i = 0; i < w2_size_; ++i) { h_w2_[i] = distrib(gen); }
    for (int i = 0; i < output_size_; ++i) { h_b2_[i] = distrib(gen); }
}

// Helper function to clean up all Host and Device memory
void NeuralNetwork::cleanup_memory() {
    delete[] h_w1_;
    delete[] h_b1_;
    delete[] h_w2_;
    delete[] h_b2_;

    CUDA_CHECK(cudaFree(d_w1_));
    CUDA_CHECK(cudaFree(d_b1_));
    CUDA_CHECK(cudaFree(d_w2_));
    CUDA_CHECK(cudaFree(d_b2_));
    CUDA_CHECK(cudaFree(d_hidden_output_));
    CUDA_CHECK(cudaFree(d_input_buffer_));
    CUDA_CHECK(cudaFree(d_target_buffer_));
    CUDA_CHECK(cudaFree(d_output_buffer_));
}

// --- (CONSTRUCTOR & DESTRUCTOR, and initialize_host_weights) ---

// Constructor: Allocates GPU memory and copies initial weights
NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
      learning_rate_(learning_rate) {

    // Calculate total sizes for 1D arrays
    w1_size_ = (size_t)input_size * hidden_size;
    w2_size_ = (size_t)hidden_size * output_size;

    // --- 1. Initialize Weights on Host (CPU) ---
    initialize_host_weights();

    // --- 2. Allocate Device (GPU) Memory ---
    // Weights and Biases (persistent during training)
    CUDA_CHECK(cudaMalloc((void**)&d_w1_, w1_size_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_b1_, hidden_size_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_w2_, w2_size_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_b2_, output_size_ * sizeof(double)));

    // Activation Buffers (used in forward/backward pass)
    CUDA_CHECK(cudaMalloc((void**)&d_hidden_output_, hidden_size_ * sizeof(double)));

    // Input/Target Buffers (used to transfer data for each training example)
    CUDA_CHECK(cudaMalloc((void**)&d_input_buffer_, input_size_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_target_buffer_, output_size_ * sizeof(double)));

    // Output Buffer (final output of the network)
    CUDA_CHECK(cudaMalloc((void**)&d_output_buffer_, output_size_ * sizeof(double)));

    // --- 3. Copy Initial Weights from Host (CPU) to Device (GPU) ---
    CUDA_CHECK(cudaMemcpy(d_w1_, h_w1_, w1_size_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1_, h_b1_, hidden_size_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2_, h_w2_, w2_size_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2_, h_b2_, output_size_ * sizeof(double), cudaMemcpyHostToDevice));

    std::cout << "MLP (CUDA) initialized: [" << input_size << " -> " << hidden_size << " -> " << output_size << "]" << std::endl;
}

// Destructor: Automatically cleans up memory when the object goes out of scope
NeuralNetwork::~NeuralNetwork() {
    cleanup_memory();
    std::cout << "MLP (CUDA) memory cleaned up." << std::endl;
}

// --- (FORWARD & TRAIN FUNCTIONS) ---

// Host function: Manages data transfer and launches forward kernels for prediction
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    if (input.size() != input_size_) {
        throw std::runtime_error("Input vector size mismatch.");
    }

    std::vector<double> output(output_size_);
    
    // 1. Copy Input Data to Device (GPU)
    CUDA_CHECK(cudaMemcpy(d_input_buffer_, input.data(), input_size_ * sizeof(double), cudaMemcpyHostToDevice));

    // 2. Launch Kernel 1 (Input -> Hidden)
    dim3 threads_h(hidden_size_);
    forward_kernel_layer1<<<1, threads_h>>>(
        d_input_buffer_, d_w1_, d_b1_,
        d_hidden_output_, input_size_, hidden_size_);
    
    // 3. Launch Kernel 2 (Hidden -> Output)
    dim3 threads_o(output_size_);
    // Reuse d_target_buffer to store the final output (a_k)
    forward_kernel_layer2<<<1, threads_o>>>(
        d_hidden_output_, d_w2_, d_b2_,
        d_output_buffer_, // Stores final output on Device
        hidden_size_, output_size_);

    // 4. Copy Output Data to Host (CPU)
    CUDA_CHECK(cudaMemcpy(output.data(), d_output_buffer_, output_size_ * sizeof(double), cudaMemcpyDeviceToHost));    
    
    return output;
}

// Host function: Manages data transfer and launches all kernels for training
void NeuralNetwork::train(const std::vector<double>& input, const std::vector<double>& target) {
    if (target.size() != output_size_ || input.size() != input_size_) {
        throw std::runtime_error("Input or Target vector size mismatch in train.");
    }
    
    // --- 1. Data Transfer (Host -> Device) ---
    // Copy current input and target data for this training example
    CUDA_CHECK(cudaMemcpy(d_input_buffer_, input.data(), input_size_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target_buffer_, target.data(), output_size_ * sizeof(double), cudaMemcpyHostToDevice));

    // --- 2. Forward Pass (Kernel 1 & 2) ---
    dim3 threads_h(hidden_size_);
    forward_kernel_layer1<<<1, threads_h>>>(
        d_input_buffer_, d_w1_, d_b1_,
        d_hidden_output_, input_size_, hidden_size_);
    
    dim3 threads_o(output_size_);
    forward_kernel_layer2<<<1, threads_o>>>(
        d_hidden_output_, d_w2_, d_b2_,
        d_output_buffer_,
        hidden_size_, output_size_);
    
    // --- 3. Backward Pass & W2/B2 Update (Kernel 3) ---
    double* d_output_deltas; // Temporary buffer for d_k
    CUDA_CHECK(cudaMalloc((void**)&d_output_deltas, output_size_ * sizeof(double)));
    
    backprop_update_layer2<<<1, threads_o>>>(
        d_target_buffer_, d_output_buffer_, d_hidden_output_, // Target and Output are both in d_target_buffer
        d_w2_, d_b2_,
        d_output_deltas, learning_rate_,
        hidden_size_, output_size_);
    
    // --- 4. W1/B1 Update (Kernel 4) ---
    backprop_update_layer1<<<1, threads_h>>>(
        d_input_buffer_, d_w1_, d_b1_,
        d_w2_, d_hidden_output_,
        d_output_deltas, learning_rate_,
        input_size_, hidden_size_, output_size_);

    // --- 5. Cleanup Temporary Device Memory ---
    CUDA_CHECK(cudaFree(d_output_deltas));
    // The weights (d_w1, d_w2, d_b1, d_b2) remain on the device for the next training iteration.
}