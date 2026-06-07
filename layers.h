#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <stdexcept>

#include "tensor.h"

// ============================================================
// ReLU Layer
// Applies: output = max(0, input)
// Used after conv1 and conv2 in your PyTorch LeNet.
// ============================================================

// for CPU
Tensor4D relu(const Tensor4D& input);
void relu_inplace(Tensor4D& input, bool openmp = false);

// for CUDA
void relu_cuda(CudaTensor4D& input, int block_size, float* compute_time_ms = nullptr);
void relu_cuda(CudaMatrix& input, int block_size, float* compute_time_ms = nullptr);

// Optional vector version, useful if you later add ReLU after FC layers.
std::vector<float> relu_vector(const std::vector<float>& input);
void relu_vector_inplace(std::vector<float>& input);


// ============================================================
// AvgPool2D Layer
// PyTorch equivalent: nn.AvgPool2d(kernel_size, stride)
// For this LeNet:
//   pool1 = AvgPool2D(2, 2)
//   pool2 = AvgPool2D(2, 2)
// ============================================================
struct AvgPool2DConfig {
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;

    AvgPool2DConfig(
        int k_h,
        int k_w,
        int s_h,
        int s_w
    );
};

class AvgPool2D {
public:
    AvgPool2DConfig cfg;

    AvgPool2D(const AvgPool2DConfig& config);

    // for CPU
    Tensor4D forward(const Tensor4D& input) const;

    // for CUDA
    CudaTensor4D forward_cuda(
        const CudaTensor4D& input,
        int block_size,
        float* compute_time_ms = nullptr,
        float* malloc_time_ms = nullptr
    ) const;

private:
    int output_height(int input_h) const;
    int output_width(int input_w) const;
};


// ============================================================
// Flatten Layer
// Converts Tensor4D into a 1D vector for Linear layer.
//
// For LeNet after pool2:
//   input shape:  [1, 16, 4, 4]
//   output size:  16 * 4 * 4 = 256
//
// This version is designed for single-image inference.
// If input.N > 1, choose which image by passing batch_index.
// ============================================================
std::vector<float> flatten(const Tensor4D& input, int batch_index = 0);


// ============================================================
// Linear / Fully Connected Layer
// PyTorch equivalent: nn.Linear(in_features, out_features)
//
// Weight layout:
//   weight[out_feature][in_feature]
// Flattened index:
//   weight[of * in_features + inf]
//
// Forward:
//   output[of] = bias[of] + sum(input[inf] * weight[of][inf])
// ============================================================
struct LinearConfig {
    int in_features;
    int out_features;
    bool use_bias;

    LinearConfig(
        int in_f,
        int out_f,
        bool bias = true
    );
};

class Linear {
public:
    LinearConfig cfg;
    std::vector<float> weight;  // [out_features][in_features]
    std::vector<float> bias;    // [out_features]

    Linear(const LinearConfig& config);

    void set_weight(const std::vector<float>& w);
    void set_bias(const std::vector<float>& b);

    // for CPU
    std::vector<float> forward(const std::vector<float>& input) const;

    // for CUDA
    CudaMatrix forward_cuda(const CudaMatrix& input,
        int block_size,
        float* compute_time_ms = nullptr,
        float* transfer_time_ms = nullptr,
        float* malloc_time_ms = nullptr
    ) const;

    int getWeightSize() const {
        return cfg.out_features * cfg.in_features;
    }
    
    int getBiasSize() const {
        return cfg.use_bias ? cfg.out_features : 0;
    }
};


// ============================================================
// Utility
// Returns the index of the largest value.
// Used for final prediction from fc3 output logits.
// ============================================================
int argmax(const std::vector<float>& input);

#endif