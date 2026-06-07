#include "layers.h"
#include "tensor.h"
#include "cuda_help.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>

// ============================================================
// ReLU Layer
// ============================================================

Tensor4D relu(const Tensor4D& input) {
    Tensor4D output(input.N, input.C, input.H, input.W);

    for (int i = 0; i < static_cast<int>(input.data.size()); ++i) {
        output.data[i] = input.data[i] > 0.0f ? input.data[i] : 0.0f;
    }

    return output;
}

void relu_inplace(Tensor4D& input, bool openmp) {
    float* ptr = input.data.data();
    int size = static_cast<int>(input.data.size());

    if (openmp) {
        #pragma omp simd
        for (int i = 0; i < size; ++i) {
            ptr[i] = ptr[i] > 0.0f ? ptr[i] : 0.0f;
        }
    }else{
        for (int i = 0; i < size; ++i) {
            ptr[i] = ptr[i] > 0.0f ? ptr[i] : 0.0f;
        }
    }
}

std::vector<float> relu_vector(const std::vector<float>& input) {
    std::vector<float> output(input.size(), 0.0f);

    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }

    return output;
}

void relu_vector_inplace(std::vector<float>& input) {
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        if (input[i] < 0.0f) {
            input[i] = 0.0f;
        }
    }
}

// ============================================================
// ReLU CUDA version
// ============================================================

__global__ void relu_kernel(float* data, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size && data[idx] < 0.0f) {
        data[idx] = 0.0f;
    }
}

void relu_cuda(CudaTensor4D& input, int block_size, float* compute_time_ms) {
    if (block_size <= 0) {
        throw std::invalid_argument("relu_cuda: block_size must be positive.");
    }

    dim3 blockSize(block_size);
    dim3 gridSize((input.size() + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
    }

    relu_kernel<<<gridSize, blockSize>>>(input.data, input.size());
    CUDA_CHECK_KERNEL();

    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_ADD_ELAPSED_TIME(start, stop, compute_time_ms);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

void relu_cuda(CudaMatrix& input, int block_size, float* compute_time_ms) {
    if (block_size <= 0) {
        throw std::invalid_argument("relu_cuda: block_size must be positive.");
    }

    dim3 blockSize(block_size);
    dim3 gridSize((input.size() + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
    }

    relu_kernel<<<gridSize, blockSize>>>(input.data, input.size());
    CUDA_CHECK_KERNEL();

    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_ADD_ELAPSED_TIME(start, stop, compute_time_ms);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

// ============================================================
// AvgPool2DConfig
// ============================================================

AvgPool2DConfig::AvgPool2DConfig(
    int k_h,
    int k_w,
    int s_h,
    int s_w
)
    : kernel_h(k_h),
      kernel_w(k_w),
      stride_h(s_h),
      stride_w(s_w) {
    if (kernel_h <= 0 || kernel_w <= 0) {
        throw std::invalid_argument("AvgPool2DConfig: kernel size must be positive.");
    }
    if (stride_h <= 0 || stride_w <= 0) {
        throw std::invalid_argument("AvgPool2DConfig: stride must be positive.");
    }
}

// ============================================================
// AvgPool2D
// ============================================================

AvgPool2D::AvgPool2D(const AvgPool2DConfig& config)
    : cfg(config) {}

int AvgPool2D::output_height(int input_h) const {
    return (input_h - cfg.kernel_h) / cfg.stride_h + 1;
}

int AvgPool2D::output_width(int input_w) const {
    return (input_w - cfg.kernel_w) / cfg.stride_w + 1;
}

Tensor4D AvgPool2D::forward(const Tensor4D& input) const {
    const int out_h = output_height(input.H);
    const int out_w = output_width(input.W);

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("AvgPool2D::forward: invalid output shape. Check kernel/stride.");
    }

    Tensor4D output(input.N, input.C, out_h, out_w);

    const float pool_area = static_cast<float>(cfg.kernel_h * cfg.kernel_w);

    for (int n = 0; n < input.N; ++n) {
        for (int c = 0; c < input.C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;

                    for (int kh = 0; kh < cfg.kernel_h; ++kh) {
                        for (int kw = 0; kw < cfg.kernel_w; ++kw) {
                            const int ih = oh * cfg.stride_h + kh;
                            const int iw = ow * cfg.stride_w + kw;
                            sum += input.at(n, c, ih, iw);
                        }
                    }

                    output.at(n, c, oh, ow) = sum / pool_area;
                }
            }
        }
    }

    return output;
}

// ============================================================
// AvgPool2D CUDA version
// ============================================================

__global__ void avgpool2d_kernel(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int out_h,
    int out_w
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputSize = N * C * out_h * out_w;

    if (idx >= outputSize) {
        return;
    }

    const int ow = idx % out_w;
    const int oh = (idx / out_w) % out_h;
    const int c = (idx / (out_h * out_w)) % C;
    const int n = idx / (C * out_h * out_w);

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int ih = oh * stride_h + kh;
            const int iw = ow * stride_w + kw;
            sum += input[((n * C + c) * H + ih) * W + iw];
        }
    }

    output[((n * C + c) * out_h + oh) * out_w + ow] =
        sum / static_cast<float>(kernel_h * kernel_w);
}

CudaTensor4D AvgPool2D::forward_cuda(const CudaTensor4D& input, int block_size, float* compute_time_ms, float* malloc_time_ms) const {
    if (block_size <= 0) {
        throw std::invalid_argument("AvgPool2D::forward_cuda: block_size must be positive.");
    }

    const int out_h = output_height(input.H);
    const int out_w = output_width(input.W);

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("AvgPool2D::forward_cuda: invalid output shape. Check kernel/stride.");
    }

    CudaTensor4D output(nullptr, input.N, input.C, out_h, out_w, true);
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output.data), output.size() * sizeof(float)));
    CUDA_MALLOC_TIMED((void**)&output.data, output.size() * sizeof(float), malloc_time_ms);

    dim3 blockSize(block_size);
    dim3 gridSize((output.size() + blockSize.x - 1) / blockSize.x);
    
    cudaEvent_t start, stop;
    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
    }

    avgpool2d_kernel<<<gridSize, blockSize>>>(
        input.data,
        output.data,
        input.N,
        input.C,
        input.H,
        input.W,
        cfg.kernel_h,
        cfg.kernel_w,
        cfg.stride_h,
        cfg.stride_w,
        out_h,
        out_w
    );

    CUDA_CHECK_KERNEL();

    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_ADD_ELAPSED_TIME(start, stop, compute_time_ms);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return output;
}

// ============================================================
// Flatten Layer
// ============================================================

std::vector<float> flatten(const Tensor4D& input, int batch_index) {
    if (batch_index < 0 || batch_index >= input.N) {
        throw std::out_of_range("flatten: batch_index out of range.");
    }

    std::vector<float> output;
    output.reserve(input.C * input.H * input.W);

    // Keep the same NCHW memory order used by Tensor4D:
    // [channel][height][width]
    for (int c = 0; c < input.C; ++c) {
        for (int h = 0; h < input.H; ++h) {
            for (int w = 0; w < input.W; ++w) {
                output.push_back(input.at(batch_index, c, h, w));
            }
        }
    }

    return output;
}


// ============================================================
// LinearConfig
// ============================================================

LinearConfig::LinearConfig(
    int in_f,
    int out_f,
    bool bias_flag
)
    : in_features(in_f),
      out_features(out_f),
      use_bias(bias_flag) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("LinearConfig: in_features and out_features must be positive.");
    }
}


// ============================================================
// Linear / Fully Connected Layer
// ============================================================

Linear::Linear(const LinearConfig& config)
    : cfg(config),
      weight(cfg.out_features * cfg.in_features, 0.0f) {
    if (cfg.use_bias) {
        bias.resize(cfg.out_features, 0.0f);
    }
}

void Linear::set_weight(const std::vector<float>& w) {
    const int expected_size = cfg.out_features * cfg.in_features;

    if (static_cast<int>(w.size()) != expected_size) {
        throw std::invalid_argument("Linear::set_weight: size mismatch.");
    }

    weight = w;
}

void Linear::set_bias(const std::vector<float>& b) {
    if (!cfg.use_bias) {
        throw std::logic_error("Linear::set_bias: this Linear layer is configured without bias.");
    }

    if (static_cast<int>(b.size()) != cfg.out_features) {
        throw std::invalid_argument("Linear::set_bias: size mismatch.");
    }

    bias = b;
}

std::vector<float> Linear::forward(const std::vector<float>& input) const {
    if (static_cast<int>(input.size()) != cfg.in_features) {
        throw std::invalid_argument("Linear::forward: input size mismatch.");
    }

    std::vector<float> output(cfg.out_features, 0.0f);

    for (int of = 0; of < cfg.out_features; ++of) {
        float sum = cfg.use_bias ? bias[of] : 0.0f;

        for (int inf = 0; inf < cfg.in_features; ++inf) {
            sum += input[inf] * weight[of * cfg.in_features + inf];
        }

        output[of] = sum;
    }

    return output;
}

// ============================================================
// Linear CUDA version
// ============================================================

__global__ void linear_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int in_features,
    int out_features,
    bool use_bias
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputSize = N * out_features;

    if (idx >= outputSize) {
        return;
    }

    const int of = idx % out_features;
    const int n = idx / out_features;

    float sum = use_bias ? bias[of] : 0.0f;

    for (int inf = 0; inf < in_features; ++inf) {
        sum += input[n * in_features + inf] * weight[of * in_features + inf];
    }

    output[n * out_features + of] = sum;
}

CudaMatrix Linear::forward_cuda(const CudaMatrix& input, int block_size, float* compute_time_ms, float* transfer_time_ms, float* malloc_time_ms) const {
    if (block_size <= 0) {
        throw std::invalid_argument("Linear::forward_cuda: block_size must be positive.");
    }
    if (input.F != cfg.in_features) {
        throw std::invalid_argument("Linear::forward_cuda: input feature size mismatch.");
    }

    CudaMatrix output(nullptr, input.N, cfg.out_features, true);
    float* d_weight = nullptr;
    float* d_bias = nullptr;

    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output.data), sizeof(float) * output.size()));
    CUDA_MALLOC_TIMED((void**)&output.data, sizeof(float) * output.size(), malloc_time_ms);

    // allocate & copy to device
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weight), sizeof(float) * weight.size()));
    CUDA_MALLOC_TIMED((void**)&d_weight, sizeof(float) * weight.size(), malloc_time_ms);
    CUDA_MEMCPY_TIMED(
        d_weight,
        weight.data(),
        sizeof(float) * weight.size(),
        cudaMemcpyHostToDevice,
        transfer_time_ms
    );

    if (cfg.use_bias) {
        // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bias), sizeof(float) * bias.size()));
        CUDA_MALLOC_TIMED((void**)&d_bias, sizeof(float) * bias.size(), malloc_time_ms);
        CUDA_MEMCPY_TIMED(
            d_bias,
            bias.data(),
            sizeof(float) * bias.size(),
            cudaMemcpyHostToDevice,
            transfer_time_ms
        );
    }

    dim3 blockSize(block_size);
    dim3 gridSize((output.size() + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
    }

    linear_kernel<<<gridSize, blockSize>>>(
        input.data,
        d_weight,
        d_bias,
        output.data,
        input.N,
        input.F,
        cfg.out_features,
        cfg.use_bias
    );

    CUDA_CHECK_KERNEL();

    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_ADD_ELAPSED_TIME(start, stop, compute_time_ms);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUDA_CHECK(cudaFree(d_weight));
    if (d_bias != nullptr) {
        CUDA_CHECK(cudaFree(d_bias));
    }

    return output;
}

// ============================================================
// Utility
// ============================================================

int argmax(const std::vector<float>& input) {
    if (input.empty()) {
        throw std::invalid_argument("argmax: input vector is empty.");
    }

    int best_index = 0;
    float best_value = input[0];

    for (int i = 1; i < static_cast<int>(input.size()); ++i) {
        if (input[i] > best_value) {
            best_value = input[i];
            best_index = i;
        }
    }

    return best_index;
}