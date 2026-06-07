#include "conv.h"
#include "cuda_help.h"

#include <stdexcept>

// =========================
// Conv2DConfig
// =========================

Conv2DConfig::Conv2DConfig(
    int in_c,
    int out_c,
    int k_h,
    int k_w,
    int s_h,
    int s_w,
    int p_h,
    int p_w,
    bool bias_flag
)
    : in_channels(in_c),
      out_channels(out_c),
      kernel_h(k_h),
      kernel_w(k_w),
      stride_h(s_h),
      stride_w(s_w),
      pad_h(p_h),
      pad_w(p_w),
      use_bias(bias_flag) {
    if (in_channels <= 0 || out_channels <= 0) {
        throw std::invalid_argument("in_channels and out_channels must be positive.");
    }
    if (kernel_h <= 0 || kernel_w <= 0) {
        throw std::invalid_argument("kernel size must be positive.");
    }
    if (stride_h <= 0 || stride_w <= 0) {
        throw std::invalid_argument("stride must be positive.");
    }
    if (pad_h < 0 || pad_w < 0) {
        throw std::invalid_argument("padding must be non-negative.");
    }
}

// =========================
// Conv2D
// =========================

Conv2D::Conv2D(const Conv2DConfig& config)
    : cfg(config),
      weight(cfg.out_channels, cfg.in_channels, cfg.kernel_h, cfg.kernel_w) {
    if (cfg.use_bias) {
        bias.resize(cfg.out_channels, 0.0f);
    }
}

void Conv2D::set_weight(const std::vector<float>& w) {
    const int expected_size =
        cfg.out_channels * cfg.in_channels * cfg.kernel_h * cfg.kernel_w;

    if ((int)w.size() != expected_size) {
        throw std::invalid_argument("set_weight: size mismatch.");
    }

    weight.data = w;
}

void Conv2D::set_bias(const std::vector<float>& b) {
    if (!cfg.use_bias) {
        throw std::logic_error("set_bias: this Conv2D is configured without bias.");
    }

    if ((int)b.size() != cfg.out_channels) {
        throw std::invalid_argument("set_bias: size mismatch.");
    }

    bias = b;
}

template <typename InputTensor>
Tensor4D conv2d_forward_impl(const Conv2D& layer, const InputTensor& input) {
    const Conv2DConfig& cfg = layer.cfg;
    const Tensor4D& weight = layer.weight;
    const std::vector<float>& bias = layer.bias;

    if (input.C != cfg.in_channels) {
        throw std::invalid_argument("Conv2D::forward: input channel mismatch.");
    }

    const int out_h = (input.H + 2 * cfg.pad_h - cfg.kernel_h) / cfg.stride_h + 1;;
    const int out_w = (input.W + 2 * cfg.pad_w - cfg.kernel_w) / cfg.stride_w + 1;

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Conv2D::forward: invalid output shape.");
    }

    Tensor4D output(input.N, cfg.out_channels, out_h, out_w);

    for (int n = 0; n < input.N; ++n) {
        for (int oc = 0; oc < cfg.out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = cfg.use_bias ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < cfg.in_channels; ++ic) {
                        for (int kh = 0; kh < cfg.kernel_h; ++kh) {
                            for (int kw = 0; kw < cfg.kernel_w; ++kw) {
                                const int ih = oh * cfg.stride_h - cfg.pad_h + kh;
                                const int iw = ow * cfg.stride_w - cfg.pad_w + kw;

                                if (ih >= 0 && ih < input.H &&
                                    iw >= 0 && iw < input.W) {
                                    sum += input.at(n, ic, ih, iw)
                                         * weight.at(oc, ic, kh, kw);
                                }
                            }
                        }
                    }

                    output.at(n, oc, oh, ow) = sum;
                }
            }
        }
    }

    return output;
}

Tensor4D Conv2D::forward(const Tensor4D& input) const {
    return conv2d_forward_impl(*this, input);
}

Tensor4D Conv2D::forward(const Tensor4DView& input) const {
    return conv2d_forward_impl(*this, input);
}

// for CUDA
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int C_in,
    int H,
    int W,
    int C_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int out_h,
    int out_w,
    bool use_bias
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputSize = N * C_out * out_h * out_w;

    if (idx >= outputSize) {
        return;
    }

    // output[idx]
    // = output[n][oc][oh][ow]
    // = output[ow + oh * out_w + oc * out_h * out_w + n * C_out * out_h * out_w]
    // = output[((n * C_out + oc) * out_h + oh) * out_w + ow]

    const int ow = idx % out_w;
    const int oh = (idx / out_w) % out_h;
    const int oc = (idx / (out_h * out_w)) % C_out;
    const int n = idx / (C_out * out_h * out_w);

    float sum = use_bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int ih = oh * stride_h - pad_h + kh;
                const int iw = ow * stride_w - pad_w + kw;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {   // index valid
                    const int input_idx = ((n * C_in + ic) * H + ih) * W + iw;
                    const int weight_idx = ((oc * C_in + ic) * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    output[((n * C_out + oc) * out_h + oh) * out_w + ow] = sum;
}

CudaTensor4D Conv2D::forward_cuda(const CudaTensor4D& input,
    int block_size,
    float* compute_time_ms,
    float* transfer_time_ms,
    float* malloc_time_ms
) const {
    if (block_size <= 0) {
        throw std::invalid_argument("Conv2D::forward_cuda: block_size must be positive.");
    }
    if (input.C != cfg.in_channels) {
        throw std::invalid_argument("Conv2D::forward_cuda: input channel mismatch.");
    }

    const int out_h = (input.H + 2 * cfg.pad_h - cfg.kernel_h) / cfg.stride_h + 1;
    const int out_w = (input.W + 2 * cfg.pad_w - cfg.kernel_w) / cfg.stride_w + 1;

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Conv2D::forward_cuda: invalid output shape.");
    }

    CudaTensor4D output(nullptr, input.N, cfg.out_channels, out_h, out_w, true);
    float* d_weight = nullptr;
    float* d_bias = nullptr;

    // allocate on device
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output.data),sizeof(float) * output.size()));
    CUDA_MALLOC_TIMED((void**)&output.data, sizeof(float) * output.size(), malloc_time_ms);

    // allocate & copy to device
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weight), sizeof(float) * weight.data.size()));
    CUDA_MALLOC_TIMED((void**)&d_weight, sizeof(float) * weight.data.size(), malloc_time_ms);
    CUDA_MEMCPY_TIMED(
        d_weight,
        weight.data.data(),
        sizeof(float) * weight.data.size(),
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

    // const int grid_size = (output.size() + block_size - 1) / block_size;
    dim3 blockSize(block_size);
    dim3 gridSize((output.size() + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    if (compute_time_ms != nullptr) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
    }

    conv2d_kernel<<<gridSize, blockSize>>>(
        input.data,
        d_weight,
        d_bias,
        output.data,
        input.N,
        input.C,
        input.H,
        input.W,
        cfg.out_channels,
        cfg.kernel_h,
        cfg.kernel_w,
        cfg.stride_h,
        cfg.stride_w,
        cfg.pad_h,
        cfg.pad_w,
        out_h,
        out_w,
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