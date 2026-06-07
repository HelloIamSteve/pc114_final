#include "layers.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

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

void relu_inplace(Tensor4D& input) {
    for (int i = 0; i < static_cast<int>(input.data.size()); ++i) {
        if (input.data[i] < 0.0f) {
            input.data[i] = 0.0f;
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