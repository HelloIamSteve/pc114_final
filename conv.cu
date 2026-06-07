#include "conv.h"
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