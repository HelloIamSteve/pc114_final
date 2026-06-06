#ifndef CONV_H
#define CONV_H

#include <vector>
#include <stdexcept>

#include "tensor.h"

struct Conv2DConfig {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    bool use_bias;

    Conv2DConfig(
        int in_c,
        int out_c,
        int k_h,
        int k_w,
        int s_h = 1,
        int s_w = 1,
        int p_h = 0,
        int p_w = 0,
        bool bias = true
    );
};

class Conv2D {
public:
    Conv2DConfig cfg;
    Tensor4D weight; // [out_c][in_c][k_h][k_w]
    std::vector<float> bias;   // [out_c]

    Conv2D(const Conv2DConfig& config);

    void set_weight(const std::vector<float>& w);
    void set_bias(const std::vector<float>& b);

    Tensor4D forward(const Tensor4D& input) const;
    Tensor4D forward(const Tensor4DView& input) const;

    int getWeightSize() const {
        return weight.N * weight.C * weight.H * weight.W;
    }
    
    int getBiasSize() const {
        return cfg.use_bias ? cfg.out_channels : 0;
    }
};

#endif