#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>

struct Tensor4D {
    int N, C, H, W;
    std::vector<float> data;

    Tensor4D();
    Tensor4D(int n, int c, int h, int w);

    float& at(int n, int c, int h, int w);
    const float& at(int n, int c, int h, int w) const;
};

#endif