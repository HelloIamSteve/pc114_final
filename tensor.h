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

Tensor4D load_image_as_tensor(
    const std::string& image_path,
    int target_h = 28,
    int target_w = 28,
    bool normalize = true
);

#endif