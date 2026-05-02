#include "tensor.h"
#include <stdexcept>

Tensor4D::Tensor4D() : N(0), C(0), H(0), W(0) {}

Tensor4D::Tensor4D(int n, int c, int h, int w)
    : N(n), C(c), H(h), W(w), data(n * c * h * w, 0.0f) {
    if (n < 0 || c < 0 || h < 0 || w < 0) {
        throw std::invalid_argument("Tensor4D dimensions must be non-negative.");
    }
}

float& Tensor4D::at(int n, int c, int h, int w) {
    if (n < 0 || n >= N || c < 0 || c >= C || h < 0 || h >= H || w < 0 || w >= W) {
        throw std::out_of_range("Tensor4D::at index out of range.");
    }
    return data[((n * C + c) * H + h) * W + w];
}

const float& Tensor4D::at(int n, int c, int h, int w) const {
    if (n < 0 || n >= N || c < 0 || c >= C || h < 0 || h >= H || w < 0 || w >= W) {
        throw std::out_of_range("Tensor4D::at index out of range.");
    }
    return data[((n * C + c) * H + h) * W + w];
}