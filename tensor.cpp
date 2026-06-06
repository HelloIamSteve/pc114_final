#include "tensor.h"
#include <stdexcept>
#include <opencv2/opencv.hpp>

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

Tensor4D load_image_as_tensor(
    const std::string& image_path,
    int target_h,
    int target_w,
    bool normalize
) {
    if (target_h <= 0 || target_w <= 0) {
        throw std::invalid_argument("load_image_as_tensor: target size must be positive.");
    }

    // Read as grayscale because LeNet for MNIST expects 1 input channel.
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("load_image_as_tensor: cannot open image: " + image_path);
    }

    // Convert uint8 [0, 255] to float [0, 1].
    cv::Mat float_image;
    image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    Tensor4D tensor(1, 1, target_h, target_w);

    for (int h = 0; h < target_h; ++h) {
        for (int w = 0; w < target_w; ++w) {
            float value = float_image.at<float>(h, w);

            // Match PyTorch:
            // transforms.Normalize((0.5), (0.5))
            if (normalize) {
                value = (value - 0.5f) / 0.5f;
            }

            tensor.at(0, 0, h, w) = value;
        }
    }

    return tensor;
}