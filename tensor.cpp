#include "tensor.h"
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <fstream>

namespace {

std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) {
        return file;
    }

    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') {
        return dir + file;
    }

    return dir + "/" + file;
}

}  // namespace

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

    // resize to 28 * 28
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_AREA);

    // Convert uint8 [0, 255] to float [0, 1].
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32F, 1.0 / 255.0);

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

TestingSet load_testing_set_as_tensor(
    const std::string& testing_set_dir,
    int max_images,
    int target_h,
    int target_w,
    bool normalize
) {
    if (max_images == 0 || max_images < -1) {
        throw std::invalid_argument("load_testing_set_as_tensor: max_images must be -1 or positive.");
    }
    if (target_h <= 0 || target_w <= 0) {
        throw std::invalid_argument("load_testing_set_as_tensor: target size must be positive.");
    }

    const std::string labels_path = join_path(testing_set_dir, "labels.txt");
    const std::string images_dir = join_path(testing_set_dir, "images");

    std::ifstream fin(labels_path);
    if (!fin.is_open()) {
        throw std::runtime_error("load_testing_set_as_tensor: cannot open labels file: " + labels_path);
    }

    std::vector<std::string> filenames;
    std::vector<int> labels;

    if (max_images > 0) {
        filenames.reserve(max_images);
        labels.reserve(max_images);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        std::string filename;
        int label = -1;

        if (!(iss >> filename >> label)) {
            throw std::runtime_error("load_testing_set_as_tensor: invalid label line: " + line);
        }

        filenames.push_back(filename);
        labels.push_back(label);

        if (max_images > 0 && static_cast<int>(labels.size()) >= max_images) {
            break;
        }
    }

    if (fin.bad()) {
        throw std::runtime_error("load_testing_set_as_tensor: error while reading labels file: " + labels_path);
    }

    const int num_images = static_cast<int>(labels.size());
    if (num_images == 0) {
        throw std::runtime_error("load_testing_set_as_tensor: no images found in: " + labels_path);
    }

    TestingSet dataset;
    dataset.images = Tensor4D(num_images, 1, target_h, target_w);
    dataset.labels = labels;
    dataset.filenames = filenames;

    for (int n = 0; n < num_images; ++n) {
        const std::string image_path = join_path(images_dir, filenames[n]);
        Tensor4D image_tensor = load_image_as_tensor(image_path, target_h, target_w, normalize);

        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                dataset.images.at(n, 0, h, w) = image_tensor.at(0, 0, h, w);
            }
        }
    }

    return dataset;
}