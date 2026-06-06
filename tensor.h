#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <string>

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

class Tensor4DView {
public:
    const float* data;
    int N, C, H, W;

    Tensor4DView(const Tensor4D& tensor, int start_n, int end_n)
        : data(tensor.data.data() + start_n * tensor.C * tensor.H * tensor.W),
          N(end_n - start_n),
          C(tensor.C),
          H(tensor.H),
          W(tensor.W) {
            if (start_n < 0 || end_n > tensor.N || start_n > end_n) {
                throw std::out_of_range("Tensor4DView: invalid batch range.");
            }
          }

    float at(int n, int c, int h, int w) const {
        return data[((n * C + c) * H + h) * W + w];
    }
};

struct TestingSet {
    Tensor4D images;
    std::vector<int> labels;
    std::vector<std::string> filenames;
};

TestingSet load_testing_set_as_tensor(
    const std::string& testing_set_dir,
    int max_images = -1,
    int target_h = 28,
    int target_w = 28,
    bool normalize = true
);

#endif