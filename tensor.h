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

// for cuda
struct CudaTensor4D {
    float* data;
    int N, C, H, W;
    bool owns_data;

    CudaTensor4D() : data(nullptr), N(0), C(0), H(0), W(0), owns_data(true){}
    CudaTensor4D(float* ptr, int n, int c, int h, int w, bool owns)
        : data(ptr), N(n), C(c), H(h), W(w), owns_data(owns) {}

    ~CudaTensor4D();

    CudaTensor4D(const CudaTensor4D&) = delete;
    CudaTensor4D& operator=(const CudaTensor4D&) = delete;

    CudaTensor4D(CudaTensor4D&& other) noexcept;
    CudaTensor4D& operator=(CudaTensor4D&& other) noexcept;

    int size() const {
        return N * C * H * W;
    }

    void release() noexcept;
};

struct CudaMatrix {
    float* data;
    int N, F;
    bool owns_data;
    
    CudaMatrix() : data(nullptr), N(0), F(0), owns_data(true){}
    CudaMatrix(float* ptr, int n, int f, bool owns)
    : data(ptr), N(n), F(f), owns_data(owns) {}
    
    ~CudaMatrix();
    
    CudaMatrix(const CudaMatrix&) = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;
    
    CudaMatrix(CudaMatrix&& other) noexcept;
    CudaMatrix& operator=(CudaMatrix&& other) noexcept;
    
    int size() const {
        return N * F;
    }

    void release() noexcept;
};

CudaTensor4D tensor4d_to_device(const Tensor4D& input);
std::vector<float> cuda_matrix_to_host(const CudaMatrix& matrix);

#endif