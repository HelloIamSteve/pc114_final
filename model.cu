#include "model.h"
#include "tensor.h"
#include "cuda_help.h"

#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <pthread.h>
#include <omp.h>

/* CPU */
/* single image */
std::vector<float> LeNet::forward(const Tensor4D& input) const{
    Tensor4D y = conv1.forward(input);
    relu_inplace(y);
    y = pool1.forward(y);

    y = conv2.forward(y);
    relu_inplace(y);
    y = pool2.forward(y);

    std::vector<float> flattened = flatten(y);

    std::vector<float> out = fc1.forward(flattened);
    out = fc2.forward(out);
    out = fc3.forward(out);

    return out;
}

/* batch inference */
std::vector<std::vector<float>> LeNet::forward_batch(const Tensor4D& input) const {
    Tensor4D y = conv1.forward(input);
    relu_inplace(y);
    y = pool1.forward(y);

    y = conv2.forward(y);
    relu_inplace(y);
    y = pool2.forward(y);

    std::vector<std::vector<float>> batch_logits;
    batch_logits.reserve(y.N);

    for (int n = 0; n < y.N; ++n) {
        std::vector<float> flattened = flatten(y, n);

        std::vector<float> out = fc1.forward(flattened);
        out = fc2.forward(out);
        out = fc3.forward(out);

        batch_logits.push_back(out);
    }

    return batch_logits;
}

void LeNet::forward_batch(const Tensor4DView& input, std::vector<std::vector<float>>& output, int output_offset, bool openmp) const {
    Tensor4D y = conv1.forward(input);
    relu_inplace(y, openmp);
    y = pool1.forward(y);
    
    y = conv2.forward(y);
    relu_inplace(y, openmp);
    y = pool2.forward(y);
    
    for (int n = 0; n < y.N; ++n) {
        std::vector<float> flattened = flatten(y, n);
        
        std::vector<float> out = fc1.forward(flattened);
        out = fc2.forward(out);
        out = fc3.forward(out);
        
        output[output_offset + n] = out;
    }
}

/* pthread version */
void* threadRunner(void* arg) {
    pthreadArg* args = static_cast<pthreadArg*>(arg);

    const LeNet* model = args->model;
    const Tensor4D& input = *(args->input);
    std::vector<std::vector<float>>& output = *(args->output);

    int start = args->batch_start;
    int end = args->batch_end;

    if (start >= end) {
        return nullptr;
    }

    Tensor4DView input_view(input, start, end);
   
    model->forward_batch(input_view, output, start); // output[start:end] = forward_batch(input[start:end])

    return nullptr;
}

std::vector<std::vector<float>> LeNet::forward_batch_pthread(const Tensor4D& input, int thread_num) const{
    if (thread_num <= 0) {
        throw std::invalid_argument("thread_num must be positive.");
    }

    if (thread_num > input.N) {
        thread_num = input.N;
    }
    
    int n = input.N;
    int chunkSize = (n + thread_num - 1) / thread_num; // ceil(n / thread_num)

    std::vector<pthread_t> tid(thread_num);
    std::vector<pthreadArg> args(thread_num);
    std::vector<pthread_attr_t> attr(thread_num);

    std::vector<std::vector<float>> output(n); // pre-allocate output for all images

    for(int i=0; i<thread_num; i++){
        int start = i * chunkSize;
        int end = std::min(start + chunkSize, n);

        args[i].batch_start = start;
        args[i].batch_end = end;
        args[i].model = this;
        args[i].input = &input;
        args[i].output = &output;
        args[i].thread_id = i;

        pthread_attr_init(&attr[i]);
        pthread_create(&tid[i], &attr[i], threadRunner, (void*)&args[i]);
    }

    for(int i=0; i<thread_num; i++){
        pthread_join(tid[i], NULL);
    }

    return output;
}

/* openmp version */
std::vector<std::vector<float>> LeNet::forward_batch_openmp(const Tensor4D& input, int thread_num) const {
    int n = input.N;
    std::vector<std::vector<float>> output(n);

    // allocate task to each cores
    int mini_batch_size = 32; 

    #pragma omp parallel for num_threads(thread_num) schedule(dynamic)
    for (int start = 0; start < n; start += mini_batch_size) {
        int end = std::min(n, start + mini_batch_size);
        Tensor4DView input_view(input, start, end);
        
        forward_batch(input_view, output, start, true);
    }

    return output;
}

/* CUDA version */
std::vector<std::vector<float>> LeNet::forward_batch_cuda(
    const Tensor4D& input,
    int block_size,
    float* cuda_compute_time_ms = nullptr,
    float* cuda_transfer_time_ms = nullptr,
    float* cuda_malloc_time_ms = nullptr
)
const{
    // check input
    if (block_size <= 0) {
        throw std::invalid_argument("block_size must be positive.");
    }

    if (input.N <= 0 || input.C <= 0 || input.H <= 0 || input.W <= 0) {
        throw std::invalid_argument("forward_batch_cuda: input tensor has invalid shape.");
    }

    // initialize timer
    if (cuda_compute_time_ms != nullptr) {
        *cuda_compute_time_ms = 0.0f;
    }

    if (cuda_transfer_time_ms != nullptr) {
        *cuda_transfer_time_ms = 0.0f;
    }

    if(cuda_malloc_time_ms != nullptr){
        *cuda_malloc_time_ms = 0.0f;
    }

    CudaTensor4D d_input = Tensor4D_to_device(input, cuda_transfer_time_ms, cuda_malloc_time_ms);

    CudaTensor4D d_y = conv1.forward_cuda(d_input, block_size, cuda_compute_time_ms, cuda_transfer_time_ms, cuda_malloc_time_ms);
    relu_cuda(d_y, block_size, cuda_compute_time_ms);
    d_y = pool1.forward_cuda(d_y, block_size, cuda_compute_time_ms, cuda_malloc_time_ms);

    d_y = conv2.forward_cuda(d_y, block_size, cuda_compute_time_ms, cuda_transfer_time_ms, cuda_malloc_time_ms);
    relu_cuda(d_y, block_size, cuda_compute_time_ms);
    d_y = pool2.forward_cuda(d_y, block_size, cuda_compute_time_ms, cuda_malloc_time_ms);

    // flatten
    CudaMatrix d_flattened(d_y.data, d_y.N, d_y.C * d_y.H * d_y.W, false);
    
    // fully-connection layers
    d_flattened = fc1.forward_cuda(d_flattened, block_size, cuda_compute_time_ms, cuda_transfer_time_ms, cuda_malloc_time_ms);
    d_flattened = fc2.forward_cuda(d_flattened, block_size, cuda_compute_time_ms, cuda_transfer_time_ms, cuda_malloc_time_ms);
    d_flattened = fc3.forward_cuda(d_flattened, block_size, cuda_compute_time_ms, cuda_transfer_time_ms, cuda_malloc_time_ms);

    CUDA_CHECK(cudaDeviceSynchronize());    // wait for all threads finish

    std::vector<float> host_logits = cuda_matrix_to_host(d_flattened, cuda_transfer_time_ms);

    std::vector<std::vector<float>> output(input.N, std::vector<float>(fc3.cfg.out_features, 0.0f));

    for(int i=0; i<input.N; i++){
        for(int j=0; j<fc3.cfg.out_features; j++){
            output[i][j] = host_logits[i * fc3.cfg.out_features + j];
        }
    }

    return output;
}