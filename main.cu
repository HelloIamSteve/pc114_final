#include <iostream>
#include <stdlib.h>
#include <vector>
#include <ctime>

#include "tensor.h"
#include "conv.h"
#include "layers.h"
#include "weight_loader.h"
#include "model.h"

#define THREAD_NUM 6

#define WEIGHTS_DIR "weights/"
#define TESTING_SET_DIR "testing_set/"

float getAccuracy(const std::vector<std::vector<float>>& logits, const std::vector<int>& labels) {
    int correct = 0;
    int n = logits.size();

    for (int i = 0; i < n; ++i) {
        int pred = argmax(logits[i]);

        if (pred == labels[i]) {
            correct++;
        }
    }

    return static_cast<float>(correct) / n * 100.0f;
}

int main(int argc, char* argv[]){
    if (argc != 1) {
        // std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        std::cerr << "Usage: " << argv[0] << '\n';
        return 1;
    }

    /* claim model */
    LeNet lenet;

    /* load weights */
    lenet.load_weights(WEIGHTS_DIR);

    // /* load input image as tensor */
    // Tensor4D img = load_image_as_tensor("test_images/0.png");
    // Tensor4D img = load_image_as_tensor(argv[1]);

    /* load testing set */
    TestingSet test_set = load_testing_set_as_tensor(TESTING_SET_DIR);
    std::cout << test_set.images.N << " images loaded from testing set." << '\n';

    /* inference */
    struct timespec t_start, t_end;

    // start time
    std::cout << "Sequential version:" << '\n';

    clock_gettime(CLOCK_REALTIME, &t_start);

    std::vector<std::vector<float>> logits = lenet.forward_batch(test_set.images);
    
    // end time
    clock_gettime(CLOCK_REALTIME, &t_end);
    double elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    
    std::cout << "Inference time: " << elapsedTime << " ms" << '\n';

    float acc = getAccuracy(logits, test_set.labels);
    std::cout << "Accuracy: " << acc << "%" << '\n';
    std::cout << "---------------" << '\n';

    /* pthread version */
    std::cout << "Pthread version:" << '\n';

    // start time
    clock_gettime(CLOCK_REALTIME, &t_start);
    std::vector<std::vector<float>> logits_pthread = lenet.forward_batch_pthread(test_set.images, THREAD_NUM);
    
    // end time
    clock_gettime(CLOCK_REALTIME, &t_end);
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    
    std::cout << "Inference time: " << elapsedTime << " ms" << '\n';

    acc = getAccuracy(logits_pthread, test_set.labels);
    std::cout << "Accuracy: " << acc << "%" << '\n';
    std::cout << "---------------" << '\n';

    /* OpenMP version */
    std::cout << "OpenMP version:" << '\n';

    // start time
    clock_gettime(CLOCK_REALTIME, &t_start);
    std::vector<std::vector<float>> logits_openmp = lenet.forward_batch_openmp(test_set.images, THREAD_NUM);
    
    // end time
    clock_gettime(CLOCK_REALTIME, &t_end);
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;

    std::cout << "Inference time: " << elapsedTime << " ms" << '\n';

    acc = getAccuracy(logits_openmp, test_set.labels);
    std::cout << "Accuracy: " << acc << "%" << '\n';
    std::cout << "---------------" << '\n';

    /* CUDA version */
    std::cout << "CUDA version:" << '\n';

    // start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start time
    cudaEventRecord(start);
    std::vector<std::vector<float>> logits_cuda = lenet.forward_batch_cuda(test_set.images, THREAD_NUM);

    // end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime_cuda;
    cudaEventElapsedTime(&elapsedTime_cuda, start, stop);
    std::cout << "Inference time: " << elapsedTime_cuda << " ms" << '\n';
    acc = getAccuracy(logits_cuda, test_set.labels);
    std::cout << "Accuracy: " << acc << "%" << '\n';

    return 0;
}