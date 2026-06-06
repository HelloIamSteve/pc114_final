#include <iostream>
#include <stdlib.h>
#include <vector>

#include "tensor.h"
#include "conv.h"
#include "layers.h"
#include "weight_loader.h"
#include "model.h"

#define WEIGHTS_DIR "weights/"

int main(int argc, char* argv[]){
    if (argc != 1) {
        // std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        std::cerr << "Usage: " << argv[0] << std::endl;
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
    TestingSet test_set = load_testing_set_as_tensor("testing_set");
    std::cout << test_set.images.N << " images loaded from testing set." << std::endl;

    /* inference */
    struct timespec t_start, t_end;

    // start time
    clock_gettime(CLOCK_REALTIME, &t_start);

    std::vector<std::vector<float>> logits = lenet.forward_batch(test_set.images);

    int correct = 0;
    for (int i = 0; i < test_set.images.N; ++i) {
        int pred = argmax(logits[i]);

        if (pred == test_set.labels[i]) {
            correct++;
        }
    }

    double acc = static_cast<double>(correct) / test_set.images.N * 100.0;

    // end time
    clock_gettime(CLOCK_REALTIME, &t_end);
    double 	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;

    // std::cout << "Predicted label: " << predicted_label << std::endl;
    std::cout << "Inference time: " << elapsedTime << " ms" << std::endl;
    std::cout << "Accuracy: " << acc << "%" << std::endl;

    return 0;
}