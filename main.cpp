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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    /* claim model */
    LeNet lenet;

    /* load weights */
    lenet.load_weights(WEIGHTS_DIR);

    /* load input image as tensor */
    // Tensor4D img = load_image_as_tensor("test_images/0.png");
    Tensor4D img = load_image_as_tensor(argv[1]);

    /* inference */
    struct timespec t_start, t_end;

    // start time
    clock_gettime(CLOCK_REALTIME, &t_start);

    std::vector<float> outputLogits = lenet.forward(img);
    int predicted_label = argmax(outputLogits);

    // end time
    clock_gettime(CLOCK_REALTIME, &t_end);
    double 	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;

    std::cout << "Predicted label: " << predicted_label << std::endl;
    std::cout << "Inference time: " << elapsedTime << " ms" << std::endl;

    return 0;
}