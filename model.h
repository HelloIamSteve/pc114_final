#ifndef MODEL_H
#define MODEL_H

#include "conv.h"
#include "layers.h"
#include "tensor.h"
#include "weight_loader.h"

#include <string>
#include <vector>

class LeNet {
public:
    Conv2D conv1;
    AvgPool2D pool1;
    Conv2D conv2;
    AvgPool2D pool2;
    Linear fc1;
    Linear fc2;
    Linear fc3;

    LeNet()
        : conv1(Conv2DConfig(1, 6, 5, 5)),
          pool1(AvgPool2DConfig(2, 2, 2, 2)),
          conv2(Conv2DConfig(6, 16, 5, 5)),
          pool2(AvgPool2DConfig(2, 2, 2, 2)),
          fc1(LinearConfig(16 * 4 * 4, 120)),
          fc2(LinearConfig(120, 84)),
          fc3(LinearConfig(84, 10)) {}

    void load_weights(const std::string& weights_dir) {
        load_lenet_weights(conv1, conv2, fc1, fc2, fc3, weights_dir);
    }

    // sequential version
    std::vector<float> forward(const Tensor4D&) const;
    std::vector<std::vector<float>> forward_batch(const Tensor4D&) const;

    // pthread version
    std::vector<std::vector<float>> forward_batch_pthread(const Tensor4D&, int) const;
    void forward_batch(const Tensor4DView&, std::vector<std::vector<float>>&, int) const;
};

typedef struct _pthreadArg{
    const LeNet* model;
    const Tensor4D* input;
    std::vector<std::vector<float>>* output;
    int batch_start;
    int batch_end;
    int thread_id;
}pthreadArg;

void* threadRunner(void*);

#endif