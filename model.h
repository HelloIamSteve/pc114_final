#ifndef MODEL_H
#define MODEL_H

#include "conv.h"
#include "layers.h"
#include "tensor.h"
#include "weight_loader.h"

#include <iostream>
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

    std::vector<float> forward(const Tensor4D& input) const {
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
};


#endif