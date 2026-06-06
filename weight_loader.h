#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include <string>
#include <vector>

#include "conv.h"
#include "layers.h"

// Load a text file containing one float per line.
// If expected_size > 0, this function checks whether the number of values is correct.
std::vector<float> load_vector_from_txt(const std::string& path, int expected_size = -1);

// Load all LeNet weights and assign them to the corresponding layers.
// weights_dir should usually be "weights".
void load_lenet_weights(
    Conv2D& conv1,
    Conv2D& conv2,
    Linear& fc1,
    Linear& fc2,
    Linear& fc3,
    const std::string& weights_dir = "weights"
);

#endif
