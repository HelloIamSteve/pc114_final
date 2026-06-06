#include "weight_loader.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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
}

std::vector<float> load_vector_from_txt(const std::string& path, int expected_size) {
    std::ifstream fin(path);

    if (!fin.is_open()) {
        throw std::runtime_error("load_vector_from_txt: cannot open file: " + path);
    }

    std::vector<float> values;
    if (expected_size > 0) {
        values.reserve(expected_size);
    }

    float value = 0.0f;
    while (fin >> value) {
        values.push_back(value);
    }

    if (fin.bad()) {
        throw std::runtime_error("load_vector_from_txt: error while reading file: " + path);
    }

    if (expected_size > 0 && static_cast<int>(values.size()) != expected_size) {
        throw std::runtime_error(
            "load_vector_from_txt: size mismatch in " + path +
            ", expected " + std::to_string(expected_size) +
            ", got " + std::to_string(values.size())
        );
    }

    return values;
}

void load_lenet_weights(
    Conv2D& conv1,
    Conv2D& conv2,
    Linear& fc1,
    Linear& fc2,
    Linear& fc3,
    const std::string& weights_dir
) {
    conv1.set_weight(load_vector_from_txt(join_path(weights_dir, "conv1_weight.txt"), conv1.getWeightSize()));
    conv1.set_bias(load_vector_from_txt(join_path(weights_dir, "conv1_bias.txt"), conv1.getBiasSize()));

    conv2.set_weight(load_vector_from_txt(join_path(weights_dir, "conv2_weight.txt"), conv2.getWeightSize()));
    conv2.set_bias(load_vector_from_txt(join_path(weights_dir, "conv2_bias.txt"), conv2.getBiasSize()));

    fc1.set_weight(load_vector_from_txt(join_path(weights_dir, "fc1_weight.txt"), fc1.getWeightSize()));
    fc1.set_bias(load_vector_from_txt(join_path(weights_dir, "fc1_bias.txt"), fc1.getBiasSize()));

    fc2.set_weight(load_vector_from_txt(join_path(weights_dir, "fc2_weight.txt"), fc2.getWeightSize()));
    fc2.set_bias(load_vector_from_txt(join_path(weights_dir, "fc2_bias.txt"), fc2.getBiasSize()));

    fc3.set_weight(load_vector_from_txt(join_path(weights_dir, "fc3_weight.txt"), fc3.getWeightSize()));
    fc3.set_bias(load_vector_from_txt(join_path(weights_dir, "fc3_bias.txt"), fc3.getBiasSize()));
}
