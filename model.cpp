#include "model.h"

#include <string>
#include <vector>

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

void LeNet::forward_batch(const Tensor4DView& input, std::vector<std::vector<float>>& output, int output_offset) const {
    Tensor4D y = conv1.forward(input);
    relu_inplace(y);
    y = pool1.forward(y);

    y = conv2.forward(y);
    relu_inplace(y);
    y = pool2.forward(y);

    for (int n = 0; n < y.N; ++n) {
        std::vector<float> flattened = flatten(y, n);

        std::vector<float> out = fc1.forward(flattened);
        out = fc2.forward(out);
        out = fc3.forward(out);

        output[output_offset + n] = out;
    }
}

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

    model->forward_batch(input_view, output, start);

    return nullptr;
}

std::vector<std::vector<float>> LeNet::forward_batch_pthread(const Tensor4D& input, int thread_num) const{
    int n = input.N;
    int chunkSize = (n + thread_num - 1) / thread_num; // ceil(n / thread_num)

    std::vector<pthread_t> tid(thread_num);
    std::vector<pthreadArg*> args(thread_num);
    std::vector<pthread_attr_t> attr(thread_num);

    std::vector<std::vector<float>> output(n); // pre-allocate output for all images
    output.reserve(n);

    for(int i=0; i<thread_num; i++){
        int start = i * chunkSize;
        int end = (i == thread_num - 1) ? n : (i + 1) * chunkSize;

        args[i] = (pthreadArg*)malloc(sizeof(pthreadArg));

        args[i]->batch_start = start;
        args[i]->batch_end = end;
        args[i]->model = this;
        args[i]->input = &input;
        args[i]->output = &output;
        args[i]->thread_id = i;

        pthread_attr_init(&attr[i]);
        pthread_create(&tid[i], &attr[i], threadRunner, (void*)args[i]);
    }

    for(int i=0; i<thread_num; i++){
        pthread_join(tid[i], NULL);
    }
    
    for(int i=0; i<thread_num; i++){
        free(args[i]);
    }

    return output;
}
