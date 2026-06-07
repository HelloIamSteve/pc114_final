#ifndef CUDA_HELP_H
#define CUDA_HELP_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr,                                           \
                         "CUDA error at %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(err));     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                  \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                 \
    do {                                                                    \
        CUDA_CHECK(cudaGetLastError());                                      \
    } while (0)

#endif