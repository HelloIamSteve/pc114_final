#ifndef CUDA_HELP_H
#define CUDA_HELP_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

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

inline void CUDA_MEMCPY_TIMED(
    void* dst,
    const void* src,
    size_t count,
    cudaMemcpyKind kind,
    float* elapsed_ms
) {
    if (elapsed_ms == nullptr) {
        CUDA_CHECK(cudaMemcpy(dst, src, count, kind));
        return;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(dst, src, count, kind));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    *elapsed_ms += ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

inline void CUDA_MALLOC_TIMED(
    void** dev_ptr,
    size_t count,
    float* elapsed_ms
) {
    if (elapsed_ms == nullptr) {
        CUDA_CHECK(cudaMalloc(dev_ptr, count));
        return;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(dev_ptr, count));
    const auto stop = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<float, std::milli> elapsed = stop - start;
    *elapsed_ms += elapsed.count();
}

inline void CUDA_ADD_ELAPSED_TIME(
    cudaEvent_t start,
    cudaEvent_t stop,
    float* elapsed_ms
) {
    if (elapsed_ms == nullptr) {
        return;
    }

    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    *elapsed_ms += ms;
}

#endif