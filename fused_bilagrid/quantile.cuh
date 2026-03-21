#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>

template<bool invert_quantile>
cudaError_t batch_quantile_masked_radix_select(
    const float* d_x,
    int B,
    int N,
    float q,
    float* d_out,
    uint32_t* temp,
    cudaStream_t stream
);
