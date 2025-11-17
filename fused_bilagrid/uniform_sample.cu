#include "uniform_sample_forward.cu"
#include "uniform_sample_backward_v1.cu"
#include "uniform_sample_backward_v2.cu"

#define PATCHED
#include "uniform_sample_forward.cu"
#include "uniform_sample_backward_v2.cu"


void bilagrid_uniform_sample_forward(
    const float* bilagrid,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    int total = N * m * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bilagrid_uniform_sample_forward_kernel<<<blocks, threads>>>(
        bilagrid, rgb, output,
        N, L, H, W, m, h, w
    );
    // cudaDeviceSynchronize();
}


void bilagrid_patched_sample_forward(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0
) {
    int total = N * m * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bilagrid_patched_sample_forward_kernel<<<blocks, threads>>>(
        bilagrid, rgb, output,
        N, L, H, W, m, h, w, h0, w0, offsets
    );
    // cudaDeviceSynchronize();
}



void bilagrid_uniform_sample_backward_v2(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w
) {
    dim3 block = { 16, 16, 1 };
    dim3 bounds = {
        (w +block.x-1)/block.x,
        (h +block.y-1)/block.y,
        (N*m +block.z-1)/block.z
    };
    bilagrid_uniform_sample_backward_v2_kernel<<<bounds, block>>>(
        bilagrid, rgb, v_output,
        v_bilagrid, v_rgb,
        N, L, H, W, m, h, w
    );
}


void bilagrid_patched_sample_backward(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0
) {
    dim3 block = { 16, 16, 1 };
    dim3 bounds = {
        (w +block.x-1)/block.x,
        (h +block.y-1)/block.y,
        (N*m +block.z-1)/block.z
    };
    bilagrid_patched_sample_backward_kernel<<<bounds, block>>>(
        bilagrid, rgb, v_output,
        v_bilagrid, v_rgb,
        N, L, H, W, m, h, w, h0, w0, offsets
    );
}
