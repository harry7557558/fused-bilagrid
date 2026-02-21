#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "config.h"

namespace cg = cooperative_groups;


template<int C>
__global__ void tv_loss_forward_kernel(
    const float* __restrict__ bilagrid,  // [N,C,L,H,W]
    float* __restrict__ tv_loss,
    int N, int L, int H, int W
) {
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    // bool inside = (wi < W && hi < H && idx < (L*C*N));
    bool inside = (wi < W && hi < H && idx < (L*N));
    int li = idx % L; idx /= L;
    // int ci = idx % C; idx /= C;
    int ni = idx;

    float tv_sum = 0.0f;

    if (inside) {
        #pragma unroll
        for (int ci = 0; ci < C; ci++) {
            
        int base = (ni*C+ci)*L*H*W;
        int cell_idx = base + (li*H+hi)*W+wi;

        float val = bilagrid[cell_idx];

        if (wi > 0) {
            float val0 = bilagrid[cell_idx - 1];
            float l2 = (val-val0) * (val-val0);
            tv_sum += l2 / (L*H*(W-1));
        }
        if (hi > 0) {
            float val0 = bilagrid[cell_idx - W];
            float l2 = (val-val0) * (val-val0);
            tv_sum += l2 / (L*(H-1)*W);
        }
        if (li > 0) {
            float val0 = bilagrid[cell_idx - W*H];
            float l2 = (val-val0) * (val-val0);
            tv_sum += l2 / ((L-1)*H*W);
        }

        }  // ci
        tv_sum /= (C*N);
    }

#if 0
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    tv_sum = cg::reduce(warp, tv_sum, cg::plus<float>());
    if (warp.thread_rank() == 0)
        atomicAdd(tv_loss, tv_sum);
#else
    __shared__ float sharedData[256];

    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    sharedData[tid] = tv_sum;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s)
            sharedData[tid] += sharedData[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(tv_loss, sharedData[0]);
#endif
}


void tv_loss_forward(
    const float* bilagrid,
    float* tv_loss,
    int N, int C, int L, int H, int W,
    cudaStream_t stream
) {
    dim3 block = { 4, 4, 4 };
    dim3 bounds = {
        (W +block.x-1)/block.x,
        (H +block.y-1)/block.y,
        // (N*C*L +block.z-1)/block.z
        (N*L +block.z-1)/block.z
    };
    if (C == 12)
        tv_loss_forward_kernel<12><<<bounds, block, 0, stream>>>(
            bilagrid, tv_loss,
            N, L, H, W
        );
    else if (C == 9)
        tv_loss_forward_kernel<9><<<bounds, block, 0, stream>>>(
            bilagrid, tv_loss,
            N, L, H, W
        );
    CHECK_DEVICE_ERROR;
}
