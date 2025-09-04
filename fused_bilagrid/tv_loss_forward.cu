#include <cuda_runtime.h>
#include <cooperative_groups.h>

__global__ void tv_loss_forward_kernel(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    float* __restrict__ tv_loss,
    int N, int L, int H, int W
) {
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    // bool inside = (wi < W && hi < H && idx < (L*12*N));
    bool inside = (wi < W && hi < H && idx < (L*N));
    int li = idx % L; idx /= L;
    // int ci = idx % 12; idx /= 12;
    int ni = idx;

    float tv_sum = 0.0f;

    if (inside) {
        #pragma unroll
        for (int ci = 0; ci < 12; ci++) {

        int base = (ni*12+ci)*L*H*W;
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
        tv_sum /= (12*N);
    }

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
}

void tv_loss_forward(
    const float* bilagrid,
    float* tv_loss,
    int N, int L, int H, int W
) {
    dim3 block = { 4, 4, 4 };
    dim3 bounds = {
        (W +block.x-1)/block.x,
        (H +block.y-1)/block.y,
        // (N*12*L +block.z-1)/block.z
        (N*L +block.z-1)/block.z
    };
    tv_loss_forward_kernel<<<bounds, block>>>(
        bilagrid, tv_loss,
        N, L, H, W
    );
}
