#include "config.h"

#include <cooperative_groups.h>


__global__ void bilagrid_uniform_sample_backward_v1_kernel_bilagrid(
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_bilagrid,  // [N,12,L,H,W]
    int N, int L, int H, int W,
    int m, int h, int w,
    int mult_x, int mult_y
) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    int xi = x_idx / mult_x, xf = x_idx % mult_x;
    int yi = y_idx / mult_y, yf = y_idx % mult_y;
    bool inside = (xi < W && yi < H && idx < (N*L));
    if (!inside && (
        mult_x*mult_y == 1 ||
        (mult_x % blockDim.x != 0 || mult_y % blockDim.y != 0)
    )) return;
    int zi = idx % L; idx /= L;
    int ni = idx;

    // Loop bounds
    float sw = float(w-1)/float(W-1);
    int block_wi0 = max((int)ceil((xi-1)*sw), 0);  // same for all threads in the block
    int block_wi1 = min((int)floor((xi+1)*sw), w-1) + 1;
    float sh = float(h-1)/float(H-1);
    int block_hi0 = max((int)ceil((yi-1)*sh), 0);
    int block_hi1 = min((int)floor((yi+1)*sh), h-1) + 1;
    int x_step = (block_wi1-block_wi0+mult_x-1)/mult_x;
    int y_step = (block_hi1-block_hi0+mult_y-1)/mult_y;

    int wi0 = block_wi0+xf*x_step;
    int hi0 = block_hi0+yf*y_step;
    // int wi1 = min(block_wi0+(xf+1)*x_step, w);
    // int hi1 = min(block_hi0+(yf+1)*y_step, h);
    int wi1 = min(block_wi0+(xf+1)*x_step, block_wi1);
    int hi1 = min(block_hi0+(yf+1)*y_step, block_hi1);

    // Result for each affine mat channel
    float accum[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Loop over all samples for this batch
    if (inside)
    for (int mi = 0; mi < m; ++mi) {
        for (int wi = wi0; wi < wi1; wi++)
        for (int hi = hi0; hi < hi1; hi++) {

            int g_off = (((ni*m + mi)*h + hi)*w + wi)*3;
            float sr = rgb[g_off+0];
            float sg = rgb[g_off+1];
            float sb = rgb[g_off+2];

            float x = (float)wi / (float)(w-1) * (float)(W-1);
            float y = (float)hi / (float)(h-1) * (float)(H-1);
            float z = (kC2G_r * sr + kC2G_g * sg + kC2G_b * sb);
            z = min(max(z, 0.0f), 1.0f) * (float)(L-1);

            int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
            int x1 = min(x0+1, W-1);
            int y1 = min(y0+1, H-1);
            int z1 = min(z0+1, L-1);
            if (zi != z0 && zi != z1) continue;

            float fx = x-x0, fy = y-y0, fz = z-z0;
            float accum_t = 0.0;
            if (zi == z0) {
                if (xi == x0 && yi == y0) accum_t += (1-fx)*(1-fy)*(1-fz);
                if (xi == x1 && yi == y0) accum_t += fx*(1-fy)*(1-fz);
                if (xi == x0 && yi == y1) accum_t += (1-fx)*fy*(1-fz);
                if (xi == x1 && yi == y1) accum_t += fx*fy*(1-fz);
            }
            if (zi == z1) {
                if (xi == x0 && yi == y0) accum_t += (1-fx)*(1-fy)*fz;
                if (xi == x1 && yi == y0) accum_t += fx*(1-fy)*fz;
                if (xi == x0 && yi == y1) accum_t += (1-fx)*fy*fz;
                if (xi == x1 && yi == y1) accum_t += fx*fy*fz;
            }

            float dr = v_output[g_off+0];
            float dg = v_output[g_off+1];
            float db = v_output[g_off+2];

            #pragma unroll
            for (int ci = 0; ci < 12; ci++) {
                int si = ci % 4;
                int di = ci / 4;

                float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
                float gout = (di==0 ? dr : di==1 ? dg : db);
                float grad_weight = r_coeff * gout;

                accum[ci] += accum_t * grad_weight;
            }

        }
    }

    // Write result

    int out_idx_start = ((ni*12*L + zi)*H + yi)*W + xi;
    int out_idx_offset = L*H*W;

    // simply write in this case
    if (mult_x*mult_y == 1) {
        #pragma unroll
        for (int ci = 0; ci < 12; ci++) {
            int out_idx = out_idx_start + ci * out_idx_offset;
            if (isfinite(accum[ci]))
                v_bilagrid[out_idx] = accum[ci];
        }
        return;
    }

    // out_idx can be different for each thread, fall back to global atomicAdd
    if (mult_x % blockDim.x != 0 || mult_y % blockDim.y != 0) {
        #pragma unroll
        for (int ci = 0; ci < 12; ci++) {
            int out_idx = out_idx_start + ci * out_idx_offset;
            if (isfinite(accum[ci]))
                atomicAdd(v_bilagrid + out_idx, accum[ci]);
        }
        return;
    }

    // fast atomicAdd

    __shared__ float sharedData[64];

    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int ci = 0; ci < 12; ci++) {
        int out_idx = out_idx_start + ci * out_idx_offset;

        sharedData[tid] = isfinite(accum[ci]) ? accum[ci] : 0.0f;
        __syncthreads();

        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s)
                sharedData[tid] += sharedData[tid + s];
            __syncthreads();
        }

        if (tid == 0)
            atomicAdd(v_bilagrid + out_idx, sharedData[0]);
    }

}


__global__ void bilagrid_uniform_sample_backward_v1_kernel_rgb(
    const float* __restrict__ bilagrid,  // [N,12,L,H,W]
    const float* __restrict__ rgb,  // [N,m,h,w,3]
    const float* __restrict__ v_output,  // [N,m,h,w,3]
    float* __restrict__ v_rgb,  // [N,m,h,w,3]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * m * h * w;
    if (idx >= total) return;

    int tmp = idx;
    int wi = tmp % w; tmp /= w;
    int hi = tmp % h; tmp /= h;
    int mi = tmp % m; tmp /= m;
    int ni = tmp;

    // input and output colors
    int g_off = (((ni * m + mi) * h + hi) * w + wi) * 3;
    float sr = rgb[g_off+0];
    float sg = rgb[g_off+1];
    float sb = rgb[g_off+2];
    float dr = v_output[g_off+0];
    float dg = v_output[g_off+1];
    float db = v_output[g_off+2]; 
    float vr = 0.0, vg = 0.0, vb = 0.0;

    // grid coords
    float x = (float)wi / (float)(w-1) * (float)(W-1);
    float y = (float)hi / (float)(h-1) * (float)(H-1);
    float z = (kC2G_r * sr + kC2G_g * sg + kC2G_b * sb) * (L-1);
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = min(x0+1, W-1);
    int y1 = min(y0+1, H-1);
    int z1 = z0 + 1;
    z0 = min(max(z0,0), L-1); z1 = min(max(z1,0), L-1);

    // fractional parts
    float fx = x - x0, fy = y - y0, fz = z - z0;
    float w000 = (1-fx)*(1-fy)*(1-fz);
    float w001 = fx*(1-fy)*(1-fz);
    float w010 = (1-fx)*fy*(1-fz);
    float w011 = fx*fy*(1-fz);
    float w100 = (1-fx)*(1-fy)*fz;
    float w101 = fx*(1-fy)*fz;
    float w110 = (1-fx)*fy*fz;
    float w111 = fx*fy*fz;

    // accumulate bilagrid gradient over 12 channels
    #pragma unroll
    for (int si = 0; si < 3; si++) {
        #pragma unroll
        for (int di = 0; di < 3; di++) {
            int ci = 4 * di + si;
            float gout = (di==0 ? dr : di==1 ? dg : db);

            int base = ((ni*12 + ci)*L*H*W);
            float val =
                bilagrid[base+(z0*H+y0)*W+x0] * w000 +
                bilagrid[base+(z0*H+y0)*W+x1] * w001 +
                bilagrid[base+(z0*H+y1)*W+x0] * w010 +
                bilagrid[base+(z0*H+y1)*W+x1] * w011 +
                bilagrid[base+(z1*H+y0)*W+x0] * w100 +
                bilagrid[base+(z1*H+y0)*W+x1] * w101 +
                bilagrid[base+(z1*H+y1)*W+x0] * w110 +
                bilagrid[base+(z1*H+y1)*W+x1] * w111;
            (si == 0 ? vr : si == 1 ? vg : vb) += val * gout;
        }
    }

    // spatial derivatives for coords
    float dwdz[8] = {
        -(1-fx)*(1-fy), -fx*(1-fy),
        -(1-fx)*fy,     -fx*fy,
         (1-fx)*(1-fy),  fx*(1-fy),
         (1-fx)*fy,      fx*fy
    };

    // accumulate gradient into coords (chain through bilagrid values and rgb)
    float gz_grad = 0.f;
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = (corner & 1) ? x1 : x0;
        int yi = (corner & 2) ? y1 : y0;
        int zi = (corner & 4) ? z1 : z0;
        float trilerp = 0.f;
        // gather the corresponding bilagrid value for each of the 12 channels
        #pragma unroll
        for (int ci = 0; ci < 12; ++ci) {
            const float* vol = bilagrid + ((ni*12 + ci)*L*H*W);
            float v = vol[(zi*H + yi)*W + xi];
            int si = ci % 4, di = ci / 4;
            float r_coeff = (si==0 ? sr : si==1 ? sg : si==2 ? sb : 1.f);
            float gout = (di==0 ? dr : di==1 ? dg : db);
            trilerp += v * r_coeff * gout;
        }
        gz_grad += dwdz[corner] * (L-1) * trilerp;
    }
    vr += kC2G_r * gz_grad;
    vg += kC2G_g * gz_grad;
    vb += kC2G_b * gz_grad;
    v_rgb[g_off+0] = isfinite(vr) ? vr : 0.0f;
    v_rgb[g_off+1] = isfinite(vg) ? vg : 0.0f;
    v_rgb[g_off+2] = isfinite(vb) ? vb : 0.0f;
}



void bilagrid_uniform_sample_backward_v1(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size
) {
    // v_bilagrid
    {
        dim3 block = { block_x, block_y, 1 };
    
        int mult_x = (2*w+W)/(block.x*W*target_tile_size);
        int mult_y = (2*h+H)/(block.y*H*target_tile_size);
        if (mult_x * mult_y < 4)
            mult_x = mult_y = 1;
        else {
            mult_x = max(mult_x, 1) * block.x;
            mult_y = max(mult_y, 1) * block.y;
        }
        // printf("mult_x: %d, mult_y: %d\n", mult_x, mult_y);

        dim3 bounds = {
            (W*mult_x +block.x-1)/block.x,
            (H*mult_y +block.y-1)/block.y,
            (N*L +block.z-1)/block.z
        };
        bilagrid_uniform_sample_backward_v1_kernel_bilagrid<<<bounds, block>>>(
            rgb, v_output, v_bilagrid,
            N, L, H, W, m, h, w, mult_x, mult_y
        );
    }

    // v_coords and v_rgb
    {
        int total = N * m * h * w;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        bilagrid_uniform_sample_backward_v1_kernel_rgb<<<blocks, threads>>>(
            bilagrid, rgb, v_output,
            v_rgb,
            N, L, H, W, m, h, w
        );
    }
}
