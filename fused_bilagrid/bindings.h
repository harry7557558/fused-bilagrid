#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


void bilagrid_sample_forward(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);


void bilagrid_sample_backward(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_coords,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);


void bilagrid_uniform_sample_forward(
    const float* bilagrid,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);


void bilagrid_patched_sample_forward(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    cudaStream_t stream
);

void bilagrid_uniform_sample_backward_v1(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size,
    cudaStream_t stream
);

void bilagrid_patched_sample_backward_v1(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size,
    const int mi_batch_size,
    cudaStream_t stream
);

void bilagrid_uniform_sample_backward_v2(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);

void bilagrid_patched_sample_backward_v2(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    cudaStream_t stream
);


void bilagrid_ppisp_sample_forward(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);

void bilagrid_ppisp_sample_backward(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_coords,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);

void bilagrid_ppisp_packed_sample_forward(
    const float* bilagrid,
    const int64_t* image_indices,
    const float* coords,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int nnz,
    cudaStream_t stream
);

void bilagrid_ppisp_packed_sample_backward(
    const float* bilagrid,
    const int64_t* image_indices,
    const float* coords,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_coords,
    float* v_rgb,
    int N, int L, int H, int W,
    int nnz,
    cudaStream_t stream
);

void bilagrid_ppisp_uniform_sample_forward(
    const float* bilagrid,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);

void bilagrid_ppisp_patched_sample_forward(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    cudaStream_t stream
);

void bilagrid_ppisp_uniform_sample_backward_v1(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size,
    cudaStream_t stream
);

void bilagrid_ppisp_patched_sample_backward_v1(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size,
    const int mi_batch_size,
    cudaStream_t stream
);


void bilagrid_loglinear_uniform_sample_forward(
    const float* bilagrid,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);

void bilagrid_loglinear_patched_sample_forward(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    cudaStream_t stream
);

void bilagrid_loglinear_uniform_sample_backward_v1(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size,
    cudaStream_t stream
);

void bilagrid_loglinear_patched_sample_backward_v1(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size,
    const int mi_batch_size,
    cudaStream_t stream
);


void bilagrid_depth_uniform_sample_forward(
    const float* bilagrid,
    const float* depth,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w,
    cudaStream_t stream
);

void bilagrid_depth_patched_sample_forward(
    const float* bilagrid,
    const float* depth,
    const int* offsets,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    cudaStream_t stream
);

void bilagrid_depth_uniform_sample_backward_v1(
    const float* bilagrid,
    const float* depth,
    const float* v_output,
    float* v_bilagrid,
    float* v_depth,
    int N, int L, int H, int W,
    int m, int h, int w,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size,
    cudaStream_t stream
);

void bilagrid_depth_patched_sample_backward_v1(
    const float* bilagrid,
    const float* depth,
    const int* offsets,
    const float* v_output,
    float* v_bilagrid,
    float* v_depth,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0,
    const unsigned block_x, const unsigned block_y,
    const int target_tile_size,
    const int mi_batch_size,
    cudaStream_t stream
);


void tv_loss_forward(
    const float* input,
    float* tv_loss,
    int N, int C, int L, int H, int W,
    cudaStream_t stream
);

void tv_loss_backward(
    const float* bilagrid,
    const float v_tv_loss,
    float* v_bilagrid,
    int N, int C, int L, int H, int W, bool inplace,
    cudaStream_t stream
);

void channel_mean_forward(
    const float* bilagrid,
    float* channel_mean,
    int N, int C, int L, int H, int W,
    cudaStream_t stream
);

void channel_mean_backward(
    const float* bilagrid,
    const float* v_tv_loss,
    float* v_bilagrid,
    int N, int C, int L, int H, int W, bool inplace,
    cudaStream_t stream
);


torch::Tensor bilagrid_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,12,L,H,W]
    torch::Tensor coords,  // [N,m,h,w,2]
    torch::Tensor rgb  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = coords.size(1), h = coords.size(2), w = coords.size(3);

    auto output = torch::empty({N, m, h, w, 3}, rgb.options());

    bilagrid_sample_forward(
        bilagrid.data_ptr<float>(),
        coords.data_ptr<float>(),
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor>
bilagrid_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,12,L,H,W]
    torch::Tensor coords,  // [N,m,h,w,2]
    torch::Tensor rgb,  // [N,m,h,w,3]
    torch::Tensor v_output,  // [N,m,h,w,3]
    bool compute_coords_grad
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = coords.size(1), h = coords.size(2), w = coords.size(3);

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,12,L,H,W}, opts);
    auto v_rgb = torch::empty({N,m,h,w,3}, opts);

    if (compute_coords_grad) {
        auto v_coords = torch::zeros({N,m,h,w,2}, opts);
        bilagrid_sample_backward(
            bilagrid.data_ptr<float>(),
            coords.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            v_coords.data_ptr<float>(),
            v_rgb.data_ptr<float>(),
            N, L, H, W, m, h, w,
            at::cuda::getCurrentCUDAStream()
        );
        return std::make_tuple(v_bilagrid, v_coords, v_rgb);
    }

    else {
        auto v_coords = std::optional<torch::Tensor>();
        bilagrid_sample_backward(
            bilagrid.data_ptr<float>(),
            coords.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            nullptr,
            v_rgb.data_ptr<float>(),
            N, L, H, W, m, h, w,
            at::cuda::getCurrentCUDAStream()
        );
        return std::make_tuple(v_bilagrid, v_coords, v_rgb);
    }
}

torch::Tensor bilagrid_ppisp_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,9,L,H,W]
    torch::Tensor coords,  // [N,m,h,w,2]
    torch::Tensor rgb  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = coords.size(1), h = coords.size(2), w = coords.size(3);

    auto output = torch::empty({N, m, h, w, 3}, rgb.options());

    bilagrid_ppisp_sample_forward(
        bilagrid.data_ptr<float>(),
        coords.data_ptr<float>(),
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor>
bilagrid_ppisp_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,9,L,H,W]
    torch::Tensor coords,  // [N,m,h,w,2]
    torch::Tensor rgb,  // [N,m,h,w,3]
    torch::Tensor v_output,  // [N,m,h,w,3]
    bool compute_coords_grad
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = coords.size(1), h = coords.size(2), w = coords.size(3);

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,9,L,H,W}, opts);
    auto v_rgb = torch::empty({N,m,h,w,3}, opts);

    if (compute_coords_grad) {
        auto v_coords = torch::zeros({N,m,h,w,2}, opts);
        bilagrid_ppisp_sample_backward(
            bilagrid.data_ptr<float>(),
            coords.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            v_coords.data_ptr<float>(),
            v_rgb.data_ptr<float>(),
            N, L, H, W, m, h, w,
            at::cuda::getCurrentCUDAStream()
        );
        return std::make_tuple(v_bilagrid, v_coords, v_rgb);
    }

    else {
        auto v_coords = std::optional<torch::Tensor>();
        bilagrid_ppisp_sample_backward(
            bilagrid.data_ptr<float>(),
            coords.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            nullptr,
            v_rgb.data_ptr<float>(),
            N, L, H, W, m, h, w,
            at::cuda::getCurrentCUDAStream()
        );
        return std::make_tuple(v_bilagrid, v_coords, v_rgb);
    }
}

torch::Tensor bilagrid_ppisp_packed_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,9,L,H,W]
    torch::Tensor image_indices,  // [nnz]
    torch::Tensor coords,  // [nnz,2]
    torch::Tensor rgb  // [nnz,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int nnz = image_indices.numel();

    auto output = torch::empty_like(rgb);

    bilagrid_ppisp_packed_sample_forward(
        bilagrid.data_ptr<float>(),
        image_indices.data_ptr<int64_t>(),
        coords.data_ptr<float>(),
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, nnz,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor>
bilagrid_ppisp_packed_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,9,L,H,W]
    torch::Tensor image_indices,  // [nnz]
    torch::Tensor coords,  // [nnz,2]
    torch::Tensor rgb,  // [nnz,3]
    torch::Tensor v_output,  // [nnz,3]
    bool compute_coords_grad
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int nnz = image_indices.numel();

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,9,L,H,W}, opts);
    auto v_rgb = torch::empty_like(rgb);

    if (compute_coords_grad) {
        auto v_coords = torch::zeros({nnz,2}, opts);
        bilagrid_ppisp_packed_sample_backward(
            bilagrid.data_ptr<float>(),
            image_indices.data_ptr<int64_t>(),
            coords.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            v_coords.data_ptr<float>(),
            v_rgb.data_ptr<float>(),
            N, L, H, W, nnz,
            at::cuda::getCurrentCUDAStream()
        );
        return std::make_tuple(v_bilagrid, v_coords, v_rgb);
    }

    else {
        auto v_coords = std::optional<torch::Tensor>();
        bilagrid_ppisp_packed_sample_backward(
            bilagrid.data_ptr<float>(),
            image_indices.data_ptr<int64_t>(),
            coords.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            nullptr,
            v_rgb.data_ptr<float>(),
            N, L, H, W, nnz,
            at::cuda::getCurrentCUDAStream()
        );
        return std::make_tuple(v_bilagrid, v_coords, v_rgb);
    }
}



torch::Tensor bilagrid_uniform_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,12,L,H,W]
    torch::Tensor rgb  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto output = torch::empty_like(rgb);

    bilagrid_uniform_sample_forward(
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor>
bilagrid_uniform_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,12,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    torch::Tensor v_output,  // [N,m,h,w,3]
    const int version,
    const int block_x, const int block_y,
    const int target_tile_size
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,12,L,H,W}, opts);
    auto v_rgb = torch::empty({N,m,h,w,3}, opts);

    if (version == 1) {
        // large image: launch from grid and traverse pixels
        bilagrid_uniform_sample_backward_v1(
            bilagrid.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            v_rgb.data_ptr<float>(),
            N, L, H, W, m, h, w,
            (unsigned)block_x, (unsigned)block_y,
            target_tile_size,
            at::cuda::getCurrentCUDAStream()
        );
    }
    else if (version == 2) {
        // small image: launch from pixels and add to grid
        bilagrid_uniform_sample_backward_v2(
            bilagrid.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            v_rgb.data_ptr<float>(),
            N, L, H, W, m, h, w,
            at::cuda::getCurrentCUDAStream()
        );
    }

    return std::make_tuple(v_bilagrid, v_rgb);
}


torch::Tensor bilagrid_ppisp_uniform_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,9,L,H,W]
    torch::Tensor rgb  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto output = torch::empty_like(rgb);

    bilagrid_ppisp_uniform_sample_forward(
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor>
bilagrid_ppisp_uniform_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,9,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    torch::Tensor v_output,  // [N,m,h,w,3]
    const int version,
    const int block_x, const int block_y,
    const int target_tile_size
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,9,L,H,W}, opts);
    auto v_rgb = torch::empty({N,m,h,w,3}, opts);

    if (version == 1 || version == 2) {
        // large image: launch from grid and traverse pixels
        bilagrid_ppisp_uniform_sample_backward_v1(
            bilagrid.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            v_rgb.data_ptr<float>(),
            N, L, H, W, m, h, w,
            (unsigned)block_x, (unsigned)block_y,
            target_tile_size,
            at::cuda::getCurrentCUDAStream()
        );
    }
    else if (version == 2) {
        // small image: launch from pixels and add to grid
        // bilagrid_ppisp_uniform_sample_backward_v2(
        //     bilagrid.data_ptr<float>(),
        //     rgb.data_ptr<float>(),
        //     v_output.data_ptr<float>(),
        //     v_bilagrid.data_ptr<float>(),
        //     v_rgb.data_ptr<float>(),
        //     N, L, H, W, m, h, w,
        //     at::cuda::getCurrentCUDAStream()
        // );
        throw std::runtime_error("Currently only supports version == 1");
    }

    return std::make_tuple(v_bilagrid, v_rgb);
}


torch::Tensor bilagrid_loglinear_uniform_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,9,L,H,W]
    torch::Tensor rgb  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto output = torch::empty_like(rgb);

    bilagrid_loglinear_uniform_sample_forward(
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor>
bilagrid_loglinear_uniform_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,9,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    torch::Tensor v_output,  // [N,m,h,w,3]
    const int version,
    const int block_x, const int block_y,
    const int target_tile_size
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,9,L,H,W}, opts);
    auto v_rgb = torch::empty({N,m,h,w,3}, opts);

    if (version == 1 || version == 2) {
        // large image: launch from grid and traverse pixels
        bilagrid_loglinear_uniform_sample_backward_v1(
            bilagrid.data_ptr<float>(),
            rgb.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            v_rgb.data_ptr<float>(),
            N, L, H, W, m, h, w,
            (unsigned)block_x, (unsigned)block_y,
            target_tile_size,
            at::cuda::getCurrentCUDAStream()
        );
    }
    else if (version == 2) {
        // small image: launch from pixels and add to grid
        // bilagrid_loglinear_uniform_sample_backward_v2(
        //     bilagrid.data_ptr<float>(),
        //     rgb.data_ptr<float>(),
        //     v_output.data_ptr<float>(),
        //     v_bilagrid.data_ptr<float>(),
        //     v_rgb.data_ptr<float>(),
        //     N, L, H, W, m, h, w,
        //     at::cuda::getCurrentCUDAStream()
        // );
        throw std::runtime_error("Currently only supports version == 1");
    }

    return std::make_tuple(v_bilagrid, v_rgb);
}


torch::Tensor bilagrid_depth_uniform_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,2,L,H,W]
    torch::Tensor depth  // [N,m,h,w,1]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = depth.size(1), h = depth.size(2), w = depth.size(3);

    auto output = torch::empty_like(depth);

    bilagrid_depth_uniform_sample_forward(
        bilagrid.data_ptr<float>(),
        depth.data_ptr<float>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor>
bilagrid_depth_uniform_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,2,L,H,W]
    torch::Tensor depth,  // [N,m,h,w,1]
    torch::Tensor v_output,  // [N,m,h,w,1]
    const int version,
    const int block_x, const int block_y,
    const int target_tile_size
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = depth.size(1), h = depth.size(2), w = depth.size(3);

    auto opts = depth.options();
    auto v_bilagrid = torch::zeros({N,2,L,H,W}, opts);
    auto v_depth = torch::empty({N,m,h,w,1}, opts);

    if (version == 1 || version == 2) {
        // large image: launch from grid and traverse pixels
        bilagrid_depth_uniform_sample_backward_v1(
            bilagrid.data_ptr<float>(),
            depth.data_ptr<float>(),
            v_output.data_ptr<float>(),
            v_bilagrid.data_ptr<float>(),
            v_depth.data_ptr<float>(),
            N, L, H, W, m, h, w,
            (unsigned)block_x, (unsigned)block_y,
            target_tile_size,
            at::cuda::getCurrentCUDAStream()
        );
    }
    else if (version == 2) {
        // small image: launch from pixels and add to grid
        // bilagrid_loglinear_uniform_sample_backward_v2(
        //     bilagrid.data_ptr<float>(),
        //     depth.data_ptr<float>(),
        //     v_output.data_ptr<float>(),
        //     v_bilagrid.data_ptr<float>(),
        //     v_depth.data_ptr<float>(),
        //     N, L, H, W, m, h, w,
        //     at::cuda::getCurrentCUDAStream()
        // );
        throw std::runtime_error("Currently only supports version == 1");
    }

    return std::make_tuple(v_bilagrid, v_depth);
}


torch::Tensor bilagrid_patched_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,12,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    int h0, int w0,
    torch::Tensor offsets  // [N,m,2]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto output = torch::empty_like(rgb);

    bilagrid_patched_sample_forward(
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        offsets.data_ptr<int>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w, h0, w0,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor>
bilagrid_patched_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,12,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    int h0, int w0,
    torch::Tensor offsets,  // [N,m,2]
    torch::Tensor v_output  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,12,L,H,W}, opts);
    auto v_rgb = torch::empty({N,m,h,w,3}, opts);

    bilagrid_patched_sample_backward_v2(
    // bilagrid_patched_sample_backward_v1(
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        offsets.data_ptr<int>(),
        v_output.data_ptr<float>(),
        v_bilagrid.data_ptr<float>(),
        v_rgb.data_ptr<float>(),
        N, L, H, W, m, h, w, h0, w0,
        // 8, 8, 4, 1,
        at::cuda::getCurrentCUDAStream()
    );

    return std::make_tuple(v_bilagrid, v_rgb);
}


torch::Tensor bilagrid_ppisp_patched_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,9,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    int h0, int w0,
    torch::Tensor offsets  // [N,m,2]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto output = torch::empty_like(rgb);

    bilagrid_ppisp_patched_sample_forward(
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        offsets.data_ptr<int>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w, h0, w0,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor>
bilagrid_ppisp_patched_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,9,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    int h0, int w0,
    torch::Tensor offsets,  // [N,m,2]
    torch::Tensor v_output  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,9,L,H,W}, opts);
    auto v_rgb = torch::empty({N,m,h,w,3}, opts);

    // bilagrid_ppisp_patched_sample_backward_v2(
    bilagrid_ppisp_patched_sample_backward_v1(  // TODO: v1 is extremely slow for patched sample
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        offsets.data_ptr<int>(),
        v_output.data_ptr<float>(),
        v_bilagrid.data_ptr<float>(),
        v_rgb.data_ptr<float>(),
        N, L, H, W, m, h, w, h0, w0,
        8, 8, 4, 1,
        at::cuda::getCurrentCUDAStream()
    );

    return std::make_tuple(v_bilagrid, v_rgb);
}


torch::Tensor bilagrid_loglinear_patched_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,9,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    int h0, int w0,
    torch::Tensor offsets  // [N,m,2]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto output = torch::empty_like(rgb);

    bilagrid_loglinear_patched_sample_forward(
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        offsets.data_ptr<int>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w, h0, w0,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor>
bilagrid_loglinear_patched_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,9,L,H,W]
    torch::Tensor rgb,  // [N,m,h,w,3]
    int h0, int w0,
    torch::Tensor offsets,  // [N,m,2]
    torch::Tensor v_output  // [N,m,h,w,3]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = rgb.size(1), h = rgb.size(2), w = rgb.size(3);

    auto opts = rgb.options();
    auto v_bilagrid = torch::zeros({N,9,L,H,W}, opts);
    auto v_rgb = torch::empty({N,m,h,w,3}, opts);

    // bilagrid_loglinear_patched_sample_backward_v2(
    bilagrid_loglinear_patched_sample_backward_v1(  // TODO: v1 is extremely slow for patched sample
        bilagrid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        offsets.data_ptr<int>(),
        v_output.data_ptr<float>(),
        v_bilagrid.data_ptr<float>(),
        v_rgb.data_ptr<float>(),
        N, L, H, W, m, h, w, h0, w0,
        8, 8, 4, 1,
        at::cuda::getCurrentCUDAStream()
    );

    return std::make_tuple(v_bilagrid, v_rgb);
}


torch::Tensor bilagrid_depth_patched_sample_forward_tensor(
    torch::Tensor bilagrid, // [N,2,L,H,W]
    torch::Tensor depth,  // [N,m,h,w,1]
    int h0, int w0,
    torch::Tensor offsets  // [N,m,2]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = depth.size(1), h = depth.size(2), w = depth.size(3);

    auto output = torch::empty_like(depth);

    bilagrid_depth_patched_sample_forward(
        bilagrid.data_ptr<float>(),
        depth.data_ptr<float>(),
        offsets.data_ptr<int>(),
        output.data_ptr<float>(),
        N, L, H, W, m, h, w, h0, w0,
        at::cuda::getCurrentCUDAStream()
    );
    
    return output;
}


std::tuple<torch::Tensor, torch::Tensor>
bilagrid_depth_patched_sample_backward_tensor(
    torch::Tensor bilagrid,  // [N,2,L,H,W]
    torch::Tensor depth,  // [N,m,h,w,1]
    int h0, int w0,
    torch::Tensor offsets,  // [N,m,2]
    torch::Tensor v_output  // [N,m,h,w,1]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);
    int m = depth.size(1), h = depth.size(2), w = depth.size(3);

    auto opts = depth.options();
    auto v_bilagrid = torch::zeros({N,2,L,H,W}, opts);
    auto v_depth = torch::empty({N,m,h,w,1}, opts);

    // bilagrid_depth_patched_sample_backward_v2(
    bilagrid_depth_patched_sample_backward_v1(  // TODO: v1 is extremely slow for patched sample
        bilagrid.data_ptr<float>(),
        depth.data_ptr<float>(),
        offsets.data_ptr<int>(),
        v_output.data_ptr<float>(),
        v_bilagrid.data_ptr<float>(),
        v_depth.data_ptr<float>(),
        N, L, H, W, m, h, w, h0, w0,
        8, 8, 4, 1,
        at::cuda::getCurrentCUDAStream()
    );

    return std::make_tuple(v_bilagrid, v_depth);
}


torch::Tensor tv_loss_forward_tensor(
    torch::Tensor bilagrid  // [N,C,L,H,W]
) {
    int N = bilagrid.size(0), C = bilagrid.size(1), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);

    auto tv_loss = torch::zeros({}, bilagrid.options());

    tv_loss_forward(
        bilagrid.data_ptr<float>(),
        tv_loss.data_ptr<float>(),
        N, C, L, H, W,
        at::cuda::getCurrentCUDAStream()
    );
    
    return tv_loss;
}


torch::Tensor tv_loss_backward_tensor(
    torch::Tensor bilagrid,  // [N,C,L,H,W]
    torch::Tensor v_tv_loss  // scalar
) {
    int N = bilagrid.size(0), C = bilagrid.size(1), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);

    auto v_bilagrid = torch::empty_like(bilagrid);

    tv_loss_backward(
        bilagrid.data_ptr<float>(),
        v_tv_loss.item<float>(),
        v_bilagrid.data_ptr<float>(),
        N, C, L, H, W, false,
        at::cuda::getCurrentCUDAStream()
    );

    return v_bilagrid;
}


torch::Tensor tv_loss_backward_inplace_tensor(
    torch::Tensor bilagrid,  // [N,C,L,H,W]
    torch::Tensor v_tv_loss,  // scalar
    torch::Tensor v_bilagrid  // [N,C,L,H,W]
) {
    int N = bilagrid.size(0), C = bilagrid.size(1), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);

    tv_loss_backward(
        bilagrid.data_ptr<float>(),
        v_tv_loss.item<float>(),
        v_bilagrid.data_ptr<float>(),
        N, C, L, H, W, true,
        at::cuda::getCurrentCUDAStream()
    );

    return v_bilagrid;
}


torch::Tensor channel_mean_forward_tensor(
    torch::Tensor bilagrid  // [N,C,L,H,W]
) {
    int N = bilagrid.size(0), C = bilagrid.size(1), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);

    auto channel_mean = torch::zeros({C}, bilagrid.options());

    channel_mean_forward(
        bilagrid.data_ptr<float>(),
        channel_mean.data_ptr<float>(),
        N, C, L, H, W,
        at::cuda::getCurrentCUDAStream()
    );
    
    return channel_mean;
}


torch::Tensor channel_mean_backward_tensor(
    torch::Tensor bilagrid,  // [N,C,L,H,W]
    torch::Tensor v_channel_mean  // scalar
) {
    int N = bilagrid.size(0), C = bilagrid.size(1), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);

    auto v_bilagrid = torch::empty_like(bilagrid);

    channel_mean_backward(
        bilagrid.data_ptr<float>(),
        v_channel_mean.data_ptr<float>(),
        v_bilagrid.data_ptr<float>(),
        N, C, L, H, W, false,
        at::cuda::getCurrentCUDAStream()
    );

    return v_bilagrid;
}


torch::Tensor channel_mean_backward_inplace_tensor(
    torch::Tensor bilagrid,  // [N,C,L,H,W]
    torch::Tensor v_channel_mean,  // scalar
    torch::Tensor v_bilagrid  // [N,C,L,H,W]
) {
    int N = bilagrid.size(0), C = bilagrid.size(1), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);

    channel_mean_backward(
        bilagrid.data_ptr<float>(),
        v_channel_mean.data_ptr<float>(),
        v_bilagrid.data_ptr<float>(),
        N, C, L, H, W, true,
        at::cuda::getCurrentCUDAStream()
    );

    return v_bilagrid;
}
