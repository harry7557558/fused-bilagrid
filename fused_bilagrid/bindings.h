#include <torch/extension.h>


void bilagrid_sample_forward(
    const float* bilagrid,
    const float* coords,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w
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
    int m, int h, int w
);


void bilagrid_uniform_sample_forward(
    const float* bilagrid,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w
);


void bilagrid_patched_sample_forward(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0
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
    const int target_tile_size
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
    const int mi_batch_size
);

void bilagrid_uniform_sample_backward_v2(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w
);

void bilagrid_patched_sample_backward_v2(
    const float* bilagrid,
    const float* rgb,
    const int* offsets,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w, int h0, int w0
);


void tv_loss_forward(
    const float* input,
    float* tv_loss,
    int N, int L, int H, int W
);


void tv_loss_backward(
    const float* bilagrid,
    const float v_tv_loss,
    float* v_bilagrid,
    int N, int L, int H, int W
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
        N, L, H, W, m, h, w
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
            N, L, H, W, m, h, w
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
            N, L, H, W, m, h, w
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
        N, L, H, W, m, h, w
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
            target_tile_size
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
            N, L, H, W, m, h, w
        );
    }

    return std::make_tuple(v_bilagrid, v_rgb);
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
        N, L, H, W, m, h, w, h0, w0
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
        N, L, H, W, m, h, w, h0, w0
        // , 8, 8, 4, 1
    );

    return std::make_tuple(v_bilagrid, v_rgb);
}


torch::Tensor tv_loss_forward_tensor(
    torch::Tensor bilagrid  // [N,12,L,H,W]
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);

    auto tv_loss = torch::zeros({}, bilagrid.options());

    tv_loss_forward(
        bilagrid.data_ptr<float>(),
        tv_loss.data_ptr<float>(),
        N, L, H, W
    );
    
    return tv_loss;
}


torch::Tensor tv_loss_backward_tensor(
    torch::Tensor bilagrid,  // [N,12,L,H,W]
    torch::Tensor v_tv_loss  // scalar
) {
    int N = bilagrid.size(0), L = bilagrid.size(2),
        H = bilagrid.size(3), W = bilagrid.size(4);

    auto v_bilagrid = torch::zeros_like(bilagrid);

    tv_loss_backward(
        bilagrid.data_ptr<float>(),
        v_tv_loss.item<float>(),
        v_bilagrid.data_ptr<float>(),
        N, L, H, W
    );

    return v_bilagrid;
}
