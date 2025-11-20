import sys
import os
sys.path += [os.path.dirname(os.path.abspath(__file__))]
from util import *

import torch
import random

from fused_bilagrid_cuda import (
    bilagrid_uniform_sample_forward,
    bilagrid_uniform_sample_backward,
    bilagrid_patched_sample_forward,
    bilagrid_patched_sample_backward
)
from fused_bilagrid import fused_bilagrid_sample


# def intersection_aabb(aabb1, aabb2):
#     # Assuming aabb is in format [x_min, y_min, x_max, y_max] for 2D
    
#     inter_min_x = max(aabb1[0], aabb2[0])
#     inter_min_y = max(aabb1[1], aabb2[1])

#     inter_max_x = min(aabb1[2], aabb2[2])
#     inter_max_y = min(aabb1[3], aabb2[3])

#     if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
#         return False

#     return True


# def generate_offsets(h0, w0, h, w, n):
#     offsets = []
#     while len(offsets) < n:
#         x = random.randint(0, w0-w-1)
#         y = random.randint(0, h0-h-1)
#         no_intersect = True
#         for (x1, y1) in offsets:
#             if intersection_aabb([x, y, x+w, y+h], [x1, y1, x1+w, y1+h]):
#                 no_intersect = False
#                 break
#         if no_intersect:
#             offsets.append((x, y))



def copy_patch(rgb_0, rgb, x0, y0):
    """
    rgb_0: (B, N, H, W, C)
    rgb:   (B, N, h, w, C)
    x0, y0: (B, N) int tensors, top-left write positions.
    """
    B, N, H, W, C = rgb_0.shape
    _, _, h, w, _ = rgb.shape
    device = rgb_0.device

    # (h, w) relative indices
    yy = torch.arange(h, device=device)[:, None]   # (h,1)
    xx = torch.arange(w, device=device)[None, :]   # (1,w)

    # (B, N, h, w) absolute positions
    Y = y0[..., None, None] + yy                   # broadcast
    X = x0[..., None, None] + xx

    # batch + sample index grids
    b_idx = torch.arange(B, device=device)[:, None, None, None]
    b_idx = b_idx.expand(B, N, h, w)

    n_idx = torch.arange(N, device=device)[None, :, None, None]
    n_idx = n_idx.expand(B, N, h, w)

    # write
    rgb_0[b_idx, n_idx, Y, X] = rgb

def extract_patch(rgb_1, x0, y0, h, w):
    """
    rgb_1: (B, N, H, W, C)
    x0, y0: (B, N) int tensors, top-left read positions.
    h, w: scalar python ints or 0D tensors.
    
    Returns patch of shape (B, N, h, w, C)
    """
    B, N, H, W, C = rgb_1.shape
    device = rgb_1.device
    
    yy = torch.arange(h, device=device)[:, None]    # (h,1)
    xx = torch.arange(w, device=device)[None, :]    # (1,w)

    Y = y0[..., None, None] + yy                    # (B,N,h,w)
    X = x0[..., None, None] + xx

    b_idx = torch.arange(B, device=device)[:, None, None, None]
    b_idx = b_idx.expand(B, N, h, w)

    n_idx = torch.arange(N, device=device)[None, :, None, None]
    n_idx = n_idx.expand(B, N, h, w)

    # read with advanced indexing
    patch = rgb_1[b_idx, n_idx, Y, X]               # (B,N,h,w,C)
    return patch


def bilagrid_patched_sample_generic(bilagrid, rgb, h0, w0, offsets):
    x0, y0 = torch.unbind(offsets, -1)
    # x1 = x0 + rgb.shape[-2]
    # y1 = y0 + rgb.shape[-3]
    rgb_0 = torch.zeros((*rgb.shape[:-3], h0, w0, 3), dtype=rgb.dtype, device=rgb.device)
    # rgb_0[..., y0:y1, x0:x1, :] = rgb
    copy_patch(rgb_0, rgb, x0, y0)
    rgb_1 = fused_bilagrid_sample(bilagrid, None, rgb_0)
    # return rgb_1[..., y0:y1, x0:x1, :]
    return extract_patch(rgb_1, x0, y0, rgb.shape[-3], rgb.shape[-2])




@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid_patched_sample():

    # N, m = 1, 1
    # L, H, W = 4, 8, 8
    # h, w = 45, 59

    N, m = 3, 29
    L, H, W = 5, 7, 15
    h0, w0 = 234, 567
    h, w = 13, 15

    # N, m = 1, 4
    # L, H, W = 2, 4, 4
    # h0, w0 = 234, 567
    # h, w = 13, 15

    torch.random.manual_seed(42)
    random.seed(42)

    bilagrid = torch.randn((N, 12, L, H, W)).cuda()
    rgb = 0.5+0.5*torch.randn((N, m, h, w, 3)).cuda()

    offsets = torch.tensor([[
        (random.randint(0, w0-w-1), random.randint(0, h0-h-1))
        for _1 in range(m)
    ] for _0 in range(N)], dtype=torch.int32).cuda()
    # print(offsets)

    bilagrid = torch.nn.Parameter(bilagrid)
    rgb = torch.nn.Parameter(rgb)

    output = bilagrid_patched_sample_generic(bilagrid, rgb, h0, w0, offsets)
    output.retain_grad()
    output.requires_grad_(True)

    print("# Test patched sample forward")
    output1 = bilagrid_patched_sample_forward(bilagrid, rgb, h0, w0, offsets)
    assert_close(output1, output, 1e-4, "output")
    print()

    weights = torch.randn_like(output)
    loss = (weights*output).mean()
    loss.backward()

    print("# Test patched sample backward")
    v_bilagrid, v_rgb = bilagrid_patched_sample_backward(bilagrid, rgb, h0, w0, offsets, output.grad)
    # print(v_bilagrid)
    # print(bilagrid.grad)
    assert_close(v_bilagrid, bilagrid.grad, 1e-8, "bilagrid.grad")
    assert_close(v_rgb, rgb.grad, 1e-8, "rgb.grad")
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def profile_patched_bilagrid_sample():

    N, m = 1, 128
    L, H, W = 8, 16, 16
    h0, w0 = 640, 480
    h, w = 64, 64

    torch.random.manual_seed(42)
    random.seed(42)

    print("# Profile patched sample")
    print()

    bilagrid = torch.randn((N, 12, L, H, W)).cuda()
    rgb = 0.5+0.5*torch.randn((N, m, h, w, 3)).cuda()

    offsets = torch.tensor([[
        (random.randint(0, w0-w-1), random.randint(0, h0-h-1))
        for _1 in range(m)
    ] for _0 in range(N)], dtype=torch.int32).cuda()

    timeits([
        # (lambda: bilagrid_patched_sample_generic(bilagrid, rgb, h0, w0, offsets), "generic forward"),
        (lambda: bilagrid_patched_sample_forward(bilagrid, rgb, h0, w0, offsets), "fused forward"),
    ])
    print()

    bilagrid = torch.nn.Parameter(bilagrid)
    rgb = torch.nn.Parameter(rgb)

    output = bilagrid_patched_sample_generic(bilagrid, rgb, h0, w0, offsets)
    output.retain_grad()
    output.requires_grad_(True)

    weight = torch.randn_like(output)
    loss = (weight*output).mean()
    loss.backward(retain_graph=True)

    timeits([
        # (lambda: loss.backward(retain_graph=True), "generic backward"),
        (lambda: bilagrid_patched_sample_backward(bilagrid, rgb, h0, w0, offsets, output.grad), "fused backward"),
    ])


if __name__ == "__main__":

    test_bilagrid_patched_sample()
    print()

    profile_patched_bilagrid_sample()
