import sys
import os
sys.path += [os.path.dirname(os.path.abspath(__file__))]
from util import *

import torch

import torch_impl
import fused_bilagrid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid():

    print("# Test bilagrid")

    N = 3
    grid_shape = (15, 16, 7)
    h, w = 345, 678
    # h, w = 5, 8
    idx = torch.tensor([0]).cuda()

    bilagrid0 = torch_impl.BilateralGrid(N, *grid_shape).cuda()
    bilagrid1 = fused_bilagrid.BilateralGrid(N, *grid_shape).cuda()

    torch.random.manual_seed(42)
    grid_data = torch.randn_like(bilagrid0.grids.data)
    bilagrid0.grids.data = grid_data
    bilagrid1.grids.data = grid_data
    assert_close(bilagrid1.grids, bilagrid0.grids, 0.0, "bilagrid")

    ni = len(idx) if idx is not None else N
    uv = 0.5 + 0.5 * torch.randn((ni, h, w, 2)).cuda()
    rgb = 0.5 + 0.5 * torch.randn((ni, h, w, 3)).cuda()
    uv0 = torch.nn.Parameter(uv.clone())
    rgb0 = torch.nn.Parameter(rgb.clone())
    uv1 = torch.nn.Parameter(uv.clone())
    rgb1 = torch.nn.Parameter(rgb.clone())

    output0 = torch_impl.slice(bilagrid0, uv0, rgb0, idx)['rgb']
    output1 = fused_bilagrid.slice(bilagrid1, uv1, rgb1, idx, compute_coords_grad=True)['rgb']

    assert_close(output1, output0, 1e-5, "output")

    weights = torch.randn_like(output0)
    (weights*output0).mean().backward()
    (weights*output1).mean().backward()

    assert_close(bilagrid1.grids.grad, bilagrid0.grids.grad, 1e-3, "bilagrid.grad")
    assert_close(uv1.grad, uv0.grad, 1e-4, "uv.grad")
    assert_close(rgb1.grad, rgb0.grad, 1e-3, "rgb.grad")
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid_uniform():

    print("# Test uniform bilagrid")

    N = 3
    W, H, L = 15, 16, 7
    # W, H, L = 4, 4, 2
    h, w = 345, 678
    # h, w = 2, 2
    # h, w = 3, 5
    idx = torch.tensor([0]).cuda()

    bilagrid0 = torch_impl.BilateralGrid(N, W, H, L).cuda()
    bilagrid1 = fused_bilagrid.BilateralGrid(N, W, H, L).cuda()

    torch.random.manual_seed(42)
    grid_data = torch.randn_like(bilagrid0.grids.data)
    bilagrid0.grids.data = grid_data
    bilagrid1.grids.data = grid_data
    assert_close(bilagrid1.grids, bilagrid0.grids, 0.0, "bilagrid")

    ni = len(idx) if idx is not None else N

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1.0, h).cuda(),
        torch.linspace(0, 1.0, w).cuda(),
        indexing="ij",
    )
    uv = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(ni, 1, 1, 1)

    rgb = 0.5 + 0.5 * torch.randn((ni, h, w, 3)).cuda()
    rgb0 = torch.nn.Parameter(rgb.clone())
    rgb1 = torch.nn.Parameter(rgb.clone())

    output0 = torch_impl.slice(bilagrid0, uv, rgb0, idx)['rgb']
    output1 = fused_bilagrid.slice(bilagrid1, None, rgb1, idx)['rgb']

    assert_close(output1, output0, 1e-4, "output")

    weights = torch.randn_like(output0)
    loss = (weights*output1).mean()

    weights = torch.randn_like(output0)
    (weights*output0).mean().backward()
    (weights*output1).mean().backward()

    # print(bilagrid1.grids.grad)
    # print(bilagrid0.grids.grad)

    assert_close(bilagrid1.grids.grad, bilagrid0.grids.grad, 1e-3, "bilagrid.grad")
    assert_close(rgb1.grad, rgb0.grad, 1e-3, "rgb.grad")
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid_ppisp_uniform():

    print("# Test uniform bilagrid (ppisp)")

    N = 3
    W, H, L = 15, 16, 7
    # W, H, L = 4, 4, 2
    h, w = 345, 678
    # h, w = 2, 2
    # h, w = 3, 5
    idx = torch.tensor([0]).cuda()

    bilagrid0 = torch_impl.BilateralGrid(N, W, H, L, bilagrid_type="ppisp").cuda()
    bilagrid1 = fused_bilagrid.BilateralGridPPISP(N, W, H, L).cuda()

    torch.random.manual_seed(42)
    grid_data = torch.randn_like(bilagrid0.grids.data)
    bilagrid0.grids.data = grid_data
    bilagrid1.grids.data = grid_data
    assert_close(bilagrid1.grids, bilagrid0.grids, 0.0, "bilagrid")

    ni = len(idx) if idx is not None else N

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1.0, h).cuda(),
        torch.linspace(0, 1.0, w).cuda(),
        indexing="ij",
    )
    uv = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(ni, 1, 1, 1)

    rgb = 0.5 + 0.5 * torch.randn((ni, h, w, 3)).cuda()
    rgb = torch.relu(rgb)
    rgb0 = torch.nn.Parameter(rgb.clone())
    rgb1 = torch.nn.Parameter(rgb.clone())

    output0 = torch_impl.slice(bilagrid0, uv, rgb0, idx)['rgb']
    output1 = fused_bilagrid.fused_bilagrid_ppisp_sample(
        bilagrid1.grids[idx], None, rgb1.unsqueeze(1)
    ).squeeze(1)
    # output0 = torch.clip(output0, 0.0, 1.0)
    # output1 = torch.clip(output1, 0.0, 1.0)

    assert_close(output1, output0, 1e-4, "output")

    weights = torch.randn_like(output0)
    loss = (weights*output1).mean()

    weights = torch.randn_like(output0)
    (weights*output0).mean().backward()
    (weights*output1).mean().backward()

    assert_close(bilagrid1.grids.grad, bilagrid0.grids.grad, 1e-3, "bilagrid.grad")
    assert_close(rgb1.grad, rgb0.grad, 1e-3, "rgb.grad")
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid_loglinear_uniform():

    print("# Test uniform bilagrid (loglinear)")

    N = 3
    W, H, L = 15, 16, 7
    # W, H, L = 4, 4, 2
    h, w = 345, 678
    # h, w = 2, 2
    # h, w = 3, 5
    idx = torch.tensor([0]).cuda()

    bilagrid0 = torch_impl.BilateralGrid(N, W, H, L, bilagrid_type="loglinear").cuda()
    bilagrid1 = fused_bilagrid.BilateralGridLoglinear(N, W, H, L).cuda()

    torch.random.manual_seed(42)
    grid_data = torch.randn_like(bilagrid0.grids.data)
    grid_data[:, 0] *= 0.2
    grid_data[:, 4] *= 0.2
    grid_data[:, 8] *= 0.2
    bilagrid0.grids.data = grid_data
    bilagrid1.grids.data = grid_data
    assert_close(bilagrid1.grids, bilagrid0.grids, 0.0, "bilagrid")

    ni = len(idx) if idx is not None else N

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1.0, h).cuda(),
        torch.linspace(0, 1.0, w).cuda(),
        indexing="ij",
    )
    uv = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(ni, 1, 1, 1)

    rgb = 0.5 + 0.5 * torch.randn((ni, h, w, 3)).cuda()
    rgb0 = torch.nn.Parameter(rgb.clone())
    rgb1 = torch.nn.Parameter(rgb.clone())

    output0 = torch_impl.slice(bilagrid0, uv, rgb0, idx)['rgb']
    output1 = fused_bilagrid.fused_bilagrid_loglinear_sample(
        bilagrid1.grids[idx], None, rgb1.unsqueeze(1)
    ).squeeze(1)

    assert_close(output1, output0, 1e-4, "output")

    weights = torch.randn_like(output0)
    loss = (weights*output1).mean()

    weights = torch.randn_like(output0)
    (weights*output0).mean().backward()
    (weights*output1).mean().backward()

    # print(bilagrid1.grids.grad)
    # print(bilagrid0.grids.grad)

    assert_close(bilagrid1.grids.grad, bilagrid0.grids.grad, 1e-3, "bilagrid.grad")
    assert_close(rgb1.grad, rgb0.grad, 1e-3, "rgb.grad")
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid_depth_uniform():

    print("# Test uniform bilagrid (depth)")

    N = 3
    W, H, L = 15, 16, 7
    # W, H, L = 4, 4, 2
    h, w = 345, 678
    # h, w = 2, 2
    # h, w = 3, 5
    idx = torch.tensor([0]).cuda()

    bilagrid0 = torch_impl.BilateralGrid(N, W, H, L, bilagrid_type="depth").cuda()
    bilagrid1 = fused_bilagrid.BilateralGridDepth(N, W, H, L).cuda()

    torch.random.manual_seed(42)
    grid_data = torch.randn_like(bilagrid0.grids.data)
    grid_data[:, 0] *= 0.2
    bilagrid0.grids.data = grid_data
    bilagrid1.grids.data = grid_data
    assert_close(bilagrid1.grids, bilagrid0.grids, 0.0, "bilagrid")

    ni = len(idx) if idx is not None else N

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1.0, h).cuda(),
        torch.linspace(0, 1.0, w).cuda(),
        indexing="ij",
    )
    uv = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(ni, 1, 1, 1)

    rgb = torch.exp(0.1*torch.randn((ni, h, w, 1))).cuda()
    rgb0 = torch.nn.Parameter(rgb.clone())
    rgb1 = torch.nn.Parameter(rgb.clone())

    output0 = torch_impl.slice(bilagrid0, uv, rgb0, idx)['rgb']
    output1 = fused_bilagrid.fused_bilagrid_depth_sample(
        bilagrid1.grids[idx], None, rgb1.unsqueeze(1)
    ).squeeze(1)

    # import matplotlib.pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(output0[0].detach().cpu().numpy())
    # # ax1.imshow(rgb1[0].detach().cpu().numpy())
    # ax2.imshow(output1[0].detach().cpu().numpy())
    # plt.show()

    assert_close(output1, output0, 1e-4, "output")

    weights = torch.randn_like(output0)
    loss = (weights*output1).mean()

    weights = torch.randn_like(output0)
    (weights*output0).mean().backward()
    (weights*output1).mean().backward()

    # print(bilagrid1.grids.grad)
    # print(bilagrid0.grids.grad)

    # assert_close(bilagrid1.grids.grad[:, 0], bilagrid0.grids.grad[:, 0], 1e-3, "bilagrid.grad[0]")
    # assert_close(bilagrid1.grids.grad[:, 1], bilagrid0.grids.grad[:, 1], 1e-3, "bilagrid.grad[1]")
    assert_close(bilagrid1.grids.grad, bilagrid0.grids.grad, 1e-3, "bilagrid.grad")
    assert_close(rgb1.grad, rgb0.grad, 1e-3, "rgb.grad")
    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_tv_loss():

    print("# Test total variation loss")

    N = 25
    W, H, L = 15, 16, 7

    bilagrid0 = torch_impl.BilateralGrid(N, W, H, L).cuda()
    bilagrid1 = fused_bilagrid.BilateralGrid(N, W, H, L).cuda()

    torch.random.manual_seed(42)
    grid_data = torch.randn_like(bilagrid0.grids.data)
    bilagrid0.grids.data = grid_data
    bilagrid1.grids.data = grid_data
    assert_close(bilagrid1.grids, bilagrid0.grids, 0.0, "bilagrid")

    output0 = bilagrid0.tv_loss()
    output1 = bilagrid1.tv_loss()

    assert_close(output1, output0, 1e-4, "output")

    weight = 1.234
    (weight*output0).backward()
    (weight*output1).backward()

    assert_close(bilagrid1.grids.grad, bilagrid0.grids.grad, 1e-3, "bilagrid.grad")
    print()



if __name__ == "__main__":

    test_bilagrid()
    test_bilagrid_uniform()
    test_bilagrid_ppisp_uniform()
    test_bilagrid_loglinear_uniform()
    test_bilagrid_depth_uniform()
    test_tv_loss()

