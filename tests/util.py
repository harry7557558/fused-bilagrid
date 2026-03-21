import pytest
from time import perf_counter

import torch
import numpy as np

from typing import List, Tuple, Callable, Literal


def assert_close(x, y, tol, name: str, reduce: Literal["mean", "max"] = "max"):
    assert x.shape == y.shape, (x.shape, y.shape)
    abserr = torch.amax(torch.abs(y - x)).item()
    if reduce == "mean":
        relerr = torch.abs(y - x).mean() / torch.fmax(torch.abs(x).mean(), torch.abs(y).mean())
    elif reduce == "max":
        relerr = abserr / torch.fmax(torch.abs(x).mean(), torch.abs(y).mean())
    print(f"{name}: abserr = {abserr:.2g}, relerr = {relerr:.2g}")
    assert relerr <= tol


def timeit(fun: Callable, name: str, repeat=20):
    torch.cuda.empty_cache()

    for i in range(2):
        fun()
    torch.cuda.synchronize()

    time0 = perf_counter()
    for i in range(repeat):
        fun()
        torch.cuda.synchronize()
    time1 = perf_counter()

    dt = 1e3 * (time1-time0) / repeat
    print(f"{name}: {dt:.2f} ms")

    return dt


def timeits(funs: List[Tuple[Callable, str]], repeat=20):
    """Less affected by fluctuations"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    names = [name for (fun, name) in funs]
    times = [[] for _ in range(len(funs))]

    def run(warmup=False):
        for idx, (fun, name) in enumerate(funs):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time0 = perf_counter()
            fun()
            torch.cuda.synchronize()
            time1 = perf_counter()
            if not warmup:
                times[idx].append(1e3*(time1-time0))

    for j in range(2):
        for i in range(2):
            run(warmup=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for i in range(repeat):
        run()

    dts = [np.median(time) for time in times]

    for dt, name in zip(dts, names):
        print(f"{name}: {dt:.2f} ms")

    return dts
