"""
Based on https://github.com/rahul-goel/fused-ssim/blob/main/setup.py
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


# Default fallback architectures
fallback_archs = [
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_120,code=sm_120",
]

nvcc_args = [
    "-O3",
    #"--maxrregcount=32",
    "--use_fast_math",
]
nvcc_args += ["-lineinfo", "--generate-line-info", "--source-in-ptx"]

detected_arch = None

if torch.cuda.is_available():
    try:
        device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(device)
        arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
        
        # Print to multiple outputs
        arch_msg = f"Detected GPU architecture: {arch}"
        print(arch_msg)
        print(arch_msg, file=sys.stderr, flush=True)
        
        nvcc_args.append(f"-arch={arch}")
        detected_arch = arch
    except Exception as e:
        error_msg = f"Failed to detect GPU architecture: {e}. Falling back to multiple architectures."
        print(error_msg)
        print(error_msg, file=sys.stderr, flush=True)
        nvcc_args.extend(fallback_archs)
else:
    cuda_msg = "CUDA not available. Falling back to multiple architectures."
    print(cuda_msg)
    print(cuda_msg, file=sys.stderr, flush=True)
    nvcc_args.extend(fallback_archs)

# Create a custom class that prints the architecture information
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        arch_info = f"Building with GPU architecture: {detected_arch if detected_arch else 'multiple architectures'}"
        print("\n" + "="*50)
        print(arch_info)
        print("="*50 + "\n")
        super().build_extensions()

setup(
    name="fused_bilagrid",
    packages=['fused_bilagrid'],
    ext_modules=[
        CUDAExtension(
            name="fused_bilagrid_cuda",
            sources=[
                "fused_bilagrid/sample_forward.cu",
                "fused_bilagrid/sample_backward.cu",
                # "fused_bilagrid/uniform_sample_forward.cu",
                # "fused_bilagrid/uniform_sample_backward_v1.cu",
                # "fused_bilagrid/uniform_sample_backward_v2.cu",
                "fused_bilagrid/uniform_sample.cu",
                "fused_bilagrid/tv_loss_forward.cu",
                "fused_bilagrid/tv_loss_backward.cu",
                "fused_bilagrid/ext.cpp"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

# Print again at the end of setup.py execution
final_msg = f"Setup completed. NVCC args: {nvcc_args}"
print(final_msg)
