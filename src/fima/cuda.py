"""Load the Linux CUDA runtimes installed in the active Python environment."""

import ctypes
import sys
from functools import cache
from importlib.metadata import distribution


@cache
def load_cuda_libraries() -> tuple[ctypes.CDLL, ...]:
    """Preload cuBLAS/cuDNN by absolute path and retain their handles.

    Changing LD_LIBRARY_PATH after Python starts does not update Linux's loader
    search path. Loading these libraries directly lets CTranslate2 find them by
    name later; their own loader paths resolve the companion NVIDIA libraries.
    """
    if sys.platform != "linux":
        return ()

    libraries = (
        ("nvidia-cublas-cu12", "nvidia/cublas/lib/libcublas.so.12"),
        ("nvidia-cudnn-cu12", "nvidia/cudnn/lib/libcudnn.so.9"),
    )
    return tuple(
        ctypes.CDLL(str(distribution(package).locate_file(relative)), mode=ctypes.RTLD_GLOBAL)
        for package, relative in libraries
    )
