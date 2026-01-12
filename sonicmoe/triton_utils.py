import paddle
import sys

def swap_torch_guard(fn):
    def wrapped_fn(*args, **kwargs):
        if "torch" not in sys.modules:
            return fn(*args, **kwargs)
        torch_module = sys.modules["torch"]
        sys.modules["torch"] = paddle
        try:
            return fn(*args, **kwargs)
        finally:
            sys.modules["torch"] = torch_module

    return wrapped_fn


def wrap_triton_kernel(triton_kernel):
    class WrappedTritonKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, index):
            return swap_torch_guard(self.kernel[index])

        def __getattr__(self, name):
            return getattr(self.kernel, name)

    return WrappedTritonKernel(triton_kernel)