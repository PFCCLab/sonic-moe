import paddle

def wrap_triton_kernel(triton_kernel):
    class WrappedTritonKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, index):
            return paddle.use_compat_guard(enable=True, silent=True)(self.kernel[index])
    return WrappedTritonKernel(triton_kernel)