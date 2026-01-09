# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import paddle
import inspect
import cutlass.cute

if not (hasattr(paddle.library.CustomOpDef, "__call__") and inspect.isfunction(paddle.library.CustomOpDef.__call__)):
    def __call__(self, *args, **kwargs):
        return getattr(getattr(paddle.ops, self._namespace), self._name)(*args, **kwargs)

    paddle.library.CustomOpDef.__call__ = __call__

def torch_compat_empty(*args, **kwargs):
    if "device" in  kwargs and kwargs["device"] == "cuda":
        del kwargs["device"]
    return paddle.empty(*args, **kwargs)

paddle.compat.proxy._extend_torch_proxy_overrides(
    {
        "torch.empty": paddle.compat.proxy.RawOverriddenAttribute(torch_compat_empty),
    }
)

def cute_tensor_init(
    self,
    tensor,
    assumed_align=None,
):
    # If tensor is already a DLPack object, use it directly
    if hasattr(tensor, "__dlpack_device__") and not hasattr(tensor, "__dlpack__"):
        self._dlpack_data = tensor
    else:
        self._dlpack_data = tensor.__dlpack__(stream=-1)  # Dont sync in cute 4.2.1, this already fixed in 4.3.0
    self._dltensor_wrapper = None
    self._assumed_align = assumed_align
    self._is_dynamic = False
    self._memref_desc = None
    self._dtype = None

cutlass.cute.runtime._Tensor.__init__ = cute_tensor_init

from .count_cumsum import count_cumsum
from .enums import KernelBackendMoE
from .functional import enable_quack_gemm, moe_TC_softmax_topk_layer
from .moe import MoE
