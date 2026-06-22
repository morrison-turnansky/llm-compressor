"""
CPU offloading for torch.compile via custom op.

Each layer is wrapped so that parameter onloading (CPU→CUDA) and
offloading (CUDA→CPU) happen inside the compiled graph as opaque
custom ops.
"""

import torch
import torch.nn as nn
from torch.library import Library, impl, register_fake, fallthrough_kernel

from compressed_tensors.offload.utils import send_tensors


lib = Library("move_data", "DEF")
lib.define("onload(Tensor x) -> Tensor", tags=[torch.Tag.pt2_compliant_tag])
lib.define("offload(Tensor x) -> Tensor", tags=[torch.Tag.pt2_compliant_tag])


@register_fake("move_data::onload")
def _onload_fake(x):
    return x.new_empty_strided(
        x.size(), x.stride(), device="cuda", dtype=x.dtype
    )


@impl("move_data::onload", "CPU")
def _onload_cpu(x):
    return send_tensors(x, device="cuda", copy=False)


@impl("move_data::onload", "CUDA")
def _onload_cuda(x):
    return x


lib.impl("onload", fallthrough_kernel, "Autograd")


@register_fake("move_data::offload")
def _offload_fake(x):
    return x.new_empty_strided(
        x.size(), x.stride(), device="cpu", dtype=x.dtype
    )


@impl("move_data::offload", "CUDA")
def _offload_cuda(x):
    return send_tensors(x, device="cpu", copy=False)


@impl("move_data::offload", "CPU")
def _offload_cpu(x):
    return x


lib.impl("offload", fallthrough_kernel, "Autograd")


class OffloadedModule(nn.Module):
    """
    Wraps a module so its parameters are stored on CPU and
    onloaded to CUDA via custom op before forward.
    Uses functional_call to avoid mutating parameters.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        if isinstance(module, OffloadedModule):
            raise TypeError("module is already wrapped with OffloadedModule")
        self.module = module
        self.module.to("cpu")

    def forward(self, *args, **kwargs):
        onloaded = {}
        for name, param in self.module.named_parameters():
            onloaded[name] = torch.ops.move_data.onload(param)
        for name, buf in self.module.named_buffers():
            onloaded[name] = torch.ops.move_data.onload(buf)

        result = torch.func.functional_call(
            self.module, onloaded, args, kwargs
        )

        for name in list(onloaded):
            torch.ops.move_data.offload(onloaded.pop(name))

        return result
