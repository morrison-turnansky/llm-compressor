"""
Unit tests for CPU offloading via custom op + functional_call.

Tests the core pattern:
    for k in layers:
        block_k(x)  # compiled with fullgraph=True, no graph breaks

    Where block_k does:
        gpu_params = offload(cpu_params_k)   # CPU -> CUDA via custom op
        return layer_k(gpu_params, x)        # compute on GPU
"""

import torch
import torch.nn as nn
import pytest
from torch._dynamo.testing import CompileCounter
from torch._inductor.utils import fresh_cache
from move_data import OffloadedModule
from compressed_tensors.offload.utils import send_tensors


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.linear(self.norm(x))

class LayeredModel(nn.Module):
    def __init__(self, n_layers, dim, block_cls=Block):
        super().__init__()
        self.layers = nn.ModuleList([block_cls(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# --- Single layer offload ---

def test_single_layer_offload():
    """Single layer: wrap, compile fullgraph, params stay on CPU."""
    layer = nn.Linear(4, 4)
    wrapped = torch.compile(OffloadedModule(layer), fullgraph=True)

    x = torch.randn(2, 4).cuda()
    with fresh_cache():
        out = wrapped(x)

    assert out.device.type == "cuda"
    assert layer.weight.device.type == "cpu"


def test_correctness_vs_cuda_reference():
    """Offloaded compiled output matches standard CUDA forward."""
    ref = nn.Linear(4, 4).cuda()
    layer = nn.Linear(4, 4)
    layer.weight.data = ref.weight.data.cpu()
    layer.bias.data = ref.bias.data.cpu()

    compiled = torch.compile(OffloadedModule(layer), fullgraph=True)
    x = torch.randn(2, 4).cuda()

    with fresh_cache():
        assert torch.allclose(ref(x), compiled(x), atol=1e-5)
    assert layer.weight.device.type == "cpu"


def test_per_layer_compiled_blocks():
    """Each layer independently wrapped + compiled. Python loop drives execution."""
    model = LayeredModel(3, 64).cuda()
    x = torch.randn(2, 64).cuda()
    ref_out = model(x)

    for i, layer in enumerate(model.layers):
        model.layers[i] = torch.compile(
            OffloadedModule(layer), fullgraph=True
        )

    with fresh_cache():
        out = model(x)
    assert torch.allclose(ref_out, out, atol=1e-4)

    for p in model.parameters():
        assert p.device.type == "cpu"


def test_call_blocks_individually():
    """Each compiled block callable directly — no model.forward needed."""
    model = LayeredModel(3, 64)
    x = torch.randn(2, 64)
    ref_out = model(x)

    blocks = [
        torch.compile(OffloadedModule(layer).cuda(), fullgraph=True)
        for layer in model.layers
    ]

    saved_tensors = [x]

    with fresh_cache():
        for i, block in enumerate(blocks):
            gpu_input_tensor = send_tensors(saved_tensors[i], "cuda")
            gpu_layer_output = block(gpu_input_tensor)
            saved_tensors.append(send_tensors(gpu_layer_output, "cpu"))
        
    assert torch.allclose(ref_out, saved_tensors[-1], atol=1e-4)
    for tensor in saved_tensors:
        assert tensor.device.type == "cpu"


def test_each_layer_executes_compiled():
    """Logging backend confirms each compiled layer runs once per forward."""
    execution_log = []

    def logging_backend(gm: torch.fx.GraphModule, example_inputs):
        graph_id = len(execution_log)

        def compiled_fn(*args, **kwargs):
            execution_log.append(graph_id)
            gm.print_readable()
            return gm.forward(*args, **kwargs)
        return compiled_fn

    model = LayeredModel(3, 64)
    for i, layer in enumerate(model.layers):
        model.layers[i] = torch.compile(
            OffloadedModule(layer), backend=logging_backend, fullgraph=True
        )

    x = torch.randn(2, 64).cuda()
    with fresh_cache():
        model(x)

    assert len(execution_log) == 3
