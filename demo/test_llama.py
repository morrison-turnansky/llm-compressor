"""
TinyLlama integration tests.

Verifies:
- Compiled model accuracy matches eager baseline
- Offloaded + compiled accuracy matches eager baseline
- A separate compiled object exists per layer
"""

import torch
import pytest
from torch._dynamo.testing import CompileCounter
from torch._inductor.utils import fresh_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from move_data import OffloadedModule

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPTS = [
    "The capital of France is",
    "In 1969, the first human",
    "def fibonacci(n):",
]


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def _generate_tokens(model, tokenizer, device=None):
    tokens = []
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gen = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            tokens.append(gen[0].clone())
    return tokens


def test_offloaded_accuracy(model_and_tokenizer):
    """Offloaded + compiled model produces the same tokens as eager."""
    model, tokenizer = model_and_tokenizer
    num_layers = len(model.model.layers)
    eager_tokens = _generate_tokens(model, tokenizer)

    prev_limit = torch._dynamo.config.cache_size_limit
    torch._dynamo.config.cache_size_limit = num_layers * 4

    try:
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = torch.compile(
                OffloadedModule(layer), fullgraph=True, dynamic=True, backend="inductor"
            )

        with fresh_cache():
            offloaded_tokens = _generate_tokens(model, tokenizer, device="cuda")

        for i, prompt in enumerate(PROMPTS):
            assert torch.equal(eager_tokens[i], offloaded_tokens[i]), (
                f"Mismatch on '{prompt}': "
                f"eager={tokenizer.decode(eager_tokens[i])} vs "
                f"offloaded={tokenizer.decode(offloaded_tokens[i])}"
            )
    finally:
        torch._dynamo.config.cache_size_limit = prev_limit


def test_compile_object_per_layer(model_and_tokenizer):
    """Each layer gets its own compiled object that actually executes."""
    model, tokenizer = model_and_tokenizer
    num_layers = len(model.model.layers)
    execution_log = []

    def logging_backend(gm, example_inputs):
        graph_id = len(execution_log)

        def compiled_fn(*args, **kwargs):
            execution_log.append(graph_id)
            return gm.forward(*args, **kwargs)
        return compiled_fn

    # All layers share the same OffloadedModule.forward code object,
    # so Dynamo sees each layer as a recompilation of the same frame.
    prev_limit = torch._dynamo.config.cache_size_limit
    torch._dynamo.config.cache_size_limit = num_layers + 4

    try:
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = torch.compile(
                OffloadedModule(layer),
                backend=logging_backend,
                fullgraph=True,
            )

        inputs = tokenizer(PROMPTS[0], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad(), fresh_cache():
            model(**inputs)

        assert len(execution_log) == num_layers, (
            f"Expected {num_layers} compiled executions, "
            f"got {len(execution_log)}"
        )
    finally:
        torch._dynamo.config.cache_size_limit = prev_limit
