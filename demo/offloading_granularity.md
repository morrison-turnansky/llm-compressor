# Why Layers Are the Wrong Granularity for Offloading

## The core mismatch

Offloading decisions need to happen at the **op level** determined by hardware
constraints, not at the **layer level** determined by model architecture. A
layer is an arbitrary abstraction boundary that rarely aligns with what the
hardware needs.

## Dynamo tracing is incompatible with layer-level tracing

Dynamo operates at the op level. It decomposes modules into ATen ops, tracks
tensor metadata via FakeTensors, and validates the graph by executing ops with
fake inputs. Making a layer opaque to Dynamo (via `allow_in_graph` or similar)
breaks this because:

- Dynamo must propagate shapes through every node in the graph. An opaque layer
  has real parameters that are not FakeTensors, causing a device/type mismatch
  when Dynamo tries to infer output shapes.
- Registering each layer as a custom op with a fake kernel is possible but
  requires per-model boilerplate and generic shape inference, which is
  non-trivial for arbitrary architectures.
- FX symbolic trace avoids this by using Proxy objects instead of FakeTensors.

The fundamental tension: Dynamo wants to see ops so it can reason about shapes
and guards. Layer-level opacity denies it that visibility. The two abstractions
work at different levels and do not compose cleanly.

## Layers are the wrong boundary

Wrapping each layer and offloading its weights as a unit forces offload
boundaries to align with module boundaries. This is wrong in both directions:

**Too coarse.** A single transformer block may contain attention + MLP + norms
with more parameters than fit on the device. You need to split within a layer,
but treating a layer as atomic prevents that.

**Too fine.** Two adjacent small layers (e.g., a projection + norm) might fit
on the device together. Offloading each independently wastes transfer bandwidth
on two small copies when one larger transfer would saturate the bus.

## The right granularity comes from constraints, not architecture

The size of each pipeline stage should be determined by:

- **VRAM budget** -- how much weight memory is available after accounting for
  activations and KV cache
- **Transfer bandwidth** -- PCIe/NVLink throughput determines the minimum
  stage size worth offloading (below some threshold, transfer latency dominates)
- **Compute overlap** -- stages should be sized so that compute on stage N
  overlaps with prefetching of stage N+1

None of these map to "one layer." The optimal stage might span 3 layers, half
a layer, or a layer plus a head -- whatever fills the device without
overflowing.

## Single-node offloading borrows from parallelism

CPU-GPU offloading on a single node is a memory management problem. The CPU is
a slow backing store. The GPU is the only compute device. Data moves between
them to work around VRAM limits, not to distribute work.

But the techniques from multi-device parallelism apply directly to this
single-node problem:

**When a single op does not fit in VRAM**, you need tensor-parallel-style
sharding -- split the weight tensor along one dimension, stream each shard
through the GPU, and accumulate partial results. This is the same partitioning
logic as multi-GPU tensor parallelism, applied to one device with CPU as the
backing store. The scheduler needs op-level visibility to know which matmul to
shard and along which dimension.

**When the activation memory exceeds VRAM**, you need context-parallel-style
splitting -- partition the sequence or batch dimension so only a slice of
activations is live at any time. Again, this requires op-level visibility into
the attention computation to know where to insert the split.

Current offloading implementations only exploit pipeline-style staging --
moving entire layers between CPU and GPU. Tensor-parallel and context-parallel
ideas are untapped at the single-node level, leaving performance on the table.

These are not separate problems. They are the same graph partitioning problem
viewed along different axes (weight dimension, sequence dimension, depth
dimension). All require op-level visibility. Layer-level wrapping hides the
information needed to make any of these decisions.

## What the right approach looks like

1. **Trace** the model to get the op graph (ATen-level, not module-level)
2. **Partition** the op graph into pipeline stages sized by VRAM and bandwidth
   constraints
3. **Schedule** prefetch and eviction to double-buffer across stage boundaries
4. **Compile** each stage independently

This is a graph partitioning and scheduling problem, not a per-layer wrapping
problem. Pipeline stages should be cut by the scheduler based on device
constraints, not by the module hierarchy.
