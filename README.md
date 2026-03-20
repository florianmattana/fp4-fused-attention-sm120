# FP4 Fused Attention for SM120 (RTX 5070 Ti / 5080)

A fused FP4 attention kernel targeting consumer Blackwell GPUs (SM120).

SM120 lacks `tcgen05.mma` (available on SM100/datacenter), so this project uses warp-level `mma.sync` with inline PTX to implement fused GEMM-softmax-GEMM attention in FP4 E2M1.

## Why this exists

There is currently no native FP4 fused attention kernel for SM120 consumer GPUs. Existing implementations (FlashAttention, SageAttention3) target SM100 datacenter hardware. SM120 users are stuck with FP16/FP8 fallbacks that cannot tap the FP4 Tensor Core capability.

## Key findings

- On SM120, FP4 E2M1 via `kind::mxf8f6f4` stores each 4-bit value in an 8-bit container (bits 5-2, with padding). This halves throughput compared to SM100's `kind::mxf4nvf4`.
- `scale_vec::2X` is not available on SM120. Block scaling is limited to `scale_vec::1X` (one UE8M0 scale factor per 32 elements).
- The MMA instruction is `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0`.

## Status

Work in progress. See `tests/` for validated MMA tests.

Full technical writeup: https://florianmattana.com/fp4-fused-attention-kernel-sm120

## Building

```bash
make test

Requires CUDA 12.8+ and an SM120 GPU.

References
gau-nernst/learn-cuda - SM120 block-scaled GEMM
SageAttention3 - FP4 attention for SM100
PTX ISA 9.2 - Instruction reference
CUTLASS - NVIDIA's template library for MMA
