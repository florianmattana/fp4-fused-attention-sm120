# FP4 Fused Attention for SM120

A fused FP4 attention kernel for consumer Blackwell GPUs (SM120), written entirely in inline PTX.

SM120 lacks `tcgen05.mma` and Tensor Memory (available on SM100 datacenter). This kernel uses warp-level `mma.sync` to implement fused GEMM-softmax-GEMM attention in FP4 E2M1, with the score matrix held in registers between both GEMMs.

## Why this exists

SageAttention3 and FlashInfer both support SM120, but their implementations are built on CUTLASS templates that abstract away the hardware details. If you want to understand how bytes are packed into MMA registers, how scale factors are distributed across lanes, or why the FP4 container format silently reads the wrong value when you shift by one bit, the abstractions do not help.

This project makes every step of the FP4 fused attention pipeline visible at the instruction level. The MMA fragment layout, container format, and scale distribution for SM120 were reverse-engineered empirically since none of it is documented in the PTX ISA.

## Key findings

- On SM120, FP4 E2M1 via `kind::mxf8f6f4` stores each 4-bit value in an 8-bit container (nibble in bits 5-2, with padding). This halves throughput compared to SM100's `kind::mxf4nvf4` which packs two FP4 values per byte.
- `scale_vec::2X` is not available on SM120. Block scaling is limited to `scale_vec::1X` (one UE8M0 scale factor per 32 elements).
- The fragment layout for `mma.sync.aligned.m16n8k32` with FP4 E2M1 on SM120 is undocumented. The A fragment requires 4 registers (not 2), the B fragment requires 2 (not 1). Row assignment is `lane / 4`, K-column group is `lane % 4`. Scale factors are read from lanes where `lane % 4 == 0` (scale A for row0) and `lane % 4 == 1` (scale A for row0+8). Lanes 2 and 3 scale values are ignored by the hardware.
- The MMA instruction is:
  ```
  mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0
  ```

## Status

The kernel is functionally complete:

- S = Q x Kt in FP4 with online softmax and 1/sqrt(d) scaling
- K tile loop (arbitrary seq_k, multiple of 64)
- V accumulation from FP16 staging in shared memory
- Multi-head and batching via template HEAD_DIM
- HEAD_DIM: any multiple of 32 (64, 96, 128, 160, 256)
- Correctness validated at cosine 1.0000 on all configurations

### Benchmark (batch=1, heads=32, seq_q=64, RTX 5070 Ti)

| head_dim | seq_k | kernel ms | TFLOPS | BW GB/s |
|----------|-------|-----------|--------|---------|
| 128 | 128 | 0.057 | 2.34 | 109 |
| 128 | 1024 | 0.443 | 2.43 | 80 |
| 64 | 128 | 0.029 | 2.34 | 109 |
| 64 | 1024 | 0.157 | 3.42 | 113 |

The kernel reaches 2 to 3 TFLOPS, roughly 4 to 5x slower than PyTorch SDPA FP16 on the same hardware. The gap is expected: the kernel quantizes Q and K on the fly inside the main loop. The quantization pipeline (scale computation, FP4 encoding, byte-by-byte shared memory access) dominates the runtime. The Tensor Cores execute 4 QMMA instructions out of ~4,200 total SASS instructions. The full NCU analysis is in the writeup.

## Building

```bash
mkdir build && cd build
cmake ..
make -j
```

Run tests:
```bash
./test_fp4         # isolated MMA tests
./test_attention   # correctness vs CPU reference
./benchmark        # performance numbers
./validate         # correctness + benchmark combined
```

Requires CUDA 12.8+ and an SM120 GPU.

## Full writeup

24 sections, 10,000+ words documenting every bug, wrong assumption, and hardware surprise:

[florianmattana.com/posts/fp4-fused-attention-kernel-sm120](https://florianmattana.com/posts/fp4-fused-attention-kernel-sm120/)

## References

- [gau-nernst/learn-cuda](https://github.com/gau-nernst/learn-cuda/tree/main/09a_block_scaled_mm_sm120) - SM120 block-scaled GEMM
- [SageAttention3](https://github.com/thu-ml/SageAttention) - FP4 attention for SM100/SM120
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) - Instruction reference
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA template library for MMA
- [CCCL issue #8146](https://github.com/NVIDIA/cccl/issues/8146) - Request for cuda::ptx wrappers for warp-level mma.sync on SM120
