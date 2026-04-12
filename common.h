#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ── Fixed MMA geometry ────────────────────────────────────────────────────────
constexpr int MMA_M         = 16;
constexpr int MMA_N         = 8;
constexpr int MMA_K         = 32;

// ── Thread configuration ──────────────────────────────────────────────────────
constexpr int WARP_SIZE     = 32;
constexpr int NUM_THREADS   = 128;   // 4 warps × 32 threads

// ── Quantization ──────────────────────────────────────────────────────────────
constexpr int BLOCK_ELEMENT = 32;

// ── Fixed Q tile ──────────────────────────────────────────────────────────────
constexpr int BQ            = 64;

// ── FP4 E2M1 encoding ─────────────────────────────────────────────────────────
__device__ inline uint8_t encode_fp4_e2m1(float val)
{
    uint8_t sign = (val < 0.f) ? 1 : 0;
    float abs_val = fabsf(val);
    uint8_t nibble;
    if      (abs_val >= 5.0f)  nibble = 0x7;
    else if (abs_val >= 3.5f)  nibble = 0x6;
    else if (abs_val >= 2.5f)  nibble = 0x5;
    else if (abs_val >= 1.75f) nibble = 0x4;
    else if (abs_val >= 1.25f) nibble = 0x3;
    else if (abs_val >= 0.75f) nibble = 0x2;
    else if (abs_val >= 0.25f) nibble = 0x1;
    else                       nibble = 0x0;
    nibble |= (sign << 3);
    return (uint8_t)(nibble << 2);
}

// ── UE8M0 block scale computation ────────────────────────────────────────────
__device__ inline uint8_t compute_scale_ue8m0(float* block)
{
    float max_abs = 0.0f;
    for (int i = 0; i < BLOCK_ELEMENT; i++) {
        float a = fabsf(block[i]);
        if (a > max_abs) max_abs = a;
    }
    int exponent = (int)ceilf(log2f(max_abs));
    return (uint8_t)(exponent + 127);
}

// ── Fused FP4 attention kernel ────────────────────────────────────────────────
// Computes Out = softmax(Q×Kᵀ / sqrt(HEAD_DIM)) × V
//
// Tensor layout (all row-major):
//   Q   : [batch, heads, seq_q, HEAD_DIM]
//   K   : [batch, heads, seq_k, HEAD_DIM]
//   V   : [batch, heads, seq_k, HEAD_DIM]
//   Out : [batch, heads, seq_q, HEAD_DIM]
//
// Launch: <<<batch * heads, NUM_THREADS>>>
//
// Constraints:
//   HEAD_DIM must be a multiple of MMA_K (32): 64, 96, 128, 160...
//   seq_k must be a multiple of BQ (64)
//
// O accumulator layout per thread:
//   Each thread owns rows (q_row0, q_row1) and HEAD_DIM/4 output column pairs.
//   O_SIZE = HEAD_DIM/4 pairs × 2 rows = HEAD_DIM/2 floats total.
//   For HEAD_DIM=128: 64 floats → comfortable register usage.
template<int HEAD_DIM>
__global__ void fused_fp4_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float*       __restrict__ Out,
    const float* __restrict__ V,
    int seq_q,
    int seq_k,
    int heads)
{
    // ── Compile-time constants ────────────────────────────────────────────────
    constexpr int TILE_SIZE        = BQ * HEAD_DIM;
    constexpr int K_TILES          = HEAD_DIM / MMA_K;    // chunks along head dim
    constexpr int N_TILES          = BQ / MMA_N;          // col tiles of S (= BQ/8)
    constexpr int V_COL_TILES      = HEAD_DIM / MMA_N;    // output col tiles
    constexpr int O_SIZE           = V_COL_TILES * 2;     // floats per row per thread
    constexpr int NUM_SCALE_BLOCKS = TILE_SIZE / BLOCK_ELEMENT;
    constexpr int BYTES_PER_REG    = 4;

    // ── Shared memory ─────────────────────────────────────────────────────────
    __shared__ float   staging [TILE_SIZE];
    __shared__ uint8_t Q_quant [TILE_SIZE];
    __shared__ uint8_t K_quant [TILE_SIZE];
    __shared__ uint8_t Q_scales[NUM_SCALE_BLOCKS];
    __shared__ uint8_t K_scales[NUM_SCALE_BLOCKS];

    // ── Block → (batch, head) mapping ────────────────────────────────────────
    int batch_idx = blockIdx.x / heads;
    int head_idx  = blockIdx.x % heads;

    int q_base   = batch_idx * heads * seq_q * HEAD_DIM + head_idx * seq_q * HEAD_DIM;
    int k_base   = batch_idx * heads * seq_k * HEAD_DIM + head_idx * seq_k * HEAD_DIM;
    int v_base_g = batch_idx * heads * seq_k * HEAD_DIM + head_idx * seq_k * HEAD_DIM;
    int o_base   = batch_idx * heads * seq_q * HEAD_DIM + head_idx * seq_q * HEAD_DIM;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane    = tid % WARP_SIZE;

    // ── Load and quantize Q ───────────────────────────────────────────────────
    for (int k = 0; k < TILE_SIZE; k += NUM_THREADS) {
        int idx   = tid + k;
        staging[idx] = Q[q_base + idx];
    }
    __syncthreads();

    for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS) {
        uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT]);
        Q_scales[i]   = scale;
        float scale_f = exp2f((float)(scale - 127));
        for (int j = 0; j < BLOCK_ELEMENT; j++) {
            float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
            Q_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
        }
    }
    __syncthreads();

    // ── Fragment layout (empirically validated — sections 12-15) ─────────────
    // lane/4  → row base within warp tile (0..7)
    // lane%4  → K-column subgroup and output-column subgroup (0..3)
    // a0=(r0,c), a1=(r0+8,c), a2=(r0,c+16), a3=(r0+8,c+16)
    int q_row0 = warp_id * MMA_M + (lane / 4);
    int q_row1 = q_row0 + 8;

    const float softmax_scale = 1.0f / sqrtf((float)HEAD_DIM);

    // ── O accumulator array (unnormalized) ───────────────────────────────────
    // O0[v_col_tile*2 + 0] = output col v_col_tile*MMA_N + (lane%4)*2     for row0
    // O0[v_col_tile*2 + 1] = output col v_col_tile*MMA_N + (lane%4)*2 + 1 for row0
    float O0[O_SIZE];
    float O1[O_SIZE];
    #pragma unroll
    for (int t = 0; t < O_SIZE; t++) { O0[t] = 0.f; O1[t] = 0.f; }

    // ── Softmax state (scalar — same for all output columns) ─────────────────
    float m0 = -INFINITY, l0 = 0.f;
    float m1 = -INFINITY, l1 = 0.f;

    int num_seq_tiles = seq_k / BQ;

    // ── K tile loop ───────────────────────────────────────────────────────────
    for (int seq_tile = 0; seq_tile < num_seq_tiles; seq_tile++)
    {
        for (int k = 0; k < TILE_SIZE; k += NUM_THREADS) {
            int idx   = tid + k;
            int g_idx = k_base + seq_tile * TILE_SIZE + idx;
            staging[idx] = K[g_idx];
        }
        __syncthreads();

        for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS) {
            uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT]);
            K_scales[i]   = scale;
            float scale_f = exp2f((float)(scale - 127));
            for (int j = 0; j < BLOCK_ELEMENT; j++) {
                float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
                K_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
            }
        }
        __syncthreads();

        // ── MMA: compute S tile ───────────────────────────────────────────────
        for (int n_tile = 0; n_tile < N_TILES; n_tile++)
        {
            float acc[4] = {0.f};

            for (int k_tile = 0; k_tile < K_TILES; k_tile++)
            {
                int q_col_base = k_tile * MMA_K + (lane % 4) * BYTES_PER_REG;
                int qi0 = q_row0 * HEAD_DIM + q_col_base;
                int qi1 = q_row1 * HEAD_DIM + q_col_base;
                int qi2 = q_row0 * HEAD_DIM + q_col_base + 16;
                int qi3 = q_row1 * HEAD_DIM + q_col_base + 16;

                auto pack = [&](int base) -> uint32_t {
                    return (uint32_t)Q_quant[base]
                         | ((uint32_t)Q_quant[base+1] << 8)
                         | ((uint32_t)Q_quant[base+2] << 16)
                         | ((uint32_t)Q_quant[base+3] << 24);
                };

                uint32_t a0 = pack(qi0), a1 = pack(qi1);
                uint32_t a2 = pack(qi2), a3 = pack(qi3);

                int k_n      = n_tile * MMA_N + (lane / 4);
                int k_k_base = k_tile * MMA_K + (lane % 4) * BYTES_PER_REG;
                int ki0 = k_n * HEAD_DIM + k_k_base;
                int ki1 = k_n * HEAD_DIM + k_k_base + 16;

                auto packK = [&](int base) -> uint32_t {
                    return (uint32_t)K_quant[base]
                         | ((uint32_t)K_quant[base+1] << 8)
                         | ((uint32_t)K_quant[base+2] << 16)
                         | ((uint32_t)K_quant[base+3] << 24);
                };

                uint32_t b0 = packK(ki0), b1 = packK(ki1);

                int sa_row = ((lane % 4) == 1) ? q_row1 : q_row0;
                uint8_t sa = Q_scales[sa_row * (HEAD_DIM / BLOCK_ELEMENT) + k_tile];
                uint8_t sb = K_scales[k_n    * (HEAD_DIM / BLOCK_ELEMENT) + k_tile];

                uint32_t scale_a = (uint32_t)sa;
                uint32_t scale_b = (uint32_t)sb;

                asm volatile(
                    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
                    "{%14},{%15,%16}, {%17},{%18,%19};"
                    :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),
                     "r"(b0),"r"(b1),
                     "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]),
                     "r"(scale_a),"h"((short)0),"h"((short)0),
                     "r"(scale_b),"h"((short)0),"h"((short)0)
                );
            }

            // Apply 1/sqrt(d) before softmax
            for (int i = 0; i < 4; i++) acc[i] *= softmax_scale;

            // ── Online softmax row0 ───────────────────────────────────────────
            float local_max0 = fmaxf(acc[0], acc[1]);
            local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xFFFFFFFF, local_max0, 1));
            local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xFFFFFFFF, local_max0, 2));

            float new_m0 = fmaxf(m0, local_max0);
            float alpha0 = expf(m0 - new_m0);
            float e0_c0  = expf(acc[0] - new_m0);
            float e0_c1  = expf(acc[1] - new_m0);

            float local_sum0 = e0_c0 + e0_c1;
            local_sum0 += __shfl_xor_sync(0xFFFFFFFF, local_sum0, 1);
            local_sum0 += __shfl_xor_sync(0xFFFFFFFF, local_sum0, 2);
            float new_l0 = alpha0 * l0 + local_sum0;

            // ── Online softmax row1 ───────────────────────────────────────────
            float local_max1 = fmaxf(acc[2], acc[3]);
            local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xFFFFFFFF, local_max1, 1));
            local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xFFFFFFFF, local_max1, 2));

            float new_m1 = fmaxf(m1, local_max1);
            float alpha1 = expf(m1 - new_m1);
            float e1_c0  = expf(acc[2] - new_m1);
            float e1_c1  = expf(acc[3] - new_m1);

            float local_sum1 = e1_c0 + e1_c1;
            local_sum1 += __shfl_xor_sync(0xFFFFFFFF, local_sum1, 1);
            local_sum1 += __shfl_xor_sync(0xFFFFFFFF, local_sum1, 2);
            float new_l1 = alpha1 * l1 + local_sum1;

            // ── Rescale all O values when m changes ───────────────────────────
            // Must apply to ALL output columns — this is why O is an array
            #pragma unroll
            for (int t = 0; t < O_SIZE; t++) {
                O0[t] *= alpha0;
                O1[t] *= alpha1;
            }

            // ── Fetch softmax weights from lane neighbors (once per n_tile) ───
            // se0/se1 for row0, se2/se3 for row1 — same for ALL output col tiles
            float se0[4], se1[4], se2[4], se3[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int src = (lane / 4) * 4 + i;
                se0[i] = __shfl_sync(0xFFFFFFFF, e0_c0, src);
                se1[i] = __shfl_sync(0xFFFFFFFF, e0_c1, src);
                se2[i] = __shfl_sync(0xFFFFFFFF, e1_c0, src);
                se3[i] = __shfl_sync(0xFFFFFFFF, e1_c1, src);
            }

            // ── V accumulation across ALL output column tiles ─────────────────
            // Each v_col_tile covers 8 output columns (MMA_N).
            // This thread contributes to columns v_col_tile*MMA_N + (lane%4)*2
            // and v_col_tile*MMA_N + (lane%4)*2 + 1
            #pragma unroll
            for (int vt = 0; vt < V_COL_TILES; vt++) {
                int out_c = vt * MMA_N + (lane % 4) * 2;
                float p0_c0 = 0.f, p0_c1 = 0.f;
                float p1_c0 = 0.f, p1_c1 = 0.f;

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int local_tok = n_tile * MMA_N + i * 2;
                    int v_tok     = seq_tile * BQ + local_tok;
                    int vrow0     = v_base_g + v_tok       * HEAD_DIM;
                    int vrow1     = v_base_g + (v_tok + 1) * HEAD_DIM;

                    p0_c0 += se0[i] * V[vrow0 + out_c]     + se1[i] * V[vrow1 + out_c];
                    p0_c1 += se0[i] * V[vrow0 + out_c + 1] + se1[i] * V[vrow1 + out_c + 1];
                    p1_c0 += se2[i] * V[vrow0 + out_c]     + se3[i] * V[vrow1 + out_c];
                    p1_c1 += se2[i] * V[vrow0 + out_c + 1] + se3[i] * V[vrow1 + out_c + 1];
                }

                O0[vt * 2]     += p0_c0;
                O0[vt * 2 + 1] += p0_c1;
                O1[vt * 2]     += p1_c0;
                O1[vt * 2 + 1] += p1_c1;
            }

            m0 = new_m0; l0 = new_l0;
            m1 = new_m1; l1 = new_l1;
        }

        __syncthreads();
    }

    // ── Normalize O by l and write output ─────────────────────────────────────
    int out_base0 = o_base + q_row0 * HEAD_DIM;
    int out_base1 = o_base + q_row1 * HEAD_DIM;

    #pragma unroll
    for (int vt = 0; vt < V_COL_TILES; vt++) {
        int out_c = vt * MMA_N + (lane % 4) * 2;
        Out[out_base0 + out_c]     = O0[vt * 2]     / l0;
        Out[out_base0 + out_c + 1] = O0[vt * 2 + 1] / l0;
        Out[out_base1 + out_c]     = O1[vt * 2]     / l1;
        Out[out_base1 + out_c + 1] = O1[vt * 2 + 1] / l1;
    }
}

// ── Debug kernel ──────────────────────────────────────────────────────────────
__global__ void debug_mma(const float* Q, const float* K, float* S_out)
{
    constexpr int HEAD_DIM      = 128;
    constexpr int K_TILES       = HEAD_DIM / MMA_K;
    constexpr int N_TILES_DBG   = BQ / MMA_N;
    constexpr int BYTES_PER_REG = 4;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int q_row0  = warp_id * MMA_M + (lane / 4);
    int q_row1  = q_row0 + 8;

    for (int n_tile = 0; n_tile < N_TILES_DBG; n_tile++)
    {
        int out_col = n_tile * MMA_N + (lane % 4) * 2;
        float acc[4] = {0.f};

        for (int k_tile = 0; k_tile < K_TILES; k_tile++)
        {
            int q_col_base = k_tile * MMA_K + (lane % 4) * BYTES_PER_REG;
            int qi0 = q_row0 * HEAD_DIM + q_col_base;
            int qi1 = q_row1 * HEAD_DIM + q_col_base;
            int qi2 = q_row0 * HEAD_DIM + q_col_base + 16;
            int qi3 = q_row1 * HEAD_DIM + q_col_base + 16;

            auto pack = [&](int base) -> uint32_t {
                return encode_fp4_e2m1(Q[base])
                     | (encode_fp4_e2m1(Q[base+1]) << 8)
                     | (encode_fp4_e2m1(Q[base+2]) << 16)
                     | (encode_fp4_e2m1(Q[base+3]) << 24);
            };

            uint32_t a0 = pack(qi0), a1 = pack(qi1);
            uint32_t a2 = pack(qi2), a3 = pack(qi3);

            int k_n      = n_tile * MMA_N + (lane / 4);
            int k_k_base = k_tile * MMA_K + (lane % 4) * BYTES_PER_REG;
            int ki0 = k_n * HEAD_DIM + k_k_base;
            int ki1 = k_n * HEAD_DIM + k_k_base + 16;

            auto packK = [&](int base) -> uint32_t {
                return encode_fp4_e2m1(K[base])
                     | (encode_fp4_e2m1(K[base+1]) << 8)
                     | (encode_fp4_e2m1(K[base+2]) << 16)
                     | (encode_fp4_e2m1(K[base+3]) << 24);
            };

            uint32_t b0 = packK(ki0), b1 = packK(ki1);
            uint32_t scale_a = 127u, scale_b = 127u;

            asm volatile(
                "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
                "{%14},{%15,%16}, {%17},{%18,%19};"
                :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),
                 "r"(b0),"r"(b1),
                 "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]),
                 "r"(scale_a),"h"((short)0),"h"((short)0),
                 "r"(scale_b),"h"((short)0),"h"((short)0)
            );
        }

        S_out[q_row0 * BQ + out_col]     = acc[0];
        S_out[q_row0 * BQ + out_col + 1] = acc[1];
        S_out[q_row1 * BQ + out_col]     = acc[2];
        S_out[q_row1 * BQ + out_col + 1] = acc[3];
    }
}