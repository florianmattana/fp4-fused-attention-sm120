#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Tile and MMA geometry
constexpr int BQ = 64;
constexpr int Bd = 128;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 32;

// Thread configuration
constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 128;

// Quantization
constexpr int BLOCK_ELEMENT = 32;

// Derived constants
constexpr int ACC_PER_THREAD = 4;
constexpr int TILE_SIZE = BQ * Bd;
constexpr int M_TILES = BQ / MMA_M;
constexpr int N_TILES = BQ / MMA_N;
constexpr int K_TILES = Bd / MMA_K;
constexpr int NUM_SCALE_BLOCKS = TILE_SIZE / BLOCK_ELEMENT;
constexpr int BYTES_PER_REG = 4;

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

__global__ void fused_fp4_attention(const float* Q, const float* K, float* Out, const float* V)
{
    __shared__ float   staging [TILE_SIZE];
    __shared__ uint8_t Q_quant [TILE_SIZE];
    __shared__ uint8_t K_quant [TILE_SIZE];
    __shared__ uint8_t Q_scales[NUM_SCALE_BLOCKS];
    __shared__ uint8_t K_scales[NUM_SCALE_BLOCKS];

    int tid = threadIdx.x;

    // Load Q → SMEM
    for (int k = 0; k < TILE_SIZE; k += NUM_THREADS) {
        int idx   = tid + k;
        int g_idx = blockIdx.x * TILE_SIZE + idx;
        staging[idx] = Q[g_idx];
    }
    __syncthreads();

    // Quantize Q
    for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS) {
        uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT]);
        Q_scales[i] = scale;
        float scale_f = exp2f((float)(scale - 127));
        for (int j = 0; j < BLOCK_ELEMENT; j++) {
            float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
            Q_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
        }
    }
    __syncthreads();

    // Load K → SMEM (reuse staging)
    for (int k = 0; k < TILE_SIZE; k += NUM_THREADS) {
        int idx   = tid + k;
        int g_idx = 0 + idx;  // first K tile only for now
        staging[idx] = K[g_idx];
    }
    __syncthreads();

    // Quantize K
    for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS) {
        uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT]);
        K_scales[i] = scale;
        float scale_f = exp2f((float)(scale - 127));
        for (int j = 0; j < BLOCK_ELEMENT; j++) {
            float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
            K_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
        }
    }
    __syncthreads();

    int warp_id = tid / WARP_SIZE;
    int lane    = tid % WARP_SIZE;

    // Layout MMA m16n8k32 (validé empiriquement) :
    //   D : row = lane/4, col = (lane%4)*2, acc[0,1] → row0, acc[2,3] → row0+8
    //   A : a0=(r0,c), a1=(r0+8,c), a2=(r0,c+16), a3=(r0+8,c+16)  avec c=(lane%4)*4
    //   B : b0=K[k_n*Bd+k_base], b1=K[k_n*Bd+k_base+16]  avec k_n=n_tile*8+lane/4
    //   Scale A : lane%4==0 → row=q_row0, lane%4==1 → row=q_row0+8
    //   Scale B : lane%4==0 → col=k_n
    int q_row0  = warp_id * MMA_M + (lane / 4);
    int q_row1  = q_row0 + 8;
    int out_col = (lane % 4) * 2;

    // softmax state — persiste entre les n_tiles
    float m0 = -INFINITY, l0 = 0.f, O0_c0 = 0.f, O0_c1 = 0.f;
    float m1 = -INFINITY, l1 = 0.f, O1_c0 = 0.f, O1_c1 = 0.f;

    for (int n_tile = 0; n_tile < N_TILES; n_tile++)
    {
        float acc[ACC_PER_THREAD] = {0.f};

        for (int k_tile = 0; k_tile < K_TILES; k_tile++)
        {
            // ── A fragment ──
            int q_col_base = k_tile * MMA_K + (lane % 4) * BYTES_PER_REG;

            int qi0 = q_row0 * Bd + q_col_base;
            int qi1 = q_row1 * Bd + q_col_base;
            int qi2 = q_row0 * Bd + q_col_base + 16;
            int qi3 = q_row1 * Bd + q_col_base + 16;

            auto pack = [&](int base) -> uint32_t {
                return (uint32_t)Q_quant[base]
                     | ((uint32_t)Q_quant[base+1] << 8)
                     | ((uint32_t)Q_quant[base+2] << 16)
                     | ((uint32_t)Q_quant[base+3] << 24);
            };

            uint32_t a0 = pack(qi0);
            uint32_t a1 = pack(qi1);
            uint32_t a2 = pack(qi2);
            uint32_t a3 = pack(qi3);

            // ── B fragment ──
            int k_n      = n_tile * MMA_N + (lane / 4);
            int k_k_base = k_tile * MMA_K + (lane % 4) * BYTES_PER_REG;

            int ki0 = k_n * Bd + k_k_base;
            int ki1 = k_n * Bd + k_k_base + 16;

            auto packK = [&](int base) -> uint32_t {
                return (uint32_t)K_quant[base]
                     | ((uint32_t)K_quant[base+1] << 8)
                     | ((uint32_t)K_quant[base+2] << 16)
                     | ((uint32_t)K_quant[base+3] << 24);
            };

            uint32_t b0 = packK(ki0);
            uint32_t b1 = packK(ki1);

            // ── Scales ──
            // scale A : lane%4==0 → row=q_row0, lane%4==1 → row=q_row1
            int sa_row = ((lane % 4) == 1) ? q_row1 : q_row0;
            uint8_t sa = Q_scales[sa_row * (Bd / BLOCK_ELEMENT) + k_tile];

            // scale B : lane%4==0 → col=k_n (autres ignorées par le hardware)
            uint8_t sb = K_scales[k_n * (Bd / BLOCK_ELEMENT) + k_tile];

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

        // warp reduce — max par ligne (butterfly)
        float local_max0 = fmaxf(acc[0], acc[1]);
        float local_max1 = fmaxf(acc[2], acc[3]);
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xFFFFFFFF, local_max0, 1));
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xFFFFFFFF, local_max0, 2));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xFFFFFFFF, local_max1, 1));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xFFFFFFFF, local_max1, 2));

        // online softmax row0
        float new_m0 = fmaxf(m0, local_max0);
        float alpha0 = expf(m0 - new_m0);
        float e0_c0  = expf(acc[0] - new_m0);
        float e0_c1  = expf(acc[1] - new_m0);

        float local_sum0 = e0_c0 + e0_c1;
        local_sum0 += __shfl_xor_sync(0xFFFFFFFF, local_sum0, 1);
        local_sum0 += __shfl_xor_sync(0xFFFFFFFF, local_sum0, 2);
        float new_l0 = alpha0 * l0 + local_sum0;

        // online softmax row1
        float new_m1 = fmaxf(m1, local_max1);
        float alpha1 = expf(m1 - new_m1);
        float e1_c0  = expf(acc[2] - new_m1);
        float e1_c1  = expf(acc[3] - new_m1);

        float local_sum1 = e1_c0 + e1_c1;
        local_sum1 += __shfl_xor_sync(0xFFFFFFFF, local_sum1, 1);
        local_sum1 += __shfl_xor_sync(0xFFFFFFFF, local_sum1, 2);
        float new_l1 = alpha1 * l1 + local_sum1;

        // accumulation O — chaque lane collecte les e des 4 lanes de son groupe
        float p0_c0 = 0.f, p0_c1 = 0.f;
        float p1_c0 = 0.f, p1_c1 = 0.f;

        for (int i = 0; i < 4; i++) {
            int src     = (lane / 4) * 4 + i;
            int src_col = n_tile * MMA_N + i * 2;
            float se0 = __shfl_sync(0xFFFFFFFF, e0_c0, src);
            float se1 = __shfl_sync(0xFFFFFFFF, e0_c1, src);
            float se2 = __shfl_sync(0xFFFFFFFF, e1_c0, src);
            float se3 = __shfl_sync(0xFFFFFFFF, e1_c1, src);
            p0_c0 += se0 * V[src_col * MMA_N + out_col]
                   + se1 * V[(src_col+1) * MMA_N + out_col];
            p0_c1 += se0 * V[src_col * MMA_N + out_col + 1]
                   + se1 * V[(src_col+1) * MMA_N + out_col + 1];
            p1_c0 += se2 * V[src_col * MMA_N + out_col]
                   + se3 * V[(src_col+1) * MMA_N + out_col];
            p1_c1 += se2 * V[src_col * MMA_N + out_col + 1]
                   + se3 * V[(src_col+1) * MMA_N + out_col + 1];
        }

        O0_c0 = (alpha0 * l0 * O0_c0 + p0_c0) / new_l0;
        O0_c1 = (alpha0 * l0 * O0_c1 + p0_c1) / new_l0;
        m0 = new_m0;
        l0 = new_l0;

        O1_c0 = (alpha1 * l1 * O1_c0 + p1_c0) / new_l1;
        O1_c1 = (alpha1 * l1 * O1_c1 + p1_c1) / new_l1;
        m1 = new_m1;
        l1 = new_l1;
    }

    // écriture finale
    Out[q_row0 * MMA_N + out_col]     = O0_c0;
    Out[q_row0 * MMA_N + out_col + 1] = O0_c1;
    Out[q_row1 * MMA_N + out_col]     = O1_c0;
    Out[q_row1 * MMA_N + out_col + 1] = O1_c1;
}

__global__ void debug_mma(const float* Q, const float* K, float* S_out)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;

    int q_row0 = warp_id * MMA_M + (lane / 4);
    int q_row1 = q_row0 + 8;

    for (int n_tile = 0; n_tile < N_TILES; n_tile++)
    {
        int out_col = n_tile * MMA_N + (lane % 4) * 2;
        float acc[4] = {0.f};

        for (int k_tile = 0; k_tile < K_TILES; k_tile++)
        {
            int q_col_base = k_tile * MMA_K + (lane % 4) * BYTES_PER_REG;

            int qi0 = q_row0 * Bd + q_col_base;
            int qi1 = q_row1 * Bd + q_col_base;
            int qi2 = q_row0 * Bd + q_col_base + 16;
            int qi3 = q_row1 * Bd + q_col_base + 16;

            auto pack = [&](int base) -> uint32_t {
                return encode_fp4_e2m1(Q[base])
                     | (encode_fp4_e2m1(Q[base+1]) << 8)
                     | (encode_fp4_e2m1(Q[base+2]) << 16)
                     | (encode_fp4_e2m1(Q[base+3]) << 24);
            };

            uint32_t a0 = pack(qi0);
            uint32_t a1 = pack(qi1);
            uint32_t a2 = pack(qi2);
            uint32_t a3 = pack(qi3);

            int k_n      = n_tile * MMA_N + (lane / 4);
            int k_k_base = k_tile * MMA_K + (lane % 4) * BYTES_PER_REG;

            int ki0 = k_n * Bd + k_k_base;
            int ki1 = k_n * Bd + k_k_base + 16;

            auto packK = [&](int base) -> uint32_t {
                return encode_fp4_e2m1(K[base])
                     | (encode_fp4_e2m1(K[base+1]) << 8)
                     | (encode_fp4_e2m1(K[base+2]) << 16)
                     | (encode_fp4_e2m1(K[base+3]) << 24);
            };

            uint32_t b0 = packK(ki0);
            uint32_t b1 = packK(ki1);

            uint32_t scale_a = 127u;
            uint32_t scale_b = 127u;

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