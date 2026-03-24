#include<cuda_runtime.h>

// Tile and MMA geometry
constexpr int BQ = 64;              // tokens per block
constexpr int Bd = 128;             // head dimension
constexpr int MMA_M = 16;           // MMA output rows for one warp
constexpr int MMA_N = 8;            // MMA output columns for one warp
constexpr int MMA_K = 32;           // MMA reduction depth for one warp

// Thread configuration
constexpr int WARP_SIZE = 32;       
constexpr int NUM_THREADS = 128;    // threads per block (4 warps)

// Quantization
constexpr int BLOCK_ELEMENT = 32;   // elements per scaling group since we ahve one UE8M0 scale for 32 FP4 values

// Derived constants
constexpr int ACC_PER_THREAD = 4;                           // FP32 accumulators per thread (MMA_M * MMA_N / WARP_SIZE)
constexpr int TILE_SIZE = BQ * Bd;                          // 8192 elements per tile
constexpr int M_TILES = BQ / MMA_M;   // 4 fragments in rows
constexpr int N_TILES = BQ / MMA_N;   // 8 fragments in columns
constexpr int K_TILES = Bd / MMA_K;   // 4 chunks along reduction dimension
constexpr int NUM_SCALE_BLOCKS = TILE_SIZE / BLOCK_ELEMENT; // 256 scaling groups per tile

constexpr int MMA_A_STRIDE = 16; // column stride between a0 and a1 registers in MMA fragment A
constexpr int BYTES_PER_REG = 4; // FP4 values per 32-bit register (8-bit containers)

__device__ uint8_t encode_fp4_e2m1(float val)
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

__device__ uint8_t compute_scale_ue8m0(float* block) 
{
	float max_abs = 0.0f;
	for (int i = 0; i < BLOCK_ELEMENT; i++) 
	{
		float a = fabsf(block[i]);
		if (a > max_abs) max_abs = a;
	}

	int exponent = (int)ceilf(log2f(max_abs));
	uint8_t ue8m0 = (uint8_t)(exponent + 127);
	return ue8m0;
}

__global__ void fused_fp4_attention (const float* Q, const float* K, float* S)
{
	__shared__ float   staging [TILE_SIZE];
	__shared__ uint8_t Q_quant [TILE_SIZE];
	__shared__ uint8_t K_quant [TILE_SIZE];
	__shared__ uint8_t Q_scales[NUM_SCALE_BLOCKS];
	__shared__ uint8_t K_scales[NUM_SCALE_BLOCKS];

	int tid = threadIdx.x;

	//Load Q from VRAM to shared memory
	for(int k = 0 ; k < TILE_SIZE ; k += NUM_THREADS)
	{
		int idx = tid + k;
		int g_idx = blockIdx.x * TILE_SIZE + idx;
		staging[idx] = Q[g_idx];
	}
	__syncthreads();

	//Quantize Q: compute scales then encode FP32 -> FP4
	for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS)
	{
		uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT]);
		Q_scales[i] = scale;
		float scale_f = exp2f((float)(scale - 127));

		for (int j = 0; j < BLOCK_ELEMENT; j++)
		{
			float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
			Q_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
		}
	}
	__syncthreads();

	// Load K from VRAM to shared memory (reuse staging buffer since Q has been quantified. Not need for Q FP32)
	for(int k = 0 ; k < TILE_SIZE ; k += NUM_THREADS)
	{
		int idx = tid + k;
		int g_idx = 0 + idx;  // first K tile only for now
		staging[idx] = K[g_idx];
	}
	__syncthreads();

	// Quantize K: same pattern as Q
	for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS)
	{
		uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT]);
		K_scales[i] = scale;
		float scale_f = exp2f((float)(scale - 127));

		for (int j = 0; j < BLOCK_ELEMENT; j++)
		{
			float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
			K_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
		}
	}
	__syncthreads();

	// MMA: compute S = Q * K^T
	// Each warp owns 16 rows of S, iterates over 8 column positions
	int warp_id = tid / WARP_SIZE;
	int lane = tid % WARP_SIZE;

	for (int n_tile = 0; n_tile < N_TILES; n_tile++)
	{
		float acc[ACC_PER_THREAD] = {0.f};

		for (int k_tile = 0; k_tile < K_TILES; k_tile++)
		{
			//  Load Q fragment into registers a0, a1
			int q_row = warp_id * MMA_M + (lane % 16);
			int q_col = k_tile * MMA_K + (lane / 16) * BYTES_PER_REG;

			int q_idx_0 = q_row * Bd + q_col;
			int q_idx_1 = q_row * Bd + q_col + MMA_A_STRIDE;

			uint32_t a0 = Q_quant[q_idx_0]
						| (Q_quant[q_idx_0 + 1] << 8)
						| (Q_quant[q_idx_0 + 2] << 16)
						| (Q_quant[q_idx_0 + 3] << 24);

			uint32_t a1 = Q_quant[q_idx_1]
						| (Q_quant[q_idx_1 + 1] << 8)
						| (Q_quant[q_idx_1 + 2] << 16)
						| (Q_quant[q_idx_1 + 3] << 24);

			// Load K fragment into register b0
			int k_row = k_tile * MMA_K + (lane % 16);
			int k_col = n_tile * MMA_N + (lane / 16) * BYTES_PER_REG;

			int k_idx_0 = k_row * BQ + k_col;

			uint32_t b0 = K_quant[k_idx_0]
						| (K_quant[k_idx_0 + 1] << 8)
						| (K_quant[k_idx_0 + 2] << 16)
						| (K_quant[k_idx_0 + 3] << 24);

			// Load scale factors
			uint8_t sa = Q_scales[q_row * (Bd / BLOCK_ELEMENT) + k_tile];
			uint8_t sb = K_scales[k_row * (BQ / BLOCK_ELEMENT) + (n_tile * MMA_N) / BLOCK_ELEMENT];

			uint32_t scale_a = sa | (sa << 8) | (sa << 16) | (sa << 24);
			uint32_t scale_b = sb | (sb << 8) | (sb << 16) | (sb << 24);

			// Execute MMA
			asm volatile(
				"mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
				".block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0"
				" {%0, %1, %2, %3},"
				" {%4, %5},"
				" {%6},"
				" {%7, %8, %9, %10},"
				" %11,"
				" %12;"
				: "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
				: "r"(a0), "r"(a1),
				"r"(b0),
				"f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
				"r"(scale_a), "r"(scale_b)
			);
		}

		// TODO: write acc to S or apply softmax
	}
}
