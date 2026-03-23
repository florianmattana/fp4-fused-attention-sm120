#include<cuda_runtime.h>

constexpr int BQ = 64;
constexpr int Bd = 128;
constexpr int MMA_M = 16;          
constexpr int MMA_N = 8;           
constexpr int MMA_K = 32;          
constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 128;
constexpr int BLOCK_ELEMENT = 32;
constexpr int TILE_SIZE = BQ * Bd; 
constexpr int NUM_SCALE_BLOCKS = TILE_SIZE / BLOCK_ELEMENT;
constexpr int ACC_PER_THREAD = 4;  
constexpr int COL_TILES = BQ / MMA_N;
constexpr int K_CHUNKS = Bd / MMA_K;

__device__ uint8_t encode_fp4_e2m1(float val)
{
	 uint8_t sign = (val < 0.f) ? 1 : 0;
	 
	 float abs_val = fabsf(val);

	 uint8_t nibble;
	 if(abs_val >= 5.0f)        nibble = 0x7;
	 else if (abs_val >= 3.5f)  nibble = 0x6;
	 else if (abs_val >= 2.5f)  nibble = 0x5;
	 else if (abs_val >= 1.75f) nibble = 0x4;  
	 else if (abs_val >= 1.25f) nibble = 0x3;  
	 else if (abs_val >= 0.75f) nibble = 0x2;  
	 else if (abs_val >= 0.25f) nibble = 0x1; 
	 else                       nibble = 0x0; 

	 nibble |= (sign << 3);

	 return (uint8_t)(nibble << 2); // shift nibble to bits [5:2] as required by MMA hardware
}


__device__ uint8_t compute_scale_ue8m0(float* block, int size) 
	{
		float max_abs = 0.0f;
		for (int i = 0; i < size; i++) {
			float a = fabsf(block[i]);
			if (a > max_abs) max_abs = a;
		}

		// Pick the smallest power-of-two that covers the max
		int exponent = (int)ceilf(log2f(max_abs));
		uint8_t ue8m0 = (uint8_t)(exponent + 127);

		return ue8m0;
	}

	__global__ void fused_fp4_attention (const float* Q, const float* K, float* S)
	{
		__shared__ float   staging[TILE_SIZE];
		__shared__ uint8_t Q_quant[TILE_SIZE];
		__shared__ uint8_t K_quant[Bd * BQ];
		__shared__ uint8_t Q_scales[TILE_SIZE / BLOCK_ELEMENT];
		__shared__ uint8_t K_scales[Bd * BQ / BLOCK_ELEMENT];

		int tid = threadIdx.x;

		for(int k = 0 ; k < TILE_SIZE ; k += NUM_THREADS)
		{
			int idx = tid + k;
			int g_idx = blockIdx.x * TILE_SIZE + idx;
			staging[idx] = Q[g_idx];
		}
		__syncthreads();

		for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS)
		{
			uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT], BLOCK_ELEMENT);
			Q_scales[i] = scale;

			float scale_f = exp2f((float)(scale - 127));

			for (int j = 0; j < BLOCK_ELEMENT; j++)
			{
				float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
				Q_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
			}
		}

	__syncthreads();

		for(int k = 0 ; k < TILE_SIZE ; k += NUM_THREADS)
			{
				int idx = tid + k;
				int g_idx = 0 + idx;
				staging[idx] = K[g_idx];
			}
		__syncthreads();

		for (int i = tid; i < NUM_SCALE_BLOCKS; i += NUM_THREADS)
		{
			uint8_t scale = compute_scale_ue8m0(&staging[i * BLOCK_ELEMENT], BLOCK_ELEMENT);
			K_scales[i] = scale;

			float scale_f = exp2f((float)(scale - 127));

			for (int j = 0; j < BLOCK_ELEMENT; j++)
			{
				float val = staging[i * BLOCK_ELEMENT + j] / scale_f;
				K_quant[i * BLOCK_ELEMENT + j] = encode_fp4_e2m1(val);
			}
		}
	
	__syncthreads();

		int warp_id = tid / WARP_SIZE;
		int lane = tid % WARP_SIZE;

		for (int col_tile = 0; col_tile < COL_TILES; col_tile++)
		{
			float acc[ACC_PER_THREAD] = {0.f};

			for (int k_chunk = 0; k_chunk < K_CHUNKS; k_chunk++)
			{
				
			}		
		}
	}
