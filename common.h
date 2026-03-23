#include<cuda_runtime.h>

constexpr int BQ = 64;
constexpr int Bd = 128;

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

	 return (uint8_t)(nibble << 2);
}


__device__ uint8_t compute_scale_ue8m0(float* block, int size) 
	{
		float max_abs = 0.0f;
		for (int i = 0; i < size; i++) {
			float a = fabsf(block[i]);
			if (a > max_abs) max_abs = a;
		}

		int exponent = (int)ceilf(log2f(max_abs));
		uint8_t ue8m0 = (uint8_t)(exponent + 127);

		return ue8m0;
	}

	__global__ void MMA (const float* Q, const float* K, float* S)
	{
		__shared__ float Q_staging [BQ * Bd];
		__shared__ uint8_t Q_quant[BQ][Bd];
		__shared__ uint8_t K_quant[Bd][BQ];
		__shared__ uint8_t Q_scales[BQ * (Bd / 32)];
		__shared__ uint8_t K_scales[Bd * (BQ / 32)];

		int lane_id = threadIdx.x;
		int NUM_THREADS = 128; 

		for(int k = 0 ; k < BQ * Bd ; k += NUM_THREADS)
		{
			int row = (lane_id + k) / Bd;
			int col = (lane_id + k) % Bd;

			int g_row = blockIdx.x * BQ + row;
			int g_col = col;

			Q_staging[row * Bd + col] = Q[g_row * Bd + g_col];

		}
		__syncthreads();
	}