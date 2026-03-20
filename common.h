#include<cuda_runtime.h>

__device__ uint8_t encode_fp4_e2m1(float val)
{
    uint8_t sign = (val < 0.f) ? 1 : 0;
    
    float abs_val = fabsf(val);

    return 0;
}