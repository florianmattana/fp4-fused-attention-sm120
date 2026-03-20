#include<cuda_runtime.h>

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