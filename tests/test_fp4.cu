#include <cstdio>
#include <cstdint>

#include<common.h>

__global__ void test_fp4()
{
    int lane = threadIdx.x;

    // Encode 8.0 / 8.0 = 1.0 en FP4
    uint8_t e0 = encode_fp4_e2m1(8.0f / 8.0f);
    uint32_t packed = e0 | (e0 << 8) | (e0 << 16) | (e0 << 24);

    uint32_t A[4] = {packed, packed, packed, packed};
    uint32_t B[2] = {packed, packed};

    // Scale factor A = 8 -> UE8M0 = 3 + 127 = 130
    uint32_t sf_a_byte = 130;
    uint32_t sf_a = sf_a_byte | (sf_a_byte << 8) | (sf_a_byte << 16) | (sf_a_byte << 24);

    // Scale factor B = 1 -> UE8M0 = 127
    uint32_t sf_b = 0x7F7F7F7F;

    float acc_D[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "{%14},{%15,%16}, {%17},{%18,%19};"
        :"=f"(acc_D[0]), "=f"(acc_D[1]), "=f"(acc_D[2]), "=f"(acc_D[3])
        :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
         "r"(B[0]), "r"(B[1]),
         "f"(acc_D[0]), "f"(acc_D[1]), "f"(acc_D[2]), "f"(acc_D[3]),
         "r"(sf_a), "h"((short)0), "h"((short)0),
         "r"(sf_b), "h"((short)0), "h"((short)0)
    );

    printf("lane %2d: %.1f %.1f %.1f %.1f\n", lane, acc_D[0], acc_D[1], acc_D[2], acc_D[3]);
}

__global__ void test_encode ()
{
   float val = 1.2f;
   uint8_t result = encode_fp4_e2m1(val);

   printf("encode(12.0/12.0) = 0x%02X\n", encode_fp4_e2m1(12.0f / 12.0f));
   printf("encode(3.0/12.0)  = 0x%02X\n", encode_fp4_e2m1(3.0f / 12.0f));

}

 int main ()
 {
    test_fp4<<<1,32>>>();
    cudaDeviceSynchronize();
    
    test_encode<<<1,1>>>();
    cudaDeviceSynchronize();
 
    return 0;
 }