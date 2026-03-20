#include <cstdio>
#include <cstdint>

 __global__ void test_fp4()
 {
    int lane = threadIdx.x;

    //We pack 4 bytes for each threads of A & B for the first MMA instruction
    uint32_t A[4] = {0x08080808, 0x08080808, 0x08080808, 0x08080808};
    uint32_t B[2] = {0x08080808, 0x08080808};
    //We set our scale factor to one to support UE8M0 instruction
    uint32_t sf_a = {0x7F7F7F7F};
    uint32_t sf_b = {0x7F7F7F7F};
    float acc_D[4] ={0.0f, 0.0f, 0.0f, 0.0f};

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

 int main ()
 {
    test_fp4<<<1,32>>>();
    cudaDeviceSynchronize();
 
    return 0;
 }