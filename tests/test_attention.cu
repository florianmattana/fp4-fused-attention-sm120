#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <common.h>

static constexpr int SEQ  = BQ;
static constexpr int HEAD = Bd;
static constexpr int VCOL = MMA_N;

void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
}

void cpu_softmax_rows(float* S, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        float* row = S + i * cols;
        float m = row[0];
        for (int j = 1; j < cols; j++) m = fmaxf(m, row[j]);
        float sum = 0.f;
        for (int j = 0; j < cols; j++) { row[j] = expf(row[j] - m); sum += row[j]; }
        for (int j = 0; j < cols; j++) row[j] /= sum;
    }
}

void cpu_attention(const float* Q, const float* K, const float* V,
                   float* Out, int seq, int head, int vcol)
{
    float* S  = new float[seq * seq];
    float* Kt = new float[seq * head];
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < head; j++)
            Kt[j*seq + i] = K[i*head + j];
    cpu_matmul(Q, Kt, S, seq, seq, head);
    cpu_softmax_rows(S, seq, seq);
    cpu_matmul(S, V, Out, seq, vcol, seq);
    delete[] S;
    delete[] Kt;
}

float cosine_similarity(const float* a, const float* b, int n)
{
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-8f);
}

void test_attention()
{
    printf("\n=== test fused_fp4_attention ===\n");

    float* Q   = new float[SEQ * HEAD];
    float* K   = new float[SEQ * HEAD];
    float* V   = new float[SEQ * VCOL];
    float* ref = new float[SEQ * VCOL];
    float* out = new float[SEQ * VCOL];

srand(42);
auto rnd = []{ 
    float vals[] = {-1.f, -0.5f, 0.f, 0.5f, 1.f};
    return vals[rand() % 5]; 
};
for (int i = 0; i < SEQ * HEAD; i++) { Q[i] = rnd(); K[i] = rnd(); }
for (int i = 0; i < SEQ * VCOL; i++) V[i] = rnd();

    cpu_attention(Q, K, V, ref, SEQ, HEAD, VCOL);

    float *dQ, *dK, *dV, *dOut;
    cudaMalloc(&dQ,   SEQ * HEAD * sizeof(float));
    cudaMalloc(&dK,   SEQ * HEAD * sizeof(float));
    cudaMalloc(&dV,   SEQ * VCOL * sizeof(float));
    cudaMalloc(&dOut, SEQ * VCOL * sizeof(float));
    cudaMemcpy(dQ, Q, SEQ * HEAD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, SEQ * HEAD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, SEQ * VCOL * sizeof(float), cudaMemcpyHostToDevice);

    fused_fp4_attention<<<1, NUM_THREADS>>>(dQ, dK, dOut, dV);
    cudaDeviceSynchronize();

    cudaMemcpy(out, dOut, SEQ * VCOL * sizeof(float), cudaMemcpyDeviceToHost);

    float cos_sim = cosine_similarity(ref, out, SEQ * VCOL);
    printf("cosine similarity : %.4f\n", cos_sim);
    printf("%s (seuil 0.99)\n", cos_sim >= 0.99f ? "PASS" : "FAIL");

    printf("ref[0..7] : ");
    for (int i = 0; i < 8; i++) printf("%.3f ", ref[i]);
    printf("\n");
    printf("out[0..7] : ");
    for (int i = 0; i < 8; i++) printf("%.3f ", out[i]);
    printf("\n");

    delete[] Q; delete[] K; delete[] V; delete[] ref; delete[] out;
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dOut);
}

int main()
{
    test_attention();
    return 0;
}